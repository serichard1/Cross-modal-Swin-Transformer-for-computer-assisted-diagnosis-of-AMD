from datetime import date as d
from modules import utils, engine_unimodal, test_metrics_unimodal
from modules.augmentation import BimodalAugm
from modules.dataset import BimodalDataset, collate_fn
from modules.model import build_swin_encoder, UnimodalEncoder, OCTswap, UnimodalEncoderv2
from modules.optimizer import build_optimizer
from os.path import join
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import SubsetRandomSampler
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import v2
import argparse
import json
import os
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim.lr_scheduler as lrs

def get_args_parser():

    parser = argparse.ArgumentParser(
        "Drusen detection from retinal images - LaBRI - 2024",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        add_help=False)

    # training hyperparameters
    parser.add_argument('--model', default='base', type=str,
        help=""" Base or tiny Swinv2 version """)
    parser.add_argument('--img_size', default=256, type=int,
        help=""" Input size for the images """)
    parser.add_argument('--learning_rate', default=5e-5, type=float, 
        help="""Initial value of the learning rate.""")
    parser.add_argument('--weight_decay', default=0.05, type=float, 
        help="""Initial value of the weight decay.""")
    parser.add_argument('--batch_size_per_gpu', default=16, type=int,
        help='Per-GPU batch-size : number of distinct images loaded on one GPU.')
    parser.add_argument('--nkfolds', default=10, type=int,
        help="""Number of splits train/val""")
    parser.add_argument('--n_epochs', default=50, type=int, 
        help='Number of epochs of training.')
    parser.add_argument("--patience",  default=8, type=int,
        help='Number of epochs to wait before stopping the training when validation loss stopped decreasing')
    parser.add_argument('--mixup', default=0, type=int,
        help="""mixup value""")
    parser.add_argument('--mean', default=[0.5, 0.5, 0.5], type=list,
        help="""mean norm aug""")
    parser.add_argument('--std', default=[0.5, 0.5, 0.5], type=list,
        help="""std norm aug""")
    parser.add_argument('--frozen_finetune', default=2, type=int,
        help="""number of epoch with frozen backbone""")
    parser.add_argument('--pos_weight', default=0.90, type=float,
        help="""pos weight BCE""")
    parser.add_argument('--modality', default='oct', type=str,
        help="""Which modality to use (cfp or oct)""")
    parser.add_argument('--tta', default=0, type=int,
        help="""test time augmentations""")
    parser.add_argument('--alienor_finetuning', default=False, action=argparse.BooleanOptionalAction,
        help="""True if using pretrained weights from intdb""")

    # training environment
    parser.add_argument('--use_fp16', default=True, action=argparse.BooleanOptionalAction,
        help="""Whether or not to use half precision for training. Improves training time and memory requirements,
        but can provoke instability and slight decay of performance.""")
    parser.add_argument('--num_workers', default=4, type=int, help='Number of data loading workers per GPU.')
    parser.add_argument('--avoid_fragmentation', default=False, action=argparse.BooleanOptionalAction, help='Whether or not to set a max split size for memory')
    parser.add_argument('--log_freq', default=10, type=int, help='Log frequency')
    parser.add_argument('--distributed', default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument('--data_path', default="data", type=str)
    parser.add_argument('--output_dir', default="./results", type=str, 
        help='Path to save tensorboard logs and model checkpoints during training.')
    parser.add_argument('--seed', default=3407, type=int, 
        help='Seed for random number generation.')
    parser.add_argument("--dist_url", default="env://", type=str, 
        help="""url used to set up distributed training; 
        see https://pytorch.org/docs/stable/distributed.html""")

    return parser


def main(args):
    if args.avoid_fragmentation:
        print('INFO: setting "PYTORCH_CUDA_ALLOC_CONF" to  "max_split_size_mb:32"')
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32"
    print('INFO: cuda available: ', torch.cuda.is_available())

    with open(Path(args.output_dir) / '_args.txt', 'w') as f:
        json.dump(args.__dict__, f, indent=2, default=lambda o: '<not serializable>')
    utils.init_distributed_mode(args)
    utils.fix_random_seeds(args.seed)
    print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
    cudnn.benchmark = True

    transforms_train = BimodalAugm(img_size=args.img_size, mean_norm=args.mean, std_norm=args.std, modality = args.modality)
    transforms_test = BimodalAugm(img_size=args.img_size, test=True, mean_norm=args.mean, std_norm=args.std, modality = args.modality)
    
    train_set = BimodalDataset(join(args.data_path, 'train'), transform=transforms_train, modality=args.modality)
    test_set = BimodalDataset(join(args.data_path, 'test'), transform=transforms_test, modality=args.modality)

    test_loader = DataLoader(
        test_set,
        shuffle=False,
        batch_size=2,
        drop_last=False,
        collate_fn=collate_fn)

    print('INFO: Dataset correctly loaded')
    print("INFO: Training dataset size: ", len(train_set))
    print("INFO: Validation dataset size: ", int(len(train_set)/args.nkfolds))
    print("INFO: Testing dataset size: ", len(test_set))
    classes = train_set.classes
    print("INFO: available classes: ", classes)
    
    dataiter = iter(test_loader)
    (cfp_imgs, nparray), _ = next(dataiter)
    # imgs, _ = next(dataiter)

    print('INFO: cfp batches of shape (batch, channels, height, width): ', cfp_imgs.shape)
    print('INFO: cfp batches of shape (batch, channels, height, width): ', nparray.shape)

    mixup = None
    if args.mixup > 0:
        mixup = v2.MixUp(num_classes=len(classes))

    criterion_ce = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([args.pos_weight]).cuda(non_blocking=True))
    writer = SummaryWriter(join(args.output_dir, 'tensorboard_logs'))
    print("INFO: Losses, scaler and logs writer ready.")
    date = d.today()

    kfold = StratifiedKFold(n_splits=args.nkfolds, shuffle=True, random_state = args.seed)
    print('> Starting Kfolds')

    for fold, (train_ids, valid_ids) in enumerate(kfold.split(train_set, train_set.targets)):
    
        print(f'> Initiating fold number {fold}: ')
        
        train_subsampler = utils.WeightedSubsetRandomSampler(train_ids, train_set.targets, rate=1.0085)
        valid_subsampler = utils.WeightedSubsetRandomSampler(valid_ids, train_set.targets, rate=1.001)

        # train_subsampler = SubsetRandomSampler(train_ids)
        # valid_subsampler = SubsetRandomSampler(valid_ids)

        if args.distributed:
            train_subsampler = utils.DistributedSamplerWrapper(train_subsampler, 
                                                               shuffle=True, 
                                                               num_replicas=dist.get_world_size(), 
                                                               rank=dist.get_rank())
            valid_subsampler  = utils.DistributedSamplerWrapper(valid_subsampler, 
                                                                shuffle=True, 
                                                                num_replicas=dist.get_world_size(), 
                                                                rank=dist.get_rank())

        train_loader = DataLoader(
        train_set,
        sampler=train_subsampler,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory = True,
        drop_last=True,
        collate_fn=collate_fn)
    
        valid_loader = DataLoader(
        train_set,
        sampler=valid_subsampler,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory = True,
        drop_last=True,
        collate_fn=collate_fn)

        encoder = build_swin_encoder(model= args.model, modality=args.modality, drop_path_rate=0.2)
        for param in encoder.parameters():
            param.requires_grad = False
        # encoder.cuda()

        # model = OCTswap()

        model = UnimodalEncoderv2(encoder)
        # if args.alienor_finetuning:
        #     print('loading model fold')
        #     checkpoint = torch.load(f'/home/marcus/Documents/thesis/code/CURTA_CrossSightv3/CrossSight/unimodal_cfp/ckpt_unimodal_FOLD{fold}_2024-04-06.pth', map_location='cpu')
        #     model.load_state_dict(checkpoint)

        model.classi_head.requires_grad_(True)
        # for name, param in model.named_parameters():
        #     print(name, param.requires_grad)

        mem = torch.cuda.mem_get_info()
        model.cuda()

        if fold==0:
            print('INFO: CUDA memory usage before loading model on gpu: free:', utils.GetHumanReadable(mem[0]), ' / total:', utils.GetHumanReadable(mem[1]))
            mem = torch.cuda.mem_get_info()
            print('INFO: CUDA memory usage after loading model: free:', utils.GetHumanReadable(mem[0]), ' / total:', utils.GetHumanReadable(mem[1]))

            n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"number of params: {n_parameters}")
            if hasattr(model, 'flops'):
                flops = model.flops()
                print(f"number of GFLOPs: {flops / 1e9}")

            print("INFO: Model successfully loaded and set on gpu(s)")

        if args.distributed:
            if utils.has_batchnorms(model):
                model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
            model = nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])

        optimizer = build_optimizer(model, args)

        lr_scheduler = None
    
        fp16_scaler = None
        if args.use_fp16:
            fp16_scaler = torch.cuda.amp.GradScaler()

        min_loss = 10000
        # max_acc = 0
        for epoch in range(args.n_epochs):
            if args.distributed:
                train_subsampler.set_epoch(epoch)
                valid_subsampler.set_epoch(epoch)

            train_subsampler.update_weights(epoch)
            valid_subsampler.update_weights(epoch)

            if epoch == args.frozen_finetune:
                print('unfreezing backbone')
                for param in model.parameters():
                    param.requires_grad = True
                optimizer = build_optimizer(model, args)
                # lr_scheduler = lrs.CosineAnnealingWarmRestarts(optimizer, int(args.n_epochs/3), eta_min=5e-6)

            train_stats = engine_unimodal.train_one_epoch(model,
                                                            train_loader, 
                                                            criterion_ce, 
                                                            epoch, 
                                                            args, 
                                                            fp16_scaler, 
                                                            optimizer,
                                                            lr_scheduler,
                                                            mixup)
            
            valid_stats = engine_unimodal.valid_one_epoch(model,
                                                            valid_loader, 
                                                            criterion_ce, 
                                                            epoch, 
                                                            args,
                                                            fp16_scaler)
            
            if valid_stats["loss"] < min_loss:
            # if valid_stats["accuracy"] > max_acc:
                min_loss = valid_stats["loss"]
                # max_acc = valid_stats["accuracy"]
                trigger_times = 0
                utils.save_on_master(model.state_dict(), join(args.output_dir, f'ckpt_unimodal_FOLD{fold}_{date}.pth'))
            else:
                trigger_times += 1

            log_stats_train = {**{f'train_{k}_FOLD{fold}': v for k, v in train_stats.items()},
                    'epoch': epoch}
            log_stats_valid = {**{f'valid_{k}_FOLD{fold}': v for k, v in valid_stats.items()},
                    'epoch': epoch}
            
            if utils.is_main_process():
                with (Path(args.output_dir) / "log.txt").open("a") as f:
                    f.write(json.dumps(log_stats_train) + "\n")
                    f.write(json.dumps(log_stats_valid) + "\n")

            writer.add_scalars(f"unimodal_{date}_FOLD{fold}", {
                                                'Loss/train': train_stats["loss"],
                                                'Loss/validation': valid_stats["loss"],
                                                }, epoch)
            
            writer.add_scalars(f"unimodal_{date}_FOLD{fold}", {
                                                'Acc_cfp/train': train_stats["accuracy"],
                                                'Acc_cfp/validation': valid_stats["accuracy"],
                                                }, epoch)
            
            print(f'LOG: Epoch {epoch}')
            print(f'Train Acc. => {round(train_stats["accuracy"],3)}%', end=' | ')
            print(f'Train Loss => {round(train_stats["loss"],5)}')
            print(f'valid Acc. => {round(valid_stats["accuracy"],3)}%', end=' | ')
            print(f'valid Loss => {round(valid_stats["loss"],5)} (earlystop => {trigger_times}/{args.patience}) \n')
        # 
            if trigger_times >= args.patience:
                print('Early stop !')
                print(f'Best validation loss was {min_loss}')
                break

        print(f'INFO: End of fold {fold}')
        best_ckpt = join(args.output_dir, f'ckpt_unimodal_FOLD{fold}_{date}.pth')
        print(f'INFO: Best weights have been saved in: ', {best_ckpt})
        # utils.save_on_master(model.state_dict(), join(args.output_dir, f'ckpt_unimodal_FOLD{fold}_{date}_endtriggered.pth'))

        print(f'> Evaluating fold {fold}...')
        state_dict = torch.load(best_ckpt)
        # state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)

        raw_output = test_metrics_unimodal.make_inferences(model, test_loader, fp16_scaler, test_set.instances, args)
        log_list_results = {f'test_results_{k}_FOLD{fold}': v for k, v in raw_output.items()}
        if utils.is_main_process():
            with (Path(args.output_dir) / "log_lists_raw_results.txt").open("a") as f:
                f.write(json.dumps(log_list_results) + "\n")
            
        test_metrics_unimodal.export_results(raw_output, ["0_control", "1_drusen"], date, args.output_dir, fold)
        torch.cuda.empty_cache()

    print(f'INFO: End of training, all folds completed')
    print(f'> exiting ...')

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Drusen detection from retinal images - LaBRI - 2024", parents=[get_args_parser()])
    args = parser.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)

