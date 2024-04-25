from .utils import MetricLogger, binary_acc
from einops import rearrange
import torch

def train_one_epoch(model,
                    data_loader, 
                    criterion_ce,
                    epoch, 
                    args, 
                    fp16_scaler, 
                    optimizer, 
                    lr_scheduler,
                    mixup):
    
    model.train()
    metric_logger = MetricLogger(delimiter="  ")
    header = 'TRAINING > Epoch: [{}/{}]'.format(epoch, args.n_epochs)

    for i, (images, labels) in enumerate(metric_logger.log_every(data_loader, args.log_freq, header)):
        
        # images, labels = images.cuda(non_blocking=True), labels.cuda(non_blocking=True).type(torch.float).unsqueeze(1)
        imgs, nparray = images
        imgs, nparray, labels = imgs.cuda(non_blocking=True), nparray.cuda(non_blocking=True), labels.cuda(non_blocking=True).type(torch.float).unsqueeze(1)
       
        if mixup is not None:
            images, labels = mixup(images, labels)

        with torch.cuda.amp.autocast(fp16_scaler is not None):
            # x = model(images)
            x = model(imgs, nparray)
            loss = criterion_ce(x, labels)
            
        optimizer.zero_grad(set_to_none=True)
        fp16_scaler.scale(loss).backward()
        fp16_scaler.step(optimizer)
        fp16_scaler.update()
        if lr_scheduler is not None:
            # print('updating lr')
            # print(lr_scheduler.get_last_lr())
            lr_scheduler.step((epoch-args.frozen_finetune ) + (i / len(data_loader)))

        torch.cuda.synchronize()

        metric_logger.update(loss=loss.item())
        acc = binary_acc(x, labels)
        metric_logger.update(accuracy=acc)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    metric_logger.synchronize_between_processes()
    print("TRAINING > Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def valid_one_epoch(model,
                    data_loader, 
                    criterion_ce, 
                    epoch, 
                    args, 
                    fp16_scaler):
    
    model.eval()
    metric_logger_val = MetricLogger(delimiter="  ")
    header_val = 'Validation > Epoch: [{}/{}]'.format(epoch, args.n_epochs)

    with torch.no_grad():
        for _, (images, labels) in enumerate(metric_logger_val.log_every(data_loader, args.log_freq, header_val)):

            # images, labels = images.cuda(non_blocking=True), labels.cuda(non_blocking=True).type(torch.float).unsqueeze(1)
            imgs, nparray = images
            imgs, nparray, labels = imgs.cuda(non_blocking=True), nparray.cuda(non_blocking=True), labels.cuda(non_blocking=True).type(torch.float).unsqueeze(1)

            with torch.cuda.amp.autocast(fp16_scaler is not None):
                # x = model(images)
                x = model(imgs, nparray)
                loss = criterion_ce(x, labels)

            torch.cuda.synchronize()

            metric_logger_val.update(loss=loss.item())
            acc = binary_acc(x, labels)
            metric_logger_val.update(accuracy=acc)

    metric_logger_val.synchronize_between_processes()
    print("Validation > Averaged stats:", metric_logger_val)
    return {k: meter.global_avg for k, meter in metric_logger_val.meters.items()}