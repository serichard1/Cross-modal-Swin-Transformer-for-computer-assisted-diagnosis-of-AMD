from csv import DictWriter
from os.path import join, isfile
from sklearn.metrics import confusion_matrix
from torchvision import transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
import sklearn.metrics as skm
import torch

def make_inferences(model, data_loader, fp16_scaler, instances, args):

    model.eval()
    y_pred, y_true, y_score = [], [], []

    if args.tta > 0:
        transform_tta = transforms.v2.Compose([
                            transforms.v2.RandomHorizontalFlip(p=0.4),
                            transforms.v2.RandomVerticalFlip(p=0.4),
                            transforms.v2.RandomApply([transforms.v2.RandomRotation(degrees=(-180,180))], p=0.3),
                            transforms.v2.RandomResizedCrop(
                                        size=(args.img_size, args.img_size),  
                                        scale=(0.80, 1.0), antialias=True),
                            transforms.v2.RandomApply([transforms.v2.ColorJitter(0.1, 0.1, 0.1, 0.1)], p=0.3),
                            transforms.v2.RandomGrayscale(p=0.1),
                        ])

    with torch.no_grad():
        for _, (images, labels) in enumerate(tqdm(data_loader, desc="inferences on test set")):
            # images, labels = images.cuda(non_blocking=True), labels.cuda(non_blocking=True).type(torch.float).unsqueeze(1)
            imgs, nparray = images
            imgs, nparray, labels = imgs.cuda(non_blocking=True), nparray.cuda(non_blocking=True), labels.cuda(non_blocking=True).type(torch.float).unsqueeze(1)


            with torch.cuda.amp.autocast(fp16_scaler is not None):
                # x = model(images).unsqueeze(0)
                x = model(imgs, nparray).unsqueeze(0)
                if args.tta > 0:
                    for i in range(args.tta):
                        xtta = model(transform_tta(images))
                        x = torch.cat((x, xtta.unsqueeze(0)))
                    # print(x)
                    x  = torch.mean(x, dim=0)

            torch.cuda.synchronize()

            sc = torch.sigmoid(x)
            # print(sc)
            y_score.extend(sc.squeeze().tolist())
            y_pred.extend(torch.round(sc).squeeze().tolist())
            y_true.extend(labels.squeeze().tolist())
            # print(y_score)

    classes = [ins.split('/')[0] for ins in instances]
    img_names = [ins.split('/')[1] for ins in instances]

    raw_output = {  'classes': classes,
                    'img_names': img_names,
                    'y_pred': y_pred,
                    'y_true': y_true,
                    'y_score': y_score,
                    }

    return raw_output

def export_results(raw_output, classes, date, output_dir, fold):

    output_dir = join(output_dir, f'FOLD_{fold}')
    os.mkdir(output_dir)

    df_global = pd.DataFrame.from_dict(raw_output)
    df_global.to_csv(join(output_dir, f'dataframe_output_unimodal_FOLD{fold}_{date}.csv'))

    df_alienor = df_global[df_global['classes'].str.contains("bimod")]
    df_intdb = df_global[df_global['classes'].str.contains("intdb")]

    dfs = {'global': df_global, 'alienor': df_alienor, 'intdb': df_intdb}
    # dfs = {'global': df_global}

    for tag, df in dfs.items():
        y_pred, y_true, y_score = df['y_pred'], df['y_true'], df['y_score']
        save_csv_metrics(get_metrics_skm(y_pred, y_true, y_score, tag), output_dir, date)
   
        save_conf_matrix(y_pred, y_true, classes, output_dir, fold, date, tag)
        save_rocauc(y_true, y_score, output_dir, fold, date, tag)


def get_metrics_skm(y_pred, y_true, y_score, tag):
    return  {
            'tag': tag,
            'f1score': skm.f1_score(y_pred, y_true),
            'accuracy': skm.accuracy_score(y_pred, y_true),
            'precision': skm.precision_score(y_pred, y_true),
            'recall': skm.recall_score(y_pred, y_true),
            'kappa': skm.cohen_kappa_score(y_pred, y_true),
            'auc': skm.roc_auc_score(y_true, y_score)
            }

def save_conf_matrix(y_pred, y_true, classes, output_dir, fold, date, tag):
    cf_matrix = confusion_matrix(y_true, y_pred, normalize="true")
    df_cm = pd.DataFrame(cf_matrix/np.sum(cf_matrix) * len(classes), index = [i for i in classes],
                        columns = [i for i in classes])
    plt.rcParams['figure.figsize'] = [15, 11]
    sns.set(font_scale=1.4)
    sns.heatmap(df_cm, annot=True)
    plt.savefig(join(output_dir, f'confusion_testset_unimodal_FOLD{fold}_{date}_{tag}.png'), dpi=300)
    plt.clf()

def save_csv_metrics(metrics, output_dir, date):
    csv_name = join(output_dir, f'metricsCSV_unimodal_{date}.csv')
    exists = isfile(csv_name)
    with open(csv_name, 'a+') as f:
        header = list(metrics.keys())
        writer = DictWriter(f, delimiter=',', lineterminator='\n',fieldnames=header)

        if not exists:
            writer.writeheader()

        writer.writerow(metrics)
        f.close()

def save_rocauc(y_true, y_score, output_dir, fold, date, tag):
    rand = [0 for _ in range(len(y_true))]
    rand_fpr, rand_tpr, _ = skm.roc_curve(y_true, rand)
    lr_fpr, lr_tpr, _ = skm.roc_curve(y_true, y_score)

    plt.plot(rand_fpr, rand_tpr, linestyle='--', label='No predictive power')
    plt.plot(lr_fpr, lr_tpr, marker='.', label=f'unimodal')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.savefig(join(output_dir, f'ROCAUC_unimodal_FOLD{fold}_{date}_{tag}.png'), dpi=300)
    plt.clf()
