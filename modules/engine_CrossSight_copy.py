import torch
from .utils import MetricLogger, binary_acc

def train_one_epoch(model, 
                    data_loader, 
                    criterion_attn, 
                    criterion_ce,
                    epoch, 
                    n_epochs, 
                    log_freq, 
                    fp16_scaler, 
                    optimizer, 
                    mixup):
    
    model.train()
    metric_logger = MetricLogger(delimiter="  ")
    header = 'TRAINING > Epoch: [{}/{}]'.format(epoch, n_epochs)

    for _, (images, labels) in enumerate(metric_logger.log_every(data_loader, log_freq, header)):
        cfp_imgs, oct_imgs = images
        cfp_imgs, oct_imgs = cfp_imgs.cuda(non_blocking=True).type(torch.float), oct_imgs.cuda(non_blocking=True).type(torch.float)
        labels = labels.cuda(non_blocking=True).type(torch.float).unsqueeze(1)

        if mixup is not None:
            cfp_imgs, labels = mixup(cfp_imgs, labels)
            oct_imgs, labels = mixup(oct_imgs, labels)

        with torch.cuda.amp.autocast(fp16_scaler is not None):
            x, (attn_cfp , attn_oct) = model(cfp_imgs, oct_imgs)
            crossmodal_loss = criterion_attn(attn_cfp, attn_oct, labels)
            loss_ce = criterion_ce(x, labels)
            loss = (crossmodal_loss + loss_ce) / 2

        optimizer.zero_grad(set_to_none=True)
        fp16_scaler.scale(loss).backward()
        fp16_scaler.step(optimizer)
        fp16_scaler.update()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss.item())
        metric_logger.update(crossmodal_loss=crossmodal_loss.item())
        metric_logger.update(loss_ce=loss_ce.item())

        acc = binary_acc(x, labels)
        metric_logger.update(accuracy=acc)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    metric_logger.synchronize_between_processes()
    print("TRAINING > Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def valid_one_epoch(model, 
                    data_loader, 
                    criterion_attn, 
                    criterion_ce, 
                    epoch, 
                    n_epochs, 
                    log_freq, 
                    fp16_scaler):
    
    model.eval()
    metric_logger_val = MetricLogger(delimiter="  ")
    header_val = 'Validation > Epoch: [{}/{}]'.format(epoch, n_epochs)

    with torch.no_grad():
        for _, (images, labels) in enumerate(metric_logger_val.log_every(data_loader, log_freq, header_val)):
            cfp_imgs, oct_imgs = images
            cfp_imgs, oct_imgs = cfp_imgs.cuda(non_blocking=True).type(torch.float), oct_imgs.cuda(non_blocking=True).type(torch.float)
            labels = labels.cuda(non_blocking=True).type(torch.float).unsqueeze(1)

            with torch.cuda.amp.autocast(fp16_scaler is not None):
                x, (attn_cfp , attn_oct) = model(cfp_imgs, oct_imgs)
                crossmodal_loss = criterion_attn(attn_cfp, attn_oct, labels)
                loss_ce = criterion_ce(x, labels)
                loss = (crossmodal_loss + loss_ce) / 2

            torch.cuda.synchronize()

            metric_logger_val.update(loss=loss.item())
            metric_logger_val.update(crossmodal_loss=crossmodal_loss.item())
            metric_logger_val.update(loss_ce=loss_ce.item())
            
            acc = binary_acc(x, labels)
            metric_logger_val.update(accuracy=acc)

    metric_logger_val.synchronize_between_processes()
    print("Validation > Averaged stats:", metric_logger_val)
    return {k: meter.global_avg for k, meter in metric_logger_val.meters.items()}