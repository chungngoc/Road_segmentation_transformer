import torch
import torch.nn as nn

from tqdm import tqdm
from utils.utils import draw_translucent_seg_maps
from utils.metrics import IOUEval

def train(
        model,
        train_dataloader,
        optimizer,
        classes_to_train,
        device
):
    print("Training")
    model.train()
    train_running_loss = 0.0
    prog_bar = tqdm(
        train_dataloader,
        total = len(train_dataloader),
        bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}'
    )
    count = 0 # keep track of batch count
    num_classes = len(classes_to_train)
    iou_eval = IOUEval(num_classes)

    for _, data in enumerate(prog_bar):
        count+=1
        pixel_values = data['pixel_values'].to(device)
        target = data['labels'].to(device)

        optimizer.zero_grad()
        outputs = model(pixel_values=pixel_values, labels=target)

        # BATCH-WISE LOSS
        loss = outputs.loss
        train_running_loss += loss.item()

        # Backpropagation and parameter update
        loss.backward()
        optimizer.step()

        logits = outputs.logits
        upsampled_logits = nn.functional.interpolate(
            logits, size=target.shape[-2:],
            mode='bilinear',
            align_corners=False
        )
        iou_eval.addBatch(upsampled_logits.max(1)[1].data, target.data)
    
    train_loss = train_running_loss / count
    overall_acc, per_class_acc, per_class_iou, mIoU = iou_eval.getMetric()

    return train_loss, overall_acc, mIoU

def validate(
        model,
        val_dataloader,
        classes_to_train,
        label_colors_list,
        epoch,
        save_dir,
        device
):
    print("Validating")
    model.eval()
    val_running_loss = 0.0
    num_classes = len(classes_to_train)
    iou_eval = IOUEval(num_classes)

    with torch.no_grad():
        prog_bar = tqdm(
            val_dataloader,
            total = len(val_dataloader),
            bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}'
        )

        count = 0
        for i, data in enumerate(prog_bar):
            count +=1
            pixel_values = data['pixel_values'].to(device)
            target = data['labels'].to(device)
            outputs = model(pixel_values=pixel_values, labels=target)

            logits = outputs.logits
            upsampled_logits = nn.functional.interpolate(
                logits,
                size=target.shape[-2:],
                mode='bilinear',
                align_corners=False
            )

            # Save the validation segmentation maps
            if i==1:
                draw_translucent_seg_maps(
                    pixel_values,
                    upsampled_logits,
                    epoch,
                    i,
                    save_dir,
                    label_colors_list
                )
            
            loss = outputs.loss
            val_running_loss += loss.item()

            iou_eval.addBatch(upsampled_logits.max(1)[1].data, target.data)
    
    val_loss = val_running_loss / count
    overall_acc, per_class_acc, per_class_iou, mIoU = iou_eval.getMetric()

    return val_loss, overall_acc, mIoU