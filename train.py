import torch
import os
import argparse

from data.dataset import get_image, get_dataset, get_data_loader
from model.model import segformer_model
from model.engine import train, validate
from utils.config import ALL_CLASSES, LABEL_COLORS_LIST
from utils.utils import save_model, SaveBestModel, save_plots, SaveBestModelIOU

from transformers import SegformerFeatureExtractor
from torch.optim.lr_scheduler import MultiStepLR

seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser()
parser.add_argument(
    '--epochs',
    default=10,
    help='number of epochs for training',
    type=int
)
parser.add_argument(
    '--lr',
    default=0.0001,
    help='learning rate for optimizer',
    type=float
)
parser.add_argument(
    '--batch',
    default=4,
    help='batch size',
    type=int
)
parser.add_argument(
    '--imgsz',
    default=[512, 416],
    help='width, height of image',
    nargs='+',
    type=int
)
parser.add_argument(
    '--scheduler',
    action='store_true'
)
parser.add_argument(
    '--scheduler-epochs',
    dest='scheduler_epochs',
    default=[30],
    nargs='+',
    type=int
)
args=parser.parse_args()
print(args)

if __name__ == '__main__':
    # Create outputs directory
    out_dir = os.path.join('outputs')
    out_dir_val_preds = os.path.join(out_dir, 'valid_preds')
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(out_dir_val_preds, exist_ok=True)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = segformer_model(classes=ALL_CLASSES).to(device)
    print(model)

    # Total parameters and trainable parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"{total_params:,} total parameters.")
    trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    print(f"{trainable_params:,} training parameters.")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    train_images, train_masks, val_images, val_masks = get_image(root_path='inputs')

    feature_extractor = SegformerFeatureExtractor(size=(args.imgsz[1], args.imgsz[0]))

    train_dataset, val_dataset = get_dataset(
        train_images,
        train_masks,
        val_images,
        val_masks,
        ALL_CLASSES,
        ALL_CLASSES,
        LABEL_COLORS_LIST,
        img_size=args.imgsz,
        feature_extractor=feature_extractor
    )

    train_dataloader, val_dataloader = get_data_loader(
        train_dataset,
        val_dataset,
        args.batch
    )

    save_best_model = SaveBestModel()
    save_best_iou = SaveBestModelIOU()

    # Learning rate Scheduler
    scheduler = MultiStepLR(
        optimizer, milestones=args.scheduler_epochs, gamma=0.1, verbose=True
    )

    train_loss, train_pix_acc, train_miou = [], [], []
    val_loss, val_pix_acc, val_miou = [], [], []

    for epoch in range(args.epochs):
        print(f"EPOCH : {epoch+1}")
        train_epoch_loss, train_epoch_pix_acc, train_epoch_miou = train(
            model,
            train_dataloader,
            optimizer,
            ALL_CLASSES,
            device
        )
        val_epoch_loss, val_epoch_pix_acc, val_epoch_miou = validate(
            model,
            val_dataloader,
            ALL_CLASSES,
            LABEL_COLORS_LIST,
            epoch=epoch,
            save_dir=out_dir_val_preds
        )

        train_loss.append(train_epoch_loss)
        train_pix_acc.append(train_epoch_pix_acc)
        train_miou.append(train_epoch_miou)

        val_loss.append(val_epoch_loss)
        val_pix_acc.append(val_epoch_pix_acc)
        val_miou.append(val_epoch_miou)

        # Save models
        save_best_model(
            val_epoch_loss, epoch, model, out_dir, name='model_loss'
        )
        save_best_iou(
            val_epoch_miou, epoch, model, out_dir, name='model_iou'
        )

        print(
            f"Train Epoch Loss: {train_epoch_loss:.4f},",
            f"Train Epoch PixAcc: {train_epoch_pix_acc:.4f},",
            f"Train Epoch mIOU: {train_epoch_miou:4f}"
        )
        print(
            f"Valid Epoch Loss: {val_epoch_loss:.4f},", 
            f"Valid Epoch PixAcc: {val_epoch_pix_acc:.4f}",
            f"Valid Epoch mIOU: {val_epoch_miou:4f}"
        )

        if args.scheduler:
            scheduler.step()
        print('-'*30)

        # Save plots
        save_plots(
            train_pix_acc,
            val_pix_acc,
            train_loss,
            val_loss,
            train_miou,
            val_miou,
            out_dir
        )
        # Save final model
        save_model(model, out_dir, name='final_model')
        print("Complet !!!!!!")