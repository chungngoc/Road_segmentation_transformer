import numpy as np
import cv2 as cv
import torch
import torch.nn as nn
import os
import matplotlib.pyplot as plt

from utils.config import *

plt.style.use('ggplot')

# FUNCTIONS TO SET CLASS VALUES
def set_class_values(all_classes, classes_to_train):
    '''
        Assigns a specific class label to the each of the classes.
        For example, `background=0`, `road=1`, and so on.

        all_classes: List containing all class names.
        classes_to_train: List containing class names to train
    '''
    class_values = [all_classes.index(cls.lower()) for cls in classes_to_train]
    return class_values

def get_label_mask(mask, class_values, label_colors_list):
    '''
        Encodes the pixels belonging to the same class in the image into the same label

        mask : Segmentation mask
        class_values : List containing class values
        label_colors_list : List contains all RGB colors for all classes.
    '''
    label_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.uint8)
    for value in class_values:
        for i, label in enumerate(label_colors_list):
            if value == label_colors_list.index(label):
                label = np.array(label)
                label_mask[np.where(np.all(mask == label, axis = -1 ))[:2]] = value
    label_mask = label_mask.astype(int)
    return label_mask

# Visualizing Validation Samples Inbetween Training
def denormalize(x, mean=None, std=None):
    # x should be a Numpy array of shape [H, W, C] 
    x = torch.tensor(x).permute(2,0,1).unsqueeze(0)
    for t, m, s in zip(x, mean, std):
        t.mul(s).add(m)
    result = torch.clamp(t, 0, 1)
    result = result.squeeze(0).permute(1,2,0).numpy()
    return result

def draw_translucent_seg_maps(data, output, epoch, i, val_seg_dir, label_colors_list):
    '''
        This function color codes the segmentation maps that is generated while validating.
        
        i : batch number
    '''
    alpha = 1 # how much transparency
    beta = 0.8 
    gamma = 0 # contrast

    seg_map = output[0]
    seg_map = torch.argmax(seg_map.squeeze(), dim = 0).detach().cpu().numpy()

    x = data[0].permute(1,2,0).cpu().numpy()
    image = denormalize(x, IMG_MEAN, IMG_STD)

    red_map = np.zeros_like(seg_map).astype(np.int8)
    green_map = np.zeros_like(seg_map).astype(np.int8)
    blue_map = np.zeros_like(seg_map).astype(np.int8)

    for label_num in range(0, len(label_colors_list)):
        index = (seg_map == label_num)
        red_map[index] = np.array(VIS_LABEL_MAP)[label_num, 0]
        green_map[index] = np.array(VIS_LABEL_MAP)[label_num, 1]
        blue_map[index] = np.array(VIS_LABEL_MAP)[label_num, 2]
    
    rgb = np.stack([red_map, green_map, blue_map], axis=2)
    rgb = np.array(rgb, dtype=np.float32)
    # convert color to BGR format for OpenCv
    rgb = cv.cvtColor(rgb, cv.COLOR_RGB2BGR)
    image = cv.cvtColor(image, cv.COLOR_RGB2BGR) * 255.0
    cv.addWeighted(image, alpha, rgb, beta,gamma, image)
    cv.imwrite(f"{val_seg_dir}/e{epoch}_b{i}.jpg", image)

# SAVING MODELS AND GRAPHS
class SaveBestModel:
    '''
        save the best model while training.
        If the current epoch's validation loss is less than the previous least less, then save the model state.
    '''
    def __init__(self, best_valid_loss=float('inf')):
        self.best_valid_loss = best_valid_loss
    
    def __call__(self, current_valid_loss, epoch, model, out_dir, name='model'):
        if current_valid_loss < self.best_valid_loss:
            self.best_valid_loss = current_valid_loss
            print(f"\nBest validation loss: {self.best_valid_loss}")
            print(f"\nSaving best model for epoch: {epoch+1}\n")
            model.save_pretrained(os.path.join(out_dir, name))

class SaveBestModelIOU:
    '''
        Save the best model while training.
        If the current epoch's IoU is higher than the previous highest, then save the model state.
    '''
    def __init__(self, best_iou=float(0)):
        self.best_iou = best_iou

    def __call__(self, current_iou, epoch, model, out_dir, name='model'):
        if current_iou > self.best_iou:
            self.best_iou = current_iou
            print(f"\nBest validation IoU: {self.best_iou}")
            print(f"\nSaving best model for epoch: {epoch+1}\n")
            model.save_pretrained(os.path.join(out_dir, name))

def save_model(model, out_dir, name='model'):
    model.save_pretrained(os.path.join(out_dir, name))

def save_plots(train_acc, val_acc, train_loss, val_loss, train_miou, val_miou, out_dir):
    '''
        Save the loss and accuracy plots
    '''
    # Accuracy plots
    plt.figure(figsize=(8,6))
    plt.plot(
        train_acc, color='tab:blue', linestyle='-',
        label='Train accuracy'
    )
    plt.plot(val_acc, color='tab:red', linestyle='-', label='Validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(os.path.join(out_dir, 'accuracy.png'))

    # Loss plots
    plt.figure(figsize=(8,6))
    plt.plot(train_loss, color='tab:blue', linestyle='-', label='Train loss')
    plt.plot(val_loss, color='tab:red', linestyle='-', label='Validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(out_dir, 'loss.png'))

    # mIOU plots
    plt.figure(figsize=(8,6))
    plt.plot(train_miou, color='tab:blue', linestyle='-', label='Train mIoU')
    plt.plot(val_miou, color='tab:red', linestyle='-', label='Validation mIoU')
    plt.xlabel('Epochs')
    plt.ylabel('mIoU')
    plt.legend()
    plt.savefig(os.path.join(out_dir, 'miou.png'))

# FUNCTIONS FOR INFERENCE
def predict(model, extractor, image, device):
    '''
        model : The Segformer model
        extractor : Srgformer feature extractor
        image : image in RGB format
        device : The compute device

        returns :
            labels: The fincal labels(classes) in h x w format
    '''
    pixel_values = extractor(image, return_tensors='pt').pixel_values.to(device)
    with torch.no_grad():
        logits = model(pixel_values).logits

    # rescale logits to original image size
    logits = nn.functional.interpolate(
        logits,
        size=image.shape[:2],
        mode='bilinear',
        align_corners=False
    )
    # Get class labels
    labels = torch.argmax(logits.squeeze(), dim=0)

    return labels

def draw_segmentation_map(labels, palette):
    '''
        params :
            labels : Label array from the model (shape : h x w)
            palette : List contains color information
    '''
    red_map = np.zeros_like(labels).astype(np.uint8)
    green_map = np.zeros_like(labels).astype(np.uint8)
    blue_map = np.zeros_like(labels).astype(np.uint8)

    for label_num in range(0, len(palette)):
        index = (labels == label_num)
        red_map[index] = np.array(palette)[label_num, 0]
        green_map[index] = np.array(palette)[label_num, 1]
        blue_map[index] = np.array(palette)[label_num, 2]
    
    seg_map = np.stack([red_map, green_map, blue_map], axis=2)
    return seg_map

def image_overlay(img, sedmented_img):
    '''
        img : Image in RGB format
        segmented_image : Segmentation map in RGB format
    '''
    alpha = 0.8 
    beta = 1
    gamma = 0

    sedmented_img = cv.cvtColor(sedmented_img, cv.COLOR_RGB2BGR)
    img = np.array(img)
    img = cv.cvtColor(img, cv.COLOR_RGB2BGR)
    cv.addWeighted(img, alpha, sedmented_img, beta, gamma, img)
    return img

