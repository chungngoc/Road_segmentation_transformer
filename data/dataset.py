import glob
import albumentations as A
import cv2 as cv
import os

from utils.utils import get_label_mask, set_class_values
from torch.utils.data import Dataset, DataLoader
from PIL import Image

def get_image(root_path):
    train_images = glob.glob(f"{root_path}/train/images/*")
    train_images.sort()
    train_masks = glob.glob(f"{root_path}/train/masks/*")
    train_masks.sort()
    val_images = glob.glob(f"{root_path}/valid/images/*")
    val_images.sort()
    val_masks = glob.glob(f"{root_path}/valid/masks/*")
    val_masks.sort()

    return train_images, train_masks, val_images, val_masks

def train_transforms(img_size):
    '''
        Transforms/Augmentations for training images and masks
    '''
    train_img_aug = A.Compose([
        A.Resize(img_size[1], img_size[0], always_apply=True),
        A.HorizontalFlip(p=0.5),
        A.OneOf(
            [
                A.ColorJitter(p=0.33),
                A.RandomBrightnessContrast(p=0.33),    
                A.RandomGamma(p=0.33)
            ],
            p=1
        ),
        A.Rotate(limit=25),
    ], is_check_shapes=False)

    return train_img_aug

def valid_transforms(img_size):
    '''
        Transforms/Augmentations for validation images and masks
    '''
    val_img_trans = A.Compose([
        A.Resize(img_size[1], img_size[0], always_apply=True),
    ], is_check_shapes=False)

    return val_img_trans

# Custom Segmentation Dataset Class
class SegmentationDataset(Dataset):
    def __init__(self,
                 image_paths,
                 mask_paths,
                 tfms,
                 label_color_list,
                 classes_to_train,
                 all_classes,
                 feature_extractor):
        '''
            image_paths, mask_paths : The list containing the image and mask paths that we get from the get_images function.
            tfms : the transforms that we want to apply.
            label_color_list : list containing the color values for each class
            classes_to_train and all_classes : lists containing the string names of classes that we want to train and all the class names in the dataset
            feature_extractor : The Transformers library provides a feature extractor class for the SegFormer model. This helps us apply the necessary ImageNet normalization.
        '''
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.tfms = tfms
        self.label_color_list = label_color_list
        self.classes_to_train = classes_to_train
        self.class_values = set_class_values(
            all_classes, self.classes_to_train
        )
        self.feature_extractor = feature_extractor
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, index):
        img = cv.imread(self.image_paths[index], cv.IMREAD_COLOR)
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB).astype('float32')
        mask = cv.imread(self.mask_paths[index], cv.IMREAD_COLOR)
        mask = cv.cvtColor(mask, cv.COLOR_BGR2RGB).astype('float32')

        transformed = self.tfms(image=img, mask = mask)
        img = transformed['image'].astype('uint8')
        mask = transformed['mask']

        # Get 2D label mask
        mask = get_label_mask(mask, self.class_values, self.label_color_list).astype('uint8')
        mask = Image.fromarray(mask)

        encoded_inputs = self.feature_extractor(
            Image.fromarray(img),
            mask,
            return_tensors='pt'
        )
        for k, _ in encoded_inputs.items():
            encoded_inputs[k].squeeze_()

        return encoded_inputs
     
def get_dataset(
        train_img_paths,
        train_mask_path,
        val_img_paths,
        val_mask_paths,
        all_classes,
        classes_to_train,
        label_colors_list,
        img_size,
        feature_extractor
):
    train_tfms = train_transforms(img_size)
    val_tfms = valid_transforms(img_size)

    train_dataset = SegmentationDataset(
        train_img_paths,
        train_mask_path,
        train_tfms,
        label_colors_list,
        classes_to_train,
        all_classes,
        feature_extractor
    )
    val_dataset = SegmentationDataset(
        val_img_paths,
        val_mask_paths,
        val_tfms,
        label_colors_list,
        classes_to_train,
        all_classes,
        feature_extractor
    )
    return train_dataset, val_dataset

def get_data_loader(train_dataset, val_dataset, batch_size):
    train_data_loader = DataLoader(
        train_dataset,
        batch_size = batch_size,
        drop_last=False,
        num_workers=8,
        shuffle=True
    )
    val_data_loader = DataLoader(
        val_dataset,
        batch_size = batch_size,
        drop_last=False,
        num_workers=8,
        shuffle=False
    )

    return train_data_loader, val_data_loader