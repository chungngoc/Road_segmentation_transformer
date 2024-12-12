"""
    Prepare data for training.
    All images are in a directory and all masks are in another one.
"""

import os
import shutil
import random
import argparse
from tqdm.auto import tqdm

seed = 42
random.seed(seed)

def copy_data(src_root_dir, dest_root_dir, images, masks, split='train'):
    # Define source path
    scr_images_path = os.path.join(src_root_dir, 'JPEGImages')
    src_masks_path = os.path.join(src_root_dir, 'SegmentationClassPNG')

    # Create destination path 
    train_images_path = os.path.join(dest_root_dir, 'train', 'images')
    train_masks_path = os.path.join(dest_root_dir, 'train', 'masks')
    val_images_path = os.path.join(dest_root_dir, 'valid', 'images')
    val_masks_path = os.path.join(dest_root_dir, 'valid', 'masks')

    os.makedirs(train_images_path, exist_ok=True)
    os.makedirs(train_masks_path, exist_ok=True)
    os.makedirs(val_images_path, exist_ok=True)
    os.makedirs(val_masks_path, exist_ok=True)

    if split == 'train':
        image_dest = train_images_path
        mask_dest = train_masks_path
    else:
        image_dest = val_images_path
        mask_dest = val_masks_path
    
    for i, data in tqdm(enumerate(images), total=len(images)):
        shutil.copy(
            src=os.path.join(scr_images_path, images[i]),
            dst=os.path.join(image_dest, images[i])
        )
        shutil.copy(
            src = os.path.join(src_masks_path, masks[i]),
            dst = os.path.join(mask_dest, masks[i])
        )
############################################################

parser = argparse.ArgumentParser()
parser.add_argument(
    '--src_root_dir',
    default='inputs/data_dataset_voc',
    help='folder contains 2 directories for images and masks',
    type=str
)
parser.add_argument(
    '--dest_root_dir',
    default='inputs',
    help='destination folder',
    type=str
)
parser.add_argument(
    '--valid_split',
    default=0.2,
    help='split ratio',
    type=float
)
args = parser.parse_args()
print(args)
if __name__ == '__main__':
    src_root_dir = args.src_root_dir
    dest_root_dir = args.dest_root_dir
    valid_split = args.valid_split

    scr_images_path = os.path.join(src_root_dir, 'JPEGImages')
    src_masks_path = os.path.join(src_root_dir, 'SegmentationClassPNG')

    all_images = os.listdir(scr_images_path)
    all_masks = os.listdir(src_masks_path)

    all_images.sort()
    all_masks.sort()

    print(all_masks[:3])
    print(all_images[:3])

    combined = list(zip(all_images, all_masks))
    random.shuffle(combined)
    shuffled_images, shuffled_masks = zip(*combined)

    print("Shuffled images example:", shuffled_images[:3])
    print("Shuffled masks example:", shuffled_masks[:3])

    train_samples = int((1 - valid_split)*len(shuffled_images))
    valid_samples = len(shuffled_images) - train_samples
    print(train_samples, valid_samples)

    final_train_images = shuffled_images[:train_samples]
    final_train_masks = shuffled_masks[:train_samples]
    final_valid_images = shuffled_images[-valid_samples:]
    final_valid_masks = shuffled_masks[-valid_samples:]

    print(len(final_train_images))
    print(len(final_train_masks))
    print(len(final_valid_images))
    print(len(final_valid_masks))

    copy_data(src_root_dir, dest_root_dir, final_train_images, final_train_masks, split='train')
    copy_data(src_root_dir, dest_root_dir, final_valid_images, final_valid_masks, split='valid')

