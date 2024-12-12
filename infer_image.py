import os
import glob
import argparse
import cv2 as cv
from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation

from utils.config import VIS_LABEL_MAP
from utils.utils import predict, image_overlay, draw_segmentation_map

parser = argparse.ArgumentParser()
parser.add_argument(
    '--input',
    help='path to the input image directory',
    default='inputs/inference_data/images'
)
parser.add_argument(
    '--device',
    default='cuda:0',
    help='compute device, cpu or cuda'
)
parser.add_argument(
    '--imgsz',
    default=None,
    type=int,
    nargs='+',
    help='image size, width height'
)
parser.add_argument(
    '--model',
    default='outputs/model_iou',
    help='directory contains best model'
)
args = parser.parse_args()

out_dir = 'outputs/inference_results_image'
os.makedirs(out_dir, exist_ok=True)

extractor = SegformerFeatureExtractor()
model = SegformerForSemanticSegmentation.from_pretrained(args.model)
model.to(args.device).eval()

image_paths = glob.glob(os.path.join(args.input, '*'))
for img_path in image_paths:
    # Get image
    img = cv.imread(img_path)
    if args.imgsz is not None:
        img = cv.resize(img, (args.imgsz[0], args.imgsz[1]))
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

    # Get labels
    labels = predict(model, extractor, img, args.device)

    # Segmentation map
    seg_map = draw_segmentation_map(
        labels.cpu(),
        VIS_LABEL_MAP
    )
    outputs = image_overlay(img, seg_map)

    cv.imshow('Image', outputs)
    cv.waitKey(1)

    # Save prediction
    img_name = img_path.split(os.path.sep)[-1]
    save_path = os.path.join(out_dir, '_'+img_name)
    cv.imwrite(save_path, outputs)