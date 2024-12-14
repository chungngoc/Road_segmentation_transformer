import os
import time
import argparse
import cv2 as cv
from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation

from utils.config import VIS_LABEL_MAP
from utils.utils import predict, image_overlay, draw_segmentation_map

parser = argparse.ArgumentParser()
parser.add_argument(
    '--input',
    help='path to the input video',
    default='inputs/inference_data/videos/video_1.mov'
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

out_dir = 'outputs/inference_results_video'
os.makedirs(out_dir, exist_ok=True)

extractor = SegformerFeatureExtractor()
model = SegformerForSemanticSegmentation.from_pretrained(args.model)
model.to(args.device).eval()

cap = cv.VideoCapture(args.input)
if args.imgsz is None:
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
else:
    frame_width = args.imgsz[0]
    frame_height = args.imgsz[1]

vid_fps = int(cap.get(5))
save_name = args.input.split(os.path.sep)[-1].split('.')[0]

out = cv.VideoWriter(f"{out_dir}/{save_name}.mp4",
                     cv.VideoWriter_fourcc(*'mp4v'),
                     30,
                     (frame_width, frame_height))

frame_count = 0
total_fps = 0
while cap.isOpened:
    ret, frame = cap.read()
    if ret:
        frame_count += 1
        img = frame
        if args.imgsz is not None:
            img = cv.resize(img, (args.imgsz[0], args.imgsz[1]))
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

        # Get labels
        start_time = time.time()
        labels = predict(model, extractor, img, args.device)
        end_time = time.time()

        fps = 1 / (end_time - start_time)
        total_fps += fps

        # Segmentation map
        seg_map = draw_segmentation_map(
            labels.cpu(),
            VIS_LABEL_MAP
        )
        outputs = image_overlay(img, seg_map)
        
        cv.putText(
            outputs,
            f"{fps:.1f} FPS",
            (15,35),
            fontFace=cv.FONT_HERSHEY_SIMPLEX,
            fontScale=1,
            color=(0, 0, 255),
            thickness=2,
            lineType=cv.LINE_AA
        )
        out.write(outputs)
        cv.imshow('Image', outputs)
        # Press 'q' to exit
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

# release VideoCapture()
cap.release()
cv.destroyAllWindows()

avg_fps = total_fps / frame_count
print(f"Average FPS: {avg_fps:.3f}")