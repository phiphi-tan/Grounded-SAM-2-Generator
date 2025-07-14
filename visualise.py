import cv2
from PIL import Image
import json
import numpy as np
import supervision as sv
from supervision.draw.color import ColorPalette
from utils.supervision_utils import CUSTOM_COLOR_MAP
import os
import pycocotools.mask as mask_util
from pathlib import Path

# input is a dictionary of format
# {"segmentation": {"size": [H, W], "counts": "{RLE_mask_encoding}"}}
def decode_segmentation(segmentation):
    size = tuple(segmentation['size'])
    counts = segmentation['counts']
    mask = mask_util.decode({'size': size, 'counts': counts.encode('utf-8')})
    return mask

OUTPUT_DIR = Path('outputs/military-assets-segmentised/test/output')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True) # create output directory

DATA_DIR = 'outputs/grounded_sam2_hf_generator/test/'

img = Image.open('outputs/military-assets-segmentised/test/002883.jpg')
img_cv2 = cv2.imread('outputs/military-assets-segmentised/test/002883.jpg')

data = json.load(open("outputs/military-assets-segmentised/test/002883_data.json"))
data_annotations = data['annotations']

input_boxes = np.array([annotation['bbox'] for annotation in data_annotations])
class_labels = np.array(["{} {:.2f} (mask {:.2f})".format(annotation['class_name'], annotation['confidence'], annotation['score'][0]) for annotation in data_annotations])
class_ids = np.array(list(range(len(class_labels))))
print(class_labels)

masks = np.array([decode_segmentation(annotation['segmentation']) for annotation in data_annotations])
scores = np.array([annotation['score'] for annotation in data_annotations])

detections = sv.Detections(
    xyxy=input_boxes,  # (n, 4)
    mask=masks.astype(bool),  # (n, h, w)
    class_id=class_ids
)

# create the
box_annotator = sv.BoxAnnotator(color=ColorPalette.from_hex(CUSTOM_COLOR_MAP))
annotated_frame = box_annotator.annotate(scene=img_cv2.copy(), detections=detections)
label_annotator = sv.LabelAnnotator(color=ColorPalette.from_hex(CUSTOM_COLOR_MAP))
annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=class_labels)

cv2.imwrite(os.path.join(OUTPUT_DIR, "bbox.jpg"), annotated_frame)

mask_annotator = sv.MaskAnnotator(color=ColorPalette.from_hex(CUSTOM_COLOR_MAP))
annotated_frame = mask_annotator.annotate(scene=annotated_frame, detections=detections)

cv2.imwrite(os.path.join(OUTPUT_DIR, "bbox_masked.jpg"), annotated_frame)