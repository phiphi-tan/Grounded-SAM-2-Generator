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

INPUT_DIR = Path('outputs/grounded_sam2_hf_generator/test/')
OUTPUT_DIR = Path('outputs/military-assets-segmentised/test/output')

OUTPUT_DIR.mkdir(parents=True, exist_ok=True) # create output directory

for file in os.listdir(INPUT_DIR):
    if not (file.endswith(".jpg")):
        continue

<<<<<<< HEAD
img = Image.open('outputs/military-assets-segmentised/test/002883.jpg')
img_cv2 = cv2.imread('outputs/military-assets-segmentised/test/002883.jpg')

data = json.load(open("outputs/military-assets-segmentised/test/002883_data.json"))
data_annotations = data['annotations']

input_boxes = np.array([annotation['bbox'] for annotation in data_annotations])
class_labels = np.array(["{} {:.2f} (mask {:.2f})".format(annotation['class_name'], annotation['confidence'], annotation['score'][0]) for annotation in data_annotations])
class_ids = np.array(list(range(len(class_labels))))
print(class_labels)
=======
    image = file
    img_cv2 = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    image_name = image.filename.split("\\")[-1].rstrip('.jpg')
    data_file_name = image_name + '_data.json'

    try:
        data = json.load(open(os.path.join(INPUT_DIR, data_file_name)))
    except FileNotFoundError:
        print("No corresponding data file found for image {}.jpg",format(image_name))
        continue
    data_annotations = data['annotations']
>>>>>>> 9055b45f6ade95ca61f49663df6bcf78c8b7dafc

    input_boxes = np.array([annotation['bbox'] for annotation in data_annotations])
    class_labels = np.array(["{} {:.2f} (mask {:.2f})".format(annotation['class_name'], annotation['confidence'], annotation['score'][0]) for annotation in data_annotations])
    class_ids = np.array(list(range(len(class_labels))))

    # mask input is a dictionary of format
    # {"segmentation": {"size": [H, W], "counts": "{RLE_mask_encoding}"}}
    def decode_segmentation(segmentation):
        size = tuple(segmentation['size'])
        counts = segmentation['counts']
        mask = mask_util.decode({'size': size, 'counts': counts.encode('utf-8')})
        return mask

    masks = np.array([decode_segmentation(annotation['segmentation']) for annotation in data_annotations])

    detections = sv.Detections(
        xyxy=input_boxes,  # (N, 4)
        mask=masks.astype(bool),
        class_id=class_ids
    )

    # create the
    box_annotator = sv.BoxAnnotator(color=ColorPalette.from_hex(CUSTOM_COLOR_MAP))
    annotated_frame = box_annotator.annotate(scene=img_cv2.copy(), detections=detections)
    label_annotator = sv.LabelAnnotator(color=ColorPalette.from_hex(CUSTOM_COLOR_MAP))
    annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=class_labels)
    mask_annotator = sv.MaskAnnotator(color=ColorPalette.from_hex(CUSTOM_COLOR_MAP))
    annotated_frame = mask_annotator.annotate(scene=annotated_frame, detections=detections)

    cv2.imwrite(os.path.join(OUTPUT_DIR, "{}_constructed.jpg".format(image_name)), annotated_frame)