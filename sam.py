import numpy as np
import torch
import cv2
import supervision as sv
from segment_anything import SamAutomaticMaskGenerator
from segment_anything import SamPredictor
from segment_anything import sam_model_registry

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
MODEL_TYPE = "vit_b"

sam = sam_model_registry[MODEL_TYPE](checkpoint='sam_vit_b_01ec64.pth')
sam.to(device=DEVICE)

mask_generator = SamAutomaticMaskGenerator(sam)
mask_predictor = SamPredictor(sam)

image_bgr = cv2.imread('perro.jpg')
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
#result = mask_generator.generate(image_rgb)

#mask_annotator = sv.MaskAnnotator(color_lookup = sv.ColorLookup.INDEX)
#detections = sv.Detections.from_sam(result)
#annotated_image = mask_annotator.annotate(image_bgr.copy(), detections)

mask_predictor.set_image(image_rgb)

box = np.array([100, 150, 716, 1226])
masks, scores, logits = mask_predictor.predict(
    box=box,
    multimask_output=True
)

box_annotator = sv.BoxAnnotator(color=sv.Color.RED, color_lookup=sv.ColorLookup.INDEX)
mask_annotator = sv.MaskAnnotator(color=sv.Color.RED, color_lookup=sv.ColorLookup.INDEX)

detections = sv.Detections(
    xyxy=sv.mask_to_xyxy(masks=masks),
    mask=masks
)
detections = detections[detections.area == np.max(detections.area)]

source_image = box_annotator.annotate(scene=image_bgr.copy(), detections=detections)
segmented_image = mask_annotator.annotate(scene=image_bgr.copy(), detections=detections)

sv.plot_images_grid(
    images=[source_image, segmented_image],
    grid_size=(1, 2),
    titles=['source image', 'segmented image']
)

#sv.plot_images_grid(
#    images= [image_bgr, annotated_image],
#    grid_size=(1, 2),
#    titles=['source image', 'segmented image']
#)