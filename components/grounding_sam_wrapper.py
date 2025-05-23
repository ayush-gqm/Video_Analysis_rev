"""GroundingSAM detector wrapper for entity detection in the video analysis pipeline."""

from typing import List, Dict
import numpy as np
import torch
import cv2 # OpenCV for image loading if needed, though PIL is used primarily
from PIL import Image
import groundingdino.datasets.transforms as T
from groundingdino.util.inference import load_model, predict
from segment_anything import sam_model_registry, SamPredictor
import logging # Added for logging

logger = logging.getLogger(__name__) # Added for logging

class GroundingSAMDetector:
    """
    Wrapper for GroundingSAM object detection model.
    Provides a detect(image) method returning a list of entity dicts.
    """
    def __init__(self, config: dict = None):
        """
        Initialize the GroundingSAMDetector.

        Args:
            config: Configuration dictionary with:
                - grounding_dino_config_path: Path to GroundingDINO config (corrected key)
                - grounding_dino_checkpoint_path: Path to GroundingDINO checkpoint (corrected key)
                - sam_checkpoint_path: Path to SAM checkpoint (corrected key)
                - sam_model_type: SAM model type (default: "vit_h")
                - box_threshold: Detection threshold (default: 0.3)
                - text_threshold: Text similarity threshold (default: 0.25)
                - device: Device to use (default: "cuda")
                - prompt: Default text prompt for detection
        """
        self.config = config or {}
        self.device = self.config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Initializing GroundingSAMDetector on device: {self.device}")

        gd_config_path = self.config.get("grounding_dino_config_path")
        gd_checkpoint_path = self.config.get("grounding_dino_checkpoint_path")
        sam_checkpoint_path = self.config.get("sam_checkpoint_path")

        if not all([gd_config_path, gd_checkpoint_path, sam_checkpoint_path]):
            logger.error("Missing one or more model paths for GroundingSAM in config.")
            raise ValueError("GroundingDINO config, checkpoint, and SAM checkpoint paths are required.")

        # Initialize GroundingDINO
        logger.info(f"Loading GroundingDINO model from config: {gd_config_path} and checkpoint: {gd_checkpoint_path}")
        self.grounding_dino_model = load_model(
            gd_config_path,
            gd_checkpoint_path,
            device=self.device
        )
        # Explicitly move model to device to be sure
        self.grounding_dino_model.to(self.device)
        
        # Initialize SAM
        sam_model_type = self.config.get("sam_model_type", "vit_h")
        logger.info(f"Loading SAM model type {sam_model_type} from checkpoint: {sam_checkpoint_path}")
        self.sam = sam_model_registry[sam_model_type](
            checkpoint=sam_checkpoint_path
        ).to(self.device)
        # Explicitly move SAM predictor's model to device to be sure
        self.sam_predictor = SamPredictor(self.sam)
        self.sam_predictor.model.to(self.device)
        
        # Detection thresholds
        self.box_threshold = self.config.get("box_threshold", 0.3)
        self.text_threshold = self.config.get("text_threshold", 0.25)
        
        # Default prompt (can be overridden in detect())
        self.default_prompt = self.config.get("prompt", "person . car . bus . truck . animal . object . building . tree")
        logger.info(f"GroundingSAM default prompt: '{self.default_prompt}'")
        logger.info(f"GroundingSAM thresholds: box_threshold={self.box_threshold}, text_threshold={self.text_threshold}")
        
        # Transform for input images
        self.transform = T.Compose([
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        logger.info("GroundingSAMDetector initialized successfully.")

    def detect(self, image: np.ndarray, prompt: str = None) -> List[Dict]:
        """
        Detect entities in an image using GroundingSAM.

        Args:
            image: Input image as a numpy array (H, W, C, RGB).
                   The GroundingDINO model expects RGB format.
            prompt: Text prompt for detection. If None, uses default_prompt.

        Returns:
            List of detection dicts, each with keys: 
            'bbox' (xyxy format, normalized), 'score', 'label', 'mask' (boolean numpy array).
        """
        current_prompt = prompt or self.default_prompt
        
        # Ensure image is RGB
        if image.shape[2] == 3 and image.dtype == np.uint8:
             # Check if BGR (common OpenCV format) and convert to RGB
            # A simple heuristic: if blue channel has higher mean than red for typical outdoor scenes
            # This is not foolproof, but a common case.
            # For more robustness, the component calling this should ensure RGB.
            # Assuming image is already in RGB as per docstring.
            pass
        else:
            logger.error(f"Input image is not in expected format (H, W, 3) uint8. Got shape {image.shape}, dtype {image.dtype}")
            return []

        image_pil = Image.fromarray(image) # Expects RGB
        
        # Note: GroundingDINO's transform handles normalization and ToTensor.
        # The input to predict is a transformed image tensor.
        image_transformed, _ = self.transform(image_pil, None) 
        
        detections_result = []
        try:
            # Predict with GroundingDINO
            # Ensure image_transformed is on the correct device before passing to predict
            image_transformed = image_transformed.to(self.device)
            
            boxes, logits, phrases = predict(
                model=self.grounding_dino_model,
                image=image_transformed, # Pass the transformed tensor
                caption=current_prompt,
                box_threshold=self.box_threshold,
                text_threshold=self.text_threshold,
                device=self.device # Device is passed here, but output device can be inconsistent
            )
            
            # Explicitly move boxes to the target device after prediction
            boxes = boxes.to(self.device)

            
            if len(boxes) == 0:
                return []
                
            # Convert boxes to image coordinates (xyxy format) relative to original image size
            H, W, _ = image.shape
            # Create the scaling tensor directly on the target device
            scaling_tensor = torch.Tensor([W, H, W, H]).to(self.device)
            boxes_xyxy_abs = boxes * scaling_tensor
            # GroundingDINO outputs boxes in center_x, center_y, width, height format.
            # Convert to x_min, y_min, x_max, y_max (xyxy)
            boxes_xyxy_abs[:, :2] -= boxes_xyxy_abs[:, 2:] / 2 # x_min = cx - w/2, y_min = cy - h/2
            boxes_xyxy_abs[:, 2:] += boxes_xyxy_abs[:, :2]    # x_max = x_min + w, y_max = y_min + h (error here, should be x_max = x_min + w, y_max = y_min + h)
                                                               # Corrected: x_max = (cx - w/2) + w = cx + w/2 ; y_max = (cy - h/2) + h = cy + h/2
                                                               # Let's re-verify the output format of `predict` and `sam_predictor` box requirements.
                                                               # The `predict` function returns boxes in [cx, cy, w, h] format.
                                                               # SAM expects boxes in [x_min, y_min, x_max, y_max] format.
            
            # Conversion from [cx, cy, w, h] to [x1, y1, x2, y2]
            # x1 = cx - w / 2
            # y1 = cy - h / 2
            # x2 = cx + w / 2
            # y2 = cy + h / 2
            
            # Detach boxes from graph and clone for manipulation
            boxes_cloned = boxes.detach().clone()
            
            # Apply scaling to absolute image dimensions
            boxes_cloned[:, 0] = boxes_cloned[:, 0] * W # cx
            boxes_cloned[:, 1] = boxes_cloned[:, 1] * H # cy
            boxes_cloned[:, 2] = boxes_cloned[:, 2] * W # w
            boxes_cloned[:, 3] = boxes_cloned[:, 3] * H # h

            # Convert to xyxy absolute
            boxes_xyxy_abs_sam = torch.zeros_like(boxes_cloned)
            boxes_xyxy_abs_sam[:, 0] = boxes_cloned[:, 0] - boxes_cloned[:, 2] / 2 # x1
            boxes_xyxy_abs_sam[:, 1] = boxes_cloned[:, 1] - boxes_cloned[:, 3] / 2 # y1
            boxes_xyxy_abs_sam[:, 2] = boxes_cloned[:, 0] + boxes_cloned[:, 2] / 2 # x2
            boxes_xyxy_abs_sam[:, 3] = boxes_cloned[:, 1] + boxes_cloned[:, 3] / 2 # y2
            
            # Get SAM masks
            self.sam_predictor.set_image(image) # SAM expects HWC, uint8, RGB
            
            # SAM's transform.apply_boxes_torch expects absolute coordinates
            transformed_boxes_for_sam = self.sam_predictor.transform.apply_boxes_torch(boxes_xyxy_abs_sam, image.shape[:2])
            
            masks, _, _ = self.sam_predictor.predict_torch(
                point_coords=None,
                point_labels=None,
                boxes=transformed_boxes_for_sam.to(self.device),
                multimask_output=False, # Get a single high-quality mask per box
            )
            
            # Convert to detection dicts
            for i in range(len(boxes_xyxy_abs_sam)):
                box_abs = boxes_xyxy_abs_sam[i].cpu().numpy()
                # Normalize bbox to [0, 1]
                bbox_normalized = [
                    float(box_abs[0] / W), # Explicitly cast to float
                    float(box_abs[1] / H), # Explicitly cast to float
                    float(box_abs[2] / W), # Explicitly cast to float
                    float(box_abs[3] / H), # Explicitly cast to float
                ]
                # Clip values to be within [0, 1] and ensure they remain floats
                bbox_normalized = [float(max(0.0, min(1.0, coord))) for coord in bbox_normalized]

                detections_result.append({
                    "bbox": bbox_normalized, # Normalized xyxy (now list of floats)
                    "score": float(logits[i]), # Confidence score from GroundingDINO (already float, but re-cast for certainty)
                    "label": phrases[i],       # Detected phrase/label
                    "mask": masks[i].squeeze().cpu().numpy().astype(bool).tolist() # Convert boolean mask to list
                })
        except Exception as e:
            logger.error(f"Error during GroundingSAM detection: {e}", exc_info=True)
            return [] # Return empty list on error
            
        return detections_result 