"""Entity detection component for the video analysis pipeline."""

import os
# Set environment variables to handle NCCL issues before importing PyTorch
os.environ["NCCL_P2P_DISABLE"] = "1"  # Disable NCCL peer-to-peer which can cause symbol errors
os.environ["NCCL_BLOCKING_WAIT"] = "0"  # Non-blocking NCCL operations

import cv2
import numpy as np
import torch
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import json
from tqdm import tqdm
import gc
import uuid
import importlib
from PIL import Image

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define entity categories for classification
ENTITY_CATEGORIES = {
    "person": ["person", "man", "woman", "child", "boy", "girl", "people", "crowd", "face", "body", 
               "human", "individual", "pedestrian", "customer", "student", "teacher", "worker", 
               "player", "audience", "spectator", "passenger", "driver", "cyclist", "baby", "teen", 
               "adult", "elderly", "senior", "group", "couple", "family", "team", "crew"],
               
    "setting": ["room", "house", "building", "office", "kitchen", "bedroom", "living room", 
                "restaurant", "cafe", "hotel", "street", "road", "park", "garden", "forest",
                "beach", "mountain", "desert", "river", "lake", "ocean", "sea", "classroom",
                "theater", "cinema", "stadium", "arena", "airport", "station", "mall", "store",
                "shop", "market", "hospital", "clinic", "gym", "pool", "church", "temple", "mosque",
                "plaza", "square", "hall", "lobby", "corridor", "elevator", "staircase", "rooftop",
                "balcony", "terrace", "patio", "yard", "lawn", "field", "meadow", "path", "trail"],
                
    "environment": ["sky", "clouds", "rain", "snow", "sun", "moon", "stars", "fog", "mist",
                    "weather", "climate", "landscape", "cityscape", "urban", "rural", "sunset",
                    "sunrise", "dawn", "dusk", "night", "day", "morning", "afternoon", "evening",
                    "season", "spring", "summer", "autumn", "winter", "wind", "storm", "lightning",
                    "thunder", "rainbow", "horizon", "atmosphere", "nature", "wilderness", "terrain",
                    "valley", "canyon", "hill", "cliff", "waterfall", "stream", "creek", "pond"],
                    
    "object": ["car", "vehicle", "furniture", "chair", "table", "desk", "computer", "phone", "book",
               "food", "drink", "animal", "dog", "cat", "tree", "plant", "flower", "grass", "bicycle",
               "motorcycle", "truck", "bus", "train", "airplane", "boat", "ship", "sofa", "couch", 
               "bed", "cabinet", "dresser", "shelf", "television", "tv", "monitor", "laptop", "tablet",
               "smartphone", "camera", "microphone", "speaker", "headphones", "keyboard", "mouse",
               "plate", "bowl", "cup", "glass", "bottle", "container", "box", "bag", "backpack",
               "luggage", "suitcase", "clothing", "shirt", "pants", "dress", "shoes", "hat", "umbrella",
               "mirror", "picture", "painting", "poster", "clock", "watch", "lamp", "light", "door", 
               "window", "curtain", "rug", "carpet", "pillow", "blanket", "towel", "appliance", 
               "refrigerator", "oven", "microwave", "dishwasher", "washing machine", "vacuum", "fan",
               "air conditioner", "bird", "fish", "horse", "cow", "sheep", "goat", "pig", "bear", 
               "lion", "tiger", "elephant", "giraffe", "zebra", "deer", "rabbit", "squirrel", "mouse",
               "rat", "insect", "butterfly", "bee", "ant", "spider", "fruit", "vegetable", "meat",
               "bread", "cake", "cookie", "candy", "snack", "meal", "water", "soda", "juice", "coffee",
               "tea", "alcohol", "wine", "beer", "sign", "banner", "flag", "tool", "weapon", "toy",
               "instrument", "guitar", "piano", "drum", "flute", "violin", "trash", "garbage", "waste",
               "fire", "smoke", "paper", "pen", "pencil", "marker", "crayon"],
               
    "action": ["walking", "running", "sitting", "standing", "eating", "drinking", "talking",
               "dancing", "singing", "driving", "riding", "sleeping", "working", "playing",
               "swimming", "jumping", "climbing", "falling", "throwing", "catching", "kicking",
               "hitting", "punching", "fighting", "hugging", "kissing", "shaking hands", "waving",
               "pointing", "smiling", "laughing", "crying", "shouting", "whispering", "cooking",
               "baking", "cleaning", "washing", "shopping", "reading", "writing", "typing",
               "drawing", "painting", "photographing", "filming", "performing", "exercising",
               "training", "competing", "celebrating", "partying", "waiting", "watching", "listening",
               "speaking", "teaching", "learning", "studying", "building", "constructing", "fixing",
               "repairing", "breaking", "opening", "closing", "pushing", "pulling", "lifting",
               "carrying", "dragging", "dropping", "serving", "cutting", "chopping", "slicing"]
}

class EntityDetector:
    """
    Entity detector that uses YOLOv5, GroundingSAM, or fallback methods for object detection and classification.
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialize the EntityDetector with configuration settings.
        
        Args:
            config: Dictionary containing configuration parameters
        """
        self.config = config or {}
        self.confidence_threshold = self.config.get("confidence_threshold", 0.5)
        # Set GroundingSAM as the default detector
        self.detector_type = self.config.get("detector", "grounding_sam").lower()
        self.classes = self.config.get("classes", None)
        if self.classes:
            logger.info(f"Entity detection filtering for classes: {', '.join(self.classes)}")
        
        self.category_mapping = self.config.get("category_mapping", "static")
        self._clip_model = None
        self._clip_preprocess = None
        if self.category_mapping == "clip_grounding":
            try:
                import open_clip
                self._clip_model, self._clip_preprocess = open_clip.create_model_and_transforms(
                    "ViT-B-32", pretrained="openai", device=self.device
                )
                self._clip_model.eval()
                logger.info("CLIP model loaded for category mapping.")
            except Exception as e:
                logger.warning(f"CLIP model unavailable for category mapping: {e}")
                self.category_mapping = "static"
        
        # Detector selection logic
        if self.detector_type == "grounding_sam":
            logger.info("Using GroundingSAMDetector for entity detection.")
            try:
                from components.grounding_sam_wrapper import GroundingSAMDetector
                self.model = GroundingSAMDetector(self.config)
                self._using_grounding_sam = True
                self.device = self.config.get("device", "cuda")
            except ImportError as e:
                logger.error(f"Failed to import GroundingSAMDetector: {e}")
                raise
        else:
            self._using_grounding_sam = False
            # Check CUDA availability and set device
            if torch.cuda.is_available():
                try:
                    # Test CUDA with a small tensor
                    test_tensor = torch.zeros(1).cuda()
                    self.device = "cuda"
                    logger.info("CUDA is available and working")
                except Exception as e:
                    logger.warning(f"CUDA is available but encountered an error: {str(e)}")
                    self.device = "cpu"
                    logger.info("Falling back to CPU")
            else:
                self.device = "cpu"
                logger.info("CUDA is not available, using CPU")
            
            # Load YOLOv5 model
            logger.info(f"Loading YOLOv5 model on {self.device}...")
            try:
                self._initialize_model()
                logger.info(f"YOLOv5 model loaded successfully on {self.device}")
            except Exception as e:
                logger.error(f"Failed to load YOLOv5: {str(e)}")
                raise
            
        logger.info("Entity detector initialized")
        
    def _initialize_model(self):
        """Initialize YOLO model for object detection."""
        logger.info(f"Initializing YOLOv5 on {self.device}")
        
        # Import torch again to ensure it's available in this scope
        import torch
        
        # Clear CUDA cache if using GPU
        if self.device == "cuda":
            torch.cuda.empty_cache()
        
        # Use a direct approach to load YOLOv5
        try:
            # Get a fresh environment for YOLO to avoid module conflicts
            import os
            import sys
            import importlib
            import shutil
            
            # Save original state
            original_path = list(sys.path)
            original_modules = set(sys.modules.keys())
            
            # Make sure our TryExcept implementation isn't interfering
            if 'utils' in sys.modules:
                # Temporarily remove utils from imported modules
                utils_module = sys.modules.pop('utils')
            else:
                utils_module = None
            
            # Clear any cached imports that might be causing conflicts
            for mod_name in list(sys.modules.keys()):
                if mod_name.startswith('utils.') or mod_name == 'utils':
                    sys.modules.pop(mod_name, None)
            
            # Try directly loading the yolov5 model
            # First, make sure the repo is properly checked out and available
            try:
                # This ensures we have a fresh clone with correct structure
                torch.hub.help("ultralytics/yolov5", "yolov5s", force_reload=True)
                logger.info("Repository structure verified")
            except Exception as verify_err:
                logger.warning(f"Repo verification warning (non-fatal): {verify_err}")
            
            # Now load the model with explicit paths to avoid module confusion
            self.model = torch.hub.load(
                'ultralytics/yolov5', 
                'yolov5s', 
                pretrained=True,
                force_reload=True,
                trust_repo=True,
                verbose=False,
                skip_validation=True  # Skip validation to avoid import issues
            )
            
            # Move model to appropriate device
            if self.device == "cuda":
                self.model.cuda()
            else:
                self.model.cpu()
            
            # Restore original modules
            if utils_module is not None:
                sys.modules['utils'] = utils_module
                
        except Exception as e:
            logger.error(f"Standard loading failed: {str(e)}")
            
            # Try alternative method using a clone of YOLOv5
            try:
                logger.info("Trying alternative model loading method...")
                
                # Create a temporary directory for YOLOv5
                import tempfile
                temp_dir = tempfile.mkdtemp(prefix="yolov5_temp_")
                
                try:
                    # Clone the repository to this temp directory
                    import subprocess
                    subprocess.check_call([
                        "git", "clone", "https://github.com/ultralytics/yolov5.git", 
                        temp_dir, "--depth", "1"
                    ])
                    
                    # Add the temp directory to python path
                    sys.path.insert(0, temp_dir)
                    
                    # Load model directly from the local clone
                    sys.path.insert(0, temp_dir)
                    from models.experimental import attempt_load
                    
                    # Get pretrained weights path
                    weights_path = os.path.join(temp_dir, 'yolov5s.pt')
                    # Download if needed
                    if not os.path.exists(weights_path):
                        import torch.hub
                        torch.hub.download_url_to_file(
                            'https://github.com/ultralytics/yolov5/releases/download/v6.0/yolov5s.pt',
                            weights_path
                        )
                    
                    # Load the model
                    self.model = attempt_load(weights_path, device=self.device)
                    logger.info("YOLOv5 model loaded using alternative method")
                    
                finally:
                    # Cleanup
                    sys.path.remove(temp_dir)
                    try:
                        shutil.rmtree(temp_dir)
                    except:
                        pass
            
            except Exception as inner_e:
                logger.error(f"Alternative loading also failed: {str(inner_e)}")
                
                # Final fallback: Use a simpler model included with OpenCV
                logger.warning("Attempting final fallback to OpenCV built-in detection...")
                try:
                    import cv2
                    import numpy as np
                    
                    # Create a very simple class detector using OpenCV's built-in models
                    # This isn't YOLOv5, but will allow the pipeline to continue with basic detection
                    class SimpleDetector:
                        def __init__(self):
                            # Try to load a pre-trained model from OpenCV
                            self.net = None
                            try:
                                # Try SSD MobileNet if available
                                self.net = cv2.dnn.readNetFromCaffe(
                                    "MobileNetSSD_deploy.prototxt.txt",
                                    "MobileNetSSD_deploy.caffemodel"
                                )
                                self.classes = ["background", "aeroplane", "bicycle", "bird", "boat",
                                              "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
                                              "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
                                              "sofa", "train", "tvmonitor"]
                                self.model_type = "ssd"
                                logger.info("Using SSD MobileNet as fallback detector")
                            except:
                                # If SSD fails, try Haar cascade for face detection only
                                try:
                                    self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
                                    self.model_type = "haar"
                                    self.classes = ["person"]  # Only detect people
                                    logger.info("Using Haar Cascade face detector as minimal fallback")
                                except:
                                    logger.error("Could not load any OpenCV models")
                                    raise
                                
                        def __call__(self, img):
                            # Simple class to mimic YOLOv5's return format
                            class Results:
                                def __init__(self, boxes, names):
                                    self.xyxy = [boxes]
                                    self.names = names
                                
                            h, w = img.shape[:2]
                            boxes = []
                            
                            if self.model_type == "ssd" and self.net is not None:
                                # Use SSD model
                                blob = cv2.dnn.blobFromImage(img, 0.007843, (300, 300), 127.5)
                                self.net.setInput(blob)
                                detections = self.net.forward()
                                
                                for i in range(detections.shape[2]):
                                    confidence = detections[0, 0, i, 2]
                                    if confidence > 0.5:  # Confidence threshold
                                        class_id = int(detections[0, 0, i, 1])
                                        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                                        x1, y1, x2, y2 = box.astype("float")
                                        conf = float(confidence)
                                        cls = float(class_id)
                                        boxes.append([x1, y1, x2, y2, conf, cls])
                                        
                            elif self.model_type == "haar" and hasattr(self, 'face_cascade'):
                                # Use Haar cascade for face detection only
                                gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                                faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
                                
                                for (x, y, w, h) in faces:
                                    boxes.append([float(x), float(y), float(x+w), float(y+h), 0.8, 0.0])
                                    
                            return Results(np.array(boxes), self.classes)
                    
                    # Create our simple detector
                    self.model = SimpleDetector()
                    self._using_opencv_detector = True
                    logger.info("Using simplified OpenCV-based detection as fallback")
                    
                except Exception as final_e:
                    logger.error(f"All YOLOv5 loading methods failed: {str(final_e)}")
                    raise RuntimeError("Failed to initialize YOLOv5 model using all methods")
        
    def detect_entities(self, keyframes_metadata: Dict, keyframes_dir: str) -> Dict:
        """
        Detect entities in keyframes using the selected detector.
        
        Args:
            keyframes_metadata: Dictionary containing keyframe metadata
            keyframes_dir: Directory containing keyframe images
            
        Returns:
            Dictionary with detected entities for each scene
        """
        keyframes_dir = Path(keyframes_dir)
        if not keyframes_dir.exists():
            raise FileNotFoundError(f"Keyframes directory not found: {keyframes_dir}")
            
        logger.info("Starting entity detection")
        
        # Ensure we have a standardized format with a "scenes" key
        if not keyframes_metadata:
            logger.warning("Empty keyframes metadata provided for entity detection. Creating empty results.")
            return {"scenes": {}}
            
        # Handle both formats: {"scenes": {scene_idx: []}} and {scene_idx: []}
        if "scenes" in keyframes_metadata:
            scenes_data = keyframes_metadata["scenes"]
        else:
            logger.warning("Keyframes metadata missing 'scenes' key, attempting to use direct scene indexing")
            scenes_data = keyframes_metadata
            
        # Check if there's actually any scene data
        if not scenes_data:
            logger.warning("No scenes data available for entity detection. Creating empty results.")
            return {"scenes": {}}
        
        # Process each scene
        results = {"scenes": {}}
        
        try:
            for scene_idx, keyframes in scenes_data.items():
                logger.info(f"Processing scene {scene_idx}")
                
                # Skip if no keyframes in this scene
                if not keyframes:
                    logger.warning(f"No keyframes in scene {scene_idx}, skipping")
                    results["scenes"][scene_idx] = []
                    continue
                
                scene_entities = []
                # Ensure scene_idx is an int for formatting
                scene_idx_int = int(scene_idx) if not isinstance(scene_idx, int) else scene_idx
                mask_dir = keyframes_dir.parent / "entity_masks" / f"scene_{scene_idx_int:04d}"
                mask_dir.mkdir(parents=True, exist_ok=True)
                
                # Process each keyframe in the scene
                for keyframe in keyframes:
                    # Check if required fields are present
                    if "path" not in keyframe:
                        logger.warning(f"Keyframe missing 'path' field in scene {scene_idx}, skipping")
                        continue
                        
                    # Load the keyframe image
                    frame_path = keyframes_dir / keyframe["path"]
                    if not frame_path.exists():
                        logger.warning(f"Keyframe image not found: {frame_path}")
                        continue
                        
                    # Read the image using OpenCV (which reads in BGR)
                    image = cv2.imread(str(frame_path))
                    if image is None:
                        logger.warning(f"Failed to load keyframe image: {frame_path}")
                        continue
                    
                    # Detect entities in the image
                    try:
                        # Use selected detector
                        if getattr(self, '_using_grounding_sam', False):
                            detections = self.model.detect(image)
                            # Save masks if present
                            for i, det in enumerate(detections):
                                mask = det.get('mask')
                                if mask is not None:
                                    # Ensure mask is numpy array before processing
                                    mask_array = np.array(mask, dtype=bool)
                                    # Save mask as PNG
                                    mask_img = Image.fromarray((mask_array * 255).astype('uint8'))
                                    mask_filename = f"mask_{uuid.uuid4().hex[:8]}.png"
                                    mask_path = mask_dir / mask_filename
                                    mask_img.save(mask_path)
                                    det['mask_path'] = str(mask_path.relative_to(keyframes_dir.parent))
                                    # Ensure mask in detection results is numpy array for consistency
                                    det['mask'] = mask_array
                        else:
                            detections = self._detect_entities_in_image(image)
                        
                        if detections:
                            # Ensure frame_idx and timestamp are properly converted
                            try:
                                frame_idx = int(keyframe.get("frame_idx", 0))
                            except (ValueError, TypeError):
                                frame_idx = 0
                                
                            try:
                                timestamp = float(keyframe.get("timestamp", 0.0))
                            except (ValueError, TypeError):
                                timestamp = 0.0
                                
                            scene_entities.append({
                                "frame_idx": frame_idx,
                                "timestamp": timestamp,
                                "entities": detections
                            })
                    except Exception as e:
                        logger.error(f"Error processing keyframe {keyframe}: {e}")
                        continue
                
                results["scenes"][scene_idx] = scene_entities
                logger.info(f"Detected entities in {len(scene_entities)} keyframes for scene {scene_idx}")
                gc.collect()  # Force garbage collection after each scene
                
        except Exception as e:
            logger.error(f"Error in entity detection: {str(e)}")
            raise
            
        logger.info(f"Completed entity detection for {len(results['scenes'])} scenes")
        return results
        
    def _detect_entities_in_image(self, image: np.ndarray) -> Dict[str, List[Dict]]:
        """
        Detect entities in an image using YOLOv5 or fallback methods.
        
        Args:
            image: Image to detect entities in
            
        Returns:
            Dictionary with detected entities by category
        """
        # Initialize empty detections
        detections = {
            "objects": [],
            "faces": [],
            "text": []  # Placeholder for future text detection
        }
        
        # Get image dimensions
        h, w = image.shape[:2]
        
        # Track person count for context
        person_count = 0
        all_detections = []
        
        try:
            # Check if we're using an OpenCV fallback detector
            if hasattr(self, '_using_opencv_detector') and self._using_opencv_detector:
                # Process with our simple OpenCV-based detector
                try:
                    # Convert BGR to RGB for consistency
                    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    
                    # Run detection using our fallback detector class
                    # (it has a __call__ method that mimics the YOLOv5 model output format)
                    results = self.model(image_rgb)
                    
                    # Process results - code reused with standard YOLOv5 section below
                    # since we made our SimpleDetector return compatible format
                    for pred in results.xyxy[0]:
                        if len(pred) >= 6:  # Make sure we have enough values
                            x1, y1, x2, y2, conf, cls = pred
                            
                            # Skip if confidence is below threshold
                            if float(conf) < self.confidence_threshold:
                                continue
                                
                            # Get class name
                            try:
                                label = results.names[int(cls)]
                            except (IndexError, ValueError):
                                label = f"class_{int(cls) if isinstance(cls, (int, float)) else 0}"
                            
                            # Apply class filtering if configured
                            if self.classes and label.lower() not in (c.lower() for c in self.classes):
                                continue
                            
                            # Convert to normalized coordinates and ensure native Python types
                            bbox = [
                                max(0, float(x1/w)),
                                max(0, float(y1/h)),
                                min(1.0, float(x2/w)), 
                                min(1.0, float(y2/h))
                            ]
                            
                            # Classify entity type
                            entity_type = self._classify_entity_type(label)
                            
                            # Create detection object with native Python types
                            detection = {
                                "label": str(label),
                                "confidence": float(conf),
                                "bbox": bbox,
                                "type": str(entity_type),
                                "area": float((x2-x1) * (y2-y1) / (w*h))  # Normalized area
                            }
                            
                            # Count persons
                            if entity_type == "person":
                                person_count += 1
                                
                            all_detections.append(detection)
                
                except Exception as ocv_e:
                    logger.error(f"Error in OpenCV fallback detection: {str(ocv_e)}")
                    
            else:
                # Use standard YOLOv5 processing
                # Convert BGR to RGB
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # Run YOLOv5 inference
                results = self.model(image_rgb)
                
                # Process results
                for pred in results.xyxy[0]:  # xyxy format
                    x1, y1, x2, y2, conf, cls = pred.cpu().numpy()
                    
                    # Skip if confidence is below threshold
                    if float(conf) < self.confidence_threshold:
                        continue
                        
                    label = results.names[int(cls)]
                    
                    # Apply class filtering if configured
                    if self.classes and label.lower() not in (c.lower() for c in self.classes):
                        continue
                    
                    # Convert to normalized coordinates and ensure native Python types
                    bbox = [
                        float(x1/w),
                        float(y1/h),
                        float(x2/w),
                        float(y2/h)
                    ]
                    
                    # Classify entity type
                    entity_type = self._classify_entity_type(label)
                    
                    # Create detection object with native Python types
                    detection = {
                        "label": str(label),
                        "confidence": float(conf),
                        "bbox": bbox,
                        "type": str(entity_type),
                        "area": float((x2-x1) * (y2-y1) / (w*h))  # Normalized area
                    }
                    
                    # Count persons
                    if entity_type == "person":
                        person_count += 1
                        
                    all_detections.append(detection)
                
            # Second pass - enhance all detections with context
            for detection in all_detections:
                # Add extra metadata for context
                detection["description"] = self._create_entity_description(detection, person_count)
                
                # Add to appropriate category
                if detection["type"] == "person":
                    detections["faces"].append(detection)
                else:
                    detections["objects"].append(detection)
            
            return detections
            
        except Exception as e:
            logger.error(f"Error in entity detection: {str(e)}")
            return {"objects": [], "faces": [], "text": []}
        
    def _classify_entity_type(self, label: str) -> str:
        """
        Classify entity labels into semantic categories using static lookup or CLIP+OpenAI grounding.

        Args:
            label: Detection label (string)

        Returns:
            Entity category (string)
        """
        label = label.lower()
        if self.category_mapping == "clip_grounding" and self._clip_model is not None:
            try:
                import torch
                import open_clip
                # Prepare candidate categories
                categories = list(ENTITY_CATEGORIES.keys())
                # Encode label and categories
                texts = [label] + categories
                text_tokens = open_clip.tokenize(texts).to(self.device)
                with torch.no_grad():
                    text_features = self._clip_model.encode_text(text_tokens)
                    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                label_feat = text_features[0]
                cat_feats = text_features[1:]
                # Compute similarity
                sims = (cat_feats @ label_feat).squeeze()
                best_idx = int(torch.argmax(sims).item())
                best_category = categories[best_idx]
                logger.debug(f"CLIP-mapped '{label}' to '{best_category}' (score={sims[best_idx]:.2f})")
                return best_category
            except Exception as e:
                logger.warning(f"CLIP category mapping failed for '{label}': {e}")
                # Fallback to static
        # --- Static fallback (original logic) ---
        # First check for exact matches
        for category, keywords in ENTITY_CATEGORIES.items():
            if label in keywords:
                return category
        # Then check for keyword matches
        best_category = "object"  # Default
        best_score = 0
        for category, keywords in ENTITY_CATEGORIES.items():
            for keyword in keywords:
                if f" {keyword} " in f" {label} " or label.startswith(f"{keyword} ") or label.endswith(f" {keyword}") or label == keyword:
                    return category
                if keyword in label:
                    score = len(keyword) / len(label)
                    if score > best_score:
                        best_score = score
                        best_category = category
        if best_score > 0.5:
            logger.debug(f"Classified '{label}' as '{best_category}' with confidence {best_score:.2f}")
        return best_category
        
    def _create_entity_description(self, detection: Dict, person_count: int) -> str:
        """
        Create a descriptive string for the entity based on its type and context.
        
        Args:
            detection: Entity detection information
            person_count: Number of people detected in the image
            
        Returns:
            Descriptive string for the entity
        """
        entity_type = detection["type"]
        label = detection["label"]
        confidence = detection["confidence"]
        area = detection["area"]
        
        # Size description
        size_desc = "small"
        if area > 0.25:
            size_desc = "large"
        elif area > 0.1:
            size_desc = "medium"
        
        # Basic description based on type
        if entity_type == "person":
            if person_count > 1:
                # Person in a group
                return f"{label} ({size_desc}, one of {person_count} people)"
            else:
                # Single person
                return f"{label} ({size_desc})"
        elif entity_type == "setting":
            return f"{label} setting"
        elif entity_type == "environment":
            return f"{label} environment"
        elif entity_type == "action":
            if person_count > 0:
                return f"{label} action (with {person_count} people)"
            else:
                return f"{label} action"
        else:
            # Default object description
            return f"{size_desc} {label}"
        
    def _save_results(self, results: Dict, output_path: str):
        """Save entity detection results to a JSON file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        class NumpyEncoder(json.JSONEncoder):
            """Custom JSON encoder for numpy types."""
            def default(self, obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                return super().default(obj)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, cls=NumpyEncoder, ensure_ascii=False)
        
        logger.info(f"Saved entity detection results to {output_path}")

    def _create_summary(self, results: Dict) -> Dict:
        """
        Create a summary of entity detection results with enhanced categorization.
        
        Args:
            results: Dictionary with detection results
            
        Returns:
            Dictionary containing summary statistics
        """
        summary = {
            "total_scenes": len(results.get("scenes", {})),
            "total_entities": {
                "objects": 0,
                "faces": 0,
                "text": 0
            },
            "entity_types": {},
            "entity_categories": {
                "person": 0,
                "setting": 0,
                "environment": 0,
                "object": 0,
                "action": 0
            },
            "entity_labels": {},  # Track specific object labels
            "scenes": {}
        }
        
        # Scene-level tracking for category distribution
        scene_category_counts = {}
        
        # Process each scene
        for scene_idx, scene_data in results.get("scenes", {}).items():
            scene_summary = {
                "num_frames": len(scene_data),
                "entities": {
                    "objects": 0,
                    "faces": 0,
                    "text": 0
                },
                "entity_types": {},
                "entity_categories": {
                    "person": 0,
                    "setting": 0,
                    "environment": 0,
                    "object": 0,
                    "action": 0
                },
                "entity_labels": {}  # Track specific object labels by scene
            }
            
            # Process each frame in the scene
            for frame in scene_data:
                frame_entities = frame.get("entities", {})
                
                # Count entities by category
                for category in ["objects", "faces", "text"]:
                    count = len(frame_entities.get(category, []))
                    scene_summary["entities"][category] += count
                    summary["total_entities"][category] += count
                
                # Count entity types and specific labels
                for obj in frame_entities.get("objects", []):
                    entity_type = obj.get("type", "unknown")
                    entity_label = obj.get("label", "unknown")
                    
                    # Scene-specific counts
                    scene_summary["entity_types"][entity_type] = scene_summary["entity_types"].get(entity_type, 0) + 1
                    
                    # Update scene category counts
                    if entity_type in scene_summary["entity_categories"]:
                        scene_summary["entity_categories"][entity_type] += 1
                    
                    # Track detailed labels with confidence
                    label_key = f"{entity_label} ({entity_type})"
                    confidence = obj.get("confidence", 0)
                    
                    if label_key not in scene_summary["entity_labels"]:
                        scene_summary["entity_labels"][label_key] = {
                            "count": 1,
                            "avg_confidence": confidence,
                            "max_confidence": confidence
                        }
                    else:
                        current = scene_summary["entity_labels"][label_key]
                        current["count"] += 1
                        # Update confidence metrics
                        current["avg_confidence"] = (NumpyEncodercurrent["avg_confidence"] * (current["count"] - 1) + confidence) / current["count"]
                        current["max_confidence"] = max(current["max_confidence"], confidence)
                    
                    # Global counts
                    summary["entity_types"][entity_type] = summary["entity_types"].get(entity_type, 0) + 1
                    
                    # Update global category counts
                    if entity_type in summary["entity_categories"]:
                        summary["entity_categories"][entity_type] += 1
                    
                    # Global label tracking
                    if label_key not in summary["entity_labels"]:
                        summary["entity_labels"][label_key] = {
                            "count": 1,
                            "avg_confidence": confidence,
                            "max_confidence": confidence
                        }
                    else:
                        current = summary["entity_labels"][label_key]
                        current["count"] += 1
                        # Update confidence metrics
                        current["avg_confidence"] = (current["avg_confidence"] * (current["count"] - 1) + confidence) / current["count"]
                        current["max_confidence"] = max(current["max_confidence"], confidence)
                
                # Process faces similarly (they are people)
                for face in frame_entities.get("faces", []):
                    entity_type = face.get("type", "person")
                    
                    # Update category counts
                    scene_summary["entity_categories"]["person"] += 1
                    summary["entity_categories"]["person"] += 1
                    
                    # Rest of face processing...
            
            # Track which scenes have which categories
            for category, count in scene_summary["entity_categories"].items():
                if count > 0:
                    if category not in scene_category_counts:
                        scene_category_counts[category] = []
                    scene_category_counts[category].append(scene_idx)
            
            summary["scenes"][scene_idx] = scene_summary
        
        # Add overall statistics
        summary["statistics"] = {
            "avg_objects_per_scene": sum(s["entities"]["objects"] for s in summary["scenes"].values()) / max(1, summary["total_scenes"]),
            "avg_faces_per_scene": sum(s["entities"]["faces"] for s in summary["scenes"].values()) / max(1, summary["total_scenes"]),
            "total_unique_entity_types": len(summary["entity_types"]),
            "total_unique_entity_labels": len(summary["entity_labels"]),
            "most_common_entity_types": sorted(
                summary["entity_types"].items(),
                key=lambda x: x[1],
                reverse=True
            )[:10],
            "most_common_entity_labels": sorted(
                [(label, data["count"]) for label, data in summary["entity_labels"].items()],
                key=lambda x: x[1],
                reverse=True
            )[:20],
            "category_distribution": {
                category: count for category, count in summary["entity_categories"].items() if count > 0
            },
            "category_scene_coverage": {
                category: {
                    "scenes": scenes,
                    "coverage_percentage": (len(scenes) / summary["total_scenes"]) * 100 if summary["total_scenes"] > 0 else 0
                } for category, scenes in scene_category_counts.items()
            }
        }
        
        # Add predominant category for each scene
        for scene_idx, scene_data in summary["scenes"].items():
            if scene_data["entity_categories"]:
                predominant = max(scene_data["entity_categories"].items(), key=lambda x: x[1])
                scene_data["predominant_category"] = predominant[0]
        
        return summary

    def _save_summary(self, summary: Dict, summary_path: str):
        """Save entity detection summary to a JSON file."""
        summary_path = Path(summary_path)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, cls=NumpyEncoder, ensure_ascii=False)
        
        logger.info(f"Saved entity detection summary to {summary_path}")

# Helper function
def detect_scene_entities(keyframes_metadata: Dict, keyframes_dir: str, config: Dict = None) -> Dict:
    """
    Convenience function to detect entities in scene keyframes.
    
    Args:
        keyframes_metadata: Dictionary containing keyframe metadata
        keyframes_dir: Directory containing keyframe images
        config: Optional configuration dictionary
        
    Returns:
        Dictionary with detected entities for each scene
    """
    detector = EntityDetector(config)
    results = detector.detect_entities(keyframes_metadata, keyframes_dir)
    return results 