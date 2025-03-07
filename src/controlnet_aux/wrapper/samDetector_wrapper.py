import os
from typing import Dict, Any, Union, Optional
import numpy as np
from PIL import Image
import torch
import logging
import warnings

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("SamDetectorWrapper")

class SamDetectorWrapper:
    """
    A wrapper class for the SamDetector that allows for easy initialization, 
    configuration management, and image processing.
    """
    
    def __init__(
        self, 
        pretrained_model_or_path: str,
        model_type: str = "vit_h", 
        filename: str = None,
        subfolder: str = None,
        cache_dir: str = None,
        force_cpu: bool = True,  # Force CPU mode by default for reliability
        **kwargs
    ):
        """
        Initialize the SamDetectorWrapper with a model and optional configurations.
        
        Args:
            pretrained_model_or_path: Path to the model weights or model ID on HuggingFace Hub
            model_type: Type of SAM model ('vit_h', 'vit_l', 'vit_b', 'vit_t')
            filename: Name of the model file (if not using default)
            subfolder: Optional subfolder within the model path
            cache_dir: Optional cache directory for downloading models
            force_cpu: Force CPU mode for reliable operation (recommended)
            **kwargs: Additional parameters for the SamAutomaticMaskGenerator
        """
        # Always use CPU mode - this is the most reliable approach for this model
        self.device = "cpu" if force_cpu else ("cuda" if torch.cuda.is_available() else "cpu")
        
        if not force_cpu and self.device != "cpu":
            warnings.warn(
                "Using GPU mode may cause device errors with SAM model. "
                "If you encounter problems, try setting force_cpu=True.",
                UserWarning
            )
        
        logger.info(f"Using device: {self.device}")
        
        self.model_params = {
            "pretrained_model_or_path": pretrained_model_or_path,
            "model_type": model_type,
            "filename": self._get_default_filename(model_type) if filename is None else filename,
            "subfolder": subfolder,
            "cache_dir": cache_dir
        }
        
        # Default detector parameters
        self.detector_params = {
            "points_per_side": 32,
            "points_per_batch": 64,
            "pred_iou_thresh": 0.88,
            "stability_score_thresh": 0.95,
            "stability_score_offset": 1.0,
            "box_nms_thresh": 0.7,
            "crop_n_layers": 0,
            "crop_nms_thresh": 0.7,
            "crop_overlap_ratio": 512/1500,
            "crop_n_points_downscale_factor": 1,
            "min_mask_region_area": 0,
            "output_mode": "binary_mask"
        }
        
        # Default image processing parameters
        self.image_params = {
            "detect_resolution": 512,
            "image_resolution": 512,
            "output_type": "pil"
        }
        
        # Update parameters with any provided kwargs
        self.update_params(reinitialize=False, **kwargs)
        
        # Store named configurations
        self.configurations = {}
        
        # Initialize detector
        self._initialize_detector()
        
    def _get_default_filename(self, model_type: str) -> str:
        """Get the default filename for a given model type."""
        mapping = {
            "vit_h": "sam_vit_h_4b8939.pth",
            "vit_l": "sam_vit_l_0b3195.pth",
            "vit_b": "sam_vit_b_01ec64.pth",
            "vit_t": "mobile_sam.pt"
        }
        return mapping.get(model_type, "sam_vit_h_4b8939.pth")
    
    def _initialize_detector(self) -> None:
        """Initialize or reinitialize the detector with current parameters."""
        # Import SamDetector here to avoid circular imports
        from controlnet_aux.segment_anything import SamDetector
        
        logger.info(f"Initializing detector with model type: {self.model_params['model_type']} on {self.device}")
        
        # Extract model parameters
        model_params = {k: v for k, v in self.model_params.items() if v is not None}
        
        # Create detector instance with model params
        self.detector = SamDetector.from_pretrained(**model_params)
        
        # Make sure model is on CPU
        if hasattr(self.detector, 'mask_generator') and hasattr(self.detector.mask_generator, 'predictor'):
            if hasattr(self.detector.mask_generator.predictor, 'model'):
                self.detector.mask_generator.predictor.model.to(self.device)
        
        # Configure detector mask generator
        if hasattr(self.detector, 'mask_generator'):
            for param, value in self.detector_params.items():
                if hasattr(self.detector.mask_generator, param):
                    setattr(self.detector.mask_generator, param, value)
        
        # Set up SAM-specific fixes
        self._setup_sam_fixes()
                    
        logger.info("Detector initialized successfully")
    
    def _setup_sam_fixes(self):
        """Apply fixes specific to SAM model to prevent device issues."""
        if not hasattr(self.detector, 'mask_generator') or not hasattr(self.detector.mask_generator, 'predictor'):
            return
            
        predictor = self.detector.mask_generator.predictor
        
        # Monkey patch the set_torch_image method to ensure all inputs go to the right device
        original_set_torch_image = predictor.set_torch_image
        
        def patched_set_torch_image(self, transformed_image, original_image_size):
            # Ensure the transformed image is on the right device
            transformed_image = transformed_image.to(self.device)
            return original_set_torch_image(transformed_image, original_image_size)
        
        # Bind the new method to the predictor instance
        import types
        predictor.set_torch_image = types.MethodType(patched_set_torch_image, predictor)
    
    def save_configuration(self, name: str) -> None:
        """
        Save the current configuration under a given name.
        
        Args:
            name: Name to identify this configuration
        """
        self.configurations[name] = {
            "model_params": self.model_params.copy(),
            "detector_params": self.detector_params.copy(),
            "image_params": self.image_params.copy(),
            "device": self.device
        }
        logger.info(f"Configuration '{name}' saved")
    
    def load_configuration(self, name: str) -> None:
        """
        Load a saved configuration.
        
        Args:
            name: Name of the configuration to load
        """
        if name not in self.configurations:
            raise ValueError(f"Configuration '{name}' not found")
        
        logger.info(f"Loading configuration '{name}'")
        config = self.configurations[name]
        self.model_params = config["model_params"].copy()
        self.detector_params = config["detector_params"].copy()
        self.image_params = config["image_params"].copy()
        
        # Only update device if staying on CPU (to avoid device issues)
        if config["device"] == "cpu":
            self.device = "cpu"
        elif self.device == "cpu":
            logger.warning("Keeping CPU device despite configuration specifying GPU")
        
        # Update detector parameters
        if hasattr(self.detector, 'mask_generator'):
            for param, value in self.detector_params.items():
                if hasattr(self.detector.mask_generator, param):
                    setattr(self.detector.mask_generator, param, value)
        
        logger.info(f"Configuration '{name}' loaded")
    
    def update_params(self, reinitialize: bool = True, **kwargs) -> None:
        """
        Update the parameters for the detector.
        
        Args:
            reinitialize: Whether to reinitialize the detector after updating params
            **kwargs: Parameters to update
        """
        # Check for device update (but only allow CPU if force_cpu is True)
        if 'device' in kwargs:
            requested_device = kwargs.pop('device')
            if requested_device != "cpu" and self.device == "cpu":
                logger.warning(f"Ignoring request to change device to {requested_device}, staying on CPU for stability")
            elif requested_device == "cpu":
                self.device = "cpu"
        
        # Handle force_cpu parameter
        if 'force_cpu' in kwargs:
            force_cpu = kwargs.pop('force_cpu')
            if force_cpu:
                self.device = "cpu"
                logger.info("Forcing CPU mode for stability")
        
        # Track if model parameters are changed
        model_params_changed = False
        
        # Update model parameters
        for param in ['pretrained_model_or_path', 'model_type', 'filename', 'subfolder', 'cache_dir']:
            if param in kwargs:
                self.model_params[param] = kwargs.pop(param)
                model_params_changed = True
        
        # Update image parameters
        for param in ['detect_resolution', 'image_resolution', 'output_type']:
            if param in kwargs:
                self.image_params[param] = kwargs.pop(param)
                logger.info(f"Image parameter {param} updated")
        
        # Update detector parameters
        detector_params_updated = False
        for param, value in kwargs.items():
            self.detector_params[param] = value
            detector_params_updated = True
            logger.info(f"Detector parameter {param} updated")
        
        # Apply updates to the detector
        if reinitialize and model_params_changed:
            logger.info("Model parameters changed, reinitializing detector")
            self._initialize_detector()
        elif reinitialize and detector_params_updated and hasattr(self.detector, 'mask_generator'):
            logger.info("Updating detector parameters")
            for param, value in self.detector_params.items():
                if hasattr(self.detector.mask_generator, param):
                    setattr(self.detector.mask_generator, param, value)
    
    def __call__(
        self, 
        input_image: Union[np.ndarray, Image.Image], 
        **kwargs
    ) -> Union[np.ndarray, Image.Image]:
        """
        Process an image with the detector.
        
        Args:
            input_image: Input image as numpy array or PIL Image
            **kwargs: Optional parameters to override for this call only
        
        Returns:
            Processed image with segmentation masks
        """
        logger.info("Processing image")
        
        # Force CPU for safety
        if hasattr(self.detector, 'mask_generator') and hasattr(self.detector.mask_generator, 'predictor'):
            if hasattr(self.detector.mask_generator.predictor, 'model'):
                current_device = next(self.detector.mask_generator.predictor.model.parameters()).device
                if str(current_device) != self.device:
                    logger.info(f"Moving model from {current_device} to {self.device}")
                    self.detector.mask_generator.predictor.model.to(self.device)
        
        # Merge image parameters with any overrides
        call_params = self.image_params.copy()
        call_params.update(kwargs)
        
        # Enable safety mechanisms to avoid crashes
        with torch.inference_mode():
            try:
                return self.detector(input_image=input_image, **call_params)
            except Exception as e:
                logger.error(f"Error during inference: {str(e)}")
                logger.info("Attempting to reinitialize detector and retry...")
                
                # Last resort - reinitialize and try again
                self._initialize_detector()
                return self.detector(input_image=input_image, **call_params)