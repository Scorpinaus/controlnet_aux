import numpy as np
import os
import logging
from PIL import Image
from typing import Union, Optional, Dict, Any, Tuple
from skimage import morphology

from src.controlnet_aux.util import HWC3, resize_image
from src.controlnet_aux.anyline import AnylineDetector

logger = logging.getLogger(__name__)

class AnylineWrapper:
    """Wrapper class for AnylineDetector with enhanced customization options.
    
    This wrapper provides a simplified interface to the AnylineDetector with
    additional control over post-processing parameters and checkpoint loading.
    """

    DEFAULT_PARAMS = {
        "detect_resolution": 1280,
        "gaussian_sigma": 2.0,
        "intensity_threshold": 3,
        "min_size": 36,
        "lower_bound": 0,
        "upper_bound": 255,
        "connectivity": 1        
    }

    def __init__(
        self, 
        checkpoint_path: str = "lllyasviel/Annotators", 
        filename: Optional[str] = None, 
        subfolder: Optional[str] = None, 
        device: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """Initialize the AnylineDetector wrapper.
        
        Args:
            checkpoint_path: Path to the model checkpoint or HuggingFace model name
            filename: Optional specific filename for the checkpoint
            subfolder: Optional subfolder within the checkpoint path
            device: Optional device to load the model on (e.g., 'cuda', 'cpu')
            config: Optional configuration dictionary to override default parameters
        
        Raises:
            FileNotFoundError: If the specified checkpoint cannot be found
            RuntimeError: If the model fails to load
        """
        try:
            logging.info(f"Loading AnylineDetector from {checkpoint_path}")
            self.detector = self._load_detector(checkpoint_path, filename, subfolder)

            if device:
                logger.debug(f"Moving model to device: {device}")
                self.detector = self.detector.to(device)

            #Store default config with overrides
            self.config = self.DEFAULT_PARAMS.copy()
            if config:
                self.config.update(config)

            logger.info("AnylineDetector loaded successfully")

        except Exception as e:
            logger.error(f"Failed to initialize AnylneDetector: {str(e)}")
            raise
    
    
    def _load_detector(self, checkpoint_path: str, filename: Optional[str] = None, 
                      subfolder: Optional[str] = None) -> AnylineDetector:
        """Load the detector with custom checkpoint configuration.
        
        Args:
            checkpoint_path: Path to the model or HuggingFace model ID
            filename: Optional specific filename for the checkpoint
            subfolder: Optional subfolder within the checkpoint path
            
        Returns:
            Loaded AnylineDetector instance
            
        Raises:
            FileNotFoundError: If the checkpoint cannot be found
        """
        # Check if it's a local path
        if os.path.exists(checkpoint_path):
            if filename:
                full_path = os.path.join(checkpoint_path, filename)
                if not os.path.exists(full_path):
                    raise FileNotFoundError(f"Checkpoint file not found: {full_path}")
                checkpoint_path = full_path
            return AnylineDetector.from_pretrained(checkpoint_path)
        else:
            # Assume it's a HuggingFace model ID
            try:
                return AnylineDetector.from_pretrained(
                    checkpoint_path, 
                    filename=filename or "anyline.pth", 
                    subfolder=subfolder
                )
            except Exception as e:
                raise FileNotFoundError(f"Failed to load model from {checkpoint_path}: {str(e)}")

    def __call__(
        self,
        input_image: Union[np.ndarray, Image.Image],
        detect_resolution: Optional[int] = None,
        gaussian_sigma: Optional[float] = None,
        intensity_threshold: Optional[int] = None,
        min_size: Optional[int] = None,
        lower_bound: Optional[int] = None, 
        upper_bound: Optional[int] = None,
        connectivity: Optional[int] = None,
        output_type: str = "pil"
    ) -> Union[np.ndarray, Image.Image]:
        """Process an image with AnylineDetector with custom parameters.
        
        Args:
            input_image: Input image as numpy array or PIL Image
            detect_resolution: Resolution for detection
            gaussian_sigma: Sigma value for Gaussian blur
            intensity_threshold: Threshold for intensity detection
            min_size: Minimum size for object removal
            lower_bound: Lower bound for intensity mask
            upper_bound: Upper bound for intensity mask
            connectivity: Connectivity parameter for object removal
            output_type: Output type, either 'pil' or 'np'
            
        Returns:
            Processed image as numpy array or PIL Image
            
        Raises:
            ValueError: If input_image is invalid or output_type is not supported
        """
        # Validate input
        if input_image is None:
            raise ValueError("Input image cannot be None")
            
        if output_type not in ["pil", "np"]:
            raise ValueError(f"Unsupported output type: {output_type}")
        
        # Use instance config as defaults, override with any provided parameters
        params = self._get_parameters(
            detect_resolution, gaussian_sigma, intensity_threshold,
            min_size, lower_bound, upper_bound, connectivity
        )
        
        try:
            # Process using the base detector first
            result = self._process_base_detection(
                input_image, 
                params["detect_resolution"], 
                params["gaussian_sigma"], 
                params["intensity_threshold"]
            )
            
            # Apply post-processing with custom parameters
            result = self._apply_post_processing(
                result, 
                params["min_size"], 
                params["lower_bound"], 
                params["upper_bound"], 
                params["connectivity"]
            )
            
            # Convert to requested output format
            return self._convert_to_output_format(result, output_type)
        except Exception as e:
            logger.error(f"Error processing image: {str(e)}")
            raise

    def _get_parameters(self, detect_resolution, gaussian_sigma, intensity_threshold,
                       min_size, lower_bound, upper_bound, connectivity) -> Dict[str, Any]:
        """Merge default parameters with any provided overrides.
        
        Args:
            Various processing parameters that may be None
            
        Returns:
            Dictionary of parameters to use for processing
        """
        params = self.config.copy()
        
        # Update with any non-None parameters
        if detect_resolution is not None:
            params["detect_resolution"] = detect_resolution
        if gaussian_sigma is not None:
            params["gaussian_sigma"] = gaussian_sigma
        if intensity_threshold is not None:
            params["intensity_threshold"] = intensity_threshold
        if min_size is not None:
            params["min_size"] = min_size
        if lower_bound is not None:
            params["lower_bound"] = lower_bound
        if upper_bound is not None:
            params["upper_bound"] = upper_bound
        if connectivity is not None:
            params["connectivity"] = connectivity
            
        return params
    
    def _process_base_detection(self, input_image, detect_resolution, gaussian_sigma, intensity_threshold):
        """Run the base detector with given parameters.
        
        Args:
            input_image: Input image
            detect_resolution: Resolution for detection
            gaussian_sigma: Sigma for Gaussian blur
            intensity_threshold: Intensity threshold
            
        Returns:
            Processed numpy array
        """
        # Ensure input is numpy array
        if isinstance(input_image, Image.Image):
            input_image = np.array(input_image)

        # Convert to HWC3 format if necessary
        input_image = HWC3(input_image)
        
        # Resize image to detection resolution
        if detect_resolution is not None:
            input_image = resize_image(input_image, detect_resolution)
            
        logger.debug(f"Running detection with resolution={detect_resolution}, "
                     f"sigma={gaussian_sigma}, threshold={intensity_threshold}")
            
        return self.detector(
            input_image=input_image,
            detect_resolution=detect_resolution,
            guassian_sigma=gaussian_sigma,
            intensity_threshold=intensity_threshold,
            output_type="np"  # Always use numpy for intermediate processing
        )

    def _apply_post_processing(self, image, min_size, lower_bound, upper_bound, connectivity):
        """Apply custom post-processing to the detection result.
        
        Args:
            image: Input image array
            min_size: Minimum size for object removal
            lower_bound: Lower bound for intensity mask
            upper_bound: Upper bound for intensity mask
            connectivity: Connectivity parameter
            
        Returns:
            Post-processed image array
        """
        logger.debug(f"Post-processing with min_size={min_size}, bounds={lower_bound}-{upper_bound}, "
                     f"connectivity={connectivity}")
                     
        # Apply intensity mask
        if hasattr(self.detector, 'get_intensity_mask'):
            result = self.detector.get_intensity_mask(
                image, lower_bound=lower_bound, upper_bound=upper_bound
            )
            
            # Remove small objects based on min_size and connectivity
            cleaned = morphology.remove_small_objects(
                result.astype(bool), min_size=min_size, connectivity=connectivity
            )
            return result * cleaned
        return image

    def _convert_to_output_format(self, image, output_type):
        """Convert the processed image to the requested output format.
        
        Args:
            image: Input image (numpy array or PIL Image)
            output_type: Desired output type ('pil' or 'np')
            
        Returns:
            Image in the requested format
        """
        if output_type == "pil" and isinstance(image, np.ndarray):
            return Image.fromarray(image)
        elif output_type == "np" and isinstance(image, Image.Image):
            return np.array(image)
        return image
    
    def update_config(self, new_config: Dict[str, Any]) -> None:
        """Update the default configuration for this detector instance.
        
        Args:
            new_config: Dictionary of parameters to update
        """
        self.config.update(new_config)
        logger.debug(f"Updated configuration: {self.config}")