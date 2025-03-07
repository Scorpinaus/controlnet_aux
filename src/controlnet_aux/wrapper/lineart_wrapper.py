from controlnet_aux import LineartDetector
from PIL import Image
import numpy as np
import io
import logging
from typing import Union, Optional, Dict, Any, Literal

# Set up logging
logger = logging.getLogger(__name__)

class LineartDetectorWrapper:
    """
    Wrapper class for LineartDetector that provides a simplified interface
    for processing images with line art detection.
    
    This wrapper handles the complexity of the underlying LineartDetector
    and provides a clean, consistent API for image processing.
    """
    
    def __init__(
        self, 
        pretrained_model_path: str = "lllyasviel/Annotators",
        coarse: bool = False, 
        detect_resolution: int = 512, 
        image_resolution: int = 512, 
        device: Optional[str] = None
    ) -> None:
        """
        Initialize the LineartDetectorWrapper.
        
        Args:
            pretrained_model_path (str): Path to pretrained model or HuggingFace repo.
                Default is "lllyasviel/Annotators".
            coarse (bool): Whether to use the coarse model. Default is False.
            detect_resolution (int): Resolution used for detection. Default is 512.
            image_resolution (int): Output image resolution. Default is 512.
            device (Optional[str]): Device to load the model on. Default is None
                which uses the default device.
        """
        logger.info(f"Initializing LineartDetectorWrapper with coarse={coarse}")
        
        self.detector = self._load_model(pretrained_model_path)
        self.config = {
            "coarse": coarse,
            "detect_resolution": detect_resolution,
            "image_resolution": image_resolution
        }
        
        if device:
            self.to(device)
    
    def _load_model(self, pretrained_model_path: str) -> LineartDetector:
        """
        Load the LineartDetector model from the specified path.
        
        Args:
            pretrained_model_path (str): Path to pretrained model.
            
        Returns:
            LineartDetector: Loaded model.
        """
        try:
            return LineartDetector.from_pretrained(pretrained_model_path)
        except Exception as e:
            logger.error(f"Failed to load LineartDetector model: {e}")
            raise RuntimeError(f"Failed to load LineartDetector: {e}")
    
    def to(self, device: str) -> 'LineartDetectorWrapper':
        """
        Move the detector to the specified device.
        
        Args:
            device (str): Device to move the model to (e.g., 'cuda', 'cpu').
            
        Returns:
            LineartDetectorWrapper: Self for method chaining.
        """
        try:
            self.detector.to(device)
            logger.info(f"Moved LineartDetector to device: {device}")
            return self
        except Exception as e:
            logger.error(f"Failed to move LineartDetector to device {device}: {e}")
            raise RuntimeError(f"Failed to move to device {device}: {e}")
    
    def process(
        self, 
        image: Union[Image.Image, bytes, np.ndarray], 
        output_format: Literal["pil", "np", "bytes"] = "pil",
        **kwargs
    ) -> Union[Image.Image, np.ndarray, bytes]:
        """
        Process an image with the LineartDetector.
        
        Args:
            image (Union[Image.Image, bytes, np.ndarray]): Input image.
            output_format (Literal["pil", "np", "bytes"]): Desired output format.
                Default is "pil" for PIL.Image.
            **kwargs: Optional parameters to override the default configuration.
                
        Returns:
            Union[Image.Image, np.ndarray, bytes]: Processed image in requested format.
            
        Raises:
            ValueError: If the input image format is not supported or on processing error.
        """
        # Prepare the image
        prepared_image = self._prepare_input(image)
        
        # Merge config with any override parameters
        process_params = {**self.config, **kwargs}
        
        # Process the image
        try:
            # Set output_type based on desired format, but LineartDetector only supports "pil" or "np"
            internal_output_type = "np" if output_format == "np" else "pil"
            
            result = self.detector(
                input_image=prepared_image,
                output_type=internal_output_type,
                **process_params
            )
            
            # Convert to the requested output format
            return self._format_output(result, output_format)
            
        except Exception as e:
            logger.error(f"Image processing failed: {e}")
            raise ValueError(f"Failed to process image: {e}")
    
    def _prepare_input(self, image: Union[Image.Image, bytes, np.ndarray]) -> Union[Image.Image, np.ndarray]:
        """
        Prepare the input image for processing.
        
        Args:
            image (Union[Image.Image, bytes, np.ndarray]): Input image.
            
        Returns:
            Union[Image.Image, np.ndarray]: Prepared image.
            
        Raises:
            ValueError: If the input format is not supported.
        """
        if isinstance(image, bytes):
            return Image.open(io.BytesIO(image)).convert("RGB")
        elif isinstance(image, np.ndarray):
            return image
        elif isinstance(image, Image.Image):
            return image
        else:
            raise ValueError(f"Unsupported image format: {type(image)}")
    
    def _format_output(
        self, 
        result: Union[Image.Image, np.ndarray], 
        output_format: Literal["pil", "np", "bytes"]
    ) -> Union[Image.Image, np.ndarray, bytes]:
        """
        Format the output according to the requested format.
        
        Args:
            result (Union[Image.Image, np.ndarray]): Processing result.
            output_format (Literal["pil", "np", "bytes"]): Desired output format.
            
        Returns:
            Union[Image.Image, np.ndarray, bytes]: Formatted output.
        """
        if output_format == "bytes":
            if isinstance(result, np.ndarray):
                result = Image.fromarray(result)
            
            buffer = io.BytesIO()
            result.save(buffer, format='PNG')
            return buffer.getvalue()
        
        elif output_format == "np" and isinstance(result, Image.Image):
            return np.array(result)
        
        elif output_format == "pil" and isinstance(result, np.ndarray):
            return Image.fromarray(result)
        
        return result
    
    def update_config(self, **kwargs) -> None:
        """
        Update the configuration parameters for the detector.
        
        Args:
            **kwargs: New parameters to update.
        """
        for key, value in kwargs.items():
            if key in self.config:
                self.config[key] = value
                logger.debug(f"Updated config parameter {key} to {value}")
            else:
                logger.warning(f"Unknown configuration parameter: {key}")
    
    def __call__(self, *args, **kwargs) -> Union[Image.Image, np.ndarray, bytes]:
        """
        Make the class callable, equivalent to the process method.
        
        Args:
            *args, **kwargs: Arguments to pass to the process method.
            
        Returns:
            Union[Image.Image, np.ndarray, bytes]: Processed image.
        """
        return self.process(*args, **kwargs)