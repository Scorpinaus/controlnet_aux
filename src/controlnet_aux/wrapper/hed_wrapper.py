from controlnet_aux import HEDdetector
from PIL import Image
import numpy as np
import io

class HEDWrapper:
    def __init__(self, pretrained_model_path="lllyasviel/Annotators", device="cpu"):
        """
        Initialize the HED wrapper.
        
        Args:
            pretrained_model_path (str): Path to the pretrained model.
            device (str): Device to run the model on ('cpu' or 'cuda').
        """
        self.detector = HEDdetector.from_pretrained(pretrained_model_path)
        self.detector.to(device)
        self.device = device
    
    def _process_image(self, image, detect_resolution=512, image_resolution=512, 
                      safe=False, scribble=False, output_type="pil"):
        """Internal method to process an image with common parameters."""
        # Handle different input types
        if isinstance(image, bytes):
            image = Image.open(io.BytesIO(image)).convert("RGB")
        
        # Process the image
        return self.detector(
            input_image=image,
            detect_resolution=detect_resolution,
            image_resolution=image_resolution,
            safe=safe,
            scribble=scribble,
            output_type=output_type
        )
        
    def detect_edges(self, image, detect_resolution=512, image_resolution=512, 
                    safe=False, output_type="pil"):
        """
        Detect edges with soft transitions.
        
        Args:
            image: Input image (PIL.Image, np.ndarray, or bytes)
            detect_resolution: Resolution for detection
            image_resolution: Output image resolution
            safe: Whether to use safe step function
            output_type: Output type, "pil" or "np"
            
        Returns:
            Processed image with soft edges
        """
        return self._process_image(
            image, 
            detect_resolution=detect_resolution,
            image_resolution=image_resolution,
            safe=safe,
            scribble=False,
            output_type=output_type
        )
    
    def detect_scribble(self, image, detect_resolution=512, image_resolution=512, 
                       safe=False, output_type="pil"):
        """
        Detect edges as binary scribble lines.
        
        Args:
            image: Input image (PIL.Image, np.ndarray, or bytes)
            detect_resolution: Resolution for detection
            image_resolution: Output image resolution
            safe: Whether to use safe step function
            output_type: Output type, "pil" or "np"
            
        Returns:
            Processed image with binary scribble lines
        """
        return self._process_image(
            image, 
            detect_resolution=detect_resolution,
            image_resolution=image_resolution,
            safe=safe,
            scribble=True,
            output_type=output_type
        )