from controlnet_aux import LineartAnimeDetector
import torch
from PIL import Image
import numpy as np
import os

class LineartAnimeWrapper:
    def __init__(self, 
                 pretrained_model_or_path="lllyasviel/Annotators",
                 filename=None, 
                 cache_dir=None, 
                 local_files_only=False,
                 device=None):
        """
        Wrapper for LineartAnimeDetector with configurable parameters
        
        Args:
            pretrained_model_or_path (str): Path to pretrained model or HuggingFace model ID
            filename (str, optional): Specific model filename. Defaults to None.
            cache_dir (str, optional): Directory to cache downloaded models. Defaults to None.
            local_files_only (bool): If True, only use local files. Defaults to False.
            device (str, optional): Device to run the model on ('cpu' or 'cuda'). Defaults to None.
        """
        self.detector = LineartAnimeDetector.from_pretrained(
            pretrained_model_or_path=pretrained_model_or_path,
            filename=filename,
            cache_dir=cache_dir,
            local_files_only=local_files_only
        )
        
        # Set device
        if device is not None:
            self.device = device
        else:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.detector.to(self.device)
    
    def __call__(self, 
                input_image, 
                detect_resolution=512, 
                image_resolution=512,
                output_type="pil",
                **kwargs):
        """
        Process an image with LineartAnimeDetector
        
        Args:
            input_image (PIL.Image or np.ndarray): Input image
            detect_resolution (int): Resolution for detection. Defaults to 512.
            image_resolution (int): Output image resolution. Defaults to 512.
            output_type (str): Output type, either "pil" or "np". Defaults to "pil".
            **kwargs: Additional arguments to pass to the detector
            
        Returns:
            PIL.Image or np.ndarray: Processed image
        """
        return self.detector(
            input_image=input_image,
            detect_resolution=detect_resolution,
            image_resolution=image_resolution,
            output_type=output_type,
            **kwargs
        )
    
    def to(self, device):
        """
        Move the detector to the specified device
        
        Args:
            device (str): Device to move to ('cpu' or 'cuda')
            
        Returns:
            LineartAnimeWrapper: Self
        """
        self.device = device
        self.detector.to(device)
        return self
    
    def process_file(self, 
                    file_path, 
                    detect_resolution=512, 
                    image_resolution=512,
                    output_type="pil",
                    save_path=None,
                    **kwargs):
        """
        Process an image file with LineartAnimeDetector
        
        Args:
            file_path (str): Path to input image file
            detect_resolution (int): Resolution for detection. Defaults to 512.
            image_resolution (int): Output image resolution. Defaults to 512.
            output_type (str): Output type, either "pil" or "np". Defaults to "pil".
            save_path (str, optional): Path to save the output image. Defaults to None.
            **kwargs: Additional arguments to pass to the detector
            
        Returns:
            PIL.Image or np.ndarray: Processed image
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Input file not found: {file_path}")
        
        # Load image
        input_image = Image.open(file_path).convert("RGB")
        
        # Process image
        processed_image = self(
            input_image=input_image,
            detect_resolution=detect_resolution,
            image_resolution=image_resolution,
            output_type=output_type,
            **kwargs
        )
        
        # Save if requested
        if save_path is not None:
            if output_type == "pil":
                processed_image.save(save_path)
            else:
                Image.fromarray(processed_image).save(save_path)
        
        return processed_image