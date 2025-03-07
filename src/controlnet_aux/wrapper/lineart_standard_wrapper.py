from controlnet_aux import LineartStandardDetector

class LineartStandardWrapper:
    def __init__(
        self,
        guassian_sigma=6.0,
        intensity_threshold=8,
        detect_resolution=512,
        output_type="pil"
    ):
        """Wrapper for LineartStandardDetector with configurable parameters
        
        Args:
            guassian_sigma (float): Gaussian blur sigma. Default: 6.0
            intensity_threshold (int): Intensity threshold for line detection. Default: 8
            detect_resolution (int): Resolution for detection. Default: 512
            output_type (str): Output type, either "pil" or "np". Default: "pil"
        """
        self.detector = LineartStandardDetector()
        self.params = {
            "guassian_sigma": guassian_sigma,
            "intensity_threshold": intensity_threshold,
            "detect_resolution": detect_resolution,
            "output_type": output_type
        }
    
    def __call__(self, image):
        """Process an image with LineartStandardDetector
        
        Args:
            image: Input image (PIL Image or numpy array)
            
        Returns:
            Processed image (PIL Image or numpy array, depending on output_type)
        """
        return self.detector(
            input_image=image,
            **self.params
        )