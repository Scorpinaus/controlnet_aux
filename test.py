from wrapper import AnylineWrapper
from PIL import Image
import numpy as np

detector = AnylineWrapper()
image = Image.open("C:/Users/hafiz/DiffusersProject/DiffuseBackEnd/input/images/cnet_openpose_test.png")

result = detector(
    input_image=image,
    detect_resolution=1024,
    gaussian_sigma=1.5,
    intensity_threshold=5,
    min_size=50,
    lower_bound=10,
    upper_bound=240,
    connectivity=2,
    output_type="pil"
)
result.save("custom_output.jpg")