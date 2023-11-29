from roboflow import Roboflow
from PIL import Image
import io

pedicel_dict = {'ped':'white',
                'sin-ped':'black',
                'doble':'purple',
                's-doble':'red'}

ROBOFLOW_API_KEY = "dewRu1upu5sImkWrjMbE"
ROBOFLOW_MODEL_ENDPOINT = "cherry-pedicel"
ROBOFLOW_VERSION = "1"

pedicel_prediction_url = "".join([
    "https://detect.roboflow.com/",
    ROBOFLOW_MODEL_ENDPOINT,
    "/",
    ROBOFLOW_VERSION,
    "?api_key=",
    ROBOFLOW_API_KEY
])

rf = Roboflow(api_key=ROBOFLOW_API_KEY)
project = rf.workspace().project(ROBOFLOW_MODEL_ENDPOINT)
pedicel_model = project.version(ROBOFLOW_VERSION).model
pedicel_model.confidence = 20
pedicel_model.overlap = 30

def compress_image(image_content, target_size_bytes=1500000):
    # Load the image from image_content
    image = Image.open(io.BytesIO(image_content))
    image_format = image.format  # Get the original image format (e.g., "JPEG", "PNG")

    # Set the target file size in bytes
    target_file_size = target_size_bytes

    # Compress the image while maintaining the aspect ratio
    quality = 90  # Initial quality factor
    while True:
        temp_file = io.BytesIO()
        image.save(temp_file, format=image_format, quality=quality)
        temp_file_size = len(temp_file.getvalue())

        if temp_file_size <= target_file_size:
            break  # Image meets the size constraint

        # Reduce the quality if the file size is still too large
        quality -= 5  # Adjust this step size as needed    

    # Return the compressed image content
    return temp_file.getvalue()