import io
import base64
import numpy as np
from PIL import Image
from keras.preprocessing import image



# default_image_size = tuple((256, 256))
#
# def convert_image(image_data):
#     image = Image.open(io.BytesIO(base64.b64decode(image_data)))
#     if image is not None:
#         image = image.resize(default_image_size, Image.ANTIALIAS)
#         image_array = image.img_to_array(image)
#         return np.expand_dims(image_array, axis=0), None
#     else:
#         return "Error loading image file",None



image1 = r"PlantVillage/test/Apple___Apple_scab/0a5e9323-dbad-432d-ac58-d291718345d9___FREC_Scab 3417.JPG"
op = convert_image(image1)