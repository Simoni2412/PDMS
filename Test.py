import cv2
import numpy as np
from keras.preprocessing import image
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.preprocessing import image

model = load_model('crop.h5')

Classes = [

    'Apple___Apple_scab',
    'Apple___Black_rot',
    'Apple___Cedar_apple_rust',
    'Apple___healthy',

    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
    'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight',
    'Corn_(maize)___healthy',

    'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)',
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    'Grape___healthy',

    'Potato___Early_blight',
    'Potato___Late_blight',
    'Potato___healthy',

    'Tomato_Bacterial_spot',
    'Tomato_Early_blight',
    'Tomato_Late_blight',
    'Tomato_Leaf_Mold',
    'Tomato_Septoria_leaf_spot',
    'Tomato_Spider_mites_Two_spotted_spider_mite',
    'Tomato__Target_Spot',
    'Tomato__Tomato_YellowLeaf__Curl_Virus',
    'Tomato__Tomato_mosaic_virus',
    'Tomato_healthy',

]

# Pre-Processing test data same as train data.
def prepare(img_path):
    img = image.load_img(img_path, target_size=(256, 256))
    x = image.img_to_array(img)
    x = x / 255
    return np.expand_dims(x, axis=0)


#image feeded to the model
inputImage = r'C:\Users\SOHAM SHAH\Downloads\plant test images\tomato bacterial spot.jpg'
result = model.predict_classes([prepare(inputImage)])


print(Classes[int(result)])
print(result)

img = cv2.imread(inputImage)


scale_percent = 220  # percent of original size
width = int(img.shape[1] * scale_percent / 100)
height = int(img.shape[0] * scale_percent / 100)
dim = (width, height)
font = cv2.FONT_HERSHEY_SIMPLEX
org = (30, 35)
fontScale = 1
color = (255, 0, 0)
thickness = 2

img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
cv2.imshow("inputImage", img)

outputImage = cv2.putText(img, str(Classes[int(result)]), org, font, fontScale, color, thickness, cv2.LINE_AA)

outputImageName= r'outputImage\img1.jpg'

cv2.imwrite(outputImageName,outputImage)

cv2.imshow("outputImage", outputImage)
cv2.waitKey(0)
