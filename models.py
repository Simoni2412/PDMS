import cv2
import keras
import tensorflow
from tkinter import *
from tkinter import messagebox
import matplotlib.pyplot as plt
import numpy as np
from keras.preprocessing import image
from tensorflow.python.keras.models import load_model,model_from_json
from tensorflow.python.keras.preprocessing import image
global graph
graph = tensorflow.get_default_graph()






# load json file before weights
loaded_json = open("crop.json", "r")
# read json architecture into variable
loaded_json_read = loaded_json.read()
# close file
loaded_json.close()
# retreive model from json
loaded_model = model_from_json(loaded_json_read)
# load weights
loaded_model.load_weights("crop_weights.h5")
model1 = load_model("crop.h5")




# plot for accuracy and loss
acc = model1.history['acc']
val_acc = model1.history['val_acc']

loss = model1.history['loss']
val_loss = model1.history['val_loss']
epochs = range(1, len(acc) + 1)

# Train and validation accuracy
plt.plot(epochs, acc, 'b', label='Training accurarcy')
plt.plot(epochs, val_acc, 'r', label='Validation accurarcy')
plt.title('Training and Validation accurarcy')
plt.legend()

plt.figure()
# Train and validation loss
plt.plot(epochs, loss, 'b', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and Validation loss')
plt.legend()
plt.show()

















#
#
#
# def leaf_predict(img_path):
#     # load image with target size
#     img = image.load_img(img_path, target_size=(256, 256))
#     # convert to array
#     img = image.img_to_array(img)
#
#     # normalize the array
#     img = img / 255
#     img1 = np.linalg.norm(img)
#     print(img1)
#
#     # expand dimensions for keras convention
#     img = np.expand_dims(img, axis=0)
#     print(img.dtype)
#
#
#     with graph.as_default():
#
#         opt = tensorflow.keras.optimizers.Adam(lr=0.001)
#         loaded_model.compile(optimizer=opt, loss='mse')
#         preds = model1.predict(img)
#
#         # preds1 = preds.astype(int)
#         print(preds.shape)
#         preds2 = np.linalg.norm(preds)
#
#
#         # opt1 = img - preds2
#         # img = np.linalg.norm(img)
#         # preds = np.linalg.norm(preds)
#         # dist = np.linalg.norm(opt1)
#         # print (img)
#         # print(preds)
#         if preds2 >= 98:
#             return "leaf"
#         else:
#
#             errormsg = Tk()
#
#             messagebox.showerror("Image is not Proper","Please Upload Proper Image Of Crop.")
#             errormsg.mainloop()
#
#
#
#         # preds = np.squeeze(preds)
#         # preds = np.asscalar(preds)
#         # print(preds.dtype)
#
#
# leaf = leaf_predict(r'other_img/62566_top-hd-marvel-wallpapers_3840x2160_h.jpg')
# #leaf = leaf_predict(r'PlantVillage/test/Apple___Apple_scab/0a5e9323-dbad-432d-ac58-d291718345d9___FREC_Scab 3417.JPG')
# print(leaf)