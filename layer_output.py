import glob
import os
from keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
import tensorflow
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D
from tensorflow.python.keras.layers import Dense, Dropout, Flatten
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from tensorflow.python.keras.models import load_model
import matplotlib.image as mpimg


global graph
graph = tensorflow.get_default_graph()

model = load_model("crop.h5")

model.summary()



