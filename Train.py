# Import Libraries
import glob
import os

import matplotlib.pyplot as plt
import tensorflow
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D
from tensorflow.python.keras.layers import Dense, Dropout, Flatten
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator


train_dir = "PlantVillage/train"
test_dir = "PlantVillage/test"


# function to get count of images
def get_files(directory):
    if not os.path.exists(directory):
        return 0
    count = 0
    for current_path, dirs, files in os.walk(directory):
        for dr in dirs:
            count += len(glob.glob(os.path.join(current_path, dr + "/*")))
    return count


train_samples = get_files(train_dir)
num_classes = len(glob.glob(train_dir + "/*"))
test_samples = get_files(
    test_dir)  # For testing i took only few samples from unseen data. we can evaluate using validation data which is part of train data.
print(num_classes, "Classes")
print(train_samples, "Train images")
print(test_samples, "Test images")

# Preprocessing data.
train_datagen = ImageDataGenerator(rescale=1. / 255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   validation_split=0.2,  # validation split 20%.
                                   horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1. / 255)

# set height and width and color of input image.
img_width, img_height = 256, 256
input_shape = (img_width, img_height, 3)
batch_size = 32

train_generator = train_datagen.flow_from_directory(train_dir,
                                                    target_size=(img_width, img_height),
                                                    batch_size=batch_size)
test_generator = test_datagen.flow_from_directory(test_dir, shuffle=True,
                                                  target_size=(img_width, img_height),
                                                  batch_size=batch_size)
print(train_generator.class_indices)

# CNN building.
model = Sequential()
model.add(Conv2D(32, (5, 5), input_shape=input_shape, activation='relu'))
model.add(MaxPooling2D(pool_size=(3, 3)))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(128, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))
model.summary()

model_layers = [layer.name for layer in model.layers]
print('layer name : ', model_layers)

# validation data.
validation_generator = train_datagen.flow_from_directory(
    train_dir,  # same directory as training data
    target_size=(img_height, img_width),
    batch_size=batch_size)

# Model building to get trained with parameters.


opt = tensorflow.keras.optimizers.Adam(lr=0.001)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
train = model.fit(train_generator,
                  epochs=6,
                  steps_per_epoch=train_generator.samples // batch_size,
                  validation_data=validation_generator,
                  validation_steps=validation_generator.samples // batch_size, verbose=1)



# Save entire model with optimizer, architecture, weights and training configuration.

model.save('crop.h5')

# Save model weights.

model.save_weights('crop_weights.h5')

# Get classes of model trained on
classes = train_generator.class_indices
print(classes)

# plot for accuracy and loss
acc = train.history['acc']
val_acc = train.history['val_acc']

loss = train.history['loss']
val_loss = train.history['val_loss']
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

# Evaluate model using unseen data.
score, accuracy = model.evaluate(test_generator, verbose=1)
print("Test score is {}".format(score))
print("Test accuracy is {}".format(accuracy))







