import tensorflow as tf
import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import cv2
import numpy as np

from keras import activations, models, optimizers, metrics, utils, layers
from keras.utils.vis_utils import model_to_dot

from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix as CM
from random import randint

def get_images(directory):
    Images = []
    Labels = []
    label = 0
    
    for labels in os.listdir(directory):
        if labels == 'glacier':
            label = 2
        elif labels == 'sea':
            label = 4
        elif labels == 'buildings':
            label = 0
        elif labels == 'forest':
            label = 1
        elif labels == 'street':
            label = 5
        elif labels == 'mountain':
            label = 3
            
        for image_file in os.listdir(directory + labels):
            image = cv2.imread(directory + labels +r'/'+image_file)
            image = cv2.resize(image, (150,150))
            
            Images.append(image)
            Labels.append(label)
            
    return shuffle(Images, Labels, random_state=817328462)

def get_classlabel(class_code):
    labels = {2:'glacier', 4:'sea', 0:'buildings', 1:'forest', 5:'street', 3:'mountain'}
    
    return labels[class_code]


Images, Labels = get_images('seg_train/seg_train/')

Images = np.array(Images)
Labels = np.array(Labels)

print('shape of Images:', Images.shape)
print('shape of Labels:', Labels.shape)

model = models.Sequential()

model.add(layers.Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same', input_shape=(150, 150, 3), kernel_initializer='he_normal'))
model.add(layers.BatchNormalization())
model.add(layers.Conv2D(32, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer='he_normal'))
model.add(layers.BatchNormalization())

model.add(layers.MaxPool2D(3, 3))

model.add(layers.Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer='he_normal'))
model.add(layers.BatchNormalization())
model.add(layers.Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer='he_normal'))
model.add(layers.BatchNormalization())
model.add(layers.Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer='he_normal'))
model.add(layers.BatchNormalization())

model.add(layers.MaxPool2D(3, 3))

model.add(layers.Conv2D(128, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer='he_normal'))
model.add(layers.BatchNormalization())
model.add(layers.Conv2D(128, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer='he_normal'))
model.add(layers.BatchNormalization())
model.add(layers.Conv2D(128, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer='he_normal'))
model.add(layers.BatchNormalization())

model.add(layers.MaxPool2D(3, 3))

model.add(layers.GlobalAveragePooling2D())
model.add(layers.Dense(200, activation='relu', kernel_initializer='he_normal'))
model.add(layers.Dropout(rate=0.8))
model.add(layers.Dense(6, activation='softmax'))

model.summary()

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(Images, Labels, epochs=35, validation_split=0.2)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
