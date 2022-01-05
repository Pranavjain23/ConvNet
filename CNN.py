"""
@author: Pranav-Jain
"""

#CNN to detect plant diseases in Maize plant leaves
#Train accuracy ~ 96.2% and Test accuracy ~ 93%

import tensorflow
from tensorflow.keras.layers import Conv2D, Flatten, Dense, Dropout,MaxPooling2D,Flatten, BatchNormalization,SpatialDropout2D
from tensorflow.keras.models import Model,Sequential

model=Sequential()

model.add(Conv2D(32,(3,3),padding='same',activation='relu',input_shape=(150,150,3)))
model.add(MaxPooling2D(2,2))
#model.add(SpatialDropout2D(0.1))

model.add(Conv2D(32,(3,3),padding='same',activation='relu'))
model.add(MaxPooling2D(2,2))
model.add(BatchNormalization())
#model.add(SpatialDropout2D(0.2))

model.add(Conv2D(32,(3,3),padding='same',activation='relu'))
model.add(MaxPooling2D(2,2))
model.add(BatchNormalization())
#model.add(SpatialDropout2D(0.2))

model.add(Conv2D(32,(3,3),padding='same',activation='relu'))
model.add(MaxPooling2D(2,2))
model.add(BatchNormalization())
#model.add(SpatialDropout2D(0.3))

model.add(Conv2D(32,(3,3),padding='same',activation='relu'))
model.add(MaxPooling2D(2,2))
model.add(BatchNormalization())
#model.add(SpatialDropout2D(0.3))

model.add(Conv2D(32,(3,3),padding='same',activation='relu'))
model.add(MaxPooling2D(2,2))
model.add(BatchNormalization())
#model.add(SpatialDropout2D(0.3))

model.add(Conv2D(32,(3,3),padding='same',activation='relu'))
model.add(MaxPooling2D(2,2))
model.add(BatchNormalization())
#model.add(SpatialDropout2D(0.5))

model.add(Flatten())
#model.add(Dropout(0.5))
model.add(Dense(128,activation='relu'))
#model.add(Dropout(0.5))
model.add(Dense(128,activation='relu'))
model.add(Dense(32,activation='relu'))
#model.add(Dropout(0.5))
model.add(Dense(4,activation='softmax'))

model.summary()

#adam=tensorflow.keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer="Adam",loss='categorical_crossentropy',metrics=['accuracy',tensorflow.keras.metrics.Precision(),tensorflow.keras.metrics.Recall(),tensorflow.keras.metrics.AUC()])

from tensorflow.keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
        '/content/drive/MyDrive/datasamenoofpics/train',
        target_size=(150,150),
        batch_size=32 ,
        class_mode='categorical')

val_set = test_datagen.flow_from_directory(
        '/content/drive/MyDrive/datasamenoofpics/test',
        target_size=(150,150),
        batch_size=32,
        class_mode='categorical')

history=model.fit_generator(
        training_set,
        steps_per_epoch=12,
        epochs=10,
        validation_data=val_set,
        validation_steps=15)

from google.colab import drive
drive.mount('/content/drive')

import matplotlib.pyplot as plt

plt.subplot(211)  
plt.plot(history.history['accuracy'])  
plt.plot(history.history['val_accuracy'])  
plt.title('model accuracy')  
plt.ylabel('accuracy')  
plt.xlabel('epoch')  
plt.legend(['train', 'test'], loc='upper left')
