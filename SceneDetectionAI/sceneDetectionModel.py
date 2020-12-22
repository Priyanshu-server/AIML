import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers

#Loading the training Data
train_datagen = ImageDataGenerator(rescale=1./255,shear_range=0.2,zoom_range=0.2,horizontal_flip=True) #training
training_set = train_datagen.flow_from_directory('seg_train/',target_size=(64,64),
                                                                batch_size= 64,class_mode= 'categorical')
#Loading the Test Data
test_datagen = ImageDataGenerator(rescale=1./255)
test_set = test_datagen.flow_from_directory('seg_test/',target_size=(64,64),
                                                        batch_size= 64,class_mode='categorical')


# print(training_set.class_indices) #{'buildings': 0, 'forest': 1, 'glacier': 2, 'mountain': 3, 'sea': 4, 'street': 5}
cnn = tf.keras.models.Sequential()

cnn.add(layers.Conv2D(filters =64,kernel_size = 3,activation = 'relu',input_shape = [64,64,3])) #64,128,256,32
cnn.add(layers.MaxPool2D(pool_size=2,strides = 2))
cnn.add(layers.Dropout(0.2))

cnn.add(layers.Conv2D(filters =128,kernel_size = 3,activation = 'relu'))
cnn.add(layers.MaxPool2D(pool_size=2,strides = 2))
cnn.add(layers.Dropout(0.2))
cnn.add(layers.Flatten())

cnn.add(layers.Dense(units = 128,activation = 'relu'))
cnn.add(layers.Dropout(0.2))

cnn.add(layers.Dense(units = 64,activation='relu'))
cnn.add(layers.Dropout(0.2))

cnn.add(layers.Dense(units = 28,activation='relu'))
cnn.add(layers.Dropout(0.2))

#Final Output Layer
cnn.add(layers.Dense(units = 6,activation='softmax'))

#Training the model
cnn.compile(optimizer='adam',loss = 'categorical_crossentropy',metrics = ['accuracy'])
cnn.fit(x = training_set,validation_data=test_set,epochs=25,shuffle = True)

cnn.save('SceneDetection.h5')
