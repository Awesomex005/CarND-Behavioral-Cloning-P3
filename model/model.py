import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

UseGivenData = False

if UseGivenData:
    csv_file_path = "F:\SDC\p3\data\driving_log.csv" 
    path_ = "F:\SDC\p3\data"
else: 
   csv_file_path = "H:/train_data/driving_log.csv"
''' without generator
lines = []

with open(csv_file_path, 'r') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)
        
center_images = []
measurements = []
if  UseGivenData:
    lines = lines[1:]
    for line in lines:
        src_path = line[0]
        file_name = src_path.split('/')[-1]
        src_path = path_+"/"+src_path
        img = mpimg.imread(src_path)
        center_images.append(img)
        measurement = float(line[3])
        measurements.append(measurement)
else:
    for line in lines:
        src_path = line[0]
        file_name = src_path.split('/')[-1]
        img = mpimg.imread(src_path)
        center_images.append(img)
        measurement = float(line[3])
        measurements.append(measurement)
    
augmented_images, augmented_measurements = [], []
for img, msm in zip(center_images, measurements):
    augmented_images.append(img)
    augmented_measurements.append(msm)
    if abs(msm) > 0.03:
        augmented_images.append(np.flip(img, 1))
        augmented_measurements.append(msm*-1.0)
    
X_train = np.array(augmented_images, dtype=np.float32)
y_train = np.array(augmented_measurements, dtype=np.float32)
'''
import sklearn
import random
import math

samples = []
with open(csv_file_path) as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)
print("train_samples: %d" %len(train_samples))
print("validation_samples: %d" %len(validation_samples))
#print(train_samples[0][0].split('/')[-1]); exit()

batch_sz = 32
train_batches = math.ceil(len(train_samples)/batch_sz)
validation_batches = math.ceil(len(validation_samples)/batch_sz)
print("train_batches: %d" %train_batches)
print("validation_batches: %d" %validation_batches)

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        random.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                name = batch_sample[0].split('/')[-1]
                center_image = mpimg.imread(name)
                center_angle = float(batch_sample[3])
                images.append(center_image)
                angles.append(center_angle)
                #if abs(center_angle) > 0.04:
                if 1: # simple data augmentation
                    images.append(np.flip(center_image, 1))
                    angles.append(center_angle*-1.0)

            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_sz)
validation_generator = generator(validation_samples, batch_sz)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Dropout, Activation, Cropping2D
from keras.layers.convolutional import Convolution2D, Conv2D
from keras.layers.pooling import MaxPooling2D

model = Sequential()
model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape=(160,320,3))) # normalize input data
model.add(Cropping2D(cropping=((70,25), (0,0)))) # select interest of region, remove distraction
model.add(Conv2D(6, (5, 5)))
model.add(Activation('relu'))
model.add(MaxPooling2D())
model.add(Conv2D(6, (5, 5)))
model.add(Activation('relu'))
model.add(MaxPooling2D())
model.add(Conv2D(12, (3, 3)))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(128))
model.add(Dropout(0.8))
model.add(Dense(84))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
'''  without generator
history_object = model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=10, verbose=1)
'''
history_object = model.fit_generator(train_generator, steps_per_epoch= \
            train_batches, validation_data=validation_generator, \
            validation_steps=validation_batches, epochs=10, verbose=1)

### print the keys contained in the history object
print(history_object.history.keys())

### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()


model.save('model.h5')

