# **Behavioral Cloning** 
---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)


[image2]: ./examples/center_lane_driving.jpg "center"
[image3]: ./examples/recover0.jpg "Recovery Image"
[image4]: ./examples/recover1.jpg "Recovery Image"
[image5]: ./examples/recover2.jpg "Recovery Image"
[image6]: ./examples/filp0.jpg "Normal Image"
[image7]: ./examples/filp1.jpg "Flipped Image"
[image8]: ./examples/crop0.jpg "before crop"
[image9]: ./examples/crop1.jpg "after crop"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
#### Files
My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### Command to run the model
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

### Model Architecture and Training Strategy

#### 1.Model architecture

My model consists of convolution layers with 5x5/3x3 filter sizes and depths 32 and 16 (model.py lines 112-118) 

The model includes RELU layers to introduce nonlinearity (code line 113, 116, 119), and the data is normalized in the model using a Keras lambda layer (code line 110). 

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 120, 123). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 65, 131). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 127).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, clockwise driving,  recovering from the left and right sides of the road, recovering from sharp turn. 

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to according to input image produce proper steer angle.

My first step was to use a convolution neural network model similar to the LeNet I thought this model might be appropriate because it is useful in extract featrues from images.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I added some dropout layers.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track, before get on the bridge, and at some sharp turns. To improve the driving behavior in these cases, I drive/record more data at these spots, especially recovering from the edges.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 112-125) consisted of 3 convolution neural network and 3 full connection layer.

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps (counter-clockwise & clockwise) on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to recover from the road edge. These images show what a recovery looks like starting from ... :

![alt text][image3]
![alt text][image4]
![alt text][image5]

To augment the data sat, I also flipped images and angles thinking that this would effectively double the data set. For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

After the collection process, I had 5073 number of data points. I then preprocessed this data by normalizing pixel value and cropping each image to focus on only the portion of the image that is useful for predicting a steering angle. 

![alt text][image8]
![alt text][image9]

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 15 as evidenced by the loss stop decrease any more. I used an adam optimizer so that manually training the learning rate wasn't necessary.
