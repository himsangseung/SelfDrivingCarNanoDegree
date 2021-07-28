**Behavioral Cloning Project**

This project focuses on controlling steering of a simulator vehicle autonomously going around a track, with NVIDIA's Deep Learning CNN model in  End-to-End Deep Learning for Self-Driving Cars[Bojarski, etc] in 2016. Steering output is predicted through the CNN using three simulation camera images. Overall, it showed a satisfactory performance with low cost function value for the cross-validation data.

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

[//]: # (Image References)

[image2]: ./img/center.jpg "Center Image"
[image3]: ./img/left.jpg "Recovery Image - left"
[image4]: ./img/right.jpg "Recovery Image - right"
[image6]: ./img/flip.png "Flip Image"
[image7]: ./img/Loss_train_val.png "loss_val"
[image8]: ./img/LeNet5.png "LeNet5 CNN architecture"
[image9]: ./img/CNN.png "NVIDIA End-to-End CNN architecture"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employe

There are two major model architectures have been tested, and the one resulting in lower validation loss is chosen for the final model: NVIDIA's ENd-to-End CNN architecture.

The first model architecture deployed is LeNet5 CNN architecture, a widely proven in MNist dataset as a classic classfication network - noting the intention to build a regression network with one output optimization. A description of the network is presented below:

![alt text][image8]

[picture of LeNet] - http://vision.stanford.edu/cs598_spring07/papers/Lecun98.pdf]

By modern standards, LeNet-5 is a very simple network. It only has 7 layers, among which there are 3 convolutional layers (C1, C3 and C5), 2 sub-sampling (pooling) layers (S2 and S4), and 1 fully connected layer (F6), that are followed by the output layer. Convolutional layers use 5 by 5 convolutions with stride 1. Sub-sampling layers are 2 by 2 average pooling layers. Tanh sigmoid activations are used throughout the network. There are several interesting architectural choices that were made in LeNet-5 that are not very common in the modern era of deep learning[https://medium.com/@pechyonkin/key-deep-learning-architectures-lenet-5-6fc3c59e6f4] 

The next candidate has clear back-up work that explicitly depicts what this project is trying to acheive. The network architecture of interest has been proposed in "End-to-End Deep Learning for Self-Driving Cars" - Bojarski.M, etc. 
This proposes a method of behavioral clonning by vision-only(three cameras), efficiently predicting steering command value. An explanation of this CNN architecture is shown below:

![alt text][image9]

[picture of NVIDIA CNN]

The first layer of the network performs image normalization. The normalizer is hard-coded and is not adjusted in the learning process. Performing normalization in the network allows the normalization scheme to be altered with the network architecture, and to be accelerated via GPU processing[https://developer.nvidia.com/blog/deep-learning-self-driving-cars/]

The convolutional layers are designed to perform feature extraction, and are chosen empirically through a series of experiments that vary layer configurations. We then use strided convolutions in the first three convolutional layers with a 2×2 stride and a 5×5 kernel, and a non-strided convolution with a 3×3 kernel size in the final two convolutional layers.

The model includes RELU layers to introduce nonlinearity (code line 85, 87, 89, 91. 93), and the data is normalized in the model using a Keras lambda layer (code line 62). 

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 88 and 97). The model was trained and validated on different data sets to ensure that the model was not overfitting(code line 104). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 104).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road.  

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to adopt a CNN architecture introduced in [NVIDIA's End-to-End Learning for Self-Driving Cars] by Bojarski.M, etc. I thought this model might be appropriate because it captures the solution to the problem statement explicitly, predicting steering commmand value with images from three cameras on a vehicle.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. To combat the overfitting, I modified the model so that it contains dropout layer for a regularization technique, preventing overfitting.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track, to improve the driving behavior in these cases, I recorded some more videos with recovering from biased lane position.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture
While overall pictorial diagram of the CNN model used is available up on the *Model Architecture and Training Strategy* section, here is a tabularized CNN architecture of the final model:

|Layer     |Description					   :|
|:---------|:----------------------------------------------|
|Input     |Input BGR image (160, 320, 3)		   |
|Lambda    |Normalize the input image by (x - 127.5) - 0.5 |
|Cropping2D|(0,0), (70,25)				   |
|Conv2D    |5x5 filter, 2x2 strides, "relu", padding="same"|
|Dropout   |probability: 0.5				   |
|Conv2D    |5x5 filter, 2x2 strides, "relu", padding="same"|
|Conv2D    |5x5 filter, 2x2 strides, "relu", padding="same"|
|Conv2D    |3x3 filter, padding="same"			   |
|Conv2D    |3x3 filter, padding="same"			   |
|MaxPooling|Maxpooling2D, padding="same"		   |
|Dropout   |probability: 0.5				   |
|Flatten   |output shape: 10240			  	   |
|Dense     |output : 100				   |
|Dense     |output : 50 				   |
|Dense     |output : 10 			    	   |
|Dense     |output : 1  				   |


To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

[Picture of image from the center camera]
I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![alt text][image3]
![alt text][image4]

[Picture of image from left/right camera]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles so that a scenario with opposite turn orientation is also captured in the data set. For example, here is an image that has then been flipped:

![alt text][image2]
![alt text][image6]

[picture of original image and the horizontally flipped image]

After the collection process, I had 10031 number of data points. I then preprocessed this data by adding a lambda layer in the model for image pixel normalization, then the image is cropped at (w=70, h=25) so that it would contain road lane images as its primary focus, filtering out unnecessary scenery such as the sky, trees, hills on the side, rocks, and etc.

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 4 as evidenced bythe loss plot below - train/validation set.
![alt text][image7]

[loss plot - train/validation set]

I used an adam optimizer so that manually training the learning rate wasn't necessary.
