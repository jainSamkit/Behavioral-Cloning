# **Behavioral Cloning** 

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Using the simulator to collect data of good driving behavior
* Building, a convolution neural network in Keras that predicts steering angles from images
* Training and validating the model with a training and validation set
* Testing that the model successfully drives around track one without leaving the road
* Summarizing the results with a written report.

---

**Submission includes all required files and can be used to run the simulator in autonomous mode**

* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

**Submission includes functional code**
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

**Submission code is usable and readable**

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

## Model Architecture and Training Strategy

I have used the popular Nvidia architectue for the self driving car.The model has three convolutional layers with filter      sizes 5x5 with stride 2 and depths being 24,36 and 48.Thereafter three more convolutional layers with filter sizes 3x3 and depths being 64 each.After that four fully connected layers has been used with depths 100,50,10 and 1 each. 

The model includes ELU layers to introduce nonlinearity , and the data is normalized in the model using a Keras lambda layer.
The input of the model was 90x320 which is cropped using the cropping2d layer in keras.The cropping removed the dashboard and the upper redundant portion of the image which didn't included mostly trees and overhead sky.

## Attempts to reduce overfitting in the model

To reduce the overfitting in the model ,I used l2 regularization after every layer including convolutional and fully connected layers as well.

The model was trained and validated on different data sets to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

## Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually.The coefficient of the l2 regularizer is set to be 0.0001.

## Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road .Also I included the images from right and left cameras each.Center lane images were also flipped to remove any directional bias learnt by the network.

For details about how I created the training data, see the next section. 

## Model Architecture and Training Strategy


The overall strategy for deriving a model architecture was to introduce the mixure of convolutional and fully connected layers using mean squared error as the error function.

My first step was to use a convolution neural network model similar to the LeNet.Since I have used the network before,it was easy to incorporate with the given data as well.However,I saw that the error didnt reduce much and the model was underperforming on training data.Nvidia had recently made the network that they used to train their self drving car public.I therefore used the same model for my training data.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model by introducing l2 regularizing after every layer.I also introduced ELU activation function to reduce the wobble of the car.

The model performed a lot better than previosuly used LeNet.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track i.e near the sea specifically.This was due to the directional bias in the network.However,flipping and introducing more datapoints helped the model learn the relevant details while cornering.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

# Final Model Architecture

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 90x160x3 RGB image   							| 
| Convolution 5x5(layer1)     	| 2x2 stride, valid padding with 24 filters|
| ELU	Activation				|	
| Convolution 5x5(layer2)	    | 2x2 stride, valid padding with 36 filters|
| ELU	Activation			|												|
| Convolution 5x5(layer3)    | 2x2 stride, valid padding with 48 filters |
| ELU	Activation			|	
| Convolution 3x3(layer4)     	| 1x1 stride, valid padding with 64 filters|
| ELU	Activation				|	
| Convolution 3x3(layer5)     	| 1x1 stride, valid padding with 64 filters|
| ELU	Activation				|	
| Flattening         | Flattened layer 3|
| Fully Connected Layer | outputs 100 layers |
| ELU	Activation			|	
| Fully Connected Layer | outputs 50 layers |
| ELU	Activation			|	
| Fully Connected Layer | outputs 10 layers |
| ELU	Activation			|	
| Fully Connected Layer | outputs 1 layers |

####  Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

<img src="center.png" width="400" alt="Center Lane Image" />

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

Etc ....

After the collection process, I had X number of data points. I then preprocessed this data by ...


I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 3 as evidenced by the gradual decrease in the loss value.I used an adam optimizer so that manually training the learning rate wasn't necessary.
