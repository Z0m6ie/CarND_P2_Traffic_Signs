# Traffic Sign Recognition Project - P2

### David Peabody

Second project of the Self-Driving Car Nanodegree.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./md_images/summary.png "Data Set Summary"
[image2]: ./md_images/example_sign.png "Example Sign"
[image3]: ./md_images/training_set.png "Training Set"
[image4]: ./md_images/validation_set.png "Validation Set"
[image5]: ./md_images/test_set.png "Test Set"
[image6]: ./md_images/gray.png "gray"
[image7]: ./md_images/normalized.png "normalized"

[image8]: ./test_images/100.jpg "web image 1"
[image9]: ./test_images/circle.jpg "web image 2"
[image10]: ./test_images/dig.jpg "web image 3"
[image11]: ./test_images/thirty.JPG "web image 4"
[image12]: ./test_images/xthing.jpg "web image 5"

---
## Writeup

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/Z0m6ie/CarND_P2_Traffic_Signs/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used simple python to calculate summary statistics of the traffic
signs data set:
* Number of training examples = 34799
* Number of testing examples = 12630
* Image data shape = (32, 32, 3)
* Number of classes = 43

code and output is shown below:

![alt text][image1]

#### 2. Include an exploratory visualization of the dataset.

Here is a few exploratory visualizations of the data set.
First I have provided a random example from the training dataset to see what the traffic signs look like.

![alt text][image2]

##### Lets look at the distribution of traffic signs

![alt text][image3]
![alt text][image4]

![alt text][image5]

While not identical the distribution of signs between the training, validation & testing sets is fairly similar.

It should be noted however that the count of each sign compared to others shows quite a large amount of variation. With some signs having as many as 700+ example and others having fewer than 100. This could cause issues where our classifier may be able to become very accurate on common signs but perform poorly on rare signs.

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale because converting from RGB to grayscale reduces the amount of information contained in the image 3 fold. In addition color is easily effected by the light and different lighting conditions could effect the accuracy.

Here is an example of a traffic sign image after grayscaling.

![alt text][image6]

As a last step, I normalized the image data by using the adaptive histogram equalization from skimage. this equalizes the light and dark areas of the grayscale image. this is beneficial on images with high or low contrast. However on some images this may still not produce optimum results. A future action could be to generate new data of different light conditions along with different rotations of the images.

![alt text][image7]


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					|
|:---------------------:|:---------------------------------------------:|
| Input         		| 32x32x1 grayscale image   							|
| Convolution    	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				|
| Convolution    | 1x1 stride, valid padding, outputs 10x10x16 		|
| RELU					|				|							
| Max pooling	      	| 2x2 stride,  outputs 5x5x16 				|
|Flatten   |output 400|
| Fully connected		| output 120       									|
| RELU			|     									|
| Fully connected		| output 84      									|
| RELU			|     									|
|Fully connected   |  43 |
|softmax   |  43 |

This model architecture is a simple convolutional model based on the LeNet-5 architecture.

#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used:
* batch size of 128
* 20 epochs
* learning rate of 0.001
* Adam optimizer

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 99.8%
* validation set accuracy of 94.9%
* test set accuracy of 94%

For this project I use the LeNet-5 architecture from a previous lab on classifying numbers from the MNIST dataset. The LeNet architecture is a convolutional neural network which as per Yann Lecun "Convolutional Neural Networks are designed to recognize visual patterns directly from pixel images with minimal preprocessing.  
They can recognize patterns with extreme variability (such as handwritten characters), and with robustness to distortions and simple geometric transformations."

I chose to stick with this architecture because it is a smaller architecture which trains fairly quickly even on a CPU. In addition, while 43 different classification outputs is not negligible, it is also not on the same scale as an imagenet challenge with 1000 classes.

Out of the box this architecture did fairly well providing an accuracy of around 92% on the validation set using only the suggested normalization and skimage's grayscale function.

To improve the model and beat the 93% accuracy required on the validation set I had a hunch that all that would be required is a normalization which deals slightly better with some of the low contrast images in the dataset. After some experimentation with histogram equalization I ultimately settled on skimage's adaptive histogram equalization using a kernel size of 8 (produced the best results to my eyes).

The results I got on the training, testing & validation set show no major signs of over or underfitting, perhaps maybe a very slight overfit on the training set (99.8% accuracy), but given a 94.9% accuracy on the validation set this seems reasonable.

I did not change any of the hyperparameters as the results achieved with the standard options were adequate.


### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image8] ![alt text][image9] ![alt text][image10]
![alt text][image11] ![alt text][image12]

I have tried to select 5 images which are fairly high quality and in good light to classify (can you blame me for wanting to stack the odds in my favour?).

The first image one would think would be fairly simple however it is a very high quality image and the test set were all low quality. I am not sure if this may make it more difficult to classify.

The second image is clear and of a similar quality to the training set. however there is a hint of another sign in the image which may complicate things.

The 3rd image is very clear. the only issue i can think of is that it is not in a square format, so after preprocessing it may be slightly distorted.

The 4th image (30kph speed limit) is clear and square, however part of the top of the sign is cut off.

the final image is clear and square and should not be an issue.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set.

Here are the results of the prediction:

| Image			        |     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| Speed limit (100km/h)   		| Slippery Road   									|
| No Vehicles    			|No Vehicles 									|
| Road Work		| Road Work										|
| Speed limit (30km/h)      		| Speed limit (30km/h) 				 				|
| No Stopping		| Keep right   							|


The model was able to correctly guess 3 of the 5 traffic signs, which gives an accuracy of 60%. While not fantastic, our test set accuracy was 94%, this is also not terrible and is a good starting point. I believe key to doing better on real world examples would be to generate much more data of various sign distortions as well as more low light or bright light (washed out) images.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability.

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were


| Image |  1st Prediction	|2nd Prediction	|   3rd Prediction	|4th Prediction	| 5th Prediction	|
|:---------------------------:|
| 1st Image Speed limit (100km/h) | 0.51 Slippery Road       | 0.33 Speed limit (60km/h)| 0.08 Dangerous curve to the right| 0.03 Speed limit (50km/h)| 0.02 Right-of-way at the next intersection|
| 2nd No Vehicles                 | 0.96 No Vehicles	       | 0.03 Priority road| 0.005 Speed limit (100km/h)| 0.004 Speed limit (50km/h)| 0.00 Keep right|
| 3rd Image Road work             | 0.99 Road work	         | 0.00 Wild animals crossing | 0.00 Speed limit (80km/h) | 0.00 Bicycles crossing| 0.00 Bumpy road|
| 4th Image Speed limit (30km/h)  | 0.99 Speed limit (30km/h)| 0.00 Speed limit (50km/h)| 0.00 Traffic signals| 0.00 Speed limit (20km/h)| 0.00 General caution|
| 5th Image No Stopping           | 0.57 Keep right   	     |0.28 Priority road | 0.06 End of all speed and passing limits| 0.04 Turn left ahead| 0.03 Ahead only|


Note: for any suitably small percentage (less than 0.004) I just showed 0.00
