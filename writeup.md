# Traffic Sign Recognition

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/visualization.png "Visualization"
[image2]: ./examples/statistics.png "Image Statistics"
[image3]: ./examples/issues.png "Image Issues"
[image4]: ./examples/classes.png "Traffic Sign Classes"
[image5]: ./examples/yuv.png "YUV"
[image6]: ./examples/statistics2.png "Image Statistics after Normalization"
[image7]: ./examples/preprocessed.png "Images before/after Preprocessing"
[image8]: ./examples/confusion.png "Problems with class 0"
[image9]: ./examples/augmentation.png "Augmentation Operations"
[image10]: ./examples/visualization2.png "Visualization after Augmenting"

[image4]: ./examples/placeholder.png "Traffic Sign 1"
[image5]: ./examples/placeholder.png "Traffic Sign 2"
[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"

---

## Rubric Points
Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

### Writeup / README

#### 1. Provide a Writeup / README

You're reading it! and here is a link to my [project code](https://github.com/WarrantyVoid/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. 

I used the numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is **34799** samples
* The size of the validation set is **4410** samples
* The size of test set is **12630** samples
* The shape of a traffic sign image is **32x32 (RGB)**
* The number of unique classes/labels in the data set is **43**

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing the total number of occurances of each unique traffic sign class in the training data set relative to the size of the data set (in %) on the x-axis. On the y axis, it depicts the identifier of the according class (0 to 42)

![Visualization][image1]

There's a few things I could use this vizualisation for:
* By comparing the occurances in this chart against the occurances in the validation and training set, I could confirm that the test and validation sets are actually a very good representation of the training set (class count-wise).
* The visualization makes it very clear that some traffic sign classes occur much more frequent that others. The classes with the most samples have about 10x more samples than the least frequent classes. This could later on bias the neural networks significatnly towards the more frequently occuring classes.

Another thing that gets clear during the exploration of the data set is, that all of the samples are sorted by class. This was another strong reminder to me that I should not forget to shuffle the data set during learning.

Here is another exploratory visualization of the data set. This is a diagram showing several statistical image values applying to a cumlated percentual amount of images:

![Image Statistics][image2]

This visualization showed me the following things:
* By comparing this statistics against the same statistics for validation and training set, I can confirm that those are also pixel-variation-wise a very good representation of the training set.
* A statistically relevant parts of the images are showing high variations in global contrast and brightness. But for the neural network I need the images to be as uniform as possible while not removing any relevant features of traffic signs. So a preprocessing of this images set can not be avoided.


I have also included an exploratory visualization of the data set that actually show five random samples for each traffic sign class in the [project code](https://github.com/WarrantyVoid/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb) and then later on dumped all images into a directory for detailed browsing.

This visualization helped a great deal. It clearly shows the upcoming challenges for the neural network, as the following example samples show:

![Image Issues][image3]

* Some signs are blurred, caused by either motion or by windshield
* Some signs are partially covered by trees, other shields or leaves
* Some signs are distorted due to non-frontal perspective
* Some signs show misleading contrast patterns caused by lighting conditions/specular highlights
* Some signs are rotated to a certain degree
* Some signs show misleading color patterns caused by stickers/damage
* Some sign vary in size so much, they are only represented by very few pixels
* Some signs don't show their expected color signature due to worn-out paint
* Some signs are significantly displaced from the picture center


### Design and Test a Model Architecture

#### Describe how you preprocessed the image data. 

As first step, based on the exploration of the data set, I identify the following hierachy for traffic sign classes:

![Traffic Sign classes][image4]

This brings up some important realizations:
* Three sign classes can be classified entirely by shape
* The 15 red-stroked triangle and the 12 red-stroked circle classes rely entirely on their black and white interior for classification
* Despite the specific considerations red/blue vs. others and white vs black, the pixel color does not help much in the classification

So the goal of my preprocessing shall be to remove all of the color information while keeping red/blue similarity as a key signature. A simple grayscale conversion is out of the question, because this would lead to the very same gray values for red, blue and green. 

The solution is already given away in the provided paper *Traffic Sign Recognition with Multi-Scale Convolutional Networks* by Sermanet and LeCun: YUV conversion. The YUV conversion is a transformation of the rgb color space, which leads to a luminance value and two chrominance values. I just drop both chrominance channels, because the luminance channel consists of different weights for red, green and blue, thus providing me already with good ingridients for classification:

![YUV][image5]

The next problem I have to address is the strong brightness and contrast variation of the images detected during data set exploration. It took me tree approaches to  end up with a satisfactory solution.

A first approach was to just normalize the gray intensity to the range 0 to 255 by substracting the minimum pixel value and dividing by the maximum pixel range. This  has lead to bad results. The approach has the big issue that it is very vulnerable again outliers: One light pixel somewhere in the image is enough to effectively disable the algorithm.

An improved approach was to perform global histogram equalization. The algorithm tranforms the gray values by streching/compressing the color histogram in order to normalize the gradient of the cumulative distance function. This works very well for most of the images and yields average results. The issue with the algorithm is, that the streching happens for the gray range which is supported by the most pixels. In images, where there is alot of light sky behind a dark sign silhuette, this in fact even lowers the contrast of the signs interior. 

The next approach I have followed was the local histogram equalization (CLAHE). It tries to circument the issues from the global histogram equalization by diving the image into a grid and calcuating local histograms for each grid cell. I have chosen a grid cell of 4 for the algorithm, which has lead to very good results:

In the final implementation, I run approach 3 and 1 in a chain. The image statistics from data set exploration now looks as following:

![Image Statistics after Normalization][image6]

All minimum values are zero, all maximum values are 255, the mean value for 90% of images lies around 127 with a standard deviation of about 50 for almost all images. I can see that the normalization went pretty well. Here are also an overview of normalized image samples:

![Images before/after Preprocessing][image7]

Before finally feeding the data into the neural net, the data is additional normalized to range -0.5 to 0.5 in order to provide a head start for learning.

#### Describe how/why you decided to generate additional data

The incentive for augmenting the raining set is one specific result of the original data exploration: The high variance in sample counts suggest that the neural network might be biased towards optimizing high frequent patterns as they will contribute much more to loss function. Also a low samples count will make it harder for the classifiers to generalze and extract the core features of a specific traffic sign class, so one should try not to fall below a certain threshold.
In fact, the confusion matrix after my first learning attemps constantly displayed a horrible precision (~25%) for trying to match class 0 (280 samples) correctly:

![Problems with class 0][image8]

Now as also suggested in the provided paper *Traffic Sign Recognition with Multi-Scale Convolutional Networks* by Sermanet and LeCun, the data set might be augmented by additional samples which are created via transformations from the core samples we got. 

My goal for the augmentation was to get an equalized number of samples for every class. To reach this goal, I needed a multiplier of 11 (280 samples minimum, vs. 2010 maximum). As inspiration for the transformations, I used the same variations found in the already existing data. This is important, as the transformations are supposed to generate as probable images as possible.
Here are the transformation operation I have implemented:

![Augmentation Operations][image9]

Let ni be the sample count of the current class and nmax the maximum sample count of any class. Then the augmentation method choses nmax // ni operations from the list for each sample of class i. For mnax % ni members of the class i, one additional operation is performed, totalling in a maximum of 11 operations.

After the augmentation, the visualization of the data set looks as following:

![Visualization after Augmenting][image10]


#### 2. Describe what your final model architecture looks like

including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 3x3     	| 1x1 stride, same padding, outputs 32x32x64 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 16x16x64 				|
| Convolution 3x3	    | etc.      									|
| Fully connected		| etc.        									|
| Softmax				| etc.        									|
|						|												|
|						|												|
 


####3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an ....

####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of ?
* validation set accuracy of ? 
* test set accuracy of ?

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
* What were some problems with the initial architecture?
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
 

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The first image might be difficult to classify because ...

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| U-turn     			| U-turn 										|
| Yield					| Yield											|
| 100 km/h	      		| Bumpy Road					 				|
| Slippery Road			| Slippery Road      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Stop sign   									| 
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|


For the second image ... 

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
####1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


