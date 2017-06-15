# Traffic Sign Recognition

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the [German Traffic Sign Data Set](https://d17h27t6h515a5.cloudfront.net/topher/2017/February/5898cd6f_traffic-signs-data/traffic-signs-data.zip)
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
[image11]: ./examples/network.png "Neural Network Design"
[image12]: ./examples/learning.png "Learning rates"
[image13]: ./examples/web_images.png "Images from Web"
[image14]: ./examples/misinterpretations.png "Misinterpretations"
[image15]: ./examples/featuremaps.png "Feature Map Convolution 1"
[image16]: ./examples/featuremaps_pool.png "Feature Map Convolution 1 Max Pooling"
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

There's a few things I could use this visualization for:
* By comparing the occurences in this chart against the occurences in the validation and training set, I could confirm that the test and validation sets are actually a very good representation of the training set (class count-wise).
* The visualization makes it very clear that some traffic sign classes occur much more frequent that others. The classes with the most samples have about 10x more samples than the least frequent classes. This could later on bias the neural networks noticeably towards the more frequently occuring classes.

Another thing that gets clear during the exploration of the data set is, that all of the samples are sorted by class. This was another strong reminder to me that I should not forget to shuffle the data set during learning.

Here is another exploratory visualization of the data set. This is a diagram showing several statistical image values applying to a cumulated percentual amount of images:

![Image Statistics][image2]

This visualization shows the following things:
* By comparing this statistics against the same statistics for validation and training set, I can confirm that those are also pixel-variation-wise a very good representation of the training set.
* A statistically relevant parts of the images are showing high variations in global contrast and brightness. But for the neural network I need the images to be as uniform as possible while not removing any relevant features of traffic signs. So a preprocessing of this images set can not be avoided.


I have also included an exploratory visualization of the data set that actually show five random samples for each traffic sign class in the [project code](https://github.com/WarrantyVoid/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb) and then later on dumped all images into a directory for detailed browsing.

This visualization helped a great deal. It clearly shows the upcoming challenges for the neural network, as the following example samples show:

![Image Issues][image3]

* Some signs are blurred, caused by either motion or by water on windshield
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

As first step, based on the exploration of the data set, I identify the following hierarchy for traffic sign classes:

![Traffic Sign classes][image4]

This brings up some important realizations:
* Three sign classes can be classified entirely by shape
* The 15 red-stroked triangle and the 12 red-stroked circle classes rely entirely on their black and white interior for classification
* Despite the specific considerations red/blue vs. others and white vs black, the pixel color does not have significant value for the classification

So the goal of my preprocessing shall be to remove all of the color information while keeping red/blue similarity as a key signature. A simple gray scale conversion is out of the question, because this would lead to the very same gray values for red, blue and green. 

The solution is already given away in the provided paper [Traffic Sign Recognition with Multi-Scale Convolutional Networks](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf): YUV conversion. The YUV conversion is a transformation of the rgb color space, which leads to a luminance value and two chrominance values. I just drop both chrominance channels, because the luminance channel yields from different weights for red, green and blue channels, thus providing me already with good ingridients for classification:

![YUV][image5]

The next problem I have to address is the strong brightness and contrast variation of the images detected during data set exploration. It took me tree approaches to  end up with a satisfactory solution.

A first approach was to just normalize the gray intensity to the range 0 to 255 by subtracting the minimum pixel value and dividing by the maximum pixel range. This  has lead to bad results. The approach has the big issue that it is very vulnerable again outliers: One light pixel somewhere in the image is enough to effectively disable the algorithm.

An improved approach was to perform global histogram equalization. The algorithm transforms the gray values by stretching/compressing the color histogram in order to normalize the gradient of the cumulative distance function. This works very well for most of the images and yields average results. The issue with the algorithm is, that the streching happens for the gray range which is supported by the most pixels. In images, where there is a lot of light sky behind a dark sign silhuette, this in fact even lowers the contrast of the signs interior. 

The next approach I have followed was [adaptive histogram equalization](https://en.wikipedia.org/wiki/Adaptive_histogram_equalization). It tries to circumvent the issues from the global histogram equalization by using a local histogram for each pixel given it's rectangular neighborhood. I have chosen a neighborhood size of 4 for the algorithm, which has lead to very good results.

In the final implementation, I run approach 3 and 1 in a chain. The image statistics from data set exploration now looks as following:

![Image Statistics after Normalization][image6]

All minimum values are zero, all maximum values are 255, the mean value for 90% of images lies around 127 with a standard deviation of about 50 for almost all images. I can see that the normalization went pretty well. Here are also an overview of normalized image samples:

![Images before/after Preprocessing][image7]

Before finally feeding the data into the neural net, the data is additional normalized to range -0.5 to 0.5 in order to provide a head start for learning.

#### Describe how/why you decided to generate additional data

The incentive for augmenting the training set is one specific result of the original data exploration: The high variance in sample counts suggest that the neural network might be biased towards optimizing high frequent patterns as they will contribute much more to mean of loss of all samples. 

Also a low samples count will make it harder for the classifiers to generalize and extract the core features of a specific traffic sign class. In fact, the confusion matrix after my first learning attempts constantly displayed a horrible precision (~25%) for trying to match class 0 (280 samples) correctly:

![Problems with class 0][image8]

Now as suggested in the provided paper [Traffic Sign Recognition with Multi-Scale Convolutional Networks](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf), the data set might be augmented by additional samples which are created via transformations from the core samples we got. 

My goal for the augmentation was to get an equalized number of samples for every class. While this might not contribute to improve the test accuracy specifically (test set shows the same distribution of samples as the training set), I do not want my model not to learn that in reality some traffic sign classes occur more frequent than others at this point.

To reach my goal, I needed a multiplier of 11 (280 samples minimum, vs. 2010 maximum). As a rule set for possible transformations, I used the variations found in the already existing data. This is important, as the output of the transformations is supposed to be as indistinguishable as possible from real samples inside the core set. Another important criteria for the transformations is, that they must not leave any significant artifacts along the boundary of the images. Here are the transformation operations I have implemented:

![Augmentation Operations][image9]

Let ni be the sample count of the current class and nmax the maximum sample count of any class. Then the augmentation method choses nmax // ni operations from the list for each sample of class i. For mnax % ni members of the class i, one additional operation is performed, totaling in a maximum of 11 operations.

After the augmentation, the data set contains 86430 samples and its visualization looks as following:

![Visualization after Augmenting][image10]


#### 2. Describe what your final model architecture looks like

There is really a world of possibilities here, but as I am still beginner with neural networks and I don't have much CPU resources available for training, I decided to not diverge much from the already proven LeNet-5 architecture. However, [Traffic Sign Recognition with Multi-Scale Convolutional Networks](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf) has proven as a great source of information and inspired me for some modifications. So my final model consist of the following layers:


| Layer 				| Description			 						| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 RGB image   							| 
| Convolution 3x3		| 1x1 stride, VALID padding, outputs 30x30x21 	|
| ReLu					|												|
| Max pooling			| 2x2 stride,  outputs 15x15x21 				|
| Dropout				| 0.8 rate 										|
| Convolution 3x3		| 1x1 stride, VALID padding, outputs 13x13x43	|
| ReLu					|												|
| Max pooling			| 2x2 stride,  outputs 7x7x43					|
| Dropout				| 0.8 rate 										|
| Convolution 3x3		| 1x1 stride, SAME padding, outputs 7x7x54		|
| ReLu					|												|
| Max pooling			| 2x2 stride,  outputs 4x4x54					|
| Dropout				| 0.8 rate 										|
| Concatenation			| c1 + c2 + c3, outputs 4x4x21+4x4x43+4x4x54	|
| Fully connected		| outputs 240      								|
| ReLu					|												|
| Dropout				| 0.5 rate				 						|
| Fully connected		| outputs 168  									|
| ReLu					|												|
| Dropout				| 0.5 rate				 						|
| Logits				| outputs 43      								|


The design looks like this:

![Neural Network Design][image11]

The number of trainable parameters of the network is **530207**:
* Convolution 1: (1 + 3 * 3 *  1) * 21 = **210**
* Convolution 2: (1 + 3 * 3 * 21) * 43 = **8170**
* Convolution 3: (1 + 3 * 3 * 43) * 54 = **20952**
* Full 1       : (1 + 4 * 4 * 21 + 4 * 4 * 43 + 4 * 4 * 54) * 240 = **453120**
* Full 2       : (1 + 240) * 168 = **40488**
* Logits       : (1 + 168) * 43 = **7267**

##### Convolutions

As already captured, I used the normalized, augmented gray scale images as input to the network.
The first tree layers are 3x3 convolutional layers consisting of the sequence:
* 2-dimensional convolution
* non-linearity (ReLu)
* pooling (max pooling)
* dropout

The filter size of the first convolution in LeNet has been originally 6 for the MNIST data set. I take it that this size was chosen in respect to the number of distinctive 5x5 patterns when considering black hand-written text on white background. But in our use case, we have not such clear structures in the input data. As established earlier, we need at least 15 unique brightness patterns to distinguish the sub class "red-stroked triangles" and 12 unique brightness patterns to distingiush "red-stroked circles". Sermanet and LeCun actually propose using a filter size of 108, but this exceeds what my CPU resources can do, so I decided to split that filter into two 3x3 convolutions which should leave room for more and also more complex pattern combinations

The filter size of the second convolution has been originally 16 for the MNIST data set. This resembles to the number of distinctive combinations of convolution 1 outputs. At this point I really want to have at least 43 filters. If there were less, our classifier layers would later on have to make up for it by determining the probability of some classes purely on linear combinations of distinctive patterns from other classes. Sermanet and LeCun again propose using a filter size of 108 here.

For the non-linearity I have tried to replicate Sermanet and LeCuns choice by replacing the ReLu with abs(gi * tanh()) followed by local normalization. Unfortunately although eventually leading to better results, this non-linearity does not only slow the learning rate but also increases the time for each learning step tremendously, so I was forced to give up on the approach. 

For the pooling I have tested with average and max pooling, but max pooling provided better results as it is more successful in carving out the edges of the images. 

In addition, as proposed by Sermanet and LeCun, I added the Multi-Scale Features design to the network and branched the output of the first two convolutions directly into the classifier. Before concatenating them with the result of the last convolution, I ran the output through additional max poolings, because I want the data of all convolutions to contribute equally to the last convolution tensor.

##### Classifier

I used a classifier with the two hidden layers as in the original LeNet. There is the option of trading one classifier layer for one additional convolution layer or even dropping the 2nd hidden layer entirely, but I did not investigate deeper in these aproaches. Each fully connected layer consists of the sequence:

* fully connected linear combinations
* non-linearity (ReLu)
* dropout

As a rule of thumb, the hidden layer of a classifier should feature a number of nodes near the mean number of input and output nodes. In fact, this formula seems to somewhat match for the default LeNet setup: *(400 + 10) / 2 ~= 120 + 84* with the ratio between first and second hidden layer being 60% to 40%. So I just adapted the number of nodes very roughly to the new input hoping this is somehow near reasonable range.

In addition to prevent the overfitting caused by the increased parameter count, I furthermore added dropouts to all fully connected layers.


#### 3. Describe how you trained your model

For training I have used the AdamOptimizer, because it is the state of the art optimizer which combines the advantages of other optimizer like e.g. AdaGrad or RMSProp. For the parameters I have chosen the default ones, although there are documentation fragments which mention that this is not optimal.

As loss function I have used the softmax cross entropy with logits. It is used to maximize the certainty of the networks resulting class probability. The advantage of the cross entropie is that it provides a big loss for probabilities which are way off and thus helps for faster converging of probabilities.

I further added a L2 regularisation loss with different factors in order to fight extreme weights in the fully connected layers and avoid overfitting.

As batch size I have chosen 430, because that nicely splits the training set into 201 batches without remainder. Since the entire data set of 86430 * 32 * 32 * 1 float would actually fit into my RAM, the batching would is not absolutely necessary. But as I have the same amount of each sample due to the augmentation, I supposed that the random selections of the batches is representative enough so I can use the batching as a way to improve the learning frequency.

For the learning rate I started with a value 0.001 as in the LeNet implementation, and then ended up adjusting the rate down each time the validation set accuracy would begin drop due to diverging weight steps. I ended up using a successively smaller learning rate the later the epoch but never tried smaller values than 0.0001.

As the number of epochs I used iterations of 10 before saving the trained weights. Each successive iteration of those 10 epochs allowed me to adjust the hyper parameters as needed. Here's a graph of the learning process with basic LeNet-5 on augmented and preprocessed data:

![Learning rates][image12]


#### 4. Describe the approach taken for finding a solution

During my quest for a good solution I've made quite some beginner errors: 
* I wasted a lot of time playing around with different network structures **before** having actually normalized and augmented the data set
* I added regularization techniques like dropout and L2 regularization already to the very first models, thus unnecessarily decreasing the learning speed
* I only compared the bare results of the learning process (accurracy/confusion matrix) after a fixed amount of epochs in order to separate good from supposedly bad models, thus misjudging models with slower learning rate but ultimately better results at higher epochs

So at some point I  had to start all over, extending the basic LeNet-5  model iteratively and watching the results more carefully. After 20 epochs the base model has lead to the following result:
* **0.995%** accuracy in training set
* **0.957%** accuracy in validation set

While this would already satisfies the minimum project requirement, looking at the learning graph has made some imminent problems visible:
* The mean cross entropy curve has fallen almost flat since about epoch 20, so it looks like learn success has already gone as far as it could
* The training accuracy of almost 100% also supports the above assumption 
* The training error and validation error constantly kept diverging since the start of learning, showing massively overfitting

In the first steps, I kept ignoring the overfitting and tried to focus to lower the validations minimum cross entropy. To achieve this, I have tried to make the model deeper, so it may adjust to more complex pattern relations. While the approach to add 1x1 convolutions might have been successful to some extent too, (being inspired by [AlexNet](http://vision.stanford.edu/teaching/cs231b_spring1415/slides/alexnet_tugce_kyunghee.pdf)) I rather followed the option to add another convolution to the pyramid. 

With this 3 layer convolution network, after 14 epochs of total training, the results were:
* **0.997%** accuracy in training set
* **0.961%** accuracy in validation set
While the minimum cross entropy did not drop much and may be within variance, the model succeeded at reaching that minimum in 30% fewer epochs than the base model did, which was a good reason for me to stick with this architecture.

So as next step I tried to implement the Multi Scale Architecture as proposed by Sermant and LeCun on top of the convolutions. For this I also had to adapt some of the layer sizes relatively towards each other. The result after just 8 epochs of training was somewhat of a double-edged sword:
* **0.997%** accuracy in training set
* **0.955%** accuracy in validation set

While models capability to reach certainty about classes has again increased by a big 40%, the overfitting has also increased slightly. So I decided to give this model another shot with slight dropouts inside the now branched convolution layers. The result after 10 epochs was:
* **0.997%** accuracy in training set
* **0.971%** accuracy in validation set

Since that worked better even as expected well, I started now adding strong dropout to both fully connected layers in order to further push the generalizing capabilities of the network and thus increase the validation accuracy. The result after 20 epochs was:
* **0.997%** accuracy in training set
* **0.989%** accuracy in validation set

In order to decrease overfitting even more, I added L2 regularisation with different factors to the loss function, but the results were all worse. I interpret it the way that in my model, dropout encourages larger absolute weight combinations to make up for lost connections, while L2 regularization penalizes growing absolute weights, so that the two do not work together very well. 

So as last attempt I increased convolution dropout a little bit more and tried to reach a better training accuracy with smaller learning steps, but to no success. Also further increasing the number of filters had no positive effect. Maybe growing deeper would have still benefited the result. Most certainly the accuracy would have improved by using the same sample count distribution as in test/validation set. My final results were:
* **0.997%** accuracy in training set
* **0.989%** accuracy in validation set
* **0.975%** accuracy in test set


### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report

I have captured 44 traffic sign images from the web:

![Images from Web][image13]

Each images refers to one traffic sign class of our network. One image does actually refer to a unknown traffic sign class.
Half of the images have been captured from google StreetView, but some classes were hard to find, so I extracted whatever google image search could find. For the class *[006]: "End of speed limit (80km/h)"* I just ended up drawing a somewhat "authentic" ;) traffic sign with [gimp](https://www.gimp.org/).

#### 2. Discuss the model's predictions on these new traffic signs

The models predictions on the web data set are all correct (all, but the unknown image). This exceeds the precision of the test set. That might be just caused by the small size of the web set, but might have been also positively influenced by the equalized sample count in training set. 

Furthermore, the predictions for all non-correct classes are almost 0, which means that the model has been very certain about its predictions. This is definitely a result of using the cross entropy as loss function. It comes at the loss that the model is also almost certain about classes it doesn't even know. On the strong side, perspective, varying backgrounds, variying contrast/brightness and even a smiley face could not detract the network from delivering correct predictions.

#### 3. Describe how certain the model is when predicting on each of the five new images

I had to set the axis to logarithmic in order to make the probabilities beyond the prediction visible.
Here are select few images and their possible misinterpretations:

![Misinterpretations][image14]

For the first image, I think that the network has learned parts of the round boundary and the "0" pattern inside the sign, which I think is why the probability of class [0] ("20km/h") and [1] ("30km/h") is still noticable. Furthermore the model seems to have learned the diagonal slash through the sign as there is also a noticable probability for class [32] ("end of all restrictions").

For the second image, the model has learned the diagonal backslash, which belongs to class [38] ("Hold right") and the left pointing arrow, which is a feature of class [37] ("Straight or left"). 


### (Optional) Visualizing the Neural Network

#### 1. Discuss the visual output of your trained network's feature maps

Lets take a look at the feature maps of input image [018]: "General caution" which is one of the images which is quite hard to classify with maximum certainty.

The feature map of the first convolution after full traning already shows that the model has various input characteristics available to identify patterns:

!["Feature Map Convolution 1"][image15]

Input characteristics with high weights (light color) are:
* The outer edges of each triangle side of the sign
* The inner edges of each triangle side of the sign
* The light background shape inside the interior of the sign
* The symbol (three circles) edges inside the sign
* The symbols interior

I did not plot the feature maps before training, but I can imagine the maps starting out just with random noise over the original image data and then the characteristic edges fleshing out step by step during the training.

Ater being fed through the map pooling, the resolution of the feature visually loses alot of resolution:

!["Feature Map Convolution 1 Max Pooling"][image16]

This explains, why the images symbol is quite hard to distiguish from e.g. [018]:    1 x "General caution" or [024]: "Road narrows on the right". We need to make sure that there are enough feature maps available to make up for the loss in resolution to avoid misclassification.

While passing through the successive convulutions, the image data is getting more and more abstract and sclided thoughout the different feature maps. What is noticeable though when browsing through the higher convolution feature maps is:

a) How the ReLu consistently seems to act as some kind of filter silencing noise in low range of the feature maps (e.h. on the outside of the traffic sign)

b) How well the model has been actually trained. Even in highest convolution layer there are no repititions in feature maps and no dead featur maps either. Almost every map has at least several brighter spots after the ReLu. This is very likely an accomplishment of the dropout technique.

