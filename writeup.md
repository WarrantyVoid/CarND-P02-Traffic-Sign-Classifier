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

This visualization shows the following things:
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

The solution is already given away in the provided paper [Traffic Sign Recognition with Multi-Scale Convolutional Networks](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf): YUV conversion. The YUV conversion is a transformation of the rgb color space, which leads to a luminance value and two chrominance values. I just drop both chrominance channels, because the luminance channel consists of different weights for red, green and blue, thus providing me already with good ingridients for classification:

![YUV][image5]

The next problem I have to address is the strong brightness and contrast variation of the images detected during data set exploration. It took me tree approaches to  end up with a satisfactory solution.

A first approach was to just normalize the gray intensity to the range 0 to 255 by substracting the minimum pixel value and dividing by the maximum pixel range. This  has lead to bad results. The approach has the big issue that it is very vulnerable again outliers: One light pixel somewhere in the image is enough to effectively disable the algorithm.

An improved approach was to perform global histogram equalization. The algorithm tranforms the gray values by streching/compressing the color histogram in order to normalize the gradient of the cumulative distance function. This works very well for most of the images and yields average results. The issue with the algorithm is, that the streching happens for the gray range which is supported by the most pixels. In images, where there is alot of light sky behind a dark sign silhuette, this in fact even lowers the contrast of the signs interior. 

The next approach I have followed was [adaptive histogram equalization](https://en.wikipedia.org/wiki/Adaptive_histogram_equalization). It tries to circument the issues from the global histogram equalization by using a local histogram for each pixel given it's rectangular neighourhood. I have chosen a neighbourhood size of 4 for the algorithm, which has lead to very good results:

In the final implementation, I run approach 3 and 1 in a chain. The image statistics from data set exploration now looks as following:

![Image Statistics after Normalization][image6]

All minimum values are zero, all maximum values are 255, the mean value for 90% of images lies around 127 with a standard deviation of about 50 for almost all images. I can see that the normalization went pretty well. Here are also an overview of normalized image samples:

![Images before/after Preprocessing][image7]

Before finally feeding the data into the neural net, the data is additional normalized to range -0.5 to 0.5 in order to provide a head start for learning.

#### Describe how/why you decided to generate additional data

The incentive for augmenting the raining set is one specific result of the original data exploration: The high variance in sample counts suggest that the neural network might be biased towards optimizing high frequent patterns as they will contribute much more to mean of loss of all samples. 

Also a low samples count will make it harder for the classifiers to generalize and extract the core features of a specific traffic sign class. In fact, the confusion matrix after my first learning attemps constantly displayed a horrible precision (~25%) for trying to match class 0 (280 samples) correctly:

![Problems with class 0][image8]

Now as suggested in the provided paper [Traffic Sign Recognition with Multi-Scale Convolutional Networks](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf), the data set might be augmented by additional samples which are created via transformations from the core samples we got. 

My goal for the augmentation was to get an equalized number of samples for every class. While this might not contribute to improve the test accuracy specifically (test set shows the same distribution of samples as the training set), I do not want the network to learn (yet) that in reality some traffic sign classes occur more frequent than others.

To reach this goal, I needed a multiplier of 11 (280 samples minimum, vs. 2010 maximum). As a rule set for possible transformations, I used the variations found in the already existing data. This is important, as the ouput of the transformations is supposed to be as indistiguishable as possible from real samples inside the core set. Another important criteria for the transformations is, that they must not leave any significant artifacts along the boundary of the images. Here are the transformation operations I have implemented:

![Augmentation Operations][image9]

Let ni be the sample count of the current class and nmax the maximum sample count of any class. Then the augmentation method choses nmax // ni operations from the list for each sample of class i. For mnax % ni members of the class i, one additional operation is performed, totalling in a maximum of 11 operations.

After the augmentation, the visualization of the data set looks as following:

![Visualization after Augmenting][image10]


#### 2. Describe what your final model architecture looks like

There is really a world of possibilities here, but as I am still beginner with neural networks and I don't have much CPU resources available for training, I decided to not diverge much from the already proven LeNet architecture. However, [Traffic Sign Recognition with Multi-Scale Convolutional Networks](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf) has proven as a geat source of information and inspired me for some modifications. So my final model consist of the following layers:


| Layer 				| Description			 						| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 RGB image   							| 
| Convolution 5x5		| 1x1 stride, VALID padding, outputs 28x28x43 	|
| RELU					|												|
| Max pooling			| 2x2 stride,  outputs 14x14x43 				|
| Convolution 5x5		| 1x1 stride, VALID padding, outputs 10x10x43	|
| RELU					|												|
| Max pooling			| 2x2 stride,  outputs 5x5x43					|
| Concatenation			| 4x4 Pooling on c1 + c2, outputs 3x3x43+5x5x43	|
| Fully connected		| outputs 666      								|
| Dropout				| 0.5 rate, outputs 666 						|
| Fully connected		| outputs 222  									|
| Dropout				| 0.5 rate, outputs 222 						|
| Logits				| outputs 43      								|
|						|												|
|						|												|

The design looks like this:

![Neural Network Design][image11]

the number of trainable parameters of the network is **1051921**:
* Convolution 1: (1 + 5 * 5 * 1) * 43 = **1118**
* Convolution 2: (1 + 5 * 5 * 43) * 43 = **46268**
* Full 1       : (1 + 3 * 3 * 43 + 5 * 5 * 43) * 542 = **792946**
* Full 2       : (1 + 542) * 361 = **196023**
* Logits       : (1 + 361) * 43 = **15566**


##### Convolutions

As already captured, I used the normalized, augmented grayscale images as input to the network.
The first two layers are 5x5 convolutional layers consisting of the sequence:
* 2-dimensional convolution (5x5, 42 filter size)
* non-linearity (ReLu)
* pooling (max pooling)

The filter size of the first convolution in LeNet has been originally 6 for the MNIST data set. I take it that this size was chosen in respect to the number of distinctive 5x5 patterns when considering black hand-written text on white background. But in our use case, we have not such clear structures in the input data. As established earlier, we need at least 15 unique brightness patterns to distiguish the sub class "red-stroked triangles" and 12 unique brightness patterns to distigiush "red-stroked circles". Sermanet and LeCun actually propose using a filter size of 108, but this exceeds what my CPU resources can do, so I decided to go with at filter size of 43 which leaves at least room for up to one unqiue 5x5 weight matrix for each class.

The filter size of the seconds convolution has been originally 16 for the MNIST data set. This resembles to the number of distinctive combinations of convolution 1 outputs. Here I really want to have at least 43 filters. If there were less, our classifier layers would later on have to make up for it by determining the probability of some classes purely on linear combinations of distinctive patterns from other classes. Sermanet and LeCun again propose using a filter size of 108 here.

For the non-linearity I have tried to replicate Sermanet and LeCuns choice by replacing the ReLu with abs(gi * tanh()) followed by local normalization. Unfortunetly although eventually leading to better results, this non-linearity does not only slow the learning rate but also increases the time for each learning step tremendously, so I was forced to give up on the approach. 

For the pooling I have tested with average and max pooling, but max pooling provided better results as it is more successful in carving out the edges of the images. 

In addition, as proposed by Sermanet and LeCuns, I added the Multi-Scale Features design to the network and branched the output of the first convolution directly into the classifier. Before concatenating it with the result of the second convolution, I ran the output through an additional 4x4 max pooling, because I wanted the major part of data for the classifier still to come from convolution 2.

##### Classifier

I used a classifier with the two hidden layers as in the original LeNet. There is the option of trading one classifier layer for one additional convolution layer or even dropping the 2nd hidden layer entirely, but I did not investigate deeper in these aproaches. Each fully connected layer consists of the sequence:

* fully connected linear combinations
* non-linearity (ReLu)
* dropout (50%)

As a rule of thumb, the hidden layer of a classifier should feature a number of nodes near the mean number of input and output nodes. In fact, this formula seems to somewhat match for the default LeNet setup: *(400 + 10) / 2 ~= 120 + 84* with the ratio between first and second hidden layer being 60% to 40%. So I adapted the number of nodes to the new input and output count: *(1763 + 43) / 2 ~= 542 + 361* hoping this is somehow near reasonable range.

In addition to prevent the strong overfitting caused by the increased parameter count, I furthermore added dropouts to all fully connected layers.


#### 3. Describe how you trained your model

For training I have used the AdamOptimizer, because it is the state of the art optimizer which combines the advantages of other optimizer like e.g. AdaGrad or RMSProp. For the parameters I have chosen the default ones, although there are documentation fragments which mention that this is not optimal.

As loss function I have used the softmax cross entroy with logits. It is used to maximize the certainty of the networks resulting class probability. The advantage of the cross entropie is that it provides a big loss for probabilities which are way off and thus helps for faster converging of probabilities.

I further added a L2 regularisation loss with a factor of 0.001 in order to fight extreme weights in the fully connected layers and avoid overfitting.

As batch size I have chosen 391, because that nicely splits the training set into 89 batches without remainer. Since the entire data set of 34799 * 32 * 32 * 1 float would actually fit into my RAM, the batching would is not absultely necessary. But as I have the same amount of each sample due to the augmentation, I supposed that the random selections of the batches is representative enough so I can use the batching as a way to improve the learning frequency.

For the learning rate I started with a value 0.001 as in the LeNet implementation, but then rather quickly ended up having to adjusts the rate down each time the validation set accuracy would begin drop due to diverging weight steps. I ended up using a learning rate of 0.0001 for the first ten epochs.

As the number of epochs I used iterations 10 before save the trained weights. Each sucessive iteration of those 10 epochs allowed me to adjust the hyper parameters as needed. Heres a graph of the learning process:

![Learning rates][image12]


#### 4. Describe the approach taken for finding a solution

and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.



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
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report

I have captured 44 traffic sign images from the web:

![Images from Web][image13]

Each images refers to one traffic sign class of our network. One images not refer to a unknown traffic sign class.
Half of the images have been captured from google StreetView, but some classes were hard to find, so I extracted whatever google image search could find. For the class *[006]: "End of speed limit (80km/h)"* I just ended up drawing a somewhat "authentic" ;) traffic sign with gimp.

#### 2. Discuss the model's predictions on these new traffic signs

The models predictions on the web data set are all correct (all, but the unknown image).This exceeds the precision of the test set, but that might be just caused by the small size of the set. Furthermore, the predictions for all non-correct classes are almost 0, which means that the model has been very certain about its predictions. This certainly a result of using the cross entropy as loss function. Perpective, varying backgrounds, variying contrast/brightness and even a smiley face could not detract the network from its correct predictions.

#### 3. Describe how certain the model is when predicting on each of the five new images

I had to set the axis to logarithmic in order to make the probabilities beyond the prediction visible.
Here are select few images and their possible misinterpretations:

![Misinterpretations][image14]

For the first image, I think that the network has learned parts of the round boundary and the "0" pattern inside the sign, which I think is why the probability of class [0] ("20km/h") and [1] ("30km/h") is still noticable. Furthermore the model seems to have learned the diagonal slash through the sign as there is also a noticable probability for class [32] ("end of all restrictions").

For the second image, the model has learned the diagonal backslash, which belongs to class [38] ("Hold right") and the left pointing arrow, which is a feature of class [37] ("Straight or left"). THe probabilities to classes [20] and [25], I cannot explain.

In general I think this section would be much more fun, if the model was less accurate in its predictons.


### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
####1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


