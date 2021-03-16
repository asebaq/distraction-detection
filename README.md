# distraction-detection
This is a driver distraction detection project.

### Introduction
Driving a car is a complex task, and it requires complete attention. 
Distracted driving is any activity that takes away the driver’s attention from the road. 

Several studies have identified three main types of distraction: 
 - visual distractions (driver’s eyes off the road)
 - manual distractions (driver’s hands off the wheel)
 - cognitive distractions (driver’s mind off the driving task).

The National Highway Traffic Safety Administration (NHTSA) reported that 
36,750 people died in motor vehicle crashes in 2018, and 12% of it was due to 
distracted driving. Texting is the most alarming distraction. 
Sending or reading a text takes your eyes off the road for 5 seconds. At 55 mph, 
that’s like driving the length of an entire football field with your eyes closed.



### Problem statement

We need to develop a model to be deployed in cars to detect if a driver is distracted from the road so that the car system can provide the necessary alerts.
The ideal input should be a stream of images from a camera facing the driver and you should output an indication of if he is distracted or not.

### Reflection on the problem statement

As my work with ***Affectiva*** as a mentor for two months I witnessed how they have done lots of research in this area especially Dr. Taniya Mishra the Ex. Director of AI Research of ***Affectiva*** and they also have done some amazing patents which worth investigating and study.

First of all the problem statement is poor as there is no specified dataset to work with. 

Second, in this type of problem we can not neglect the **temporal information** so instead of working with images we have to work with videos, also the problem is considered an image classification problem which is obscure and does not make much sense, in these situation we may use **pose estimation** or **facial landmarks** or maybe combine both of them to detect wither the driver is distracted or not.

Third, the task needs a very big amount of data with different distribution to be done as a classification task of course with an accepted accuracy.

With that said let's proceed to work on a dataset with specified problem statement.

### Data

We used the [StateFarm dataset](https://www.kaggle.com/c/state-farm-distracted-driver-detection/data) which contained snapshots from a video captured by a camera mounted in the car. The training set has ~22.4 K labeled samples with equal distribution among the classes and 79.7 K unlabeled test samples. 
There are 10 classes of images:

 - Safe driving
 - Texting - Right
 - Talking on the Phone - Right
 - Texting - Left
 - Talking on the Pone - Left
 - Operating the radio
 - Drinking
 - Reaching behind
 - Hair and makeup
 - Talking to passengers

### Evaluation metric
Before proceeding to build models, it’s important to choose the right metric to measure the model performance. 
**Accuracy** is the first metric that comes to mind. But, accuracy is not the best metric for classification problems. 

Accuracy only takes into account the correctness of the prediction, whether the predicted label is the same as the true label. But, the confidence with which we classify a driver’s action as distracted is very important in evaluating the performance of the model. 
Thankfully, we have a metric that captures just that — Log Loss.

**Logarithmic loss** (related to cross-entropy) measures the performance of a classification model where the prediction input is a probability value between 0 and 1. The goal of our machine learning models is to minimize this value. A perfect model would have a log loss of 0 and it increases as the predicted probability diverges from the actual label. So predicting a probability of 0.3 
when the actual observation label is 1 would result in a high log loss.

### Use pickle files and callbacks

We will save the necessary files as **pickle** files. This way, we can pick up where you left off by directly loading the pickle files.

We can use the **CallBacks** feature in Keras which saves the weights of a model only if it sees improvement after an epoch. 

### Building and training models
I proceeded to build the **CNN** models from scratch. I added the usual components like convolution, max pooling, and dense layers. We build a model similar to the **AlexNet**.

![Test data](basic_model_aug\\basic_distraction_model_aug.png)

The results were outstanding with a loss around **0.014** and accuracy around **99.6%** on the validation set in just **5** epochs.

Well, this is just too good to be true so contemplated for a second about accidentally building the best CNN architecture 
the world has ever seen. So, we predicted the classes for the unlabeled test set using this model.

![Test data](basic_model_aug\\basic_model_test_samples.png)

So, we looked deeper into what could have gone wrong and we found that our training data has multiple images of the same person within a class with slight changes of angle and/or shifts in height or width.

This was causing a data leakage problem as the similar images were in validation as well, in other words the model was trained much of the same information that it was trying to predict.  

Well. There was no serendipity after all.



### Solutions to data leakage

In our *first trail* we split the data randomly with 80-20 split which is the main source of data leakage.

**Stratified sampling**

For the *second trial* we tried to overcome the data leakage problem, we split the images based on the person IDs (stratified sampling) and then do a random 80–20 split.

After training for **10** epoch we achieve **39.2%** accuracy which seems reasonable.

To improve the results further, we explored some techniques to increase the accuracy.

**Image Augmentation**

Now comes the *third trial* Since our training image set had only ~22K images, we wanted to synthetically get more images from the training set to make sure the models don’t overfit as the neural nets have millions of parameters. 

Image Augmentation is a technique that creates more images from the original by performing actions such as shifting width and/or height, rotation, and zoom.

For our project, Image Augmentation had a few additional advantages. Sometimes, the difference between images from the two different classes can be very subtle. In such cases, getting multiple looks at the same image through different angles will help.

Now after **10** epochs we achieve accuracy of **37.4%** which is accepted as augmentation may increase the problem complexity especially for basic models like the one we have. 

Here we use augmentation techniques like: shearing, zooming, and horizontal flip.

![Image augmentation](basic_model_aug\\data_augmentation.png "Image augmentation")

**Transfer learning**

The *fifth trial* was to combine data augmentation with transfer learning which is a method where a model developed for a related task is reused as the starting point for a model on a second task. We can re-use the model weights from pre-trained models that were developed for standard computer vision benchmark datasets, such as the ImageNet image recognition challenge. 

Generally, the final layer with Softmax activation is replaced to suit the number of classes in our dataset. In most cases, extra layers are also added to tailor the solution to the specific task.

It is a popular approach in deep learning considering the vast compute and time resources required to develop neural network models for image classification. Moreover, these models are usually trained on millions of images which helps especially when your training set is small. Most of these model architectures are proven winners - VGG16, RESNET50, Inception, Xception and Mobilenet models that we leveraged gave exceptional results on the ImageNet challenge.

**Optimizer**

Optimizers minimize an objective function parameterized by a model’s parameters by updating the parameters in the opposite direction of the gradient of the objective function w.r.t. to the parameters.

The most popular algorithm in the deep learning world is Adam which combines SGD and RMS Prop. It has been consistently performing better than other optimizers for *most* problems. However, in our case, Adam showed erratic pattern of descent while SGD was learning gradually. By doing some literature survey, I found that in few cases SGD is superior to Adam because SGD generalizes better ([link](https://arxiv.org/abs/1705.08292)). As SGD was giving stable results, we used it for all our models.

**Other architectures**

I tried only one transfer learning model with the weights from training on the ImageNet dataset (pre-trained weights).

**InceptionV3**
Inception v3 is a widely-used image recognition model that has been shown to attain greater than 78.1% accuracy on the ImageNet dataset. The model is the culmination of many ideas developed by multiple researchers over the years. It is based on the original paper: ["Rethinking the Inception Architecture for Computer Vision"](https://arxiv.org/abs/1512.00567) by Szegedy, et. al.

The model itself is made up of symmetric and asymmetric building blocks, including convolutions, average pooling, max pooling, concats, dropouts, and fully connected layers. Batchnorm is used extensively throughout the model and applied to activation inputs. Loss is computed via Softmax.

We first we freeze the deep layers and only train the top ones(which were randomly initialized).

At this point, the top layers are well trained and we can start fine-tuning convolutional layers from inception V3 the bottom N layers and continue train the remaining top layers.

Now after **5** epochs for finetuning and **10** epochs of full training we achieve accuracy of **57.4%** which is acceptable for this little number of epochs, of course if we increased the number the accuracy will also increase. 



### Conclusion

As we can see that the driver distraction detection is a tricky task to handle the temporal information is important also using facial landmarks or pose estimation will help.

But if we want to tackle it from just a classification point of view I suggest first we train the data on different known architectures and analysis their error second we using ensembling to but them all in one model.