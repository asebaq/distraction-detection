# distraction-detection
This is a driver distraction detection project, at first let's start by studying the problem at hand.

### Introduction
<p>For me in my working with <b><em>Affectiva</em></b> as a mentor for two months I witnessed how they are big research
in this area especially Dr. Taniya Mishra the Ex. Director of AI Research of <b><em>Affectiva</em></b> and they also have done
some amazing patents which worth investigating and study.
</p>

<p> Driving a car is a complex task, and it requires complete attention. 
Distracted driving is any activity that takes away the driver’s attention from the road. 
</p>
Several studies have identified three main types of distraction: 
 - visual distractions (driver’s eyes off the road)
 - manual distractions (driver’s hands off the wheel)
 - cognitive distractions (driver’s mind off the driving task).
<p> The National Highway Traffic Safety Administration (NHTSA) reported that 
36,750 people died in motor vehicle crashes in 2018, and 12% of it was due to 
distracted driving. Texting is the most alarming distraction. 
Sending or reading a text takes your eyes off the road for 5 seconds. At 55 mph, 
that’s like driving the length of an entire football field with your eyes closed.
</p>


### Problem statement
<p>We need to devel a model to be deployed in cars to detect if a driver 
is distracted from the road so that the car system can provide the necessary alerts.
Your ideal input should be a stream of images from a camera facing the driver 
and you should output an indication of if he is distracted or not. </p>

### Data
We used the [StateFarm dataset](https://www.kaggle.com/c/state-farm-distracted-driver-detection/data) 
which contained snapshots from a video captured 
by a camera mounted in the car. The training set has ~22.4 K labeled samples 
with equal distribution among the classes and 79.7 K unlabeled test samples. 
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

### Evaluation Metric
Before proceeding to build models, it’s important to choose the right metric to measure its performance. 
Accuracy is the first metric that comes to mind. But, accuracy is not the best metric for classification 
problems. Accuracy only takes into account the correctness of the prediction i.e. whether 
the predicted label is the same as the true label. But, the confidence with which we classify 
a driver’s action as distracted is very important in evaluating the performance of the model. 
Thankfully, we have a metric that captures just that — Log Loss.
Logarithmic loss (related to cross-entropy) measures the performance of a classification model 
where the prediction input is a probability value between 0 and 1. The goal of our machine learning 
models is to minimize this value. A perfect model would have a log loss of 0 and it increases as 
the predicted probability diverges from the actual label. So predicting a probability of 0.3 
when the actual observation label is 1 would result in a high log loss

### Building models
I proceeded to build the CNN models from scratch. 
I added the usual components — convolution batch normalization, 
max pooling, and dense layers. 
The results were outstanding with a loss around 0.014 and accuracy around 99.6% on 
the validation set in just 3 epochs.



Well, this is just too good to be true so contemplated for 
a second about accidentally building the best CNN architecture 
the world has ever seen. 
So, we predicted the classes for the unlabeled test set 
using this model.

Oh, well. There was no serendipity after all. 
So, we looked deeper into what could have gone wrong and 
we found that our training data has multiple images of 
the same person within a class with slight changes of angle 
and/or shifts in height or width. 
This was causing a data leakage problem as the similar images were in validation as well,
i.e. the model was trained much of the same information that it was trying to predict.



### Solution to Data Leakage

To counter the issue of data leakage, we split the images based on the person IDs instead of using a random 80–20 split.

Now, we see more realistic results when we fit our model with the modified training and validation sets. We achieved a loss of 1.76 and an accuracy of 38.5%.



To improve the results further, we explored using the tried and tested deep neural nets architectures.

**Transfer Learning**

Transfer learning is a method where a model developed for a related task is reused as the starting point for a model on a second task. We can re-use the model weights from pre-trained models that were developed for standard computer vision benchmark datasets, such as the ImageNet image recognition challenge. Generally, the final layer with softmax activation is replaced to suit the number of classes in our dataset. In most cases, extra layers are also added to tailor the solution to the specific task.

It is a popular approach in deep learning considering the vast compute and time resources required to develop neural network models for image classification. Moreover, these models are usually trained on millions of images which helps especially when your training set is small. Most of these model architectures are proven winners — VGG16, RESNET50, Xception and Mobilenet models that we leveraged gave exceptional results on the ImageNet challenge.

**Image Augmentation**

Since our training image set had only ~22K images, we wanted to synthetically get more images from the training set to make sure the models don’t overfit as the neural nets have millions of parameters. Image Augmentation is a technique that creates more images from the original by performing actions such as shifting width and/or height, rotation, and zoom. Refer to this [article](https://machinelearningmastery.com/how-to-configure-image-data-augmentation-when-training-deep-learning-neural-networks/) to know more about Image Augmentation.

For our project, Image Augmentation had a few additional advantages. Sometimes, the difference between images from the two different classes can be very subtle. In such cases, getting multiple looks at the same image through different angles will help. If you look at the images below, we see that they are almost similar but in the first picture the class is ‘Talking on the Phone — Right’ and the second picture belongs to the ‘Hair and Makeup’ class.



**Which Optimizer to Use?**

Optimizers minimize an objective function parameterized by a model’s parameters by updating the parameters in the opposite direction of the gradient of the objective function w.r.t. to the parameters. To know more about how different optimizers work, you can refer to [this blog](https://ruder.io/optimizing-gradient-descent/index.html#adam).

The most popular algorithm in the deep learning world is Adam which combines SGD and RMS Prop. It has been consistently performing better than other optimizers for *most* problems. However, in our case, Adam showed erratic pattern of descent while SGD was learning gradually. By doing some literature survey, we found that in few cases SGD is superior to Adam because SGD generalizes better ([link](https://arxiv.org/abs/1705.08292)). As SGD was giving stable results, we used it for all our models.

**Which Architectures to Use?**

We tried multiple transfer learning models with the weights from training on the ImageNet dataset i.e. pre-trained weights.

- [**Xception**](https://arxiv.org/pdf/1610.02357.pdf)**
  **While RESNET was created with the intention of getting ***deeper\*** networks, Xception was created for getting ***wider\*** networks by introducing ***depthwise separable convolutions\***. By decomposing a standard convolution layer into depthwise and pointwise convolutions, the number of computations reduces significantly. The performance of the model also improves because of having multiple filters looking at the same level.