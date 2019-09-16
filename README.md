# Rubix ML - MNIST Handwritten Digit Recognizer
The [MNIST](https://en.wikipedia.org/wiki/MNIST_database) dataset is a set of 70,000 human labeled 28 x 28 greyscale images of individual handwritten digits. It is a subset of a larger dataset available from NIST - The National Institute of Standards and Technology. In this tutorial, you'll create your own handwritten digit recognizer using a multi layer neural network trained on the MNIST dataset in Rubix ML.

- **Difficulty:** Hard
- **Training time:** Hours
- **Memory needed:** 3G

## Installation
Clone the project locally with [Git](https://git-scm.com/):
```sh
$ git clone https://github.com/RubixML/MNIST
```

Install project dependencies with [Composer](https://getcomposer.org/):
```sh
$ composer install
```

## Requirements
- [PHP](https://php.net) 7.1.3 or above

## Tutorial

### Introduction
Through deep learning, computers are able to build and compose higher order representations of the world given raw data. To train a computer program to see and distinguish objects in images, is truly an amazing accomplishment. In this tutorial, we'll use Rubix ML to train a deep learning model known as a [Multi Layer Perceptron](https://docs.rubixml.com/en/latest/classifiers/multi-layer-perceptron.html) to recognize the numbers in handwritten digits. Along the way, you'll learn about higher order feature representations and neural network architectures.

Deep Learning employs subsequent layers of computation that break down the feature space into what are known as *higher order* representations. For the MNIST problem, a classifier will need to be able to learn the lines, edges, corners, and combinations thereof in order to distinguish the numbers in the images. In the figure below, we see a snapshot of the features at one of the hidden layers of a neural network trained on the MNIST dataset. The idea is that at each layer, the learner builds more detailed depictions of the training data until the digits are easily distinguishable by a [Softmax](https://docs.rubixml.com/en/latest/neural-network/activation-functions/softmax.html) output layer.

![MNIST Deep Learning](https://github.com/RubixML/MNIST/blob/master/docs/images/mnist-deep-learning.png?raw=true)

> **Note:** The source code for this example can be found in the [train.php](https://github.com/RubixML/MNIST/blob/master/train.php) file in project root.

### Extracting the Data
The MNIST dataset comes to us in the form of 60,000 training and 10,000 testing images organized into folders where the folder name is the human annotated label. We'll use the `imagecreatefrompng()` function from the [GD library](https://www.php.net/manual/en/book.image.php) to load the images into PHP as resources and assign a label based on the folder.

```php
$samples = $labels = [];

for ($label = 0; $label < 10; $label++) {
    foreach (glob(__DIR__ . "/training/$label/*.png") as $file) {
        $samples[] = [imagecreatefrompng($file)];
        $labels[] = $label;
    }
}
```

Then, instantiate a new [Labeled](https://docs.rubixml.com/en/latest/datasets/labeled.html) datset object from the samples and labels.

```php
use Rubix\ML\Datasets\Labeled;

$dataset = new Labeled($samples, $labels);
```

### Dataset Preparation
We're going to use a transformer [Pipeline](https://docs.rubixml.com/en/latest/pipeline.html) to shape the dataset into the correct format for our learner automatically. We know that the size of each sample image in the MNIST dataset is 28 x 28 pixels, however, to make sure that future samples are always the correct size we'll throw in an [Image Resizer](https://docs.rubixml.com/en/latest/transformers/image-resizer.html) just in case. Then, to convert the image into raw pixel data we'll use the [Image Vectorizer](https://docs.rubixml.com/en/latest/transformers/image-vectorizer.html) which extracts the raw color channel data from the image. Since the sample images are black and white, we only need to use 1 color channel per pixel. At the end of the pipeline we'll center and scale the dataset using the [Z Scale Standardizer](https://docs.rubixml.com/en/latest/transformers/z-scale-standardizer.html) to help speed up convergence of the network.

### Instantiating the Learner
Let's consider a neural network architecture suited for the MNIST problem consisting of 3 groups of [Dense](https://docs.rubixml.com/en/latest/neural-network/hidden-layers/dense.html) neurons, followed by a [Leaky ReLU](https://docs.rubixml.com/en/latest/neural-network/activation-functions/leaky-relu.html) activation layer, and then a mild [Dropout](https://docs.rubixml.com/en/latest/neural-network/hidden-layers/dropout.html) layer to act as a regularizer. In theory, the third layer will form representations from the second and from the first by proxy and each subsequent layer becomes a more complex feature detector. The output layer adds an additional layer of neurons with a [Softmax](https://docs.rubixml.com/en/latest/neural-network/activation-functions/softmax.html) activation making this particular network archtecture 4 layers deep.

Next, we'll set the batch size to 200 - meaning that 200 random samples from the training set will be sent through the network at a time during training. Lastly, the [Adam](https://docs.rubixml.com/en/latest/neural-network/optimizers/adam.html) optimizer determines the update step of the Gradient Descent algorithm. Adam stands for Adaptive Moment Estimation and uses a combination of [Momentum](https://docs.rubixml.com/en/latest/neural-network/optimizers/momentum.html) and [RMS Prop](https://docs.rubixml.com/en/latest/neural-network/optimizers/rms-prop.html) to make its updates. It uses a global *learning rate* to control the size of the step which we'll set to 0.001 for this example.

```php
use Rubix\ML\Pipeline;
use Rubix\ML\PersistentModel;
use Rubix\ML\Transformers\ImageResizer;
use Rubix\ML\Transformers\ImageVectorizer;
use Rubix\ML\Transformers\ZScaleStandardizer;
use Rubix\ML\Classifiers\MultiLayerPerceptron;
use Rubix\ML\NeuralNet\Layers\Dense;
use Rubix\ML\NeuralNet\Layers\Dropout;
use Rubix\ML\NeuralNet\Layers\Activation;
use Rubix\ML\NeuralNet\ActivationFunctions\LeakyReLU;
use Rubix\ML\NeuralNet\Optimizers\Adam;
use Rubix\ML\Persisters\Filesystem;

$estimator = new PersistentModel(
    new Pipeline([
        new ImageResizer(28, 28),
        new ImageVectorizer(1),
        new ZScaleStandardizer(),
    ], new MultiLayerPerceptron([
        new Dense(100),
        new Activation(new LeakyReLU()),
        new Dropout(0.2),
        new Dense(100),
        new Activation(new LeakyReLU()),
        new Dropout(0.2),
        new Dense(100),
        new Activation(new LeakyReLU()),
        new Dropout(0.2),
    ], 200, new Adam(0.001))),
    new Filesystem('mnist.model', true)
);
```

To allow us to save and load the model from storage, we'll wrap the entire pipeline in a [Persistent Model](https://docs.rubixml.com/en/latest/persistent-model.html) meta-estimator. Persistent Model provides `save()` and `load()` methods on top of the base estimator's methods. It needs a Persister object to tell it where the model is be stored. For our purposes, we'll use the [Filesystem](https://docs.rubixml.com/en/latest/persisters/filesystem.html) persister which takes a path to the file where the serialized model data is stored on disk. Setting history to true means that the persister will keep track of every save.

### Training
To start training the neural network, call the `train()` method with the training set as input.
```php
$estimator->train($dataset);
```

### Validation Score and Loss
We can visualize the training progress at each stage by dumping the values of the loss function and validation metric after training. The `steps()` method will output an array containing the values of the default [Cross Entropy](https://docs.rubixml.com/en/latest/neural-network/cost-functions/cross-entropy.html) cost function and the `scores()` method will return an array of scores from the default [FBeta](https://docs.rubixml.com/en/latest/cross-validation/metrics/f-beta.html) validation metric.

```php
$steps = $estimator->steps();

$scores = $estimator->scores();
```

Then we can plot the values using our favorite plotting software. If all goes well, the value of the loss should go down as the value of the validation score goes up.

![Cross Entropy Loss](https://raw.githubusercontent.com/RubixML/MNIST/master/docs/images/training-loss.svg?sanitize=true)

![F1 Score](https://raw.githubusercontent.com/RubixML/MNIST/master/docs/images/validation-score.svg?sanitize=true)

### Saving
Lastly, we can save the trained network by calling the `save()` method provided by the [Persistent Model](https://docs.rubixml.com/en/latest/persistent-model.html) wrapper. The model will be saved in a compact serialized format such as the [Native](https://docs.rubixml.com/en/latest/persisters/serializers/native.html) PHP serialization format.

```php
$estimator->save();
```

### Cross Validation
On the map ...


## Original Dataset
Yann LeCun, Professor
The Courant Institute of Mathematical Sciences
New York University
Email: yann 'at' cs.nyu.edu 

Corinna Cortes, Research Scientist
Google Labs, New York
Email: corinna 'at' google.com

### References
>- Y. LeCun et al. (1998). Gradient-based learning applied to document recognition.