# MNIST Handwritten Digit Recognizer
The [MNIST](https://en.wikipedia.org/wiki/MNIST_database) dataset is a set of 70,000 human labeled 28 x 28 greyscale images of handwritten digits. It is a subset of a larger dataset available from NIST - The National Institute of Standards and Technology. In this tutorial, you'll create your own handwritten digit recognizer using a multi layer neural network trained on the MNIST dataset in Rubix ML.

- **Difficulty**: Hard
- **Training time**: < 3 Hours
- **Memory needed**: < 4G

## Installation

Clone the project locally with [Git](https://git-scm.com/):
```sh
$ git clone https://github.com/RubixML/MNIST
```

Install project dependencies with [Composer](http://getcomposer.com):
```sh
$ composer install
```

## Requirements
- [PHP](https://php.net) 7.1.3 or above

## Tutorial
Through the discovery of deep learning, computers are able to build and compose representations of the world through raw data. To train a computer program to see an image and to recognize what it is, is truly an amazing accomplishment. In this tutorial, we'll use Rubix ML to train a deep learning model known as a [Multi Layer Perceptron](https://github.com/RubixML/RubixML#multi-layer-perceptron) to distinguish the numbers in handwritten digits. Along the way, you'll learn about higher order feature representations and how to build a neural network architecture to achieve a classification accuracy of over 99%.

Deep Learning involves subsequent layers of computation that break down the feature space into what are called *higher order representations*. For the MNIST problem, a classifier will need to be able to learn the lines, edges, corners, and combinations thereof in order to distinguish numbers from the images. In the figure below, we see a snapshot of the features at one of the hidden layers of a neural network trained on the MNIST dataset. The idea is that at each layer, the learner builds more detailed depictions of the training data until the digits are easily distinguishable by a [SoftMax](https://github.com/RubixML/RubixML#softmax) output layer.

![MNIST Deep Learning](https://github.com/RubixML/MNIST/blob/master/docs/images/mnist-deep-learning.png?raw=true)

### Training
The MNIST dataset comes to us in the form of 60,000 training, and 10,000 testing images organized into folders where the folder name is the label given to the sample by a human. We'll use the `imagecreatefrompng()` function from the [GD library](https://www.php.net/manual/en/book.image.php) to load the images into PHP as resources. Then we'll instantiate a new [Labeled](https://github.com/RubixML/RubixML#labeled) dataset object with the samples and labels from the training set.

> Source code can be found in the [train.php](https://github.com/RubixML/MNIST/blob/master/train.php) file in project root.

```php
use Rubix\ML\Datasets\Labeled;

$samples = $labels = [];

for ($label = 0; $label < 10; $label++) {
    foreach (glob(__DIR__ . "/training/$label/*.png") as $file) {
        $samples[] = [imagecreatefrompng($file)];
        $labels[] = $label;
    }
}

$dataset = new Labeled($samples, $labels);
```

Next we'll instantiate the neural network learner and wrap it in a transformer [Pipeline](https://github.com/RubixML/RubixML#pipeline) that will resize, vectorize, and center the image samples automatically for us. We'll start by considering a neural network hidden layer architecture suited for the MNIST problem which consists of 3 layers of [Dense](https://github.com/RubixML/RubixML#dense) neurons, followed by a [Leaky ReLU](https://github.com/RubixML/RubixML#leaky-relu) activation function, and then a mild [Dropout](https://github.com/RubixML/RubixML#dropout) to improve the network's generalization ability. The [AdaMax](https://github.com/RubixML/RubixML#adamax) optimizer is a Gradient Descent optimizer based on the Adam algorithm that we use to update the weights of the network. We've found that this architecture and learning rate works quite well for this problem but feel free to experiment on your own with different architectures and hyperparameters.

```php
use Rubix\ML\Pipeline;
use Rubix\ML\PersistentModel;
use Rubix\ML\Persisters\Filesystem;
use Rubix\ML\NeuralNet\Layers\Dense;
use Rubix\ML\NeuralNet\Layers\Dropout;
use Rubix\ML\Transformers\ImageResizer;
use Rubix\ML\NeuralNet\Layers\Activation;
use Rubix\ML\NeuralNet\Optimizers\AdaMax;
use Rubix\ML\Transformers\ImageVectorizer;
use Rubix\ML\Transformers\ZScaleStandardizer;
use Rubix\ML\Classifiers\MultiLayerPerceptron;
use Rubix\ML\NeuralNet\ActivationFunctions\LeakyReLU;

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
    ], 100, new AdaMax(0.001))),
    new Filesystem('mnist.model', true)
);
```

Lastly, to save our model we'll wrap the entire pipeline in a Persistent Model meta-estimator so we can call the `save()` method on it after training. To start training, pass the training dataset into the estimator's `train()` method.

```php
$estimator->train($dataset);
```

We can visualize the training progress at each stage by dumping the values of the loss function and validation metric to load into plotting software. In this case, the `steps()` method will output an array containing the values of the default [Cross Entropy](https://github.com/RubixML/RubixML#cross-entropy) cost function and the `scores()` will return an array of scores from the default [FBeta](https://github.com/RubixML/RubixML#f-beta) validation metric.

```php
$steps = $estimator->steps();

$scores = $estimator->scores();
```

Here is what a typical training run looks like in terms of the Cross Entropy cost function.

![Cross Entropy Loss](https://raw.githubusercontent.com/RubixML/MNIST/master/docs/images/cross-entropy-loss.svg?sanitize=true)

And here is the F Beta validation score at each epoch.

![F Beta Score](https://raw.githubusercontent.com/RubixML/MNIST/master/docs/images/f-beta-score.svg?sanitize=true)

Finally, we save the trained network by calling the `save()` method provided by the [Persistent Model](https://github.com/RubixML/RubixML#persistent-model) wrapper.

```php
$estimator->save();
```

### Validation

Coming soon ...

## Original Dataset
Yann LeCun, Professor
The Courant Institute of Mathematical Sciences
New York University
Email: yann 'at' cs.nyu.edu 

Corinna Cortes, Research Scientist
Google Labs, New York
Email: corinna 'at' google.com

### References
>- [1] Y. LeCun et al. (1998). Gradient-based learning applied to document recognition.