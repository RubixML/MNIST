# MNIST Handwritten Digit Recognizer

The [MNIST](https://en.wikipedia.org/wiki/MNIST_database) dataset is a set of 70,000 human-labeled 28 x 28 greyscale images of individual handwritten digits. It is a subset of a larger dataset available from NIST - The National Institute of Standards and Technology. In this tutorial, you'll create your own handwritten digit recognizer using a multilayer neural network trained on the MNIST dataset.

## Requirements

- [PHP](https://php.net) 8.3 or above
- [Tensor extension 4.0+](https://github.com/RubixML/Tensor) for faster training and inference
- [GD extension](https://www.php.net/manual/en/book.image.php)

## Installation

Clone the project locally using [Composer](https://getcomposer.org/):

```sh
$ composer create-project rubix/mnist
```

> **Note:** Installation may take longer than usual due to the large dataset.

Then, install the Tensor Ext 4.x and GD extensions if they have not been installed yet. You can install the Tensor Ext extension using [PIE](https://github.com/php/pie) like in the example below:

```sh
pie install rubix/tensor_ext
```

## Tutorial

### Introduction

In this tutorial, we'll use Rubix ML to train a deep learning model called a Multilayer Perceptron to recognize the numbers in handwritten digits. For this problem, a classifier will need to be able to learn lines, edges, corners, and a combinations thereof in order to distinguish the numbers in the images. In the figure below, we see a snapshot of the features at one layer of a neural network trained on the MNIST dataset. The illustration shows that at each layer, the network builds a more detailed depiction of the training data until the digits are distinguishable by a [Softmax](https://rubixml.github.io/ML/3.0/classifiers/multilayer-perceptron.html) layer at the output.

![MNIST Deep Learning](https://github.com/RubixML/MNIST/blob/master/docs/images/mnist-deep-learning.png?raw=true)

> **Note:** The source code for this example can be found in the [train.php](https://github.com/RubixML/MNIST/blob/master/train.php) file in project root.

### Extracting the Data

The MNIST dataset comes to us in the form of 60,000 training and 10,000 testing images organized into subfolders where the folder name is the human-annotated label given to the sample. We'll use the `imagecreatefrompng()` function from the [GD library](https://www.php.net/manual/en/book.image.php) to load the images into our script and assign them a label based on the subfolder they are in.

```php
$samples = $labels = [];

for ($label = 0; $label < 10; $label++) {
    foreach (glob("training/$label/*.png") as $file) {
        $samples[] = [imagecreatefrompng($file)];
        $labels[] = "#$label";
    }
}
```

Then, we can instantiate a new [Labeled](https://rubixml.github.io/ML/3.0/datasets/labeled.html) dataset object from the samples and labels using the standard constructor.

```php
use Rubix\ML\Datasets\Labeled;

$dataset = new Labeled($samples, $labels);
```

### Dataset Preparation

We're going to use a transformer [Pipeline](https://rubixml.github.io/ML/3.0/pipeline.html) to shape the dataset into the correct format for our learner. We know that the size of each sample image in the MNIST dataset is 28 x 28 pixels, but just to make sure that future samples are always the correct input size we'll add an [Image Resizer](https://rubixml.github.io/ML/3.0/transformers/image-resizer.html). Then, to convert the image into raw pixel data we'll use the [Image Vectorizer](https://rubixml.github.io/ML/3.0/transformers/image-vectorizer.html) which extracts continuous raw color channel values from the image. Since the sample images are black and white, we only need to use 1 color channel per pixel. Rubix ML 3.0 uses a high-level type system where features are typed as either categorical or continuous, so we'll run the vectorized output through the [Float Type Converter](https://rubixml.github.io/ML/3.0/transformers/float-type-converter.html) to make sure the pixel values are interpreted as continuous float features. At the end of the pipeline we'll center and scale the dataset using the [Z Scale Standardizer](https://rubixml.github.io/ML/3.0/transformers/z-scale-standardizer.html) to help speed up the convergence of the neural network.

```php
use Rubix\ML\Pipeline;
use Rubix\ML\Transformers\ImageResizer;
use Rubix\ML\Transformers\ImageVectorizer;
use Rubix\ML\Transformers\FloatTypeConverter;
use Rubix\ML\Transformers\ZScaleStandardizer;

$transformers = [
    new ImageResizer(28, 28),
    new ImageVectorizer(true),
    new FloatTypeConverter(),
    new ZScaleStandardizer(),
];
```

### Instantiating the Learner

Now, we'll go ahead and instantiate our [Multilayer Perceptron](https://rubixml.github.io/ML/3.0/classifiers/multilayer-perceptron.html) classifier. Let's consider a neural network architecture suited for the MNIST problem consisting of 4 groups of [Dense](https://rubixml.github.io/ML/3.0/neural-network/hidden-layers/dense.html) neuronal layers of 256 neurons each, each followed by a [GELU](https://rubixml.github.io/ML/3.0/neural-network/activation-functions/gelu.html) activation layer and a mild [Dropout](https://rubixml.github.io/ML/3.0/neural-network/hidden-layers/dropout.html) layer set to 0.1 to act as a regularizer. GELU is a smooth, non-monotonic activation function that has been shown to train faster and generalize better than the classic [ReLU](https://rubixml.github.io/ML/3.0/neural-network/activation-functions/relu.html). The third group disables the bias term of its Dense layer and is followed by a [Batch Norm](https://rubixml.github.io/ML/3.0/neural-network/hidden-layers/batch-norm.html) layer which normalizes the activations of the previous layer such that the mean activation is close to 0 and the standard deviation is close to 1. Batch Norm reduces the amount of covariate shift within the network, which makes it possible to converge faster under some circumstances. The output layer adds an additional layer of neurons with a [Softmax](https://rubixml.github.io/ML/3.0/classifiers/multilayer-perceptron.html) activation making this particular network architecture 6 layers deep.

Next, we'll set the batch size to 32. The batch size is the number of samples sent through the network at a time. A smaller batch keeps memory usage in check but introduces more noise into each gradient estimate. To compensate, we'll set the gradient accumulation steps to 4, which waits until the gradients across 4 batches have been accumulated before applying a single update - giving us an effective batch size of 128 while only ever keeping 32 samples in memory at once.

We'll also specify an optimizer and a learning rate scheduler which determines the update step of the Gradient Descent algorithm. The [Adam](https://rubixml.github.io/ML/3.0/neural-network/optimizers/adam.html) optimizer uses a combination of [Momentum](https://rubixml.github.io/ML/3.0/neural-network/optimizers/momentum.html) and [RMS Prop](https://rubixml.github.io/ML/3.0/neural-network/optimizers/rms-prop.html) to make its updates and usually converges faster than standard *stochastic* Gradient Descent. In Rubix ML 3.0 the optimizer takes a [Scheduler](https://rubixml.github.io/ML/3.0/neural-network/schedulers/constant.html) object which dictates the learning rate over the course of training. We'll use the simple [Constant](https://rubixml.github.io/ML/3.0/neural-network/schedulers/constant.html) scheduler which holds the learning rate at 0.0001 for the duration of the training run. To keep the updates numerically stable we'll cap the magnitude of the gradients by setting the maximum gradient norm to 1.0.

Finally, we'll tell the learner to train for a maximum of 100 epochs, to evaluate the validation score on a 10% hold-out set every epoch, and to stop early if the score fails to improve over a window of 10 epochs by at least 1e-5. This keeps the run from overfitting by effectively *unlearning* some of the noise in the dataset.

```php
use Rubix\ML\Loggers\Screen;
use Rubix\ML\PersistentModel;
use Rubix\ML\Pipeline;
use Rubix\ML\Transformers\ImageResizer;
use Rubix\ML\Transformers\ImageVectorizer;
use Rubix\ML\Transformers\FloatTypeConverter;
use Rubix\ML\Transformers\ZScaleStandardizer;
use Rubix\ML\Classifiers\MultilayerPerceptron;
use Rubix\ML\NeuralNet\Layers\Dense;
use Rubix\ML\NeuralNet\Layers\Dropout;
use Rubix\ML\NeuralNet\Layers\Activation;
use Rubix\ML\NeuralNet\Layers\BatchNorm;
use Rubix\ML\NeuralNet\ActivationFunctions\GELU;
use Rubix\ML\NeuralNet\Optimizers\Schedulers\Constant;
use Rubix\ML\NeuralNet\Optimizers\Adam;
use Rubix\ML\Persisters\Filesystem;

$logger = new Screen();

$estimator = new PersistentModel(
    new Pipeline([
        new ImageResizer(28, 28),
        new ImageVectorizer(grayscale: true),
        new FloatTypeConverter(),
        new ZScaleStandardizer(),
    ], new MultilayerPerceptron(
        hiddenLayers: [
            new Dense(256),
            new Activation(new GELU()),
            new Dropout(0.1),
            new Dense(256),
            new Activation(new GELU()),
            new Dropout(0.1),
            new Dense(256, bias: false),
            new BatchNorm(),
            new Activation(new GELU()),
            new Dropout(0.1),
            new Dense(256),
            new Activation(new GELU()),
            new Dropout(0.1),
            new Dense(10),
        ],
        batchSize: 32,
        gradientAccumulationSteps: 4,
        optimizer: new Adam(new Constant(0.0001)),
        maxGradientNorm: 1.0,
        epochs: 100,
        minChange: 1e-5,
        evalInterval: 1,
        window: 10,
        holdOut: 0.1
    )),
    new Filesystem('mnist.rbx', true)
);

$estimator->setLogger($logger);
```

To allow us to save and load the model from storage, we'll wrap the entire pipeline in a [Persistent Model](https://rubixml.github.io/ML/3.0/persistent-model.html) meta-estimator. Persistent Model provides additional `save()` and `load()` methods on top of the base estimator's methods. It needs a Persister object to tell it where the model is to be stored. For our purposes, we'll use the [Filesystem](https://rubixml.github.io/ML/3.0/persisters/filesystem.html) persister which takes a path to the model file on disk. Setting history mode to true means that the persister will keep track of every past save.

### Training

To start training the neural network, call the `train()` method on the Estimator instance with the training set as an argument.

```php
$estimator->train($dataset);
```

### Validation Score and Loss

We can visualize the training progress at each stage by dumping the values of the loss function and validation metric after training. The `progress()` method will output an iterator that yields one row per epoch containing the epoch number, the value of the default [Multiclass Cross Entropy](https://rubixml.github.io/ML/3.0/neural-network/cost-functions/multiclass-cross-entropy.html) cost function, the gradient norm, and the score of the [F Beta](https://rubixml.github.io/ML/3.0/cross-validation/metrics/f-beta.html) metric. If you'd prefer to work with the validation metric on its own, the `scores()` method will return a flat array of just the F Beta scores from the last training session.

> **Note:** You can change the cost function and validation metric by setting them as hyper-parameters of the learner.

```php
use Rubix\ML\Extractors\CSV;

$extractor = new CSV('progress.csv', true);

$extractor->export($estimator->progress());
```

Then, we can plot the values using our favorite plotting software such as [Tableu](https://public.tableau.com/en-us/s/) or [Excel](https://products.office.com/en-us/excel-a). If all goes well, the value of the loss should go down as the value of the validation score goes up. Due to snapshotting, the epoch at which the validation score is highest and the loss is lowest is the point at which the values of the network parameters are taken for the final model. This prevents the network from overfitting the training data by effectively *unlearning* some of the noise in the dataset.

![Multiclass Cross Entropy Loss](https://raw.githubusercontent.com/RubixML/MNIST/master/docs/images/training-losses.png)

![F1 Score](https://raw.githubusercontent.com/RubixML/MNIST/master/docs/images/validation-scores.png)

### Saving

We can save the trained network by calling the `save()` method provided by the [Persistent Model](https://rubixml.github.io/ML/3.0/persistent-model.html) wrapper. The model will be serialized using the [RBX](https://rubixml.github.io/ML/3.0/serializers/rbx.html) serializer, a compact binary format that is the default choice in Rubix ML 3.0. Alternatives like [Native](https://rubixml.github.io/ML/3.0/serializers/native.html) PHP serialization also exist if you prefer a more human-readable or language-agnostic encoding.

```php
$estimator->save();
```

Since training a network on the full MNIST dataset can take hours, we don't want to commit a model to disk on every run. Instead, the script will ask you interactively whether you'd like to keep the model before it saves, defaulting to no.

Now we're ready to execute the training script from the command line.

### Cross Validation

Cross Validation is a technique for assessing how well the Estimator can generalize its training to an independent dataset. The goal is to identify problems such as underfitting, overfitting, or selection bias that would cause the model to perform poorly on new unseen data.

Fortunately, the MNIST dataset includes an extra 10,000 labeled images that we can use to test the model. Since we haven't used any of these samples to train the network with, we can use them to test the generalization performance of the model. To start, we'll extract the testing samples and labels from the `testing` folder into a [Labeled](https://rubixml.github.io/ML/3.0/datasets/labeled.html) dataset object.

```php
use Rubix\ML\Datasets\Labeled;

$samples = $labels = [];

for ($label = 0; $label < 10; $label++) {
    foreach (glob("testing/$label/*.png") as $file) {
        $samples[] = [imagecreatefrompng($file)];
        $labels[] = "#$label";
    }
}

$dataset = new Labeled($samples, $labels);
```

### Load Model from Storage

In our training script we made sure to save the model before we exited. In our validation script, we'll load the trained model from storage and use it to make predictions on the testing set. The static `load()` method on [Persistent Model](https://rubixml.github.io/ML/3.0/persistent-model.html) takes a [Persister](https://rubixml.github.io/ML/3.0/persisters/api.html) object pointing to the model in storage as its only argument and returns the loaded estimator instance.

Once the model is loaded, we'll call `cleanup()` on it. After training, the optimizer keeps around a set of internal state buffers (momentum and norm estimates) used to make its updates. Since we're only doing inference now and will never update the weights again, we can ask the optimizer to flush those buffers and release the memory they occupy.

```php
use Rubix\ML\PersistentModel;
use Rubix\ML\Persisters\Filesystem;

$estimator = PersistentModel::load(new Filesystem('mnist.rbx'));

$estimator->cleanup();
```

### Make Predictions

Now we can use the estimator to make predictions on the testing set. The `predict()` method takes a dataset as input and returns an array of predictions.

```php
$predictions = $estimator->predict($dataset);
```

### Generating the Report

The cross validation report we'll generate is actually a combination of two reports - [Multiclass Breakdown](https://rubixml.github.io/ML/3.0/cross-validation/reports/multiclass-breakdown.html) and [Confusion Matrix](https://rubixml.github.io/ML/3.0/cross-validation/reports/confusion-matrix.html). We'll wrap each report in an [Aggregate Report](https://rubixml.github.io/ML/3.0/cross-validation/reports/aggregate-report.html) to generate both reports at once under their own key.

```php
use Rubix\ML\CrossValidation\Reports\AggregateReport;
use Rubix\ML\CrossValidation\Reports\ConfusionMatrix;
use Rubix\ML\CrossValidation\Reports\MulticlassBreakdown;

$report = new AggregateReport([
    'breakdown' => new MulticlassBreakdown(),
    'matrix' => new ConfusionMatrix(),
]);
```

To generate the report, pass in the predictions along with the labels from the testing set to the `generate()` method on the report. We can print the report to the terminal and then encode the results as JSON and save them to `report.json` for later inspection.

```php
$results = $report->generate($predictions, $dataset->labels());

echo $results;

$results->toJSON()->saveTo(new Filesystem('report.json'));
```

Now we're ready to run the validation script from the command line.

```sh
$ php validate.php
```

Below is an excerpt from an example report. As you can see, our model was able to achieve 99.5% accuracy on the testing set.

```json
{
    "breakdown": {
        "overall": {
            "accuracy": 0.9954799999999999,
            "balanced accuracy": 0.9872982617942677,
            "f1 score": 0.9771892383306813,
            "precision": 0.9772820640419069,
            "recall": 0.9771069272854186,
            "specificity": 0.9974895963031166,
            "negative predictive value": 0.9974918048097698,
            "false discovery rate": 0.02271793595809326,
            "miss rate": 0.022893072714581464,
            "fall out": 0.002510403696883279,
            "false omission rate": 0.0025081951902301116,
            "mcc": 0.9746831111939634,
            "informedness": 0.9745965235885354,
            "markedness": 0.9747738688516765,
            "true positives": 9774,
            "true negatives": 89774,
            "false positives": 226,
            "false negatives": 226,
            "cardinality": 10000
        },
        "classes": {
            "#0": {
                "accuracy": 0.9973,
                "balanced accuracy": 0.9925912937236978,
                "f1 score": 0.9862315145334012,
                "precision": 0.9857288481141692,
                "recall": 0.986734693877551,
                "specificity": 0.9984478935698448,
                "negative predictive value": 0.9985585985142477,
                "false discovery rate": 0.014271151885830835,
                "miss rate": 0.013265306122449028,
                "fall out": 0.0015521064301552423,
                "false omission rate": 0.001441401485752336,
                "informedness": 0.9851825874473956,
                "markedness": 0.9842874466284168,
                "mcc": 0.9847349153256292,
                "true positives": 967,
                "true negatives": 9006,
                "false positives": 14,
                "false negatives": 13,
                "cardinality": 980,
                "proportion": 0.098
            },
            "#5": {
                "accuracy": 0.9951,
                "balanced accuracy": 0.9836577413834189,
                "f1 score": 0.9724564362001124,
                "precision": 0.9751972942502819,
                "recall": 0.9697309417040358,
                "specificity": 0.9975845410628019,
                "negative predictive value": 0.99703719960496,
                "false discovery rate": 0.0248027057497181,
                "miss rate": 0.030269058295964157,
                "fall out": 0.0024154589371980784,
                "false omission rate": 0.002962800395040044,
                "informedness": 0.9673154827668378,
                "markedness": 0.9722344938552419,
                "mcc": 0.9697718694549535,
                "true positives": 865,
                "true negatives": 9086,
                "false positives": 22,
                "false negatives": 27,
                "cardinality": 892,
                "proportion": 0.0892
            }
        }
    },
    "matrix": {
        "#0": {
            "#0": 967,
            "#5": 1,
            "#2": 2,
            "#4": 2,
            "#8": 2,
            "#7": 0,
            "#6": 4,
            "#1": 0,
            "#3": 0,
            "#9": 3
        },
        "#5": {
            "#0": 3,
            "#5": 865,
            "#2": 1,
            "#4": 0,
            "#8": 5,
            "#7": 1,
            "#6": 7,
            "#1": 0,
            "#3": 3,
            "#9": 2
        }
    }
}
```

### Next Steps

Congratulations on completing the MNIST tutorial on handwritten digit recognition in Rubix ML. We highly recommend browsing the [documentation](https://rubixml.github.io/ML/) to get a better feel for what the neural network subsystem can do. What other problems would deep learning be suitable for?

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

## License

The code is licensed [MIT](LICENSE) and the tutorial is licensed [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/).
