<?php

include __DIR__ . '/vendor/autoload.php';

use Rubix\ML\Loggers\Screen;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\PersistentModel;
use Rubix\ML\Transformers\PersistentTransformer;
use Rubix\ML\Transformers\Pipeline;
use Rubix\ML\Transformers\ImageResizer;
use Rubix\ML\Transformers\ImageVectorizer;
use Rubix\ML\Transformers\ZScaleStandardizer;
use Rubix\ML\Transformers\FloatTypeConverter;
use Rubix\ML\Classifiers\MultilayerPerceptron;
use Rubix\ML\NeuralNet\Layers\Dense;
use Rubix\ML\NeuralNet\Layers\Dropout;
use Rubix\ML\NeuralNet\Layers\Activation;
use Rubix\ML\NeuralNet\Layers\BatchNorm;
use Rubix\ML\NeuralNet\ActivationFunctions\GELU;
use Rubix\ML\NeuralNet\Optimizers\Schedulers\Constant;
use Rubix\ML\NeuralNet\Optimizers\Adam;
use Rubix\ML\Persisters\Filesystem;
use Rubix\ML\Extractors\CSV;

ini_set('memory_limit', '-1');

$logger = new Screen();

$transformer = new PersistentTransformer(
    base: new Pipeline([
        new ImageResizer(28, 28),
        new ImageVectorizer(grayscale: true),
        new FloatTypeConverter(),
        new ZScaleStandardizer(),
    ]),
    persister: new Filesystem('transformer.rbx', true)
);

$estimator = new PersistentModel(
    base: new MultilayerPerceptron(
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
        window: 5,
    ),
    persister: new Filesystem('model.rbx', true)
);

$estimator->setLogger($logger);

$logger->info('Loading data into memory');

$datasets = [];

foreach (['training', 'testing'] as $dir) {
    $samples = $labels = [];

    for ($label = 0; $label < 10; $label++) {
        foreach (glob("$dir/$label/*.png") as $file) {
            $samples[] = [imagecreatefrompng($file)];
            $labels[] = "#$label";
        }
    }

    $datasets[] = new Labeled($samples, $labels);
}

[$training, $testing] = $datasets;

$transformer->fit($training);

$logger->info('Preprocessing dataset');

$training->apply($transformer);
$testing->apply($transformer);

$estimator->setValidationDataset($testing);

$estimator->train($training);

$extractor = new CSV('progress.csv', true);

$extractor->export($estimator->progress(), overwrite: true);

$logger->info('Progress saved to progress.csv');

if (strtolower(trim(readline('Save this model? (y|[n]): '))) === 'y') {
    $transformer->save();
    $estimator->save();
}
