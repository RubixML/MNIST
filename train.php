<?php

include __DIR__ . '/vendor/autoload.php';

use Rubix\ML\Pipeline;
use Rubix\ML\PersistentModel;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Other\Loggers\Screen;
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
use League\Csv\Writer;

const MODEL_FILE = 'mnist.model';
const PROGRESS_FILE = 'progress.csv';

echo '╔═════════════════════════════════════════════════════╗' . PHP_EOL;
echo '║                                                     ║' . PHP_EOL;
echo '║ MNIST Handwritten Digit Recognizer                  ║' . PHP_EOL;
echo '║                                                     ║' . PHP_EOL;
echo '╚═════════════════════════════════════════════════════╝' . PHP_EOL;
echo PHP_EOL;

echo 'Loading data into memory ...' . PHP_EOL;

$samples = $labels = [];

for ($label = 0; $label < 10; $label++) {
    foreach (glob(__DIR__ . "/training/$label/*.png") as $file) {
        $samples[] = [imagecreatefrompng($file)];
        $labels[] = $label;
    }
}

$dataset = new Labeled($samples, $labels);

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
    new Filesystem(MODEL_FILE, true)
);

$estimator->setLogger(new Screen('MNIST'));

$estimator->train($dataset);

$writer = Writer::createFromPath(PROGRESS_FILE, 'w+');
$writer->insertOne(['score', 'loss']);
$writer->insertAll(array_map(null, $estimator->scores(), $estimator->steps()));

if (strtolower(trim(readline('Save this model? (y|[n]): '))) === 'y') {
    $estimator->save();
}
