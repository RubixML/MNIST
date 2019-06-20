<?php

include __DIR__ . '/vendor/autoload.php';

use Rubix\ML\PersistentModel;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Persisters\Filesystem;
use Rubix\ML\CrossValidation\Reports\AggregateReport;
use Rubix\ML\CrossValidation\Reports\ConfusionMatrix;
use Rubix\ML\CrossValidation\Reports\MulticlassBreakdown;

const MODEL_FILE = 'mnist.model';
const REPORT_FILE = 'report.json';

ini_set('memory_limit', '-1');

echo '╔═════════════════════════════════════════════════════╗' . PHP_EOL;
echo '║                                                     ║' . PHP_EOL;
echo '║ MNIST Handwritten Digit Recognizer                  ║' . PHP_EOL;
echo '║                                                     ║' . PHP_EOL;
echo '╚═════════════════════════════════════════════════════╝' . PHP_EOL;
echo PHP_EOL;

echo 'Loading data into memory ...' . PHP_EOL;

$samples = $labels = [];

for ($label = 0; $label < 10; $label++) {
    foreach (glob(__DIR__ . "/testing/$label/*.png") as $file) {
        $samples[] = [imagecreatefrompng($file)];
        $labels[] = $label;
    }
}

$dataset = new Labeled($samples, $labels);

$estimator = PersistentModel::load(new Filesystem(MODEL_FILE));

$predictions = $estimator->predict($dataset);

$report = new AggregateReport([
    new MulticlassBreakdown(),
    new ConfusionMatrix(),
]);

$results = $report->generate($predictions, $dataset->labels());

file_put_contents(REPORT_FILE, json_encode($results, JSON_PRETTY_PRINT));

echo 'Report saved to ' . REPORT_FILE . PHP_EOL;
