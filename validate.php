<?php

include __DIR__ . '/vendor/autoload.php';

use Rubix\ML\Datasets\Labeled;
use Rubix\ML\PersistentModel;
use Rubix\ML\Persisters\Filesystem;
use Rubix\ML\CrossValidation\Reports\AggregateReport;
use Rubix\ML\CrossValidation\Reports\ConfusionMatrix;
use Rubix\ML\CrossValidation\Reports\MulticlassBreakdown;

ini_set('memory_limit', '-1');

echo 'Loading data into memory ...' . PHP_EOL;

$samples = $labels = [];

for ($label = 0; $label < 10; $label++) {
    foreach (glob("testing/$label/*.png") as $file) {
        $samples[] = [imagecreatefrompng($file)];
        $labels[] = "$label";
    }
}

$dataset = new Labeled($samples, $labels);

$estimator = PersistentModel::load(new Filesystem('mnist.model'));

echo 'Making predictions ...' . PHP_EOL;

$predictions = $estimator->predict($dataset);

$report = new AggregateReport([
    'breakdown' => new MulticlassBreakdown(),
    'matrix' => new ConfusionMatrix(),
]);

$results = $report->generate($predictions, $dataset->labels());

echo $results;

$results->toJSON()->write('report.json');

echo 'Report saved to report.json' . PHP_EOL;
