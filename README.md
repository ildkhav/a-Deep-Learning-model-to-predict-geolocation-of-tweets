# A CNN to predict geolocation provided tweet text

The method employs CNN's ability to detect features spread over data while training. No sophistication in the model's 
architecture, but using pretty standard approach adjusted to the task nature.

The work is done with Keras API.

## Input data format

Data are expected to be csv with `text` column. The csv file names to follow naming convention - 
`<latitude value>_-<longitude value>.csv`. That is a given file consists tweets only of this coordinate.

## Data preprocessing

The text is transformed to Unicode code points, then normalized to [0, 1] range and padded with 0 to have predefined 
length. Geolocation data are normalized to [0, 1] for regression, this proved to be efficient.

## A model's overview

There are two models, they differ in amount of convolutional layers, which are the only layers there (except the final 
Dense layer), but work somewhat the same, when it comes to evaluation (200km mean error and 0.86 accuracy), at least 
on that amount of data and filters I was able to train. The single layer model though goes faster (fewer epochs) 
to its best state, which is expected.

The first model is based on idea that we want to detect location-specific phrases, names etc. at once and go to the exit
right away. It has only one Conv1D layer, kernel_size 15 (max length of the phrase, name) and 1000 filters (amount of 
different phrases, names). Obviously the more filters the better and that will affect training performance.

The second model tries to detect syllables on the first layer (max 3 letters syllables, 1000 distinct syllables), then 
the second layer looks for combinations of syllables (max 2 syllables combinations, 1000 distinct combinations), then 
sort of words (max 2 combinations mix, 1000 distinct mixes/words), then finally phrases, names (that would consist of 
max 2 words, 1000 distinct), which could be mapped to locations.

Pooling layers were not used, because that would mangle phrases, mixes to be detected (MHO). Characters' values are not 
the same as pixels' values, which can be pooled based on maximum, mean etc.

ReLU activation is applied, it proved to be more efficient than linear with coordinates having negative values.

## Regression & classification

Both methods have been tried. Regression method tries to predict exact coordinate. Classification method yields a score 
among classes, amount of which depends on needed precision. That is 1 degree latitude, longitude precision would have 
(180+1)x(360+1) (65,341) classes (including 0 coordinate), which may sound crazy and there is scale parameter introduced
to make it more than 1 degree. Anyway this method is more computationally expensive in comparison to regression.

The classification might be preferable when needed to see the model's confidence between possible coordinates, which 
regression does not provide.

## Optimizers, Losses & Metrics

Nadam was chosen as an optimizer, it showed minimizing better. There is pretty standard MSE for regression losses and 
sparse categorical cross entropy for classification.

Keras RMSE and MAE were used as metrics for regression. Need to note that due to coordinates normalization Keras RMSE 
will probably almost always have a bit greater value than Keras MAE because latitude and longitude errors are averaged.
That is for example one degree error on latitude will have 1/180 normalized value and one degree on longitude - 1/360. 
Then ```sqrt(mean([(1/180)**2, (1/360)**2]))``` will be greater than ```mean([1/180, 1/360])```. But it is just for 
information, the loss function still minimizes the error. Custom RMSE, MAE could be used for clarity, but no need 
really.

A custom regression metric was introduced to calculate the error in kilometers provided coordinates. It takes earth 
radius as 6,371 km.

## Installation

Pull the code, go into the code directory then

```commandline
python3 -m venv ./venv
pip install -r ./requirements.txt
```

## Usage

### Create

Build a model.

```commandline
$ source ./venv/bin/activate
$ python ./modeling.py create -h
usage: modeling.py create [-h] --model-type {classification,regression} --layers-number {single,multi} 
                          --model-path MODEL_PATH [--max-tweet-length MAX_TWEET_LENGTH] --data-path DATA_PATH 
                          [--test-fraction {10,20,30}] [--precision {1,2,3,4}] [--filters FILTERS]
                          [--epochs EPOCHS] [--charts-directory CHARTS_DIRECTORY]

optional arguments:
  -h, --help            show this help message and exit
  --model-type {classification,regression}
                        A type of the model - classification or regression.
  --layers-number {single,multi}
                        A number of layers to be used to build the model, single layer of multi.
  --model-path MODEL_PATH
                        A path to save the model to.
  --max-tweet-length MAX_TWEET_LENGTH
                        Max tweet length allowed while building the model, longer than that will be cut.
  --data-path DATA_PATH
                        A directory to get data.
  --test-fraction {10,20,30}
                        A fraction of data (per cent) to be used for testing.
  --precision {1,2,3,4}
                        Geolocation precision in degrees for classification.
  --filters FILTERS     Number of CNN filters.
  --epochs EPOCHS       Number of epochs to train the model.
  --charts-directory CHARTS_DIRECTORY
                        A directory to save charts.
```

### Predict

```commandline
$ source ./venv/bin/activate
$ python ./modeling.py predict -h
usage: modeling.py predict [-h] --model-path MODEL_PATH --tweet-text TWEET_TEXT [--charts-directory CHARTS_DIRECTORY]

optional arguments:
  -h, --help            show this help message and exit
  --model-path MODEL_PATH
                        A path to the saved model.
  --tweet-text TWEET_TEXT
                        A tweet text to predict coordinates for.
  --charts-directory CHARTS_DIRECTORY
                        A directory to save a chart for classification confidence distribution.
```

### Evaluate

```commandline
$ source ./venv/bin/activate
$ python ./modeling.py evaluate -h
usage: modeling.py evaluate [-h] --model-path MODEL_PATH --data-path DATA_PATH

optional arguments:
  -h, --help            show this help message and exit
  --model-path MODEL_PATH
                        A path to the saved model.
  --data-path DATA_PATH
                        A directory to get data to evaluate.
```
