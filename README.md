# A CNN to predict geolocaiton provided tweet text

The method employs CNN's ability to detect features spread over data while training. No sophistication in the model's architecture, but using pretty standard approach adjusted to the task nature.

The work is done with Keras API.

## Data preprocessing

The text is transformed to Unicode code points, then normalized to [0, 1] range and padded with 0 to have predefined length. Geolocation data are normalized to [0, 1] for regression, this proved to be efficient.

## A model's overview

There are two models, they differ in amount of convolutional layers, which are the only layers there (except the final Dense layer), but work somewhat the same, when it comes to evaluation (200km mean error and 0.86 accuracy), at least on that amount of data and filters I was able to train. The single layer model though goes faster (less epochs) to its best state, which is expected.

The first model is based on idea that we want to detect location-specific phrases, names etc. at once and go to the exit right away. It has only one Conv1D layer, kernel_size 15 (max length of the phrase, name) and 1000 filters (amount of different phrases, names). Obviously the more fliters the better and that will affect training performance.

The second model tries to detect syllables on the first layer (max 3 letters syllables, 1000 distinct syllables), then the second layer looks for combinations of syllables (max 2 syllables combinations, 1000 distinct combinations), then sort of words (max 2 combinations mix, 1000 distinct mixes/words), then finally phrases, names (that would consists of max 2 words, 1000 distinct), which could be mapped to locations.

Pooling layers were not used, because that would mangle phrases, mixes to be detected (MHO). Characters' values are not the same as pixels' values, which can be pooled based on maximum, mean etc.

ReLU activation is applied, it proved to be more efficient than linear with coordinates having negative values.

## Regression & classification

Both methods have been tried. Regression method tries to predict exact coordinate. Classificaion method yields a score among classes, amount of which depends on needed precision. That is 1 degree lattitude, longtitude precision would have (180+1)x(360+1) (65,341) classes (inlcuding 0 coordinate), which may sound crazy and there is scale parameter introduced to make it more than 1 degree. Anyway this method is more computationally expensive in comparison to regression.

The classification might be preferable when needed to see the model's confidence between possible coordinates, which regression does not provide.

## Optimizers, Losses & Metrics

Nadam was chosen as an optimizer, it showed minimizing better. There is pretty standard MSE for regression losses and sparse categorical crossentropy for classification.

Keras RMSE and MAE were used as metrics for regression. Need to note that due to coordinates normalization Keras RMSE will probably almost always have a bit greater value than Keras MAE becase lattitude and longtitude errors are averaged. That is for example one degree error on lattitude will have 1/180 normalized value and one degree on longtitude - 1/360. Then ```sqrt(mean([(1/180)**2, (1/360)**2]))``` will be greater than ```mean([1/180, 1/360])```. But it is just for information, the loss function still minimizes the error. Custom RMSE, MAE could be used for clarity, but no need really.

A custom regression metric was introduced to calculate the error in kilometers provided coordinates. It takes earth radius as 6,371 km.

## Prediction & evaluation

```predict.py``` takes as arguments the model name (path) as it was saved after training, the text (single tweet), max tweet length as it was configured for the model and an optional scale parameter for the classification model. It would yield coordinates then, the classification model additionaly plots a confidense chart.

```evaluate.py``` takes the model name (path), max tweet length as it was configured for the model and an optional scale parameter for the classification model. It then reads csv in the current directoy the same format as input data including file names and yields mean error in kilometers for regression model and accuracy for classification. Custom metric for classification can be created though to show kilometers for the most confident prediction, not implemented.
