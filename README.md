# landmarks_salience_prediction
#### Master Thesis Information Science, University of Regensburg, 2021
 
The projects aim is to test whether it is possible to predict a salience score
for landmark images. Landmarks are a concept which is frequently used in pedestrian
navigation. The predictions are made using methods from deep learning.

## Project Structure

We provide 3 main modules where functionality is implemented:
* `image_data_module.py`
* `feature_extraction_module.py`
* `salience_prediction_module.py`

A csv file with salience values is located in the `salience_csv` folder.
The raw image data (298x224 pixels) is available at: LINK

The module code merging is implemented in the jupyter notebooks available in
this projects root directory.

### `image_data_module.py`

This module contains functionality to load the images as numpy arrays and
the associated salience labels as X and Y data. It is also possible to load
binary labels which are produced by clustering and density estimation (EM)
implemented in `scripts/saleince_clustering.py`. There are also a number of
parameters that can be used when calling data loading functions, for example 
whether the image data should be RGB or grayscale as well as the cross-validation
split that should be loaded.

Example call to load data of cross-validation split 0 in color, with salience
labels:

```python
from image_data_module import TrainTestData

data_class = TrainTestData()
(X_train, Y_train), (X_test, Y_test) = data_class.get_train_test_salience(cv_name="0", gray=False)
```

### `feature_extraction_module.py`

This module offers a number of methods to extract features from the image data.
For detailed information on the functions available as well as documentation 
on how to use them, have a look at the comments in the modules code.

Example for image contrast extraction of X_train data, loaded before.

```python
from feature_extraction_module import FeatureExtractor

feature_extractor = FeatureExtractor()
contrast = feature_extractor.contrast(X_test, mode='detailed')
```


### `salience_prediction_module.py`

In this module all functionality for model training and prediction making is
located. There are functions for the initialization of DNNs, CNNs for training
from scratch with your own input vectors or transfer learning with image data.
Again, there are a number of parameters for each function to influence
the process of initialization and training. For further information have a look
at the file itself, the methods are well commented and documented.

Example for transfer learning with the data loaded before.

```python
from salience_prediction_module import SaliencePrediction

prediction_module = SaliencePrediction()
#data are scaled to values in interval [0,1] for model training 
(X_train, Y_train), (X_test, Y_test) = prediction_module.scale_data(X_train, Y_train, X_test, Y_test, labels='regression')

cnn = prediction_module.initialize_cnn_for_regression()
prediction_module.train_cnn_for_regression(cnn, X_train, Y_train, X_test, Y_test, epochs=25, batch_size=16, save=False,
                                           evaluate=True, verbose=0)
```