# landmarks-salience-prediction
 
The project's goal is to evaluate the proportion of landmark salience that is encoded by visual object information. We test whether it is possible to predict landmark related salience based on image data (visual information). Landmarks are a frequently used concept in pedestrian navigation to help people orientating in unfamiliar environment. We make predictions using methods of machine learning, particularly of deep learning.

Image dataset: https://drive.google.com/file/d/1GsP-0QIO7FlCV5450AZA-hmFUi1X8uNT/view?usp=sharing

## Project Structure

3 main modules are provided implementing the relevant functionality:
* `image_data_module.py`
* `feature_extraction_module.py`
* `salience_prediction_module.py`

A csv file with salience values is located in the `salience_csv` folder.
The raw image data (298x224 pixels) can be downloaded via https://drive.google.com/file/d/1GsP-0QIO7FlCV5450AZA-hmFUi1X8uNT/view?usp=sharing.

Methods from the 3 main modules are merged in jupyter notebooks (`dnn_classifier.ipynb`, `feature_ranking.ipynb`, `transfer_learning.ipynb`, `XAI.ipynb`).

### `image_data_module.py`

This module contains functionality to load the images as numpy arrays and
the associated salience labels as X and Y data. It is also possible to load
binary labels which are produced by k-means clustering and expectation maximization density estimation
implemented in `scripts/salience_clustering.py`. There are also a number of
parameters that can be used when calling data loading functions, for example 
whether the image data should be RGB or grayscale as well as the cross-validation
split that should be loaded. It is also possible to only load image_is, salience values
or binary labels for sepcific cross-validation splits.

Example call to load data of cross-validation split 0 in color, with salience
labels:

```python
from image_data_module import TrainTestData

data_class = TrainTestData()
(X_train, Y_train), (X_test, Y_test) = data_class.get_train_test_salience(cv_name="0", gray=False)
```

### `feature_extraction_module.py`

This module offers a number of methods to extract features from the image data.
For detailed information on the implementation as well as documentation 
on how to use the methods, we refer to the module code.

Example for image contrast extraction of X_train data, loaded in previous code example.

```python
from feature_extraction_module import FeatureExtractor

feature_extractor = FeatureExtractor()
contrast = feature_extractor.contrast(X_train, mode='detailed')
```


### `salience_prediction_module.py`

This module contains functionality to initialize and train neural networks, scale data and estimate salience value importance using random forest estimators.
We provide methods for DNN initialization and training from scratch with own input vectors as well as CNN-based transfer learning with image data.
Once again, there are a number of parameters for each function that influence
the process of initialization and training. For further information have a look
at the well commented and documented methods in the file itself.

Example for transfer learning (using VGG19 CNN) with the data loaded beforehand.

```python
from salience_prediction_module import SaliencePrediction

prediction_module = SaliencePrediction()
#data are scaled to values in interval [0,1] for model training 
(X_train, Y_train), (X_test, Y_test) = prediction_module.scale_data(X_train, Y_train, X_test, Y_test, labels='regression')

vgg19 = prediction_module.initialize_cnn_for_regression(model_name='vgg19')
prediction_module.train_cnn_for_regression(vgg19, X_train, Y_train, X_test, Y_test, epochs=25, batch_size=16, save=False,
                                           evaluate=True, verbose=0)
```

### `plots/`

Contains plots of model training processes (evaluation/results).
