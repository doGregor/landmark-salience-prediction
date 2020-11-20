import os
import tensorflow as tf
import numpy as np
from scipy import stats
from image_data_module import TrainTestData
from salience_prediction_module import SaliencePrediction


##### REGRESSION TEST #####
'''
data_class = TrainTestData()
(X_train, Y_train), (X_test, Y_test) = data_class.get_train_test_salience()

prediction_module = SaliencePrediction()
(X_train, Y_train), (X_test, Y_test) = prediction_module.scale_data(X_train, Y_train, X_test, Y_test)

cnn = prediction_module.initialize_cnn_for_regression()

prediction_module.train_cnn(cnn, X_train, Y_train, X_test, Y_test, batch_size=4, save=True)

predictions = prediction_module.predict("regression_cnn_images_directly", X_test, Y_test)

for value in zip(Y_test, predictions):
    print(value)

print("min", np.min(predictions))
print("max", np.max(predictions))
print("mean", np.mean(predictions))
print("median", np.median(predictions))

print("min", np.min(Y_test))
print("max", np.max(Y_test))
print("mean", np.mean(Y_test))
print("median", np.median(Y_test))
'''

##### CLASSIFICATION TEST #####
data_class = TrainTestData()
(X_train, Y_train), (X_test, Y_test) = data_class.get_train_test_binary()

prediction_module = SaliencePrediction()
(X_train, Y_train), (X_test, Y_test) = prediction_module.scale_data(X_train, Y_train, X_test, Y_test)

cnn = prediction_module.initialize_cnn_for_classification()

prediction_module.train_cnn_for_classifiaction(cnn, X_train, Y_train, X_test, Y_test, batch_size=16, save=True)
