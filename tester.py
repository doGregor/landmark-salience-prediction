import os
import tensorflow as tf
import numpy as np
from scipy import stats
from image_data_module import TrainTestData
from salience_prediction_module import SaliencePrediction

'''
images = os.listdir("LM_Szenen_2_bis_6")
img_path = "LM_Szenen_2_bis_6/" + images[10]
img = tf.keras.preprocessing.image.load_img(img_path)
img = tf.keras.preprocessing.image.img_to_array(img)

print(img)
'''

data_class = TrainTestData()
(X_train, Y_train), (X_test, Y_test) = data_class.get_train_test_salience()

prediction_module = SaliencePrediction()
(X_train, Y_train), (X_test, Y_test) = prediction_module.scale_data(X_train, Y_train, X_test, Y_test)

cnn = prediction_module.initialize_cnn_for_regression()

prediction_module.train_cnn(cnn, X_train, Y_train, X_test, Y_test)
