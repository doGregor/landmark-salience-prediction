import os
import tensorflow as tf
import numpy as np
from scipy import stats
from image_data_module import TrainTestData
from salience_prediction_module import SaliencePrediction

'''
### plot feature maps in feature extraction module
for idx_layer, output in enumerate(style_outputs):
    print(output.shape)
    images_to_plot = []
    indices = np.random.randint(0, output.shape[2], 9)
    for idx in indices:
        images_to_plot.append(output[:, :, idx])
    title = "Layer: block5_conv1, Shape: (18, 14), Number: 512"

    fig = plt.figure(figsize=(6, 6))
    for i in range(1, 3 * 3 + 1):
        fig.add_subplot(3, 3, i)
        plt.imshow(images_to_plot[i - 1], cmap='viridis')
    plt.suptitle(title)
    plt.show()
'''

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
'''
from sklearn.utils.class_weight import compute_class_weight

data_class = TrainTestData()
(X_train, Y_train), (X_test, Y_test) = data_class.get_train_test_binary(gray=False)

prediction_module = SaliencePrediction()
(X_train, Y_train), (X_test, Y_test) = prediction_module.scale_data(X_train, Y_train, X_test, Y_test)


print(Y_train.shape)

cnn = prediction_module.initialize_cnn_for_classification()
prediction_module.train_cnn_for_classifiaction(cnn, X_train, Y_train, X_test, Y_test, batch_size=16, save=True)
'''
