import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import sys


class SaliencePrediction():
    def __init__(self):
        pass

    def scale_data(self, train_data, train_labels, test_data, test_labels, labels='classification'):
        if labels == 'regression':
            train_labels, test_labels = train_labels / 5.0, test_labels / 5.0
        elif labels == 'classification':
            b_train = np.zeros((train_labels.size, train_labels.max()+1))
            b_train[np.arange(train_labels.size), train_labels] = 1
            b_test = np.zeros((test_labels.size, test_labels.max()+1))
            b_test[np.arange(test_labels.size), test_labels] = 1
            train_labels = b_train
            test_labels = b_test
        else:
            print("Wrong parameter for data preprocessing:", sys.exc_info()[0])
        train_data, test_data = train_data / 255.0, test_data / 255.0
        return (train_data, train_labels), (test_data, test_labels)

    def initialize_cnn_for_regression(self, image_shape=(298, 224, 3)):
        model = tf.keras.models.Sequential()
        model.add(tf.keras.applications.VGG19(include_top=False, weights='imagenet', input_shape=image_shape))
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dropout(0.5))
        model.add(tf.keras.layers.Dense(1, activation="linear"))

        model.compile(loss="mean_squared_error", optimizer='adam', metrics=['mse', 'mae', 'mape'])
        model.summary()

        return model

    def train_cnn_for_regression(self, model, train_data, train_labels, test_data, test_labels, epochs=10, batch_size=16,
                                 save=False, evaluate=True):
        history = model.fit(train_data, train_labels, epochs=epochs,
                            validation_data=(test_data, test_labels),
                            batch_size=batch_size)

        if evaluate:

            evaluation = model.evaluate(test_data, test_labels)

            plt.plot(history.history['mse'], label="train mse")
            plt.plot(evaluation.history['mse'], label="test mse")
            plt.savefig('plots/mse_regression_cnn.png')
            plt.plot(history.history['mae'], label="train mae")
            plt.plot(evaluation.history['mae'], label="test mae")
            plt.savefig('plots/mae_regression_cnn.png')
            plt.plot(history.history['mape'], label="train mape")
            plt.plot(evaluation.history['mape'], label="test mape")
            plt.savefig('plots/mape_regression_cnn.png')

        if save:
            model.save('nn_models/regression_cnn_images_directly.h5')
            del model

    def predict(self, model_name, X_data):
        model_path = "nn_models/" + model_name + ".h5"
        model = tf.keras.models.load_model(model_path)

        predictions = model.predict(X_data)
        return predictions

    def initialize_cnn_for_classification(self, image_shape=(298, 224, 3)):
        model = tf.keras.models.Sequential()
        model.add(tf.keras.applications.VGG19(include_top=False, weights='imagenet', input_shape=image_shape))
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(10000, activation="relu"))
        model.add(tf.keras.layers.Dense(1, activation="sigmoid"))

        opt = tf.keras.optimizers.SGD(lr=0.01, momentum=0.9)
        model.compile(loss="binary_crossentropy", optimizer=opt, metrics=['accuracy'])
        model.summary()

        return model

    def train_cnn_for_classifiaction(self, model, train_data, train_labels, test_data, test_labels, epochs=10, batch_size=16, save=False):
        history = model.fit(train_data, train_labels, validation_data=(test_data, test_labels),
                            epochs=epochs, batch_size=batch_size, verbose=1)
        # evaluate the model
        _, train_acc = model.evaluate(train_data, train_labels, verbose=1)
        _, test_acc = model.evaluate(test_data, test_labels, verbose=1)
        print('Train: %.3f, Test: %.3f' % (train_acc, test_acc))
        # plot loss during training
        plt.subplot(211)
        plt.title('Loss')
        plt.plot(history.history['loss'], label='train')
        plt.plot(history.history['val_loss'], label='test')
        plt.legend()
        # plot accuracy during training
        plt.subplot(212)
        plt.title('Accuracy')
        plt.plot(history.history['accuracy'], label='train')
        plt.plot(history.history['val_accuracy'], label='test')
        plt.legend()
        plt.savefig('plots/classification_cnn.png')

        if save:
            model.save('nn_models/classification_cnn_images_directly.h5')
            del model


if __name__ == '__main__':
    prediction_module = SaliencePrediction()
    cnn = prediction_module.initialize_cnn_for_regression()
