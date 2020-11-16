import tensorflow as tf
import matplotlib.pyplot as plt


class SaliencePrediction():
    def __init__(self):
        pass

    def scale_data(self, train_data, train_labels, test_data, test_labels):
        train_labels, test_labels = train_labels / 5.0, test_labels / 5.0
        train_data, test_data = train_data / 255.0, test_data / 255.0
        return (train_data, train_labels), (test_data, test_labels)

    def initialize_cnn_for_regression(self, image_shape=(298, 224, 3)):
        model = tf.keras.models.Sequential()
        model.add(tf.keras.applications.VGG19(include_top=False, weights='imagenet', input_shape=image_shape))
        model.add(tf.keras.layers.Flatten())
        #model.add(tf.keras.layers.Dense(16000, activation='relu'))
        #model.add(tf.keras.layers.Dense(4000, activation='relu'))
        #model.add(tf.keras.layers.Dense(500, activation='relu'))
        model.add(tf.keras.layers.Dense(1, activation="linear"))

        model.compile(loss="mean_squared_error", optimizer='adam', metrics=['mse', 'mae', 'mape'])
        model.summary()

        return model

    def train_cnn(self, model, train_data, train_labels, test_data, test_labels, epochs=10, batch_size=16):
        history = model.fit(train_data, train_labels, epochs=epochs,
                            validation_data=(test_data, test_labels),
                            batch_size=batch_size)
        plt.plot(history.history['mean_squared_error'])
        plt.plot(history.history['mean_absolute_error'])
        plt.plot(history.history['mean_absolute_percentage_error'])
        plt.show()


if __name__ == '__main__':
    prediction_module = SaliencePrediction()
    cnn = prediction_module.initialize_cnn_for_regression()
