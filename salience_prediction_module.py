import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import sys
from collections import Counter, OrderedDict
import math
from sklearn.utils.class_weight import compute_class_weight


class SaliencePrediction():
    def __init__(self):
        pass

    def scale_data(self, train_data, train_labels, test_data, test_labels, labels='classification'):
        if labels == 'regression':
            train_labels, test_labels = train_labels / 5.0, test_labels / 5.0
        elif labels == 'classification':
            '''
            b_train = np.zeros((train_labels.size, train_labels.max()+1))
            b_train[np.arange(train_labels.size), train_labels] = 1
            b_test = np.zeros((test_labels.size, test_labels.max()+1))
            b_test[np.arange(test_labels.size), test_labels] = 1
            train_labels = b_train
            test_labels = b_test
            '''
            train_labels = train_labels
            test_labels = test_labels
        else:
            sys.exit("Wrong parameter for data preprocessing")
        train_data, test_data = train_data / 255.0, test_data / 255.0
        return (train_data, train_labels), (test_data, test_labels)
    
    def create_class_weight(self, labels_list):
        labels_dict = dict(Counter(labels_list))
        sortDic = sorted(labels_dict.items())
        labels_dict = dict(sortDic)
        keys = list(labels_dict.keys())
        
        class_weight = compute_class_weight(class_weight='balanced', classes=keys, y=labels_list)
        class_weight = dict(enumerate(class_weight))

        return class_weight

    def initialize_cnn_for_regression(self, summary=False, image_shape=(298, 224, 3)):
        #lrelu = lambda x: tf.keras.activations.relu(x, alpha=0.1)
        model = tf.keras.models.Sequential()
        model.add(tf.keras.applications.VGG19(include_top=False, weights='imagenet', input_shape=image_shape))
        model.add(tf.keras.layers.Flatten())
        #model.add(tf.keras.layers.Dense(200, activation='relu'))
        #model.add(tf.keras.layers.Dropout(0.5))
        #model.add(tf.keras.layers.Dense(50, activation='relu'))
        model.add(tf.keras.layers.Dense(4096, activation='relu'))
        model.add(tf.keras.layers.Dense(2048, activation='relu'))
        model.add(tf.keras.layers.Dense(1, activation="linear"))

        opt = tf.keras.optimizers.SGD(lr=0.001)
        model.compile(loss='mean_absolute_error', optimizer=opt, metrics=['mse', 'mae', 'mape'])
        
        if summary:
            model.summary()

        return model

    def train_cnn_for_regression(self, model, train_data, train_labels, test_data, test_labels, epochs=10, batch_size=16,
                                 save=False, evaluate=True, save_name='regression', verbose=1, delete=True):
        history = model.fit(train_data, train_labels, epochs=epochs,
                            validation_data=(test_data, test_labels),
                            batch_size=batch_size, verbose=verbose)

        if evaluate:
            results_train_evaluation = model.evaluate(train_data, train_labels, batch_size=batch_size)
            print("train loss, train acc:", results_train_evaluation)
            
            results_test_evaluation = model.evaluate(test_data, test_labels, batch_size=batch_size)
            print("test loss, test acc:", results_test_evaluation)
            
            plt.plot(history.history['loss'])
            plt.plot(history.history['val_loss'])
            plt.title('model loss')
            plt.ylabel('loss')
            plt.xlabel('epoch')
            plt.legend(['train', 'val'], loc='upper left')
            axes = plt.gca()
            axes.set_ylim([0, 0.35])
            path = "plots/" + save_name + "_loss.png"
            plt.savefig(path)
            plt.clf()
            
            plt.plot(history.history['mse'])
            plt.plot(history.history['val_mse'])
            plt.title('model mse')
            plt.ylabel('mse')
            plt.xlabel('epoch')
            plt.legend(['train', 'val'], loc='upper left')
            axes = plt.gca()
            axes.set_ylim([0, 0.05])
            path = "plots/" + save_name + "_mse.png"
            plt.savefig(path)
            plt.clf()
            
            plt.plot(history.history['mae'])
            plt.plot(history.history['val_mae'])
            plt.title('model mae')
            plt.ylabel('mae')
            plt.xlabel('epoch')
            plt.legend(['train', 'val'], loc='upper left')
            axes = plt.gca()
            axes.set_ylim([0, 0.35])
            path = "plots/" + save_name + "_mae.png"
            plt.savefig(path)
            plt.clf()
            
            plt.plot(history.history['mape'])
            plt.plot(history.history['val_mape'])
            plt.title('model mape')
            plt.ylabel('mape')
            plt.xlabel('epoch')
            plt.legend(['train', 'val'], loc='upper left')
            axes = plt.gca()
            axes.set_ylim([0, 50])
            path = "plots/" + save_name + "_mape.png"
            plt.savefig(path)
            plt.clf()

        if save:
            path = "nn_models/" + save_name + ".h5"
            model.save(path)
            del model
        
        if delete:
            del model

    def predict(self, model_name, X_data):
        model_path = "nn_models/" + model_name + ".h5"
        model = tf.keras.models.load_model(model_path)

        predictions = model.predict(X_data)
        return predictions

    def initialize_cnn_for_classification(self, summary=False, image_shape=(298, 224, 3)):
        model = tf.keras.models.Sequential()
        model.add(tf.keras.applications.VGG19(include_top=False, weights='imagenet', input_shape=image_shape))
        model.add(tf.keras.layers.Flatten())
        #model.add(tf.keras.layers.Dense(200, activation='relu'))
        #model.add(tf.keras.layers.Dropout(0.5))
        #model.add(tf.keras.layers.Dense(50, activation='relu'))
        model.add(tf.keras.layers.Dense(4096, activation='relu'))
        model.add(tf.keras.layers.Dense(2048, activation='relu'))
        #model.add(tf.keras.layers.Dense(2, activation="softmax"))
        model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

        opt = tf.keras.optimizers.SGD(lr=0.001)
        loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
        
        if summary:
            model.summary()

        return model

    def train_cnn_for_classification(self, model, train_data, train_labels, test_data, test_labels, epochs=10,
                                     class_weights=None, batch_size=16, save=False, evaluate=False, save_name='classification', verbose=1, delete=True):
        history = model.fit(train_data, train_labels, validation_data=(test_data, test_labels),
                            epochs=epochs, batch_size=batch_size, class_weight=class_weights, verbose=verbose)

        if evaluate:
            results_train_evaluation = model.evaluate(train_data, train_labels, batch_size=batch_size)
            print("train loss, train acc:", results_train_evaluation)
            
            results_test_evaluation = model.evaluate(test_data, test_labels, batch_size=batch_size)
            print("test loss, test acc:", results_test_evaluation)
            
            plt.plot(history.history['loss'])
            plt.plot(history.history['val_loss'])
            plt.title('model loss')
            plt.ylabel('loss')
            plt.xlabel('epoch')
            plt.legend(['train', 'val'], loc='upper left')
            #axes = plt.gca()
            #axes.set_ylim([0, 0.35])
            path = "plots/" + save_name + "_loss.png"
            plt.savefig(path)
            plt.clf()

            plt.plot(history.history['accuracy'])
            plt.plot(history.history['val_accuracy'])
            plt.title('model accuracy')
            plt.ylabel('accuracy')
            plt.xlabel('epoch')
            plt.legend(['train', 'val'], loc='upper left')
            # axes = plt.gca()
            # axes.set_ylim([0, 0.35])
            path = "plots/" + save_name + "_accuracy.png"
            plt.savefig(path)
            plt.clf()

        if save:
            path = "nn_models/" + save_name + ".h5"
            model.save(path)
            del model
            
        if delete:
            del model


if __name__ == '__main__':
    prediction_module = SaliencePrediction()
    cnn = prediction_module.initialize_cnn_for_regression()
