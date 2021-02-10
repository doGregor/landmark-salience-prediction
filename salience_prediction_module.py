import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import sys
from collections import Counter, OrderedDict
import math
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics


class SaliencePrediction():
    def __init__(self):
        pass

    def scale_data(self, train_data, train_labels, test_data, test_labels, labels='classification'):
        """
        Scales image data to values in interval [0,1] and also regression labels to value between
        [0,1] if specified
        :param train_data: Batch of image data (train)
        :param train_labels: Batch of labels, either salience values or binary labels (train)
        :param test_data: Batch of image data (test)
        :param test_labels: Batch of labels, either salience values or binary labels (train)
        :param labels: Specification whether data should be scaled for regression or classification
        accepted parameters: ['regression', 'classification']
        :return: Scaled data and labels in (X_train_scaled, Y_train_scaled), (X_test_scaled, Y_test_scaled)
        """
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
    
    def compute_class_weights(self, labels_list):
        """
        Computes class weights for model training when performing classification
        :param labels_list: Label batch of train data
        :return: dict with labels as keys and class weights as values
        """
        labels_dict = dict(Counter(labels_list))
        sortDic = sorted(labels_dict.items())
        labels_dict = dict(sortDic)
        keys = list(labels_dict.keys())
        
        class_weight = compute_class_weight(class_weight='balanced', classes=keys, y=labels_list)
        class_weight = dict(enumerate(class_weight))

        return class_weight
    
    def data_augmentation(self, data_batch, label_batch, num_images):
        """
        Performs predefined image data augmentation on data batches
        :param data_batch: Image data batch
        :param label_batch: Associated label batch
        :param num_images: number of images to generate
        :return: Returns augmented data batch of near size num_images and associated label batch
        """
        datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            rotation_range=40,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            fill_mode='nearest'
        ) 
        
        datagen.fit(data_batch)
        x_batches = []
        y_batches = []
        images = 0
        for x_batch, y_batch in datagen.flow(data_batch, label_batch, batch_size=25):
            images += 25
            if images < num_images:
                x_batches.append(x_batch)
                y_batches.append(y_batch)
            else:
                break
        
        x_batches = np.concatenate(x_batches)
        y_batches = np.concatenate(y_batches)

        return x_batches, y_batches

    def regression_metrics_plots(self, history, save_name, save_path='plots/'):
        """
        Generates and saves plots of metrics resulting from model training (regression)
        :param history: model.fit() result from keras model training
        :param save_name: name of the figures
        :param save_path: path to folder where figures are saved
        :return: returns nothing
        """
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        axes = plt.gca()
        #axes.set_ylim([0, 0.35])
        path = save_path + save_name + "_loss.png"
        plt.savefig(path)
        plt.clf()

        plt.plot(history.history['mse'])
        plt.plot(history.history['val_mse'])
        plt.title('model mse')
        plt.ylabel('mse')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        axes = plt.gca()
        #axes.set_ylim([0, 0.05])
        path = save_path + save_name + "_mse.png"
        plt.savefig(path)
        plt.clf()

        plt.plot(history.history['mae'])
        plt.plot(history.history['val_mae'])
        plt.title('model mae')
        plt.ylabel('mae')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        axes = plt.gca()
        #axes.set_ylim([0, 0.35])
        path = save_path + save_name + "_mae.png"
        plt.savefig(path)
        plt.clf()

        plt.plot(history.history['mape'])
        plt.plot(history.history['val_mape'])
        plt.title('model mape')
        plt.ylabel('mape')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        axes = plt.gca()
        #axes.set_ylim([0, 50])
        path = save_path + save_name + "_mape.png"
        plt.savefig(path)
        plt.clf()

    def classification_metrics_plots(self, history, save_name, save_path='plots/'):
        """
        Generates and saves plots of metrics resulting from model training (classification)
        :param history: model.fit() result from keras model training
        :param save_name: name of the figures
        :param save_path: path to folder where figures are saved
        :return: returns nothing
        """
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        # axes = plt.gca()
        # axes.set_ylim([0, 0.35])
        path = save_path + save_name + "_loss.png"
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
        path = save_path + save_name + "_accuracy.png"
        plt.savefig(path)
        plt.clf()

    def initialize_cnn_for_regression(self, model_name='vgg19', freeze=True, summary=False, image_shape=(298, 224, 3)):
        """
        Function to initialize a cnn with pre-trained base for regression transfer learning
        :param model_name: pre trained base of the model (vgg19 or resnet152)
        :param freeze: Whether weights of pre-trained layers should be trainable or not
        :param summary: Whether summary of initialized model should be displayed
        :param image_shape: Image shape for model input
        :return: returns the initialized model with pre-trained base
        """
        if model_name == 'vgg19':
            conv_base = tf.keras.applications.VGG19(include_top=False, weights='imagenet', input_shape=image_shape)
        elif model_name == 'resnet152':
            conv_base = tf.keras.applications.ResNet152(include_top=False, weights='imagenet', input_shape=image_shape)
        else:
            sys.exit("Wrong parameter for model initialization")
        if freeze:
            conv_base.trainable = False

        model = tf.keras.models.Sequential()
        model.add(conv_base)
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(256, activation='relu'))
        model.add(tf.keras.layers.Dropout(0.5))
        model.add(tf.keras.layers.Dense(1, activation="linear"))

        #0.00001
        model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=0.00002),
                      loss='mean_absolute_error',
                      #loss='mean_squared_error',
                      metrics=['mse', 'mae', 'mape'])
        
        if summary:
            model.summary()

        return model

    def train_cnn_for_regression(self, model, train_data, train_labels, test_data, test_labels, epochs=10, batch_size=16,
                                 save=False, evaluate=True, save_name='regression', verbose=1, delete=True, plot=False):
        """
        Training of previously initialized model for regression task
        :param model: Initialized model from func:initialize_cnn_for_regression
        :param train_data: normalized batch of train images
        :param train_labels: labels associated with train_data, float value array
        :param test_data: normalized batch of test images (are also used for validation
        shared test/val set)
        :param test_labels: labels associated with test_data, float value array
        :param epochs: Number of epochs for model fine-tuning/training
        :param batch_size: Number of samples used for each step in each epoch
        :param save: Whether model should be saved to .h5 file
        :param evaluate: Whether model should be evaluated after training, saves plots of metrics
        :param save_name: Name for saved .h5 file or/and metric plots
        :param verbose: Degree of information details for training process
        :param delete: Whether model should be deleted after evaluation (clear space in memory)
        :return: returns nothing
        """
        history = model.fit(train_data, train_labels, epochs=epochs,
                            validation_data=(test_data, test_labels),
                            batch_size=batch_size, verbose=verbose)

        if evaluate:
            results_train_evaluation = model.evaluate(train_data, train_labels, batch_size=batch_size)
            print("train loss, train mse, train mae, train mape:", results_train_evaluation)
            
            results_test_evaluation = model.evaluate(test_data, test_labels, batch_size=batch_size)
            print("test loss, test mse, test mae, test mape:", results_test_evaluation)
            
        if plot:
            self.regression_metrics_plots(history=history, save_name=save_name)

        if save:
            path = "nn_models/" + save_name + ".h5"
            model.save(path)
        
        if delete:
            del model

    def predict(self, model_name, X_data):
        model_path = "nn_models/" + model_name + ".h5"
        model = tf.keras.models.load_model(model_path)

        predictions = model.predict(X_data)
        return predictions

    def initialize_cnn_for_classification(self, model_name='vgg19', freeze=True, summary=False, image_shape=(298, 224, 3)):
        """
        Function to initialize a cnn with pre-trained base for classification transfer learning
        :param model_name: pre trained base of the model (vgg19 or resnet50)
        :param freeze: Whether weights of pre-trained layers should be trainable or not
        :param summary: Whether summary of initialized model should be displayed
        :param image_shape: Image shape for model input
        :return: returns the initialized model with pre-trained base
        """
        if model_name == 'vgg19':
            conv_base = tf.keras.applications.VGG19(include_top=False, weights='imagenet', input_shape=image_shape)
            
            if freeze:
                conv_base.trainable = False
            
            model = tf.keras.models.Sequential()
            model.add(conv_base)
            model.add(tf.keras.layers.Flatten())
            model.add(tf.keras.layers.Dense(256, activation='relu'))
            model.add(tf.keras.layers.Dropout(0.5))
            model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
            
            model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=0.000005),
                          loss='binary_crossentropy',
                          metrics=['accuracy'])
            
        elif model_name == 'resnet50':
            conv_base = tf.keras.applications.ResNet50(include_top=False, weights='imagenet', input_shape=image_shape)
            
            if freeze:
                conv_base.trainable = False
                
            model = tf.keras.models.Sequential()
            model.add(conv_base)
            model.add(tf.keras.layers.Flatten())
            model.add(tf.keras.layers.Dense(256, activation='relu'))
            model.add(tf.keras.layers.Dropout(0.5))
            #model.add(tf.keras.layers.Dense(256, activation='relu'))
            model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
            
            model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=2e-5),
                          loss='binary_crossentropy',
                          metrics=['accuracy'])
                
        else:
            sys.exit("Wrong parameter for model initialization")

        '''
        #opt = tf.keras.optimizers.RMSprop(lr=0.000001)
        #opt = tf.keras.optimizers.SGD(lr=0.01)
        opt = tf.keras.optimizers.Adam(learning_rate=0.000001)
        #loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
        '''
        
        if summary:
            model.summary()

        return model

    def train_cnn_for_classification(self, model, train_data, train_labels, test_data, test_labels, epochs=30,
                                     class_weights=None, batch_size=16, save=False, evaluate=False, early_stopping=True,
                                     save_name='classification', verbose=1, delete=True, plot=False):
        """
        Training of previously initialized model for classification task
        :param model: Initialized model from func:initialize_cnn_for_classification
        :param train_data: normalized batch of train images
        :param train_labels: labels associated with train_data, float value array
        :param test_data: normalized batch of test images
        :param test_labels: normalized batch of test images (are also used for validation
        shared test/val set)
        :param epochs: Number of epochs for model fine-tuning/training
        :param class_weights: Weights for appearance of labels in train data
        :param batch_size: Number of samples used for each step in each epoch
        :param save: Whether model should be saved to .h5 file
        :param evaluate: Whether model should be evaluated after training, saves plots of metrics
        :param early_stopping: Whether early stopping should be used during training
        :param save_name: Name for saved .h5 file or/and metric plots
        :param verbose: Degree of information details for training process
        :param delete: Whether model should be deleted after evaluation (clear space in memory)
        :return: returns nothing
        """
        if early_stopping:
            es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)
            history = model.fit(train_data, train_labels, validation_data=(test_data, test_labels),
                            epochs=epochs, batch_size=batch_size, class_weight=class_weights, verbose=verbose, callbacks=[es])
        else:
            history = model.fit(train_data, train_labels, validation_data=(test_data, test_labels),
                                epochs=epochs, batch_size=batch_size, class_weight=class_weights, verbose=verbose)

        if evaluate:
            results_train_evaluation = model.evaluate(train_data, train_labels, batch_size=batch_size)
            print("train loss, train acc:", results_train_evaluation)
            
            results_test_evaluation = model.evaluate(test_data, test_labels, batch_size=batch_size)
            print("test loss, test acc:", results_test_evaluation)
            
        if plot:
            self.classification_metrics_plots(history=history, save_name=save_name)

        if save:
            path = "nn_models/" + save_name + ".h5"
            model.save(path)
            
        if delete:
            del model
            
    def fine_tuning(self, model_name, image_shape=(298, 224, 3)):
        model_path = "nn_models/" + model_name + ".h5"
        model = tf.keras.models.load_model(model_path)
        
        conv_base = tf.keras.applications.VGG19(include_top=False, weights='imagenet', input_shape=image_shape)
        
        new_model = tf.keras.models.Sequential()
        
        n_layers_conv_base = len(conv_base.layers)
        for layer_idx, layer in enumerate(conv_base.layers):
            new_layer = conv_base.get_layer(layer.name)
            if layer_idx < n_layers_conv_base-5:
                new_layer.trainable=False
            else:
                new_layer.trainable=True
            new_model.add(new_layer)
        
        for layer in model.layers[1:]:
            new_layer = model.get_layer(layer.name)
            new_model.add(new_layer)
            
        return new_model        

    def _mean_absolute_percentage_error(self, y_true, y_pred):
        """
        Computes MAPE between true and predicted data arrays
        :param y_true: Array of the true floats
        :param y_pred: Array of predicted floats
        :return: returns calculated MAPE
        """
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    def feature_importance(self, X_train, Y_train, X_test, Y_test, n_estimators=250):
        """
        Estimates feature importance for extracted feature batches via random forest for regression
        :param X_train: Batch of train data features
        :param Y_train: Regression labels for train data
        :param X_test: Batch of test data features
        :param Y_test: Regression labels for test data
        :param n_estimators: number of trees in random forrest
        :return: returns dict with MAE, MSE and MAPE as keys and calculated metric as value
        """
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)

        model = RandomForestRegressor(n_estimators=n_estimators, bootstrap=False, random_state=0)
        model.fit(X_train, Y_train)

        pred = model.predict(X_test)

        results = {}
        results['MAE'] = metrics.mean_absolute_error(Y_test, pred)
        results['MSE'] = metrics.mean_squared_error(Y_test, pred)
        results['MAPE'] = self._mean_absolute_percentage_error(Y_test, pred)

    def initialize_dnn(self, input_shape, task='regression', summary=False, own_layers=None):
        """
        Initialize MLP as final model for training with combined features
        :param input_shape: Input shape of the feature vectors
        :param task: Whether it should be initialized for regression or classification
        :param summary: Whether summary of model should be displayed
        :param own_layers: List of keras layers to initialize model with if default is not wanted
        :return: returns initialized MLP
        """
        model = tf.keras.models.Sequential()

        if own_layers is not None:
            for layer in own_layers:
                model.add(layer)
        else:
            model.add(tf.keras.layers.InputLayer(input_shape=input_shape))
            model.add(tf.keras.layers.Dense(250, activation='relu'))
            model.add(tf.keras.layers.Dense(250, activation='relu'))
            model.add(tf.keras.layers.Dense(100, activation='relu'))
            model.add(tf.keras.layers.Dropout(0.5))

        if task == 'regression':
            model.add(tf.keras.layers.Dense(1, activation='linear'))
            opt = tf.keras.optimizers.SGD(lr=0.001)
            model.compile(loss='mean_absolute_error', optimizer=opt, metrics=['mse', 'mae', 'mape'])
        elif task == 'classification':
            model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
            model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=2e-5),
                          loss='binary_crossentropy',
                          metrics=['accuracy'])
        else:
            sys.exit("Wrong parameter for model initialization")

        if summary:
            model.summary()

        return model


if __name__ == '__main__':
    prediction_module = SaliencePrediction()
    #cnn = prediction_module.initialize_cnn_for_regression()
