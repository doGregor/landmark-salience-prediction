import os
import numpy as np
import tensorflow as tf
import math
import matplotlib.pyplot as plt
import pickle
from sklearn.decomposition import NMF
from sklearn.decomposition import PCA
from sklearn.decomposition import FastICA
from sklearn.decomposition import MiniBatchDictionaryLearning
import datetime


class FeatureExtractor():
    def __init__(self):
        pass

    def _save_image(self, image_data, title, rows=3, columns=3, cmap='gray'):
        fig = plt.figure(figsize=(6, 6))
        for i in range(1, columns * rows + 1):
            fig.add_subplot(rows, columns, i)
            plt.imshow(image_data[i - 1], cmap=cmap)
        st = str(datetime.datetime.now())
        st = st.replace('.', '_')
        st = st.replace(':', '_')
        st = st.replace(' ', '_')
        st = st.replace('-', '_')
        fig_name = "plots/" + title + "_" + st + ".png"
        fig.savefig(fig_name)

    def get_shape_vgg19(self, image, input_shape=(298, 224, 3)):
        x = tf.keras.preprocessing.image.img_to_array(image)
        x = tf.image.resize(x, input_shape[:2])
        x = np.array([x])
        # x = x * 255.0
        x = tf.keras.applications.vgg19.preprocess_input(x)

        content_layers = ['block5_conv2']

        content_extractor = self._vgg_layers(content_layers, target_size=input_shape)
        content_outputs = content_extractor(x)

        ###############
        # plots
        '''
        import matplotlib.pyplot as plt
        for idx_layer, output in enumerate(content_outputs):
            print(output.shape)
            images_to_plot = []
            indices = np.random.randint(0, output.shape[2], 16)
            for idx in indices:
                images_to_plot.append(output[:, :, idx])
            fig = plt.figure(figsize=(6, 6))
            columns = 4
            rows = 4
            for i in range(1, columns * rows + 1):
                fig.add_subplot(rows, columns, i)
                plt.imshow(images_to_plot[i - 1], cmap="gray")
            plt.suptitle(str(content_layers[idx_layer]) + " Size: " + str(output[:, :, 0].shape) + " Number: " + str(output[0, 0, :].shape))
            plt.show()
        '''
        ###############

        return content_outputs

    def get_style_vgg19(self, image, input_shape=(298, 224, 3), save_fig=True):
        x = tf.image.resize(image, input_shape[:2])
        x = np.array([x])

        style_layers = ['block1_conv1',
                        'block2_conv1',
                        'block3_conv1',
                        'block4_conv1',
                        'block5_conv1']

        style_extractor = self._vgg_layers(style_layers, target_size=input_shape)
        style_outputs = style_extractor(x)

        '''
        for output in style_outputs:
            output = np.average(output[0], axis=2)
            #print(output.shape)
            plt.imshow(output)
            plt.show()
        '''

        if save_fig:
            for idx_layer, output in enumerate(style_outputs):
                images_to_plot = []
                indices = np.random.randint(0, output.shape[3], 9)
                for idx in indices:
                    images_to_plot.append(output[0, :, :, idx])
                title = str(style_layers[idx_layer]) + " Size: " + str(output[0, :, :, 0].shape) + " Number: " + str(output[0, 0, 0, :].shape)
                self._save_image(images_to_plot, title, cmap='viridis')

        ###############
        # plots
        '''
        import matplotlib.pyplot as plt
        for idx_layer, output in enumerate(style_outputs):
            images_to_plot = []
            indices = np.random.randint(0, output.shape[3], 16)
            for idx in indices:
                images_to_plot.append(output[0, :, :, idx])
            fig = plt.figure(figsize=(6, 6))
            columns = 4
            rows = 4
            for i in range(1, columns * rows + 1):
                fig.add_subplot(rows, columns, i)
                plt.imshow(images_to_plot[i - 1])
            plt.suptitle(str(style_layers[idx_layer]) + " Size: " + str(output[0, :, :, 0].shape) + " Number: " + str(output[0, 0, 0, :].shape))
            plt.show()
        '''
        ###############

        style_outputs = [self._gram_matrix(style_output)
                         for style_output in style_outputs]

        for output in style_outputs:
            plt.imshow(output[0])
            plt.show()

        ###############
        # plots
        '''
        for idx_layer, output in enumerate(style_outputs):
            import matplotlib.pyplot as plt
            plt.imshow(np.asarray(output[0]))
            plt.title("Gram matrix for " + str(style_layers[idx_layer]))
            plt.show()
        '''
        ###############

        return style_outputs

    def _vgg_layers(self, layer_names, target_size=(298, 224, 3)):
        vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet', input_shape=target_size)
        vgg.trainable = False
        outputs = [vgg.get_layer(name).output for name in layer_names]
        model = tf.keras.Model([vgg.input], outputs)
        return model

    def _gram_matrix(self, input_tensor):
        result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
        input_shape = tf.shape(input_tensor)
        num_locations = tf.cast(input_shape[1] * input_shape[2], tf.float32)
        return result / (num_locations)

    def dictionary_learning(self, train_data, test_data, components=50, save_fig=True, save_model=True):
        height, width = train_data.shape[1], train_data.shape[2]
        train_data = train_data.reshape(train_data.shape[0], train_data.shape[1] * train_data.shape[2])
        test_data = test_data.reshape(test_data.shape[0], test_data.shape[1] * test_data.shape[2])

        dictionary = MiniBatchDictionaryLearning(n_components=components)
        train_data_dl = dictionary.fit_transform(train_data)
        test_data_dl = dictionary.transform(test_data)

        if save_model:
            dict_results = {}
            dict_results["model"] = dictionary
            with open(r"learning_output/dictionary_learning.pickle", "wb") as output_file:
                pickle.dump(dict_results, output_file)

        if save_fig:
            components = dictionary.components_
            index = np.random.choice(components.shape[0], 9, replace=False)
            images_to_plot = components[index]
            images_to_plot = images_to_plot.reshape(9, height, width)
            self._save_image(images_to_plot, "DL_components")

        return train_data_dl, test_data_dl

    def get_dictionary(self):
        with open(r"learning_output/dictionary_learning.pickle", "rb") as input_file:
            dict_results = pickle.load(input_file)
        return dict_results["model"]

    def ICA(self, train_data, test_data, components=10, save_fig=True, save_model=True):
        height, width = train_data.shape[1], train_data.shape[2]
        train_data = train_data.reshape(train_data.shape[0], train_data.shape[1] * train_data.shape[2])
        test_data = test_data.reshape(test_data.shape[0], test_data.shape[1] * test_data.shape[2])

        ica = FastICA(n_components=components, max_iter=1000, tol=0.001)
        train_data_ica = ica.fit_transform(train_data)
        test_data_ica = ica.transform(test_data)

        if save_model:
            ICA_results = {}
            ICA_results["model"] = ica
            with open(r"learning_output/ICA.pickle", "wb") as output_file:
                pickle.dump(ICA_results, output_file)

        if save_fig:
            components = ica.components_
            index = np.random.choice(components.shape[0], 9, replace=False)
            images_to_plot = components[index]
            images_to_plot = images_to_plot.reshape(9, height, width)
            self._save_image(images_to_plot, "ICA_components")

        return train_data_ica, test_data_ica

    def PCA(self, train_data, test_data, components=10, save_fig=True, save_model=True):
        height, width = train_data.shape[1], train_data.shape[2]
        train_data = train_data.reshape(train_data.shape[0], train_data.shape[1] * train_data.shape[2])
        test_data = test_data.reshape(test_data.shape[0], test_data.shape[1] * test_data.shape[2])

        pca = PCA(n_components=components)
        train_data_pca = pca.fit_transform(train_data)
        test_data_pca = pca.transform(test_data)

        if save_model:
            PCA_results = {}
            PCA_results["model"] = pca
            with open(r"learning_output/PCA.pickle", "wb") as output_file:
                pickle.dump(PCA_results, output_file)

        if save_fig:
            components = pca.components_
            index = np.random.choice(components.shape[0], 9, replace=False)
            images_to_plot = components[index]
            images_to_plot = images_to_plot.reshape(9, height, width)
            self._save_image(images_to_plot, "PCA_components")

        return train_data_pca, test_data_pca

    def NMF(self, train_data, test_data, components=10, save_fig=True, save_model=True):
        height, width = train_data.shape[1], train_data.shape[2]
        train_data = train_data.reshape(train_data.shape[0], train_data.shape[1] * train_data.shape[2])
        test_data = test_data.reshape(test_data.shape[0], test_data.shape[1] * test_data.shape[2])

        nmf = NMF(n_components=components)
        train_data_nmf = nmf.fit_transform(train_data)
        test_data_nmf = nmf.transform(test_data)

        if save_model:
            NMF_results = {}
            NMF_results["model"] = nmf
            with open(r"learning_output/NMF.pickle", "wb") as output_file:
                pickle.dump(NMF_results, output_file)

        if save_fig:
            components = nmf.components_
            index = np.random.choice(components.shape[0], 9, replace=False)
            images_to_plot = components[index]
            images_to_plot = images_to_plot.reshape(9, height, width)
            self._save_image(images_to_plot, "NMF_components")

        return train_data_nmf, test_data_nmf


if __name__ == '__main__':
    feature_extractor = FeatureExtractor()

    '''
    images = os.listdir("LM_Images_downscaled")
    img_path = "LM_Images_downscaled/" + images[205]
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=(298, 224))
    '''

    # feature_extractor.get_shape_vgg19(img, input_shape=(224, 224, 3))
    # feature_extractor.get_shape_vgg19(img)
    # feature_extractor.get_style_vgg19(img)

    '''
    import matplotlib.pyplot as plt
    plt.imshow(np.asarray(img))
    plt.title("Beispielbild")
    plt.show()
    '''

    '''
    (X_train, Y_train), (X_test, Y_test) = data_loader.get_train_test_binary(gray=True)

    X_train, X_test = feature_extractor.PCA(X_train, X_test, components=2, save_fig=False, save_model=False)

    x, y = X_test.T
    plt.scatter(x, y, c=Y_test)
    plt.show()
    '''

    from image_data_module import TrainTestData
    from salience_prediction_module import SaliencePrediction

    data_loader = TrainTestData()
    salience_predictor = SaliencePrediction()

    (X_train, Y_train), (X_test, Y_test) = data_loader.get_train_test_salience(gray=False)
    (X_train, Y_train), (X_test, Y_test) = salience_predictor.scale_data(X_train, Y_train, X_test, Y_test, labels='regression')

    #X_train[5]
    style_output = feature_extractor.get_style_vgg19(X_train[0], save_fig=False)
