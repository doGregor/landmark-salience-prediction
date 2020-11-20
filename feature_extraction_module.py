import os
import numpy as np
import tensorflow as tf
from sklearn.decomposition import MiniBatchDictionaryLearning
import math
import matplotlib.pyplot as plt
import pickle
from sklearn.decomposition import NMF
from sklearn.decomposition import PCA
from sklearn.decomposition import FastICA
import datetime


class FeatureExtractor():
    def __init__(self):
        pass

    def _save_image(self, image_data, rows=3, columns=3):
        fig = plt.figure(figsize=(6, 6))
        for i in range(1, columns * rows + 1):
            fig.add_subplot(rows, columns, i)
            plt.imshow(image_data[i - 1], cmap='gray')
        st = str(datetime.datetime.now())
        st = st.replace('.', '_')
        st = st.replace(':', '_')
        st = st.replace(' ', '_')
        fig_name = "plots/" + st + ".png"
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

    def get_style_vgg19(self, image, input_shape=(298, 224, 3)):
        x = tf.keras.preprocessing.image.img_to_array(image)
        x = tf.image.resize(x, input_shape[:2])
        x = np.array([x])
        # x = x * 255.0
        x = tf.keras.applications.vgg19.preprocess_input(x)

        style_layers = ['block1_conv1',
                        'block2_conv1',
                        'block3_conv1',
                        'block4_conv1',
                        'block5_conv1']

        style_extractor = self._vgg_layers(style_layers, target_size=input_shape)
        style_outputs = style_extractor(x)

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

    def apply_dictionary_learning(self, image_batch, components=50):
        height, width = image_batch.shape[1], image_batch.shape[2]
        image_batch = image_batch.reshape(image_batch.shape[0], image_batch.shape[1] * image_batch.shape[2])
        dict_results = {}

        dictionary = MiniBatchDictionaryLearning(n_components=components)
        dictionary.fit(image_batch)
        dict_results["model"] = dictionary

        components = dictionary.components_
        dict_results["components"] = components

        with open(r"learning_output/dictionary_learning.pickle", "wb") as output_file:
            pickle.dump(dict_results, output_file)

        ###############
        # plots
        '''
        components_image_shape = components.reshape(components.shape[0], height, width)
        index = np.random.choice(components_image_shape.shape[0], 9, replace=False)
        images_to_plot = components_image_shape[index]
        fig = plt.figure(figsize=(6, 6))
        columns = 3
        rows = 3
        for i in range(1, columns * rows + 1):
            fig.add_subplot(rows, columns, i)
            plt.imshow(images_to_plot[i - 1], cmap='gray')
        plt.show()
        '''
        ###############

    def get_dictionary_components(self):
        with open(r"learning_output/dictionary_learning.pickle", "rb") as input_file:
            dict_results = pickle.load(input_file)
        return dict_results["components"]

    def apply_ICA(self, image_batch, components=10):
        height, width = image_batch.shape[1], image_batch.shape[2]
        image_batch = image_batch.reshape(image_batch.shape[0], image_batch.shape[1] * image_batch.shape[2])

        model = self.get_ICA_model()
        if model == "":
            print("[INFO] ICA model is fitted from scratch")
            ICA_results = {}
            ica = FastICA(n_components=components)
            ica.fit(image_batch)
            ICA_results["model"] = ica
            with open(r"learning_output/ICA.pickle", "wb") as output_file:
                pickle.dump(ICA_results, output_file)
            transformed_data = ica.transform(image_batch)
        else:
            print("[INFO] saved ICA model is used")
            ica = model
            transformed_data = ica.transform(image_batch)

        return transformed_data, ica.components_

        ###############
        # plots
        '''
        components_image_shape = components.reshape(components.shape[0], height, width)
        index = np.random.choice(components_image_shape.shape[0], 9, replace=False)
        images_to_plot = components_image_shape[index]
        fig = plt.figure(figsize=(6, 6))
        columns = 3
        rows = 3
        for i in range(1, columns * rows + 1):
            fig.add_subplot(rows, columns, i)
            plt.imshow(images_to_plot[i - 1], cmap='gray')
        plt.show()
        '''
        ###############

    def get_ICA_model(self):
        try:
            with open(r"learning_output/ICA.pickle", "rb") as input_file:
                dict_results = pickle.load(input_file)
                model = dict_results["model"]
        except:
            model = ""
        return model

    def apply_PCA(self, image_batch, components=10):
        height, width = image_batch.shape[1], image_batch.shape[2]
        image_batch = image_batch.reshape(image_batch.shape[0], image_batch.shape[1] * image_batch.shape[2])

        model = self.get_PCA_model()
        if model == "":
            print("[INFO] PCA model is fitted from scratch")
            PCA_results = {}
            pca = PCA(n_components=components)
            pca.fit(image_batch)
            PCA_results["model"] = pca
            with open(r"learning_output/PCA.pickle", "wb") as output_file:
                pickle.dump(PCA_results, output_file)
            transformed_data = pca.transform(image_batch)
        else:
            print("[INFO] saved PCA model is used")
            pca = model
            transformed_data = pca.transform(image_batch)

        return transformed_data, pca.components_

        ###############
        # plots
        '''
        components_image_shape = components.reshape(components.shape[0], height, width)
        index = np.random.choice(components_image_shape.shape[0], 9, replace=False)
        images_to_plot = components_image_shape[index]
        fig = plt.figure(figsize=(6, 6))
        columns = 3
        rows = 3
        for i in range(1, columns * rows + 1):
            fig.add_subplot(rows, columns, i)
            plt.imshow(images_to_plot[i - 1], cmap='gray')
        plt.show()
        '''
        ###############

    def get_PCA_model(self):
        try:
            with open(r"learning_output/PCA.pickle", "rb") as input_file:
                dict_results = pickle.load(input_file)
                model = dict_results["model"]
        except:
            model = ""
        return model


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

    from image_data_module import TrainTestData
    data_loader = TrainTestData()

    (X_train, Y_train), (X_test, Y_test) = data_loader.get_train_test_salience(gray=True)

    X_train_pca_transformed, components = feature_extractor.apply_PCA(X_train, components=5)
    print(X_train_pca_transformed.shape)
