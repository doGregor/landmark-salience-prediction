import os
import numpy as np
import tensorflow as tf
import math
import matplotlib.pyplot as plt
import pickle
from sklearn.decomposition import NMF, PCA, FastICA, MiniBatchDictionaryLearning
import datetime
from scipy import ndimage
import cv2
import sys


class FeatureExtractor():
    def __init__(self):
        pass

    def _save_image(self, image_data, title, rows=3, columns=3, cmap='gray'):
        """
        Saves a plot of images.
        :param image_data: List of images [len(list) == rows*columns]
        :param title: Title to include in plot and file name
        :param rows: number of rows in plot if multiple images
        :param columns: number of columns in plot if multiple images
        :param cmap: colormap from matplotlib, e.g. 'gray' if grayscale
        :return: returns nothing, saves plot to /plots folder
        """
        fig = plt.figure(figsize=(6, 6))
        for i in range(1, columns * rows + 1):
            fig.add_subplot(rows, columns, i)
            plt.imshow(image_data[i - 1], cmap=cmap)
            plt.suptitle(title)
        st = str(datetime.datetime.now())
        st = st.replace('.', '_')
        st = st.replace(':', '_')
        st = st.replace(' ', '_')
        st = st.replace('-', '_')
        fig_name = "plots/" + title + "_" + st + ".png"
        fig.savefig(fig_name)

    def save_to_pickle(self, data, filename, folder='learning_output'):
        """
        Can be used to save any type of data in format of a dictionary to a pickle file.
        :param data: Data to save; dictionary is recommended
        :param filename: Name of the file to save
        :param folder: Name of the folder where the file will be saved in
        :return: Returns nothing
        """
        print("[INFO] Saving data to pickle")
        path = folder + "/" + filename + ".pickle"
        with open(path, "wb") as output_file:
            pickle.dump(data, output_file)
        print("[INFO] Finished data saving")

    def get_content_vgg19(self, image_batch, input_shape=(298, 224, 3), batch_size=16):
        """
        Computes gram matrices for image batch by extracting content features from hidden vgg19 CNN layer.
        :param image_batch: Image data batch in (x,y,z,3) RGB format
        :param input_shape: Single image's shape (y,z,3)
        :param batch_size: Number of images that are processed at a time
        :return: Gram matrices batch in (x,512,512,1) format
        """
        print("[INFO] Starting VGG19 Content Matrix Extraction")
        if image_batch.shape[0] <= batch_size:
            x = [image_batch]
        else:
            n_batches = math.ceil(image_batch.shape[0] / batch_size)
            x = np.array_split(image_batch, n_batches)

        content_layers = ['block5_conv2']

        content_extractor = self._vgg_layers(content_layers, input_shape=input_shape)

        gram_outputs = []
        for batch in x:
            content_outputs = content_extractor(batch)

            for content_out in content_outputs:
                content_out = np.array([content_out])
                gm = self._gram_matrix(content_out)
                gram_outputs.append(np.asarray(gm[0]).reshape(gm[0].shape[0], gm[0].shape[1], 1))

        print("[INFO] Finished VGG19 Content Matrix Extraction")
        return np.asarray(gram_outputs)

    def get_style_vgg19(self, image_batch, input_shape=(298, 224, 3), batch_size=16):
        """
        Computes gram matrices for image batch by extracting stylistic features from hidden vgg19 CNN layer.
        :param image_batch: Image data batch in (x,y,z,3) RGB format
        :param input_shape: Single image's shape (y,z,3)
        :param batch_size: Number of images that are processed at a time
        :return: Gram matrices batch in (x,512,512,1) format
        """
        print("[INFO] Starting VGG19 Style Matrix Extraction")
        if image_batch.shape[0] <= batch_size:
            x = [image_batch]
        else:
            n_batches = math.ceil(image_batch.shape[0]/batch_size)
            x = np.array_split(image_batch, n_batches)

        style_layers = ['block5_conv1']

        style_extractor = self._vgg_layers(style_layers, input_shape=input_shape)

        gram_outputs = []
        for batch in x:
            style_outputs = style_extractor(batch)

            for style_out in style_outputs:
                style_out = np.array([style_out])
                gm = self._gram_matrix(style_out)
                gram_outputs.append(np.asarray(gm[0]).reshape(gm[0].shape[0], gm[0].shape[1], 1))

        print("[INFO] Finished VGG19 Style Matrix Extraction")
        return np.asarray(gram_outputs)

    def _vgg_layers(self, layer_names, input_shape=(298, 224, 3)):
        vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet', input_shape=input_shape)
        vgg.trainable = False
        outputs = [vgg.get_layer(name).output for name in layer_names]
        model = tf.keras.Model([vgg.input], outputs)
        return model

    def _gram_matrix(self, input_tensor):
        result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
        input_shape = tf.shape(input_tensor)
        num_locations = tf.cast(input_shape[1] * input_shape[2], tf.float32)
        return result / (num_locations)

    def dictionary_learning(self, train_data, test_data, components=100, save_fig=True, save_model=True):
        """
        Learns a dictionary from train data and applies it to train and test data.
        :param train_data: Image batch in (x,y,z,1) grayscale format (train)
        :param test_data: Image batch in (x,y,z,1) grayscale format (test)
        :param components: Number of atoms in dictionary to be extracted
        :param save_fig: If true 9 random components are plotted
        :param save_model: If true fitted dictionary model is saved as pickle file
        :return: returns transformed train and test data in (x,components) format
        """
        print("[INFO] Starting Dictionary Learning")
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

        print("[INFO] Finished Dictionary Learning")
        return train_data_dl, test_data_dl

    def get_dictionary(self):
        """
        Function to return the learnt dictionary.
        :return: sklearn fitted dictionary learning model
        """
        with open(r"learning_output/dictionary_learning.pickle", "rb") as input_file:
            dict_results = pickle.load(input_file)
        return dict_results["model"]

    def ICA(self, train_data, test_data, components=50, save_fig=True, save_model=True):
        """
        Applies Independent Component Analysis on train data and fits test data on learnt model.
        :param train_data: Image batch in (x,y,z,1) grayscale format (train)
        :param test_data: Image batch in (x,y,z,1) grayscale format (test)
        :param components: Number of model's components to learn
        :param save_fig: If true 9 random components are plotted
        :param save_model: If true ICA model is saved as pickle file
        :return: returns transformed train and test data in (x,components) format
        """
        print("[INFO] Starting ICA")
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

        print("[INFO] Finished Dictionary Learning")
        return train_data_ica, test_data_ica

    def PCA(self, train_data, test_data, components=50, save_fig=True, save_model=True):
        """
        Applies Principal Components Analysis on train data and fits test data on learnt model.
        :param train_data: Image batch in (x,y,z,1) grayscale format (train)
        :param test_data: Image batch in (x,y,z,1) grayscale format (test)
        :param components: Number of components to be kept during fitting process.
        :param save_fig: If true 9 random components are plotted
        :param save_model: If true PCA model is saved as pickle file
        :return: returns transformed train and test data in (x,components) format
        """
        print("[INFO] Starting PCA")
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

        print("[INFO] Finished PCA")
        return train_data_pca, test_data_pca

    def NMF(self, train_data, test_data, components=50, save_fig=True, save_model=True):
        """
        Applies Non Negative Matrix Factorization on train data and fits test data on learnt model.
        :param train_data: Image batch in (x,y,z,1) grayscale format (train)
        :param test_data: Image batch in (x,y,z,1) grayscale format (test)
        :param components: Matrix dimension (components) for learnt Matrices
        :param save_fig: If true 9 random components are plotted
        :param save_model: If true NMF model is saved as pickle file
        :return: returns transformed train and test data in (x,components) format
        """
        print("[INFO] Starting NMF")
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

        print("[INFO] Finished NMF")
        return train_data_nmf, test_data_nmf

    def sobel_filter(self, image_batch):
        """
        Applies sobel edge computation on data image batches.
        :param image_batch: Image data batch in (x,y,z,1) GRAY format
        :return: returns np array for input image batch with sobel filter output in shape (x, y, z, 1)
        """
        print("[INFO] Starting Sobel Detection")
        dim_1 = image_batch.shape[1]
        dim_2 = image_batch.shape[2]
        data_out = np.zeros(image_batch.shape)
        for idx in range(image_batch.shape[0]):
            image = image_batch[idx][:, :, 0]
            im = image.astype('int32')
            dx = ndimage.sobel(im, 0)
            dy = ndimage.sobel(im, 1)
            mag = np.hypot(dx, dy)
            mag *= 255.0 / np.max(mag)
            mag = mag.reshape(dim_1, dim_2, 1)
            data_out[idx] = mag
        print("[INFO] Finished Sobel Detection")
        return data_out
    
    def complexity(self, image_batch, mode='avg', grid_dim=(6, 3)):
        """
        Computes complexity of an image.
        :param image_batch: Image data batch in (x,y,z,1) GRAY format
        :param mode: how detailed complexity should be computed : 'avg' or 'detailed' or 'grid'
        :param grid_dim: (n_rows, n_columns) for grid split
        :return: Batch with complexity values in (x,y)[detailed] or (x)[avg] or
        (x,grid_dim[0]*grid_dim[1])[grid] format
        """
        print("[INFO] Starting Complexity Computation")
        image_batch = self.sobel_filter(image_batch)
        data_out = []
        for idx in range(image_batch.shape[0]):
            image = image_batch[idx][:, :, 0]
            if mode == 'avg':
                data_out.append(np.average(image))
            elif mode == 'detailed':
                data_out.append(np.average(image, axis=1))
            elif mode == 'grid':
                grids = self.split_image_grid(image, grid_dim[0], grid_dim[1])
                grids_complexity = []
                for p in grids:
                    grids_complexity.append(np.average(p))
                data_out.append(grids_complexity)
            else:
                sys.exit("Wrong parameter for data preprocessing")
        print("[INFO] Finished Complexity Computation")
        return np.asarray(data_out)

    def SIFT_learning(self, image_batch, n_features=50):
        """
        Extraction of SIFT from images.
        :param image_batch: Image data batch in (x,y,z,1) GRAY format
        :param n_features: Number of keypoints to extract from each image
        :return: returns np array with keypoints for each image in shape (x, VAR, 128)
        [INFO: Number of extracted keypoints cannot be held constant]
        """
        print("[INFO] Staring SIFT extraction")
        input_data = image_batch.astype(np.uint8)
        sift = cv2.SIFT_create(nfeatures=n_features)
        data_out = []
        for idx in range(input_data.shape[0]):
            image = input_data[idx]
            kp, des = sift.detectAndCompute(image, None)
            data_out.append(np.asarray(des))
        print("[INFO] Finished SIFT extraction")
        return np.asarray(data_out)

    def color_histogram_2d(self, image_batch):
        """
        Applies color histogram computation on data image batches.
        :param image_batch: Image data batch in (x,y,z,3) RGB format
        :return: returns a np array for input image batch with color histograms in shape (x, 180, 256, 1)

        INFO: use plt.imshow(image, interpolation='nearest') to plot images for better results
        """
        print("[INFO] Starting Color Histogram 2d Computation")
        input_data = image_batch.astype(np.uint8)
        data_out = []
        for idx in range(input_data.shape[0]):
            image = input_data[idx]
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            hist = cv2.calcHist([hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])
            data_out.append(hist.reshape(180, 256, 1))
        print("[INFO] Finished Color Histogram 2d Computation")
        return np.asarray(data_out)

    def color_histogram_1d(self, image_batch):
        """
        Computes 1d color histograms for RGB channels.
        :param image_batch: Image data batch in (x,y,z,3) RGB format
        :return: returns a np array for input image batch with color histograms in shape (x, 768)
        [INFO: 768/n_channels=256]
        """
        print("[INFO] Starting Color Histogram 1d Computation")
        input_data = image_batch.astype(np.uint8)
        data_out = []
        for idx in range(input_data.shape[0]):
            image = input_data[idx]
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            single_hist = []
            color = ('b', 'g', 'r')
            for i, col in enumerate(color):
                hist = cv2.calcHist([image], [i], None, [256], [0, 256])
                single_hist = single_hist + list(hist.flatten())
            data_out.append(single_hist)
        print("[INFO] Finished Color Histogram 1d Computation")
        return np.asarray(data_out)

    def split_image_grid(self, matrix, n_rows=6, n_cols=3):
        """
        Splits a matrix of size NxM into n_rows*n_cols grids with height N/n_rows
        and width M/n_cols
        :param matrix: Matrix of size (NxM)
        :param n_rows: number of rows for splitting
        :param n_cols: number of columns for splitting
        :return: List of extracted grid fields
        """
        h, w = matrix.shape
        block_width = int(w / n_cols)
        block_height = int(h / n_rows)

        blocks = []

        for col_id in range(0, n_rows):
            for row_id in range(0, n_cols):
                if col_id < n_rows - 1 and row_id < n_cols - 1:
                    curr_block = matrix[col_id * block_height:col_id * block_height + block_height,
                                 row_id * block_width:row_id * block_width + block_width]
                    blocks.append(curr_block)
                elif col_id < n_rows - 1 and row_id == n_cols - 1:
                    curr_block = matrix[col_id * block_height:col_id * block_height + block_height,
                                 row_id * block_width:]
                    blocks.append(curr_block)
                elif col_id == n_rows - 1 and row_id < n_cols - 1:
                    curr_block = matrix[col_id * block_height:, row_id * block_width:row_id * block_width + block_width]
                    blocks.append(curr_block)
                else:
                    curr_block = matrix[col_id * block_height:, row_id * block_width:]
                    blocks.append(curr_block)
        return blocks

    def brightness(self, image_batch, mode='avg', grid_dim=(6, 3)):
        """
        Computes brightness values for image batch.
        :param image_batch: Image data batch in (x,y,z,1) GRAY format
        :param mode: how detailed brightness should be computed : 'avg' or 'detailed' or 'grid'
        :param grid_dim: (n_rows, n_columns) for grid split
        :return: Batch with brightness values in (x,y)[detailed] or (x)[avg] or
        (x,grid_dim[0]*grid_dim[1])[grid] format
        """
        print("[INFO] Starting Brightness Computation")
        data_out = []
        for idx in range(image_batch.shape[0]):
            image = image_batch[idx][:, :, 0]
            if mode == 'avg':
                data_out.append(np.average(image))
            elif mode == 'detailed':
                data_out.append(np.average(image, axis=1))
            elif mode == 'grid':
                grids = self.split_image_grid(image, grid_dim[0], grid_dim[1])
                grids_brightness = []
                for p in grids:
                    grids_brightness.append(np.average(p))
                data_out.append(grids_brightness)
            else:
                sys.exit("Wrong parameter for data preprocessing")
        print("[INFO] Finished Brightness Computation")
        return np.asarray(data_out)

    def contrast(self, image_batch, mode='avg', grid_dim=(6, 3)):
        """
        Computes contrast values for image batch.
        :param image_batch: Image data batch in (x,y,z,3) RGB format
        :param mode: how detailed contrast should be computed: 'avg' or 'detailed' or 'grid'
        :param grid_dim: (n_rows, n_columns) for grid split
        :return: Batch with brightness values in (x,y,3)[detailed] or (x,3)[avg] format or
        (x,grid_dim[0]*grid_dim[1],3)[grid] format
        """
        print("[INFO] Starting Contrast Computation")
        data_out = []
        for idx in range(image_batch.shape[0]):
            image_R = image_batch[idx][:, :, 0]
            image_G = image_batch[idx][:, :, 1]
            image_B = image_batch[idx][:, :, 2]
            if mode == 'avg':
                R = np.max(image_R) - np.min(image_R)
                G = np.max(image_G) - np.min(image_G)
                B = np.max(image_B) - np.min(image_B)
                data_out.append([R, G, B])
            elif mode == 'detailed':
                R = np.max(image_R, axis=1) - np.min(image_R, axis=1)
                G = np.max(image_G, axis=1) - np.min(image_G, axis=1)
                B = np.max(image_B, axis=1) - np.min(image_B, axis=1)
                data_out.append(np.stack((R, G, B), axis=1))
            elif mode == 'grid':
                color_mode_results = []
                for color_mode_image in [image_R, image_G, image_B]:
                    grids = self.split_image_grid(color_mode_image, grid_dim[0], grid_dim[1])
                    grids_contrast = []
                    for p in grids:
                        grids_contrast.append(np.max(p) - np.min(p))
                    color_mode_results.append(grids_contrast)
                color_mode_results = np.asarray(color_mode_results)
                data_out.append(np.stack((color_mode_results[0], color_mode_results[1],
                                          color_mode_results[2]), axis=1))
            else:
                sys.exit("Wrong parameter for data preprocessing")
        print("[INFO] Finished Contrast Computation")
        return np.asarray(data_out)


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
    #(X_train, Y_train), (X_test, Y_test) = salience_predictor.scale_data(X_train, Y_train, X_test, Y_test, labels='regression')

    contrast = feature_extractor.contrast(X_test, mode='detailed')
    print(contrast.shape)

    #style_output = feature_extractor.get_style_vgg19(X_test)
    #content_output = feature_extractor.get_content_vgg19(X_test)

    '''
    image = X_train[5]
    image = image.astype(np.uint8)
    #image = image.reshape(image.shape[0], image.shape[1])
    plt.imshow(image)
    plt.show()
    '''

