from scipy import stats
import numpy as np
import os
import pandas as pd
import tensorflow.keras as keras
import pickle


class TrainTestData():
    def __init__(self):
        self.lm_images_df = pd.read_csv(
            "BA_Richter/download_2020-10-14_08-33-39/2_Datensätze/Landmarken mit Faktorenwerten.csv",
            sep=";",
            decimal=",")
        self.lm_images_source = "LM_images_downscaled"
        self.id_column = self.lm_images_df["ID"]
        self.salience_column = self.lm_images_df["Salienz_gerundet"]
        try:
            self.train_test_split = self.open_train_test_pickle()
        except:
            self.train_test_split = {}

    def open_train_test_pickle(self):
        with open(r"train_test_data/train_test_split.pickle", "rb") as input_file:
            train_test_split = pickle.load(input_file)
        return train_test_split

    def split_images(self, train=0.7):
        """
        Only call this function to initially create train/test split (IF NOT YET AVAILABLE!!!)
        :param train: fraction of train data
        :return: returns nothing
        """
        image_data = []
        salience_data = []
        binary_salience_data = []

        for idx, image_id in enumerate(self.id_column):
            image_numbers = []
            for value_images in range(1, 4):
                image_number = str(image_id) + "." + str(value_images) + ".jpeg"
                image_numbers.append(image_number)

            all_in = True
            files_available = os.listdir(self.lm_images_source)
            for image_number in image_numbers:
                if image_number in files_available:
                    pass
                else:
                    all_in = False

            if all_in:
                for im_no in image_numbers:
                    image_data.append(im_no)
                    salience_data.append(self.salience_column[idx])
                    if self.salience_column[idx] < 2.8:
                        binary_salience_data.append(0)
                    else:
                        binary_salience_data.append(1)

        num_data = len(image_data)
        max_train = int(num_data*train)
        indices = np.arange(int(num_data))
        np.random.shuffle(indices)
        train_split = indices[:max_train]
        test_split = indices[max_train:]

        train_images = [image_data[i] for i in train_split]
        test_images = [image_data[i] for i in test_split]
        train_salience = [salience_data[i] for i in train_split]
        test_salience = [salience_data[i] for i in test_split]
        train_binary = [binary_salience_data[i] for i in train_split]
        test_binary = [binary_salience_data[i] for i in test_split]

        self.train_test_split["train_images"] = train_images
        self.train_test_split["test_images"] = test_images
        self.train_test_split["train_salience"] = train_salience
        self.train_test_split["test_salience"] = test_salience
        self.train_test_split["train_binary"] = train_binary
        self.train_test_split["test_binary"] = test_binary

        with open(r"train_test_data/train_test_split.pickle", "wb") as output_file:
            pickle.dump(self.train_test_split, output_file)

    def get_train_test_salience(self, im_target_size=(298, 224), gray=False):
        train_images = self.train_test_split["train_images"]
        test_images = self.train_test_split["test_images"]
        train_salience = self.train_test_split["train_salience"]
        test_salience = self.train_test_split["test_salience"]

        train_images_out = []
        test_images_out = []

        for im_no in train_images:
            im_path = self.lm_images_source + "/" + im_no
            if gray:
                image = keras.preprocessing.image.load_img(im_path, target_size=im_target_size,
                                                           color_mode='grayscale')
            else:
                image = keras.preprocessing.image.load_img(im_path, target_size=im_target_size)
            train_images_out.append(keras.preprocessing.image.img_to_array(image))

        for im_no in test_images:
            im_path = self.lm_images_source + "/" + im_no
            if gray:
                image = keras.preprocessing.image.load_img(im_path, target_size=im_target_size,
                                                           color_mode='grayscale')
            else:
                image = keras.preprocessing.image.load_img(im_path, target_size=im_target_size)
            test_images_out.append(keras.preprocessing.image.img_to_array(image))

        return (np.asarray(train_images_out), np.asarray(train_salience)),\
               (np.asarray(test_images_out), np.asarray(test_salience))

    def get_train_test_binary(self, im_target_size=(298, 224), gray=False):
        train_images = self.train_test_split["train_images"]
        test_images = self.train_test_split["test_images"]
        train_binary = self.train_test_split["train_binary"]
        test_binary = self.train_test_split["test_binary"]

        train_images_out = []
        test_images_out = []

        for im_no in train_images:
            im_path = self.lm_images_source + "/" + im_no
            if gray:
                image = keras.preprocessing.image.load_img(im_path, target_size=im_target_size,
                                                           color_mode='grayscale')
            else:
                image = keras.preprocessing.image.load_img(im_path, target_size=im_target_size)
            train_images_out.append(keras.preprocessing.image.img_to_array(image))

        for im_no in test_images:
            im_path = self.lm_images_source + "/" + im_no
            if gray:
                image = keras.preprocessing.image.load_img(im_path, target_size=im_target_size,
                                                           color_mode='grayscale')
            else:
                image = keras.preprocessing.image.load_img(im_path, target_size=im_target_size)
            test_images_out.append(keras.preprocessing.image.img_to_array(image))

        return (np.asarray(train_images_out), np.asarray(train_binary)),\
               (np.asarray(test_images_out), np.asarray(test_binary))


if __name__ == '__main__':
    data_class = TrainTestData()
    (X_train, Y_train), (X_test, Y_test) = data_class.get_train_test_salience()
