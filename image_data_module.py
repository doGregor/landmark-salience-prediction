from scipy import stats
import numpy as np
import os
import pandas as pd
import tensorflow.keras as keras
import pickle
from pathlib import Path
import sys
from random import shuffle
import itertools


class TrainTestData():
    def __init__(self):
        self.lm_images_df = pd.read_csv(
            "salience_csv/Landmarken_mit_Faktorenwerten.csv",
            sep=";",
            decimal=",")
        self.lm_images_source = "LM_Images_downscaled"
        self.id_column = self.lm_images_df["ID"]
        self.salience_column = self.lm_images_df["Salienz_gerundet"]

    def open_train_test_pickle(self, cv_name="0"):
        file_path = "train_test_data/cv_" + str(cv_name) + ".pickle"
        if Path(file_path).is_file():
            with open(file_path, "rb") as input_file:
                train_test_split = pickle.load(input_file)
            return train_test_split
        else:
            sys.exit("Selected cross validation split does not exist")

    def cv_split_images(self, folds=5, salience_threshold=2.8):
        """
        Call this function to create x-fold train/test cross-validation splits.
        :param folds: number of cross-validation folds
        :param salience_threshold: Salience value at which samples should be split in (1)/(0) for binary labels
        :return: returns nothing
        """
        image_data = []

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
                image_data.append(image_id)

        fold_size = int(len(image_data)/folds)
        shuffle(image_data)

        splits = []
        for idx_set in range(0, folds):
            idx_0 = idx_set * fold_size
            if idx_set == folds - 1:
                splits.append(image_data[idx_0:])
            else:
                idx_1 = (idx_set + 1) * fold_size
                splits.append(image_data[idx_0:idx_1])

        for idx_split, split in enumerate(splits):
            train_test_split = {}
            test = split
            train = [x for i, x in enumerate(splits) if i != idx_split]
            train = list(itertools.chain.from_iterable(train))

            train_images = []
            test_images = []
            train_salience = []
            test_salience = []
            train_binary = []
            test_binary = []
            for im_no in test:
                for val_im in range(1, 4):
                    image_number = str(im_no) + "." + str(val_im) + ".jpeg"
                    test_images.append(image_number)
            shuffle(test_images)

            for im_no in train:
                for val_im in range(1, 4):
                    image_number = str(im_no) + "." + str(val_im) + ".jpeg"
                    train_images.append(image_number)
            shuffle(train_images)

            for im_no in train_images:
                salience_idx = '.'.join(im_no.split('.')[:2])
                salience_key = np.where(self.id_column == salience_idx)[0]
                train_salience.append(float(self.salience_column[salience_key]))
                if float(self.salience_column[salience_key]) <= salience_threshold:
                    train_binary.append(0)
                else:
                    train_binary.append(1)

            for im_no in test_images:
                salience_idx = '.'.join(im_no.split('.')[:2])
                salience_key = np.where(self.id_column == salience_idx)[0]
                test_salience.append(float(self.salience_column[salience_key]))
                if float(self.salience_column[salience_key]) <= salience_threshold:
                    test_binary.append(0)
                else:
                    test_binary.append(1)

            train_test_split["train_images"] = train_images
            train_test_split["test_images"] = test_images
            train_test_split["train_salience"] = train_salience
            train_test_split["test_salience"] = test_salience
            train_test_split["train_binary"] = train_binary
            train_test_split["test_binary"] = test_binary

            file_path = "train_test_data/cv_" + str(idx_split) + ".pickle"
            print("[INFO] Created split " + str(idx_split) + " at:", file_path)
            with open(file_path, "wb") as output_file:
                pickle.dump(train_test_split, output_file)

    def get_train_test_salience(self, cv_name="0", im_target_size=(298, 224), gray=False):
        """
        Load image and regression salience label data for train/test of specific data split.
        :param cv_name: name of cross validation split (same as chosen in split_images())
        :param im_target_size: size of the images when loading them
        :param gray: If images should be loaded gray-scales (True) or in RGB (False)
        :return: Returns two tuples where tuple0 = train and tuple1 = test; each in (data, labels[float])
        """
        train_test_split = self.open_train_test_pickle(cv_name=cv_name)
        train_images = train_test_split["train_images"]
        test_images = train_test_split["test_images"]
        train_salience = train_test_split["train_salience"]
        test_salience = train_test_split["test_salience"]

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

    def get_train_test_binary(self, cv_name="0", im_target_size=(298, 224), gray=False):
        """
        Load image and binary label data for train/test of specific data split.
        :param cv_name: name of cross validation split (same as chosen in split_images())
        :param im_target_size: size of the images when loading them
        :param gray: If images should be loaded gray-scales (True) or in RGB (False)
        :return: Returns two tuples where tuple0 = train and tuple1 = test; each in (data, labels[binary])
        """
        train_test_split = self.open_train_test_pickle(cv_name=cv_name)
        train_images = train_test_split["train_images"]
        test_images = train_test_split["test_images"]
        train_binary = train_test_split["train_binary"]
        test_binary = train_test_split["test_binary"]

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

    def get_raw_data(self, im_target_size=(298, 224), gray=False):
        train_test_split = self.open_train_test_pickle(cv_name="0")
        image_ids = train_test_split["train_images"] + train_test_split["test_images"]
        binary_labels = train_test_split["train_binary"] + train_test_split["test_binary"]
        salience_labels = train_test_split["train_salience"] + train_test_split["test_salience"]

        images = []

        for im_no in image_ids:
            im_path = self.lm_images_source + "/" + im_no
            if gray:
                image = keras.preprocessing.image.load_img(im_path, target_size=im_target_size,
                                                           color_mode='grayscale')
            else:
                image = keras.preprocessing.image.load_img(im_path, target_size=im_target_size)
            images.append(keras.preprocessing.image.img_to_array(image))

        output_dict = {}
        output_dict['image_ids'] = image_ids
        output_dict['images'] = images
        output_dict['salience'] = salience_labels
        output_dict['binary'] = binary_labels

        return output_dict


if __name__ == '__main__':
    data_class = TrainTestData()
    (X_train, Y_train), (X_test, Y_test) = data_class.get_train_test_salience()
