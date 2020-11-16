import pandas as pd
import numpy as np
import os

DEBUG_INFO = False

lm_images_df = pd.read_csv("BA_Richter/download_2020-10-14_08-33-39/2_Datensätze/Landmarken mit Faktorenwerten.csv",
                           sep=";",
                           decimal=",")
id_column = lm_images_df["ID"]
salience_column = lm_images_df["Salienz_gerundet"]

salience_w_image = []
salience_wo_image = []
image_names_with_salience_score = []

salience_scores = []
salience_score_in_image_amount = []

for idx, value in enumerate(id_column):
    scene = int(value.split(".")[0])
    if 2 <= scene <= 6:
        folder = "LM_Szenen_2_bis_6"
    if 7 <= scene <= 11:
        folder = "LM_Szenen_7_bis_11"
    if 12 <= scene <= 16:
        folder = "LM_Szenen_12_bis_16"
    if 17 <= scene <= 21:
        folder = "LM_Szenen_17_bis_21"

    image_numbers = []
    for value_images in range(1,4):
        image_number = str(value) + "." + str(value_images) + ".jpeg"
        image_numbers.append(image_number)

    all_in = True
    files_available = os.listdir(folder)
    for image_number in image_numbers:
        if image_number in files_available:
            pass
        else:
            all_in = False

    if all_in:
        salience_w_image.append(value)
        salience_scores.append(salience_column[idx])
        for i in range(3):
            salience_score_in_image_amount.append(salience_column[idx])
        for im_no in image_numbers:
            image_names_with_salience_score.append(im_no)
    else:
        salience_wo_image.append(value)

if DEBUG_INFO:
    print("Number of salience with images:", len(salience_w_image))
    print("Salience scores:", len(salience_scores))

    print("No images:", len(image_names_with_salience_score))
    print("No salience scores:", len(salience_score_in_image_amount))


def get_data_and_labels():
 return image_names_with_salience_score, salience_score_in_image_amount
