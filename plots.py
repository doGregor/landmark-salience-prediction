import cv2
import numpy as np
import os


def plot_multiple_lm_images(rows, columns, save_image=False):
    print("[INFO] preparing image with " + str(rows) + " rows and " + str(columns) + " cols")
    image_direcrtory = "LM_Images_downscaled"
    no_images = int(rows) * int(columns)
    images_available = os.listdir(image_direcrtory)
    no_images_available = len(images_available)
    images_to_select = np.random.randint(0, no_images_available, no_images)
    selected_images = [images_available[i] for i in images_to_select]
    selected_images = np.reshape(selected_images, (rows, columns))

    image_rows = []
    for idx_row in range(int(rows)):
        for idx_col in range(int(columns)):
            path = image_direcrtory + "/" + selected_images[idx_row, idx_col]
            image = cv2.imread(path)
            if idx_col == 0:
                image_rows.append(image)
            else:
                image_rows[idx_row] = np.hstack((image_rows[idx_row], image))

    final_image = []
    for idx, image_row in enumerate(image_rows):
        if idx == 0:
            final_image.append(image_rows[0])
        else:
            final_image[0] = np.vstack((final_image[0], image_rows[idx]))

    if save_image:
        im_path = "output_images/" + "plot_" + "rows" + str(rows) + "_cols" + str(columns) + ".png"
        image_to_save = final_image[0]
        scale_percent = 0.25
        width = int(image_to_save.shape[1] * scale_percent)
        height = int(image_to_save.shape[0] * scale_percent)
        dim = (width, height)
        resized = cv2.resize(image_to_save, dim)
        cv2.imwrite(im_path, resized)
    else:
        cv2.imshow('Images', final_image[0])
        cv2.waitKey()


if __name__ == '__main__':
    plot_multiple_lm_images(rows=4, columns=10, save_image=True)
