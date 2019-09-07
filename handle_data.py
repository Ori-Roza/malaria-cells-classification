import os

import cv2
from keras_preprocessing.image import img_to_array, load_img


def convert_dataset(directory_path):
    data = []
    labels = []
    # loop over the input images
    for image_dir in os.listdir(directory_path):
        potential_dir = os.path.join(directory_path, image_dir)
        if not os.path.isdir(potential_dir):
            continue
        for image_name in os.listdir(potential_dir)[:1000]:
            image_path = os.path.join(directory_path, image_dir, image_name)
            # load the image, pre-process it, and store it in the data list
            try:
                image = cv2.imread(image_path)
                image = img_to_array(image)
                data.append(image)
                # extract the class label from the image path and update the
                # labels list
                label = 1 if image_dir == "Parasitized" else 0
                labels.append(label)
            except:
                continue
    return data, labels
