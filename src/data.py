import os
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split


def load_data(data_dir: str, img_size=(64, 64), test_size=0.2):
    """Load images and split into train/test sets.

    Parameters
    ----------
    data_dir : str
        Root directory containing class subfolders.
    img_size : tuple
        Desired image size as (width, height).
    test_size : float
        Proportion of data to use for testing.

    Returns
    -------
    X_train, X_test, y_train, y_test : np.ndarray
        Arrays ready for model training and evaluation.
    """
    images = []
    labels = []
    class_names = sorted([d for d in os.listdir(data_dir)
                          if os.path.isdir(os.path.join(data_dir, d))])
    class_to_idx = {cls: idx for idx, cls in enumerate(class_names)}

    for cls in class_names:
        cls_dir = os.path.join(data_dir, cls)
        for fname in os.listdir(cls_dir):
            if fname.lower().endswith('.png'):
                path = os.path.join(cls_dir, fname)
                img = load_img(path, target_size=img_size)
                arr = img_to_array(img) / 255.0
                images.append(arr)
                labels.append(class_to_idx[cls])

    X = np.array(images, dtype='float32')
    y = to_categorical(np.array(labels), num_classes=len(class_names))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=labels, random_state=42
    )
    return X_train, X_test, y_train, y_test
