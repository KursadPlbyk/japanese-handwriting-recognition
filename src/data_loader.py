import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import TensorDataset, DataLoader


def load_hiragana(file_path):
    """
    Loads the Kaggle Hiragana images from the character subfolders, converts
    them to 28x28 grayscale images, and inverts the pixel values (white
    background/dark stroke -> black background/light stroke).

    Args:
        file_path (str): path to the root folder containing the 46 character subfolders

    Returns:
        images (np.array): all images, shape (N, 28, 28)
        labels (np.array): label (0-45) per image, shape (N,)
        class_names (list): sorted folder names, class_names[label] = character
    """

    images = []
    labels = []
    class_names = sorted(
        d for d in os.listdir(file_path)
        if os.path.isdir(os.path.join(file_path, d))
    )

    for label_index, class_name in enumerate(class_names):
        class_folder_path = os.path.join(file_path, class_name)

        for filename in os.listdir(class_folder_path):
            image_path = os.path.join(class_folder_path, filename)

            img = Image.open(image_path).convert("L")
            img = img.resize((28, 28))
            img_array = np.array(img)
            img_array = 255 - img_array

            images.append(img_array)
            labels.append(label_index)

    return np.array(images), np.array(labels), class_names


def load_kanji(file_path):
    """
    Loads the Kuzushiji-Kanji images from the Unicode subfolders and
    converts them to 28x28 grayscale images. No inversion needed,
    since the polarity (dark background/light stroke) is already correct.

    Args:
        file_path (str): path to the root folder containing the 49 kanji subfolders
                          (folder names are Unicode code points like "U+4E00")

    Returns:
        images (np.array): all images, shape (N, 28, 28)
        labels (np.array): label (0-48) per image, shape (N,)
        class_names (list): sorted folder names (Unicode code points)
    """

    images = []
    labels = []
    class_names = sorted(
        d for d in os.listdir(file_path)
        if os.path.isdir(os.path.join(file_path, d))
    )

    for label_index, class_name in enumerate(class_names):
        class_folder_path = os.path.join(file_path, class_name)

        for filename in os.listdir(class_folder_path):
            image_path = os.path.join(class_folder_path, filename)

            img = Image.open(image_path).convert("L")
            img = img.resize((28, 28))
            img_array = np.array(img)

            images.append(img_array)
            labels.append(label_index)

    return np.array(images), np.array(labels), class_names


def combine_datasets():
    """
    Loads the Hiragana and Kanji data, shifts the Kanji labels so they don't
    collide with the Hiragana labels, merges both datasets, and shuffles
    them in a shared random order.

    Args: none

    Returns:
        combined_X (np.array): all images (Hiragana + Kanji), shape (N, 28, 28)
        combined_y (np.array): labels 0-45 (Hiragana) and 46-94 (Kanji), shape (N,)
        all_classes (list): combined class names, same order as the labels
    """

    hiragana_X, hiragana_y, hiragana_classes = load_hiragana("data/kaggle_hiragana")
    kanji_X, kanji_y, kanji_classes = load_kanji("data/kuzushiji_kanji/kkanji2")
    kanji_y = kanji_y + len(hiragana_classes)
    combined_X = np.concatenate((hiragana_X, kanji_X))
    combined_y = np.concatenate((hiragana_y, kanji_y))
    np.random.seed(42)
    idx = np.random.permutation(len(combined_X))
    combined_X = combined_X[idx]
    combined_y = combined_y[idx]
    all_classes = hiragana_classes + kanji_classes

    return combined_X, combined_y, all_classes


def prepare_data(combined_X, combined_y, train_ratio=0.8):
    """
    Normalizes the pixel values to 0.0-1.0, adds the channel dimension
    required by PyTorch CNNs, and splits the data into train and test sets.

    Args:
        combined_X (np.array): images, shape (N, 28, 28), values 0-255
        combined_y (np.array): labels, shape (N,)
        train_ratio (float): fraction of data used for training (rest = test)

    Returns:
        X_train, y_train, X_test, y_test (np.array): split images/labels,
        X arrays have shape (N, 1, 28, 28), values 0.0-1.0
    """

    combined_X = combined_X / 255.0
    combined_X = combined_X.reshape(-1, 1, 28, 28)
    split_index = int(len(combined_X) * train_ratio)

    X_train, X_test = combined_X[:split_index], combined_X[split_index:]
    y_train, y_test = combined_y[:split_index], combined_y[split_index:]

    return X_train, y_train, X_test, y_test


def create_dataloaders(X_train, y_train, X_test, y_test, batch_size=64):
    """
    Converts the NumPy arrays to PyTorch tensors and builds DataLoaders
    from them that automatically serve batches during training.

    Args:
        X_train, y_train, X_test, y_test (np.array): from prepare_data()
        batch_size (int): number of images per batch

    Returns:
        train_loader (DataLoader): serves shuffled training batches
        test_loader (DataLoader): serves unshuffled test batches
    """

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader





if __name__ == "__main__":
    print("Loading hiragan datasets...\n")
    X, y, hiragana_classes = load_hiragana("data/kaggle_hiragana")
    print(X.shape)
    print(y.shape)
    print(hiragana_classes[:len(hiragana_classes)])

    print("----------------------------------------------------------------\n")

    print("Loading kanji datasets...\n")
    X_kanji, y_kanji, kanji_classes = load_kanji("data/kuzushiji_kanji/kkanji2")
    print(X_kanji.shape)
    print(y_kanji.shape)
    print(kanji_classes[:len(kanji_classes)])

    print("----------------------------------------------------------------\n")

    print("Combining datasets...\n")
    combined_X, combined_y, all_classes = combine_datasets()
    print(combined_X.shape)          # sollte ca. (44971, 28, 28) sein
    print(combined_y.shape)          # sollte ca. (44971,) sein
    print(all_classes[:len(all_classes)])  # sollte 95 sein
    print(len(set(combined_y)))      # sollte 95 sein

    print("----------------------------------------------------------------\n")

    print("Preparing data...\n")
    X_train, y_train, X_test, y_test = prepare_data(combined_X, combined_y)
    print(X_train.shape)   # expected: (35976, 1, 28, 28)
    print(y_train.shape)   # expected: (35976,)
    print(X_test.shape)    # expected: (8995, 1, 28, 28)
    print(y_test.shape)    # expected: (8995,)

    print("----------------------------------------------------------------\n")

    print("Creating dataloaders...\n")
    train_loader, test_loader = create_dataloaders(X_train, y_train, X_test, y_test)
    print(len(train_loader.dataset))   # expected: 35976
    print(len(test_loader.dataset))    # expected: 8995


