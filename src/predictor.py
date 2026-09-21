import torch
from PIL import Image
import numpy as np
import sys
import random
import matplotlib.pyplot as plt
sys.path.append("src")
from model import CNN


def class_to_char(class_name):
    """
    Converts a class name to a displayable character 
    (kanji unicode codes -> the actual character, hiragana romaji names stay as-is).
    """

    if class_name.startswith("U+"):
        return chr(int(class_name[2:], 16))
    return class_name

def load_model(model_path, device):
    """
    Loads the trained CNN weights from disk into a fresh model instance,
    ready for prediction.

    Args:
        model_path (str): path to the saved .pth file
        device (torch.device): where to put the model (cpu/mps/cuda)

    Returns:
        CNN: the trained model, in evaluation mode
    """

    model = CNN().to(device)  # fresh model, still random weights at this point
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)  # overwrite random weights with the trained ones
    model.eval()  # inference-only, never trained further from here

    return model


def load_class_names(path):
    """
    Reads the saved class name list back from disk (one name per line,
    same order as the label indices 0-94 used during training).

    Args:
        path (str): path to the class_names.txt file

    Returns:
        list of str: class names, class_names[i] = character for label i
    """

    with open(path) as f:
        class_names = [line.strip() for line in f]  # strip() removes the trailing "\n"

    return class_names


def preprocess_img(img_path):
    """
    Loads an arbitrary image file and converts it into the exact tensor
    format the model expects (28x28 grayscale, dark background/light
    stroke, values 0.0-1.0, shape (1, 1, 28, 28)).

    Polarity (background/stroke color) is detected automatically instead
    of assumed: a mostly-bright image (mean > 127) is treated as a photo
    of dark ink on light paper and gets inverted; a mostly-dark image is
    left as-is. This works because in a single-character image the
    background always covers most of the pixels.

    Args:
        img_path (str): path to any image file

    Returns:
        Tensor: preprocessed image, shape (1, 1, 28, 28)
    """

    img = Image.open(img_path).convert("L")
    img = img.resize((28, 28))
    img_array = np.array(img)

    if img_array.mean() > 127:
        img_array = 255 - img_array

    img_array = img_array / 255.0
    img_tensor = torch.tensor(img_array, dtype=torch.float32)
    img_tensor = img_tensor.unsqueeze(0).unsqueeze(0)  # add batch + channel dims

    return img_tensor


def predict(model, img_tensor, class_names, device):
    """
    Runs one preprocessed image through the model and returns the
    predicted character plus a real confidence percentage.

    Args:
        model (CNN): trained model, in eval mode
        img_tensor (Tensor): output of preprocess_img(), shape (1, 1, 28, 28)
        class_names (list of str): from load_class_names()
        device (torch.device): where the model lives (cpu/mps/cuda)

    Returns:
        predicted_class (str): the predicted character/class name
        confidence (float): probability of that class, between 0.0 and 1.0
    """

    img_tensor = img_tensor.to(device)

    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted_idx = torch.max(probabilities, 1)

    predicted_class = class_names[predicted_idx.item()]

    return predicted_class, confidence.item()


if __name__ == "__main__":
    plt.rcParams["font.family"] = "Hiragino Sans"  # so kanji/hiragana render in the plot titles

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    model = load_model("models/cnn_handwriting_model.pth", device)
    class_names = load_class_names("models/class_names.txt")

    # Important: pick samples from the actual held-out TEST set (X_test),
    # not random files from the raw data folders. Those folders contain
    # both training and test images mixed together, so a random file
    # would very likely be an image the model already saw while training.
    from data_loader import combine_datasets, prepare_data

    combined_X, combined_y, all_classes = combine_datasets()
    _, _, X_test, y_test = prepare_data(combined_X, combined_y)

    sample_indices = random.sample(range(len(X_test)), 10)

    fig, axes = plt.subplots(2, 5, figsize=(14, 6))
    for ax, idx in zip(axes.flat, sample_indices):
        img_tensor = torch.tensor(X_test[idx], dtype=torch.float32).unsqueeze(0)  # (1,28,28) -> (1,1,28,28)
        predicted_class, confidence = predict(model, img_tensor, class_names, device)

        ax.imshow(X_test[idx][0], cmap="gray")
        true_char = class_to_char(all_classes[y_test[idx]])
        predicted_char = class_to_char(predicted_class)
        ax.set_title(f"Echt: {true_char}\nVorhersage: {predicted_char} ({confidence:.1%})", fontsize=10)
        ax.axis("off")

    plt.suptitle("Beispiel-Vorhersagen")
    plt.tight_layout()
    plt.savefig("results/sample_predictions.png")
    plt.close()
