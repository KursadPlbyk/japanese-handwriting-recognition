import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import sys
sys.path.append("src")
from model import CNN
from data_loader import combine_datasets, prepare_data, create_dataloaders
from predictor import class_to_char



def get_predictions(model, test_loader, device):
    """
    Runs the whole test set through the model and collects every
    prediction alongside its true label, for later metrics like
    classification_report and confusion_matrix.

    Args:
        model (CNN): trained model
        test_loader (DataLoader): serves the held-out test batches
        device (torch.device): where the model/data live (cpu/mps/cuda)

    Returns:
        predictions (np.array): predicted class index per test image, shape (N,)
        labels (np.array): true class index per test image, shape (N,)
    """

    model.eval() 
    with torch.no_grad(): 
        all_predictions = []
        all_labels = []
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)             # forward pass -> raw scores (batch, 95)
            _, predicted = torch.max(outputs, 1)  # highest score per image -> predicted class

            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return np.array(all_predictions), np.array(all_labels)



def get_misclassified_examples(model, test_loader, device):
    """
    Scans the test set batch by batch and collects the first few images
    the model got wrong, together with their true and predicted labels,
    for a visual error analysis (e.g. a grid of misclassified examples).

    Args:
        model (CNN): trained model
        test_loader (DataLoader): serves the held-out test batches
        device (torch.device): where the model/data live (cpu/mps/cuda)

    Returns:
        wrong_images (list of np.array): misclassified images, each shape (1, 28, 28)
        true_labels (list of int): the correct class index for each image
        predicted_labels (list of int): what the model predicted instead
    """

    num_examples = 10
    model.eval()
    with torch.no_grad():
        wrong_images = []
        true_labels = []
        predicted_labels = []

        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            # Boolean mask: True where the prediction was wrong for that image.
            mismatch = predicted != labels

            # Boolean indexing: images[mismatch] keeps only the images where
            # mismatch is True, discarding every correctly classified one.
            wrong_images.extend(images[mismatch].cpu().numpy())
            true_labels.extend(labels[mismatch].cpu().numpy())
            predicted_labels.extend(predicted[mismatch].cpu().numpy())

            # Stop early once we have enough examples, no need to scan
            # the whole test set for just 10 images.
            if len(wrong_images) >= num_examples:
                break

    return wrong_images[:num_examples], true_labels[:num_examples], predicted_labels[:num_examples]






if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    state_dict = torch.load("models/cnn_handwriting_model.pth", map_location="cpu")
    model = CNN().to(device)
    model.load_state_dict(state_dict)

    combined_X, combined_y, all_classes = combine_datasets()
    X_train, y_train, X_test, y_test = prepare_data(combined_X, combined_y)
    train_loader, test_loader = create_dataloaders(X_train, y_train, X_test, y_test)

    predictions, labels = get_predictions(model, test_loader, device)

    print(classification_report(labels, predictions, target_names=all_classes))

    cm = confusion_matrix(labels, predictions)
    print("Confusion Matrix Shape:", cm.shape)

    # Volle 95x95 Matrix -- einzelne Labels nicht lesbar, aber zeigt das
    # Gesamtmuster (Diagonale = richtige Treffer)
    plt.figure(figsize=(10, 10))
    plt.imshow(cm, cmap="Blues")
    plt.xlabel("Vorhergesagte Klasse")
    plt.ylabel("Echte Klasse")
    plt.title("Confusion Matrix (alle 95 Klassen)")
    plt.colorbar()
    plt.savefig("results/confusion_matrix_full.png")
    plt.close()

    # Kleinere, lesbare Teilmenge: die schwaechsten Kanji + ihre haeufigsten
    # Verwechslungspartner von vorhin
    plt.rcParams["font.family"] = "Hiragino Sans"  # damit Kanji in den Achsen lesbar sind

    interessante_zeichen = ["U+5019", "U+65E5", "U+4E4B", "U+4E5F", "U+4E91", "U+4E94", "U+5973"]
    indices = [all_classes.index(name) for name in interessante_zeichen]

    cm_subset = cm[np.ix_(indices, indices)]
    labels_lesbar = [chr(int(name[2:], 16)) for name in interessante_zeichen]

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(cm_subset, cmap="Blues")

    # Anzahl in jede Zelle schreiben, Textfarbe je nach Zell-Helligkeit
    max_value = cm_subset.max()
    for row in range(cm_subset.shape[0]):
        for col in range(cm_subset.shape[1]):
            value = cm_subset[row, col]
            text_color = "white" if value > max_value / 2 else "black"
            ax.text(col, row, str(value), ha="center", va="center", color=text_color)

    ax.set_xticks(range(len(labels_lesbar)))
    ax.set_yticks(range(len(labels_lesbar)))
    ax.set_xticklabels(labels_lesbar)
    ax.set_yticklabels(labels_lesbar)
    ax.set_xlabel("Vorhergesagte Klasse")
    ax.set_ylabel("Echte Klasse")
    ax.set_title("Confusion Matrix (ausgewählte Zeichen)")
    plt.savefig("results/confusion_matrix_subset.png")
    plt.close()

    # Confusion Matrix nur fuer die 46 Hiragana-Klassen (Index 0-45)
    hiragana_indices = list(range(46))
    cm_hiragana = cm[np.ix_(hiragana_indices, hiragana_indices)]
    hiragana_labels = all_classes[:46]  # Romaji-Namen, z.B. "aa", "chi"

    fig, ax = plt.subplots(figsize=(14, 14))
    ax.imshow(cm_hiragana, cmap="Blues")

    max_value_hiragana = cm_hiragana.max()
    for row in range(cm_hiragana.shape[0]):
        for col in range(cm_hiragana.shape[1]):
            value = cm_hiragana[row, col]
            if value == 0:
                continue  # Nullen weglassen, sonst zu voll (46x46 = 2116 Zellen)
            text_color = "white" if value > max_value_hiragana / 2 else "black"
            ax.text(col, row, str(value), ha="center", va="center", color=text_color, fontsize=7)

    ax.set_xticks(range(46))
    ax.set_yticks(range(46))
    ax.set_xticklabels(hiragana_labels, fontsize=7, rotation=90)
    ax.set_yticklabels(hiragana_labels, fontsize=7)
    ax.set_xlabel("Vorhergesagte Klasse")
    ax.set_ylabel("Echte Klasse")
    ax.set_title("Confusion Matrix (Hiragana)")
    plt.tight_layout()
    plt.savefig("results/confusion_matrix_hiragana.png")
    plt.close()

    # Fehleranalyse: 10 falsch klassifizierte Beispiele als Grid
    wrong_images, true_labels, predicted_labels = get_misclassified_examples(model, test_loader, device)

    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    for ax, image, true_label, predicted_label in zip(axes.flat, wrong_images, true_labels, predicted_labels):
        ax.imshow(image[0], cmap="gray")  # image shape (1, 28, 28) -> [0] fuer das eigentliche 28x28 Bild
        true_char = class_to_char(all_classes[true_label])
        predicted_char = class_to_char(all_classes[predicted_label])
        ax.set_title(f"Echt: {true_char}\nVorhersage: {predicted_char}", fontsize=10)
        ax.axis("off")

    plt.suptitle("Falsch klassifizierte Beispiele")
    plt.tight_layout()
    plt.savefig("results/misclassified_examples.png")
    plt.close()

    # Confidence distribution: how sure was the model across ALL test predictions,
    # not just whether it was right or wrong
    model.eval()
    confidences = []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            probabilities = torch.softmax(outputs, dim=1)
            batch_confidences, _ = torch.max(probabilities, 1)
            confidences.extend(batch_confidences.cpu().numpy())

    plt.hist(confidences, bins=20, edgecolor="white", linewidth=0.8)
    plt.yscale("log")  # linear scale hides the small bins next to the ~8700 peak
    plt.xlabel("Confidence der Top-Vorhersage")
    plt.ylabel("Anzahl Testbilder (log-Skala)")
    plt.title("Verteilung der Vorhersage-Konfidenz")
    plt.savefig("results/confidence_distribution.png")
    plt.close()
