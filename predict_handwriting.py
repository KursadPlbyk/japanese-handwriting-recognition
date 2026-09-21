import sys
sys.path.append("src")
import torch
from predictor import load_model, load_class_names, preprocess_img, predict, class_to_char

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

model = load_model("models/cnn_handwriting_model.pth", device)
class_names = load_class_names("models/class_names.txt")

image_path = input("Gib den Pfad zu einem Bild ein: ")

img_tensor = preprocess_img(image_path)

predicted_class, confidence = predict(model, img_tensor, class_names, device)

predicted_char = class_to_char(predicted_class)

print("=" * 40)
print("HANDWRITING PREDICTION")
print("=" * 40)
print(f"Bild: {image_path}")
print(f"Vorhergesagtes Zeichen: {predicted_char} ({predicted_class})")
print(f"Konfidenz: {confidence:.2%}")
print("=" * 40)
