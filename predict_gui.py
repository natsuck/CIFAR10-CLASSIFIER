import torch
import torchvision.transforms as transforms
from PIL import Image, ImageTk
from models.cnn import SimpleCNN
import tkinter as tk
from tkinter import filedialog, Label, Button

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# CIFAR-10 class labels
classes = ['airplane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck']

# Load model
model = SimpleCNN().to(device)
model.load_state_dict(torch.load("outputs/best_model.pth", map_location=device))
model.eval()

# Preprocess image
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((32,32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
    ])
    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0).to(device), Image.open(image_path)

# Predict function
def predict_image(image_path):
    tensor, img = preprocess_image(image_path)
    with torch.no_grad():
        outputs = model(tensor)
        _, predicted = torch.max(outputs, 1)
    return classes[predicted.item()], img

def browse_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        predicted_class, img = predict_image(file_path)
        img = img.resize((256,256))
        img_tk = ImageTk.PhotoImage(img)
        image_label.config(image=img_tk)
        image_label.image = img_tk
        result_label.config(text=f"Predicted Class: {predicted_class}")

root = tk.Tk()
root.title("CIFAR-10 Image Classifier")

image_label = Label(root)
image_label.pack(pady=10)

result_label = Label(root, text="Predicted Class: ", font=("Helvetica", 16))
result_label.pack(pady=10)

browse_btn = Button(root, text="Select Image", command=browse_image)
browse_btn.pack(pady=10)

root.mainloop()
