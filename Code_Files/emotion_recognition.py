# === Test on Custom Image ===
from PIL import Image
import matplotlib.pyplot as plt

# Replace with your actual filename
custom_image_path = "my_face.jpg"

# Load and preprocess
custom_img = Image.open(custom_image_path).convert("RGB")
custom_tensor = transform(custom_img).unsqueeze(0).to(DEVICE)

# Inference
model.eval()
with torch.no_grad():
    output = model(custom_tensor)
    pred_idx = torch.argmax(output, dim=1).item()

# Show result
label_map = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
plt.imshow(custom_img)
plt.title(f"Predicted Emotion: {label_map[pred_idx]}")
plt.axis('off')
plt.show()
