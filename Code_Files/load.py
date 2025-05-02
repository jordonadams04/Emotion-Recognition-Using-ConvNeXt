import numpy as np
import cv2

data = np.load('/content/processed_data.npz')
x_train = data['x_train']  # shape: (28709, 48, 48, 3)
x_test = data['x_test']
y_train = data['y_train']
y_test = data['y_test']

# Upsampling
def resize_images(images, target_size=(128, 128)):
    return np.array([cv2.resize(img, target_size) for img in images])

print("Resizing training and test images to 128x128...")
x_train_resized = resize_images(x_train)
x_test_resized = resize_images(x_test)

save_path = '/content/processed_data_128.npz'
np.savez_compressed(save_path, x_train=x_train_resized, x_test=x_test_resized, y_train=y_train, y_test=y_test)
print(f"Saved resized data to {save_path}")

# Define the load path
load_path = '/content/processed_data_128.npz'

# Load the data
data = np.load(load_path)

# Access individual arrays
x_train = data['x_train']
x_test = data['x_test']
y_train = data['y_train']
y_test = data['y_test']

print("Data loaded successfully!")