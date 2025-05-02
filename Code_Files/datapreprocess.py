import kagglehub

# Download latest version
path = kagglehub.dataset_download("msambare/fer2013")

print("Path to dataset files:", path)

TRAIN_DIR = "/kaggle/input/fer2013/train/"
TEST_DIR = "/kaggle/input/fer2013/test/"

def load_image(directory):
    image_paths = []
    labels = []
    for label in os.listdir(directory):
        for filename in os.listdir(directory+label):
            image_path = os.path.join(directory, label , filename)
            image_paths.append(image_path)
            labels.append(label)

        print(label, "completed")

    return image_paths , labels

train = pd.DataFrame()
train['image'],train['label'] = load_image(TRAIN_DIR)

train = train.sample(frac=1).reset_index(drop = True)
train.head()

# Save the DataFrame to a CSV file in Kaggle
train.to_csv('/content/train_data.csv', index=False)

print("Train DataFrame saved to train_data.csv in Kaggle's working directory")

test = pd.DataFrame()
test['image'],test['label'] = load_image(TEST_DIR)

test.head()

# Save the DataFrame to a CSV file in Kaggle
train.to_csv('/content/test_data.csv', index=False)

print("Train DataFrame saved to test_data.csv in Kaggle's working directory")

sns.countplot(x='label', data=train)

from PIL import Image

plt.figure(figsize=(25,25))
files = train.iloc[0:25]

for index , file , label in files.itertuples():
    plt.subplot(5,5,index+1)
    img = load_img(file)
    img = np.array(img)
    plt.imshow(img)
    plt.title(label)
    plt.axis('off')


def extractfeatures(images):
    features = []
    for image in tqdm(images):
        img = load_img(image, target_size=(48, 48), color_mode='grayscale')  # Load as grayscale
        img = np.array(img)
        img = np.stack((img,) * 3, axis=-1)  # Convert grayscale to RGB by repeating the channel
        features.append(img)
    features = np.array(features)
    return features

# Load and preprocess the training and test images
train_features = extractfeatures(train['image'])
test_features = extractfeatures(test['image'])

x_train = train_features / 255.0
x_test = test_features / 255.0

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
le.fit(train['label'])
y_train = le.transform(train["label"])
y_test = le.transform(test['label'])

y_train = to_categorical(y_train, num_classes = 7)
y_test = to_categorical(y_test, num_classes = 7)

# Define the save path
import numpy as np

# Define the save path
save_path = '/content/processed_data.npz'

# Save the data in compressed format
np.savez_compressed(save_path, x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test)
print(f"Data saved to: {save_path}")

