import os
import cv2
import requests
import numpy as np
import pandas as pd
from PIL import Image
from io import BytesIO
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

train_images_dir = "/content/train_images"
test_images_dir = "/content/test_images"
os.makedirs(train_images_dir, exist_ok=True)
os.makedirs(test_images_dir, exist_ok=True)

train_data = pd.read_csv("/content/train.csv")
test_data = pd.read_csv("/content/test.csv")

def download_image(url, save_path):
    """Download an image from a URL and save it locally."""
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            image = Image.open(BytesIO(response.content))
            image.save(save_path)
            return True
        else:
            print(f"Failed to download image from {url} with status code {response.status_code}")
            return False
    except Exception as e:
        print(f"Error downloading image from {url}: {e}")
        return False

def preprocess_image(image_path):
    """Preprocess an image for the CNN model."""
    try:
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Failed to load image at path {image_path}.")
            return None
        image = cv2.resize(image, (224, 224))  
        image = image / 255.0  
        return image
    except Exception as e:
        print(f"Error processing image at path {image_path}: {e}")
        return None

train_image_paths = []
for idx, row in train_data.iterrows():
    image_url = row['image_link']
    image_path = os.path.join(train_images_dir, f"{idx}.jpg")
    if download_image(image_url, image_path):
        train_image_paths.append(image_path)

image_data = [preprocess_image(path) for path in train_image_paths]
image_data = [img for img in image_data if img is not None] 

def extract_numeric(value):
    try:
        cleaned_value = str(value).replace('[', '').replace(']', '').replace(',', '').strip()
        number = float(cleaned_value.split()[0])  
        return number
    except (ValueError, IndexError) as e:
        print(f"Error processing value: {value} -> {e}")
        return np.nan

train_data['numeric_value'] = train_data['entity_value'].apply(extract_numeric)

valid_mask = train_data['numeric_value'].notna()

valid_indices = train_data.index[valid_mask].tolist()
train_data_filtered = train_data.loc[valid_mask].reset_index(drop=True)

train_image_paths = [train_image_paths[i] for i in valid_indices]
image_data = [image_data[i] for i in valid_indices]  

X = np.array(image_data) 
y = train_data_filtered['numeric_value'].values[:len(X)]  

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training set size: {len(X_train)}, Validation set size: {len(X_val)}")

def create_cnn_model(input_shape):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='linear')  
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

cnn_model = create_cnn_model((224, 224, 3))
cnn_model.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val))

val_loss = cnn_model.evaluate(X_val, y_val)
print(f"Validation Loss: {val_loss}")

test_image_paths = []
for idx, row in test_data.iterrows():
    image_url = row['image_link']
    image_path = os.path.join(test_images_dir, f"{idx}.jpg")
    if download_image(image_url, image_path):
        test_image_paths.append(image_path)

test_images = [preprocess_image(path) for path in test_image_paths]
test_images = [img for img in test_images if img is not None]  

if len(test_images) == 0:
    print("No valid test images found for prediction.")
else:
    X_test = np.array(test_images)

    predictions = cnn_model.predict(X_test)
    predictions_formatted = [f"{pred[0]:.1f} gram" for pred in predictions]

    output = pd.DataFrame({
        'index': range(len(predictions_formatted)),
        'prediction': predictions_formatted
    })
    output.to_csv("predictions.csv", index=False)
