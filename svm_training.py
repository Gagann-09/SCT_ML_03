import os
import cv2
import numpy as np
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Configuration and Dataset Path
DATASET_PATH = r"C:\Users\gagan\OneDrive\Desktop\SCT_ML_03\dataset"
IMAGE_SIZE = (64, 64)
ORIENTATIONS = 9
PIXELS_PER_CELL = (8, 8)
CELLS_PER_BLOCK = (2, 2)

data = []
labels = []
categories = ['cat', 'dog']

def load_and_preprocess_image(image_path, image_size=IMAGE_SIZE):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load image at {image_path}")
        return None
    img_resized = cv2.resize(img, image_size, interpolation=cv2.INTER_AREA)
    gray_img = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    features = hog(gray_img,
                   orientations=ORIENTATIONS,
                   pixels_per_cell=PIXELS_PER_CELL,
                   cells_per_block=CELLS_PER_BLOCK,
                   block_norm='L2-Hys',
                   feature_vector=True)
    return features

print("Loading and preprocessing images...")

for category_idx, category in enumerate(categories):
    category_path = os.path.join(DATASET_PATH, category)
    if not os.path.exists(category_path):
        print(f"Warning: Directory not found: {category_path}. Skipping.")
        continue
    # Recursively walk through all subdirectories and files
    for root, dirs, files in os.walk(category_path):
        for image_name in files:
            if image_name.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                image_path = os.path.join(root, image_name)
                features = load_and_preprocess_image(image_path)
                if features is not None:
                    data.append(features)
                    labels.append(category_idx)

data = np.array(data)
labels = np.array(labels)

print(f"Finished loading {len(data)} images.")
print(f"Data shape: {data.shape}")
print(f"Labels shape: {labels.shape}")

if len(data) == 0:
    print("No images were loaded. Please check your dataset path and organization.")
    exit()

X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)
print(f"Training set size: {len(X_train)} samples")
print(f"Testing set size: {len(X_test)} samples")

print("Building and training SVM model...")
svm_model = SVC(kernel='linear', random_state=42)
svm_model.fit(X_train, y_train)
print("SVM model trained successfully!")

print("Evaluating model performance...")
y_pred = svm_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy on Test Data: {accuracy * 100:.2f}%")

print("\n--- Testing prediction on a sample image ---")

if len(X_test) > 0:
    sample_index = np.random.randint(0, len(X_test))
    sample_features = X_test[sample_index]
    actual_label = y_test[sample_index]
    prediction = svm_model.predict(sample_features.reshape(1, -1))[0]
    predicted_category = categories[prediction]
    actual_category = categories[actual_label]
    print(f"Sample image - Actual: {actual_category}, Predicted: {predicted_category}")

    # Example new cat image prediction
    new_cat_image_path = os.path.join(DATASET_PATH, 'cat', '10051.jpg')
    if os.path.exists(new_cat_image_path):
        print(f"\nPredicting category for: {new_cat_image_path}")
        new_cat_features = load_and_preprocess_image(new_cat_image_path)
        if new_cat_features is not None:
            new_cat_prediction = svm_model.predict(new_cat_features.reshape(1, -1))[0]
            new_cat_predicted_category = categories[new_cat_prediction]
            print(f"Predicted: {new_cat_predicted_category}")
            img_to_show = cv2.imread(new_cat_image_path)
            img_to_show = cv2.cvtColor(img_to_show, cv2.COLOR_BGR2RGB)
            plt.imshow(img_to_show)
            plt.title(f"Predicted: {new_cat_predicted_category}")
            plt.axis('off')
            plt.show()
        else:
            print("Could not preprocess new cat image.")
    else:
        print(f"New cat image not found at {new_cat_image_path}. Skipping display.")

    # Example new dog image prediction
    new_dog_image_path = os.path.join(DATASET_PATH, 'dog', '1004.jpg')
    if os.path.exists(new_dog_image_path):
        print(f"\nPredicting category for: {new_dog_image_path}")
        new_dog_features = load_and_preprocess_image(new_dog_image_path)
        if new_dog_features is not None:
            new_dog_prediction = svm_model.predict(new_dog_features.reshape(1, -1))[0]
            new_dog_predicted_category = categories[new_dog_prediction]
            print(f"Predicted: {new_dog_predicted_category}")
            img_to_show = cv2.imread(new_dog_image_path)
            img_to_show = cv2.cvtColor(img_to_show, cv2.COLOR_BGR2RGB)
            plt.imshow(img_to_show)
            plt.title(f"Predicted: {new_dog_predicted_category}")
            plt.axis('off')
            plt.show()
        else:
            print("Could not preprocess new dog image.")
    else:
        print(f"New dog image not found at {new_dog_image_path}. Skipping display.")
else:
    print("Not enough test data to demonstrate prediction on a sample.")
