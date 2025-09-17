# SCT_ML_03

# Cat vs. Dog Image Classification using Support Vector Machine (SVM) 🐱🐶

This project implements a Support Vector Machine (SVM) to classify images of cats and dogs. The model is trained on the popular "Cats and Dogs" dataset from Kaggle. The primary goal is to build a foundational understanding of using classical machine learning algorithms for computer vision tasks.

## 📜 Table of Contents

  * [Problem Statement](https://www.google.com/search?q=%23-problem-statement)
  * [Dataset](https://www.google.com/search?q=%23-dataset)
  * [Methodology](https://www.google.com/search?q=%23-methodology)
  * [Project Structure](https://www.google.com/search?q=%23-project-structure)
  * [Getting Started](https://www.google.com/search?q=%23-getting-started)
  * [Results and Evaluation](https://www.google.com/search?q=%23-results-and-evaluation)
  * [Future Improvements](https://www.google.com/search?q=%23-future-improvements)

-----

## 🎯 Problem Statement

The goal is to build a binary classification model that can accurately distinguish between images of cats and dogs. This project tackles the problem using a traditional machine learning approach where raw pixel values are used as features for a Support Vector Machine classifier.

-----

## 📊 Dataset

The dataset used is the **Asirra (Dogs vs. Cats) dataset**, available on Kaggle. It contains thousands of labeled images of cats and dogs, making it ideal for a binary classification task.

  * **Source**: [Kaggle Dogs vs. Cats Dataset](https://www.kaggle.com/c/dogs-vs-cats/data)
  * **Contents**:
      * `train.zip`: Contains 25,000 images, split evenly between cats and dogs. File names are `cat.1.jpg`, `dog.123.jpg`, etc.
      * `test1.zip`: Contains 12,500 unlabeled images for testing.

For this project, we will use a subset of the training data to keep training times manageable.

-----

## 🛠️ Methodology

The workflow is broken down into three main stages: data preprocessing, model training, and evaluation.

### 1\. Data Preprocessing

Since machine learning models require numerical input, the images must be converted into feature vectors.

  * **Image Loading**: Images are loaded from their respective directories.
  * **Resizing**: All images are resized to a uniform dimension (e.g., **64x64 pixels**) to ensure that each feature vector has the same length. This standardization is crucial for the SVM model.
  * **Grayscale Conversion**: Images are converted to grayscale to reduce the dimensionality of the feature space. Instead of 3 channels (R, G, B), we only have 1, simplifying the problem.
  * **Flattening**: The 2D grayscale image array (64x64) is flattened into a 1D vector of **4096 pixels** (64 \* 64). Each pixel's intensity value becomes a feature.
  * **Data Splitting**: The dataset is split into a **training set** and a **testing set** (e.g., an 80/20 split) to evaluate the model's performance on unseen data.

### 2\. Model Training

A **Support Vector Machine (SVM)** was chosen for this classification task.

  * **Why SVM?** SVMs are highly effective in high-dimensional spaces, making them suitable for problems where the number of features (pixels) is large. They work by finding an optimal hyperplane that best separates the data points of the two classes (cats and dogs).
  * **Implementation**: We use the `SVC` (Support Vector Classifier) class from the **scikit-learn** library. A linear kernel is used as a baseline.
  * **Training**: The model is trained using the `fit()` method on the preprocessed training data (flattened image vectors) and their corresponding labels.

### 3\. Model Evaluation

The model's performance is assessed on the test set using standard classification metrics:

  * **Accuracy**: The percentage of correctly classified images.
  * **Confusion Matrix**: A table showing the number of true positives, true negatives, false positives, and false negatives. It gives a detailed view of where the model is making errors.
  * **Classification Report**: A summary that includes **precision**, **recall**, and **F1-score** for each class.

-----

## 📁 Project Structure

Here is the recommended directory structure for the project:

```
cat-dog-svm-classifier/
│
├── data/
│   ├── train/
│   │   ├── cats/
│   │   └── dogs/
│   └── test/
│       ├── cats/
│       └── dogs/
│
├── notebooks/
│   └── exploration.ipynb
│
├── src/
│   ├── preprocess.py
│   └── train.py
│
├── requirements.txt
└── README.md
```

-----

## 🚀 Getting Started

Follow these steps to set up and run the project on your local machine.

### Prerequisites

  * Python 3.8+
  * Access to the Kaggle dataset

### Installation

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/your-username/cat-dog-svm-classifier.git
    cd cat-dog-svm-classifier
    ```

2.  **Set up a virtual environment (recommended):**

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the required libraries:**

    ```bash
    pip install -r requirements.txt
    ```

    The `requirements.txt` file should contain:

    ```
    numpy
    scikit-learn
    matplotlib
    Pillow
    ```

### How to Run

1.  **Download and organize the dataset**: Download the data from Kaggle and place a subset of cat and dog images into the `data/train/cats` and `data/train/dogs` folders.
2.  **Run the training script**:
    ```bash
    python src/train.py
    ```
    The script will preprocess the data, train the SVM model, and print the evaluation results to the console.

-----

## 📈 Results and Evaluation

After training on a subset of the data, the model achieved the following performance on the test set:

  * **Accuracy**: Approximately **60-65%**.

**Confusion Matrix:**
(Provide your confusion matrix results here)

|                | Predicted Dog | Predicted Cat |
| :------------- | :-----------: | :-----------: |
| **Actual Dog** |      ...      |      ...      |
| **Actual Cat** |      ...      |      ...      |

**Analysis**:
The accuracy, while better than random guessing (50%), is modest. This is expected because raw pixel values are sensitive to variations in lighting, rotation, and object position. The model is learning basic patterns but struggles with the complexity of real-world images.

-----

## 💡 Future Improvements

This baseline model can be significantly improved in several ways:

1.  **Advanced Feature Extraction**: Instead of using raw pixels, use engineered features that are more robust. Techniques like **Histogram of Oriented Gradients (HOG)** or **SIFT** can capture texture and shape information more effectively.
2.  **Hyperparameter Tuning**: Use techniques like `GridSearchCV` from scikit-learn to find the optimal SVM parameters (e.g., `C` and `gamma`) and kernel (e.g., `rbf`).
3.  **Use More Data**: Training on a larger portion of the 25,000 images will likely improve generalization.
4.  **Deep Learning Approach**: For best-in-class performance, a **Convolutional Neural Network (CNN)** would be the next step. CNNs are specifically designed for image data and can automatically learn the most relevant features.

-----