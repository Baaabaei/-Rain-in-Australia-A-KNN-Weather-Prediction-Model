 🌧️ Rain in Australia: A KNN Weather Prediction Model

![alt text](https://img.shields.io/badge/Python-3.9%2B-blue?style=for-the-badge&logo=python)
![alt text](https://img.shields.io/badge/scikit--learn-1.0.2-orange?style=for-the-badge&logo=scikit-learn)
![alt text](https://img.shields.io/badge/Pandas-1.3.5-blueviolet?style=for-the-badge&logo=pandas)
![alt text](https://img.shields.io/badge/NumPy-1.21.5-success?style=for-the-badge&logo=numpy)

📖 Overview

Have you ever wondered if it will rain tomorrow? This project tackles that exact question by building a machine learning model to predict rainfall in Australia. Using a comprehensive weather dataset, we explore, clean, and preprocess the data to train a K-Nearest Neighbors (KNN) classifier.

The goal is not just to build a model, but to walk through the entire process of handling a real-world dataset, which includes dealing with missing values and converting categorical data into a machine-readable format.

✨ Key Features

Data Cleaning: Robust handling of missing data using median imputation.

Feature Engineering: One-Hot Encoding for categorical features to prepare the data for the model.

Classification Model: Implementation of a K-Nearest Neighbors (KNN) algorithm to predict the target variable.

Performance Metrics: Evaluation of the model using both Accuracy and F1-Score to provide a balanced view of its performance.

📊 The Dataset

The project utilizes the "Rain in Australia" dataset (BOM.csv), which contains daily weather observations from numerous locations across Australia.

Target Variable: RainTomorrow (Yes/No)

Features: A wide range of meteorological measurements such as MinTemp, MaxTemp, Rainfall, WindGustDir, Humidity, and Pressure.

⚙️ Project Workflow

The project follows a systematic approach to build and evaluate the prediction model.

1. 🧹 Data Preprocessing and Cleaning

The initial dataset contained a significant number of missing values and categorical text data. The following steps were taken to prepare it:

Handling Missing Values: Missing numerical values were filled with the median of their respective columns. The RainTomorrow target column had its missing entries dropped to ensure a clean training set.

Encoding Categorical Data:

The target variable RainTomorrow ('Yes'/'No') was converted into numerical format (1/0) using an OrdinalEncoder.

All other categorical features (like Location and WindGustDir) were transformed using One-Hot Encoding to create binary columns that the model can understand.

Dropping Irrelevant Features: The Date column was removed as it was not used in this model.

2. 🧠 Building the KNN Model

With the data cleaned and prepared, the next step was to train the classifier.

Train-Test Split: The data was divided into an 80% training set and a 20% testing set to evaluate the model's performance on unseen data.

K-Nearest Neighbors (KNN): A KNeighborsClassifier with n_neighbors=5 was chosen for this classification task. KNN works by finding the 'k' most similar instances in the training data and making a prediction based on their majority class.

3. 📈 Evaluating Performance

The model's ability to predict tomorrow's rain was measured using two key metrics:

Accuracy: ~83.25%

This tells us the overall percentage of correct predictions.

F1-Score: ~0.5716

This is the harmonic mean of precision and recall. It's particularly useful here because it provides a better measure of performance when the classes might be imbalanced (i.e., more non-rainy days than rainy ones).

🚀 How to Run This Project

To replicate this project, follow these simple steps:

Clone the repository:

code
Bash
download
content_copy
expand_less
git clone https://github.com/your-username/your-repository-name.git
cd your-repository-name

Set up a virtual environment (recommended):

code
Bash
download
content_copy
expand_less
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

Install the necessary libraries:

code
Bash
download
content_copy
expand_less
pip install pandas numpy scikit-learn jupyter

Launch Jupyter Notebook and run the file:

code
Bash
download
content_copy
expand_less
jupyter notebook 2.ipynb
💡 Further Insights

While KNN provides a solid baseline, other models could also be effective. For instance:

For real-time applications where prediction speed is critical, a Decision Tree could be an excellent alternative. Decision Trees are often faster to train and predict because they create a simple set of rules, requiring less computational power compared to distance-based algorithms like KNN.

This insight highlights the trade-offs between different algorithms in real-world scenarios.
