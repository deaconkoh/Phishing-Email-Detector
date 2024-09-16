
# Phishing Email Detection Project

This project focuses on building a phishing email detection system using machine learning techniques. The goal is to classify emails as either phishing or legitimate (ham) based on the content of the message. This project serves as a practical example of natural language processing (NLP) applied to cybersecurity.

## Project Overview

The main steps in the project are as follows:

1. **Data Loading and Preprocessing**:
   - The dataset contains labeled email messages, with each message classified as either spam (phishing) or ham (legitimate).
   - Preprocessing involves cleaning the text data by removing stopwords, stemming, and tokenizing it as well as encoding the labels for analysis.

2. **Exploratory Data Analysis (EDA)**:
   - We use `word clouds` and `Matplotlib` to visualize the most common words in phishing (spam) and legitimate (ham) emails and generate a visual representation of the percentage of Spam/Ham respectively.

3. **Feature Engineering**:
   - Additional features such as the **length of the message** and specific terms that commonly appear in phishing emails are extracted.
   - The feature extraction functions help capture essential characteristics of the emails that improve classification accuracy.

4. **Model Building**:
   - We apply a several machine learning model to classify emails based on the preprocessed text and engineered features such as **LogisticRegression**,**NaiveBayes**, **DecisionTreeClassifier**, **RandomForestClassifier** and **k-nearest neighbors (KNN)**.
   - The Logistic Regression model is trained using the features derived from the email messages, and it outputs probabilities of whether a message is phishing or legitimate.

5. **Evaluation**:
   - The model's performance is evaluated using several metrics:
     - **Accuracy**: The proportion of correctly classified emails.
     - **Precision**: The proportion of true positives (correctly classified phishing emails) among all positives.
     - **Recall**: The proportion of true positives among all actual phishing emails.
     - **F1-score**: The harmonic mean of precision and recall, which balances both metrics.

6. **Visualization of Results**:
   - The results of the classification are visualized using confusion matrices, showcasing the number of true positives, false positives, true negatives, and false negatives.
   - We also provide the distribution of message lengths for both spam and ham emails using `Seaborn` histogram graph.

## Key Functions in the Code

### `preprocess_text(message)`
- **Description**: This function takes in an email message, cleans the text by removing unnecessary characters, converting it to lowercase, and tokenizing it.
- **Purpose**: Prepares the raw text data for analysis by converting it into a usable format for the machine learning model.

### `create_features(df)`
- **Description**: This function extracts key features such as message length, the presence of specific keywords, and term frequency-inverse document frequency (TF-IDF) scores.
- **Purpose**: Enhances the dataset with additional meaningful features that improve model performance.

### `train_logistic_regression(X_train, y_train)`
- **Description**: Trains a Logistic Regression classifier on the training data.
- **Purpose**: The classifier predicts whether an email is phishing or legitimate based on the provided features.

### `evaluate_model(model, X_test, y_test)`
- **Description**: Evaluates the trained model on the test dataset using accuracy, precision, recall, and F1-score metrics.
- **Purpose**: Assesses the performance of the model in distinguishing between phishing and legitimate emails.

### `plot_word_cloud(category)`
- **Description**: Generates word clouds for a specific category (either phishing or legitimate) using the `WordCloud` library.
- **Purpose**: Visualizes the most frequent words in the phishing or legitimate emails to provide insights into the data.
