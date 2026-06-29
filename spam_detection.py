#!/usr/bin/env python3
"""
================================================================================
AI-Powered Email Spam Detection using Machine Learning and Deep Learning
================================================================================
Course Project: Final Year Mini-Project
Description: This script provides a complete end-to-end framework to classify 
             SMS/Email messages as Spam or Ham (Not Spam). It implements a
             comparative study between a classical Machine Learning model 
             (Logistic Regression) and a Deep Learning model (Artificial 
             Neural Network - ANN) using the Kaggle SMS Spam Collection dataset.

Steps Included:
  1. Programmatic installation and setup of NLTK text processing resources.
  2. Data loading, cleaning, column renaming, and label encoding.
  3. Exploratory Data Analysis (EDA) with saved distribution plots.
  4. Advanced text cleaning: lowercase, punctuation/numbers removal, 
     tokenization, stopword removal, and stemming (PorterStemmer).
  5. Vectorization using TF-IDF (Term Frequency-Inverse Document Frequency).
  6. Dataset splitting into Train and Test sets (80:20).
  7. ML Model: Logistic Regression training, validation, and metrics.
  8. DL Model: Artificial Neural Network (Keras Sequential) with ReLU & Sigmoid.
  9. Evaluation & Comparative performance table (Accuracy, Precision, Recall, etc.).
 10. Visualization of training curves, ROC-AUC, and Confusion Matrices.
 11. Serialization of models (saving Keras model and TF-IDF vectorizer).
 12. Interactive CLI prediction function for real-time testing.
================================================================================
"""

# Import standard library modules
import os
import re
import ssl
import sys
import string
import warnings

# Ignore warnings for clean console outputs
warnings.filterwarnings('ignore')

# Import core scientific and data libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

# Import NLTK modules for Natural Language Processing (NLP)
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize

# Import scikit-learn modules for Machine Learning and evaluation
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, roc_curve
)

# Import TensorFlow and Keras modules for Deep Learning
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam

# Fix NLTK SSL certificate download issues on macOS
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context


def print_section_header(title):
    """Prints a formatted header to separate different project stages in the console."""
    border = "=" * 80
    print(f"\n{border}")
    print(f" {title.upper()}")
    print(border)


def download_nltk_dependencies():
    """
    Downloads required NLTK packages (stopwords, punkt, and punkt_tab) programmatically.
    Ensures the script runs seamlessly on any notebook or IDE environment.
    """
    print_section_header("Step 1: Setting up NLP Dependencies")
    print("Checking and downloading NLTK datasets ('punkt', 'punkt_tab', and 'stopwords')...")
    try:
        nltk.download('punkt', quiet=True)
        nltk.download('punkt_tab', quiet=True)
        nltk.download('stopwords', quiet=True)
        print("[SUCCESS] NLTK resources downloaded successfully.")
    except Exception as e:
        print(f"[WARNING] NLTK download failed: {e}. Attempting standard fallback...")
        nltk.download('punkt')
        nltk.download('punkt_tab')
        nltk.download('stopwords')


def load_and_preprocess_data(file_path):
    """
    Loads the SMS Spam collection dataset, cleans unnecessary columns, 
    renames variables, handles missing values, and encodes text labels to numeric values.
    
    Parameters:
        file_path (str): Path to the spam.csv file.
        
    Returns:
        pd.DataFrame: Cleaned and processed DataFrame.
    """
    print_section_header("Step 2: Data Loading and Preprocessing")
    
    if not os.path.exists(file_path):
        print(f"[ERROR] Dataset file not found at {file_path}. Please check path.")
        sys.exit(1)
        
    print(f"Loading dataset from: {file_path}")
    # Read the dataset (latin-1 encoding is required for this Kaggle dataset)
    df = pd.read_csv(file_path, encoding='latin-1')
    
    print("\n--- Raw Dataset Overview ---")
    print(f"Initial Shape: {df.shape}")
    print(df.head())
    
    # 1. Remove unnecessary columns (keep only the first two columns)
    # The dataset often has 3 empty trailing columns: Unnamed: 2, Unnamed: 3, Unnamed: 4
    df = df.iloc[:, :2]
    
    # 2. Rename columns to 'label' and 'message'
    df.columns = ['label', 'message']
    print("\n[INFO] Renamed columns to ['label', 'message'] and dropped extra columns.")
    
    # 3. Handle missing values
    print("\n--- Missing Values Audit ---")
    missing_counts = df.isnull().sum()
    print(missing_counts)
    
    if missing_counts.sum() > 0:
        print("[WARNING] Missing values detected. Dropping incomplete rows...")
        df.dropna(subset=['label', 'message'], inplace=True)
    else:
        print("[INFO] No missing values detected in the critical columns.")
        
    # 4. Encode labels: Spam = 1, Ham = 0
    df['label_encoded'] = df['label'].map({'ham': 0, 'spam': 1})
    print("\n[INFO] Encoded labels: 'ham' -> 0, 'spam' -> 1")
    
    # 5. Display general dataset info and class distribution
    print("\n--- Dataset Summary Information ---")
    df.info()
    
    print("\n--- Target Class Distribution ---")
    class_counts = df['label'].value_counts()
    class_pcts = df['label'].value_counts(normalize=True) * 100
    for idx in class_counts.index:
        print(f"Class '{idx}': {class_counts[idx]} messages ({class_pcts[idx]:.2f}%)")
        
    return df


def perform_eda(df, output_dir="outputs"):
    """
    Performs Exploratory Data Analysis (EDA). Computes lengths and word counts,
    and saves charts representing the text distributions.
    
    Parameters:
        df (pd.DataFrame): Preprocessed DataFrame.
        output_dir (str): Folder path to save the generated figures.
    """
    print_section_header("Step 3: Exploratory Data Analysis (EDA)")
    os.makedirs(output_dir, exist_ok=True)
    print(f"Generating and saving EDA plots in directory: '{output_dir}/'")
    
    # Calculate word count and message character length
    df['message_len'] = df['message'].apply(len)
    df['word_count'] = df['message'].apply(lambda x: len(x.split()))
    
    # Set plot style parameters for rich aesthetics
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    plt.rcParams['figure.facecolor'] = '#f8f9fa'
    plt.rcParams['axes.facecolor'] = '#ffffff'
    plt.rcParams['font.size'] = 11
    
    # 1. Bar Chart of Spam vs Ham Counts
    plt.figure(figsize=(7, 5))
    colors = ['#1f77b4', '#d62728'] # Classic blue and red
    counts = df['label'].value_counts()
    bars = plt.bar(counts.index, counts.values, color=colors, edgecolor='black', alpha=0.85, width=0.5)
    plt.title("Class Frequency: Ham vs Spam Messages", fontsize=13, fontweight='bold', pad=15)
    plt.xlabel("Message Type", labelpad=10)
    plt.ylabel("Number of Messages", labelpad=10)
    # Add values on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.annotate(f'{height}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
    plt.tight_layout()
    bar_path = os.path.join(output_dir, "bar_chart.png")
    plt.savefig(bar_path, dpi=150)
    plt.close()
    print(f" - Saved Bar Chart to: {bar_path}")

    # 2. Pie Chart of Label Distribution
    plt.figure(figsize=(6, 6))
    plt.pie(
        counts.values, 
        labels=['Ham (Not Spam)', 'Spam'], 
        autopct='%1.2f%%', 
        startangle=140, 
        colors=['#a1c9f4', '#ff9f9b'], 
        explode=(0, 0.1), 
        shadow=True,
        textprops={'fontweight': 'bold', 'fontsize': 11}
    )
    plt.title("Proportion of Spam vs Ham Messages", fontsize=13, fontweight='bold', pad=15)
    plt.tight_layout()
    pie_path = os.path.join(output_dir, "pie_chart.png")
    plt.savefig(pie_path, dpi=150)
    plt.close()
    print(f" - Saved Pie Chart to: {pie_path}")

    # 3. Message Length Distribution Histogram
    plt.figure(figsize=(9, 5))
    plt.hist(df[df['label_encoded'] == 0]['message_len'], bins=50, alpha=0.7, label='Ham (Not Spam)', color='#2ca02c', edgecolor='none')
    plt.hist(df[df['label_encoded'] == 1]['message_len'], bins=50, alpha=0.7, label='Spam', color='#d62728', edgecolor='none')
    plt.title("Message Length Distribution", fontsize=13, fontweight='bold', pad=15)
    plt.xlabel("Number of Characters in Message", labelpad=10)
    plt.ylabel("Frequency", labelpad=10)
    plt.legend(frameon=True, facecolor='white', edgecolor='gray')
    plt.xlim(0, 300)  # Limit to 300 characters for clear visualization of the main distribution
    plt.tight_layout()
    len_path = os.path.join(output_dir, "msg_length_dist.png")
    plt.savefig(len_path, dpi=150)
    plt.close()
    print(f" - Saved Message Length Distribution to: {len_path}")

    # 4. Word Count Distribution Histogram
    plt.figure(figsize=(9, 5))
    plt.hist(df[df['label_encoded'] == 0]['word_count'], bins=40, alpha=0.7, label='Ham (Not Spam)', color='#17becf', edgecolor='none')
    plt.hist(df[df['label_encoded'] == 1]['word_count'], bins=40, alpha=0.7, label='Spam', color='#ff7f0e', edgecolor='none')
    plt.title("Word Count Distribution", fontsize=13, fontweight='bold', pad=15)
    plt.xlabel("Number of Words in Message", labelpad=10)
    plt.ylabel("Frequency", labelpad=10)
    plt.legend(frameon=True, facecolor='white', edgecolor='gray')
    plt.xlim(0, 60)  # Limit to 60 words for clearer plotting
    plt.tight_layout()
    word_path = os.path.join(output_dir, "word_count_dist.png")
    plt.savefig(word_path, dpi=150)
    plt.close()
    print(f" - Saved Word Count Distribution to: {word_path}")


def clean_text(text, stemmer, stop_words):
    """
    Cleans a text message by:
    1. Converting to lowercase.
    2. Removing HTML tags/special characters/numbers/punctuation.
    3. Tokenizing into words.
    4. Removing stop words.
    5. Stemming words using PorterStemmer.
    
    Parameters:
        text (str): Raw input string.
        stemmer (PorterStemmer): PorterStemmer object.
        stop_words (set): English stop words list.
        
    Returns:
        str: Processed text string.
    """
    if not isinstance(text, str):
        return ""
        
    # Convert to lowercase
    text = text.lower()
    
    # Remove HTML tags if present
    text = re.sub(r'<.*?>', ' ', text)
    
    # Remove punctuation, numbers, and special characters, leaving only English alphabet letters
    text = re.sub(r'[^a-z\s]', ' ', text)
    
    # Tokenize the sentence into words
    tokens = word_tokenize(text)
    
    # Remove stopwords and perform stemming
    processed_tokens = [
        stemmer.stem(word) 
        for word in tokens 
        if word not in stop_words and len(word) > 1
    ]
    
    # Join tokens back to form a single cleaned string
    return " ".join(processed_tokens)


def preprocess_text_dataset(df):
    """
    Applies the text cleaning function to the entire dataset.
    
    Parameters:
        df (pd.DataFrame): Dataframe with raw messages.
        
    Returns:
        pd.DataFrame: Dataframe with added 'cleaned_message' column.
    """
    print_section_header("Step 4: Text Preprocessing & Tokenization")
    print("Instantiating PorterStemmer and Loading Stopwords...")
    stemmer = PorterStemmer()
    stop_words = set(stopwords.words('english'))
    
    print("Preprocessing text dataset (this may take a few seconds)...")
    # Clean and stem all messages
    df['cleaned_message'] = df['message'].apply(lambda x: clean_text(x, stemmer, stop_words))
    
    print("\n--- Preprocessing Comparison ---")
    for i in range(3):
        idx = df.index[i]
        print(f"\n[Original Message {i+1}]: {df['message'].loc[idx]}")
        print(f"[Cleaned & Stemmed]:  {df['cleaned_message'].loc[idx]}")
        
    return df, stemmer, stop_words


def vectorize_and_split(df, output_dir="."):
    """
    Extracts features using TF-IDF Vectorizer and splits the dataset
    into training and testing subsets (80:20). Saves the vectorizer for prediction.
    
    Parameters:
        df (pd.DataFrame): Preprocessed Dataframe.
        output_dir (str): Directory where the vectorizer will be saved.
        
    Returns:
        X_train, X_test, y_train, y_test, vectorizer
    """
    print_section_header("Step 5: Feature Extraction & Dataset Splitting")
    
    X = df['cleaned_message']
    y = df['label_encoded']
    
    # Split the dataset: 80% Training, 20% Testing with stratify to keep class distribution balanced
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    
    print(f"Training set size: {X_train_raw.shape[0]} samples")
    print(f"Testing set size:  {X_test_raw.shape[0]} samples")
    
    # Initialize TF-IDF Vectorizer
    # We restrict max_features to 4000 to keep the input feature space robust but manageable for our ANN
    print("\nInitializing TF-IDF Vectorizer (max_features=4000)...")
    vectorizer = TfidfVectorizer(max_features=4000)
    
    # Fit vectorizer on training data and transform both train and test data
    X_train_vec = vectorizer.fit_transform(X_train_raw).toarray()
    X_test_vec = vectorizer.transform(X_test_raw).toarray()
    
    print(f"Vectorized Training Shape: {X_train_vec.shape}")
    print(f"Vectorized Testing Shape:  {X_test_vec.shape}")
    
    # Save the TF-IDF vectorizer using joblib
    vectorizer_path = os.path.join(output_dir, "tfidf_vectorizer.joblib")
    joblib.dump(vectorizer, vectorizer_path)
    print(f"\n[SUCCESS] TF-IDF Vectorizer saved to: {vectorizer_path}")
    
    return X_train_vec, X_test_vec, y_train, y_test, vectorizer


def train_logistic_regression(X_train, y_train, X_test, y_test):
    """
    Trains a Logistic Regression model on the TF-IDF feature vectors
    and evaluates it using standard classification metrics.
    
    Parameters:
        X_train, y_train: Training vectors and labels.
        X_test, y_test: Test vectors and labels.
        
    Returns:
        dict: Performance metrics of the ML model.
        np.array: Predicted labels.
        np.array: Predicted probabilities.
    """
    print_section_header("Step 6: Machine Learning Model (Logistic Regression)")
    print("Training Logistic Regression model...")
    
    # Initialize Logistic Regression Model
    lr_model = LogisticRegression(solver='liblinear', random_state=42)
    lr_model.fit(X_train, y_train)
    
    print("Evaluating Logistic Regression model on test dataset...")
    # Predict classes and probability estimates
    y_pred = lr_model.predict(X_test)
    y_prob = lr_model.predict_proba(X_test)[:, 1]
    
    # Calculate performance metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)
    cm = confusion_matrix(y_test, y_pred)
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc,
        'confusion_matrix': cm
    }
    
    print("\n--- Logistic Regression Metrics ---")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"ROC-AUC:   {roc_auc:.4f}")
    print("Confusion Matrix:")
    print(cm)
    
    return lr_model, metrics, y_pred, y_prob


def train_ann_model(X_train, y_train, X_test, y_test, input_dim, output_dir="."):
    """
    Designs, compiles, trains, and saves an Artificial Neural Network (ANN) using Keras.
    
    Parameters:
        X_train, y_train: Train vectors and labels.
        X_test, y_test: Test vectors and labels.
        input_dim (int): Number of features in TF-IDF (Input size).
        output_dir (str): Output folder to save the .keras model file.
        
    Returns:
        Sequential: Trained Keras model.
        History: Training history object.
        dict: Performance metrics of the DL model.
        np.array: Predicted labels.
        np.array: Predicted probabilities.
    """
    print_section_header("Step 7: Deep Learning Model (Artificial Neural Network)")
    print(f"Building Artificial Neural Network structure. Input Dimension = {input_dim}")
    
    # 1. Design ANN Model
    model = Sequential([
        Input(shape=(input_dim,)),
        
        # Hidden Layer 1
        Dense(64, activation='relu'),
        Dropout(0.3),  # Dropout regularization to prevent overfitting
        
        # Hidden Layer 2
        Dense(32, activation='relu'),
        Dropout(0.3),
        
        # Output Layer with Sigmoid activation (Binary classification)
        Dense(1, activation='sigmoid')
    ])
    
    # Display the Keras model summary
    print("\n--- Keras Model Architecture Summary ---")
    model.summary()
    
    # 2. Compile model
    print("\nCompiling model with Adam optimizer and binary crossentropy loss...")
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    # 3. Train model
    epochs = 15
    batch_size = 32
    print(f"Training ANN for {epochs} epochs with batch size {batch_size} and 15% validation split...")
    
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.15,
        verbose=1
    )
    
    # 4. Predict on Test Set
    print("\nEvaluating Artificial Neural Network on test dataset...")
    # Get raw probabilities
    y_prob = model.predict(X_test).flatten()
    # Threshold probabilities at 0.5 to convert to binary labels
    y_pred = (y_prob >= 0.5).astype(int)
    
    # Calculate performance metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)
    cm = confusion_matrix(y_test, y_pred)
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc,
        'confusion_matrix': cm
    }
    
    print("\n--- ANN Deep Learning Metrics ---")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"ROC-AUC:   {roc_auc:.4f}")
    print("Confusion Matrix:")
    print(cm)
    
    # 5. Save the trained ANN model
    model_path = os.path.join(output_dir, "spam_ann_model.keras")
    model.save(model_path)
    print(f"\n[SUCCESS] Trained Keras ANN model saved to: {model_path}")
    
    return model, history, metrics, y_pred, y_prob


def plot_performance_graphs(history, y_test, lr_probs, ann_probs, lr_preds, ann_preds, lr_cm, ann_cm, output_dir="outputs"):
    """
    Plots and saves training/validation performance curves, ROC curves,
    and confusion matrices for model comparison.
    """
    print_section_header("Step 8: Generating Performance Evaluation Graphics")
    os.makedirs(output_dir, exist_ok=True)
    
    # Set styling parameters
    plt.rcParams['figure.facecolor'] = '#f8f9fa'
    plt.rcParams['font.size'] = 10
    
    # ==================== Graph 1: Training History (Loss and Accuracy Curves) ====================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Accuracy Plot
    ax1.plot(history.history['accuracy'], label='Train Accuracy', color='#1f77b4', linewidth=2, marker='o')
    ax1.plot(history.history['val_accuracy'], label='Val Accuracy', color='#ff7f0e', linewidth=2, marker='x')
    ax1.set_title('ANN Model Accuracy Curve', fontsize=12, fontweight='bold', pad=10)
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Accuracy')
    ax1.legend(loc='lower right', frameon=True, facecolor='white')
    ax1.grid(True, linestyle='--', alpha=0.6)
    
    # Loss Plot
    ax2.plot(history.history['loss'], label='Train Loss', color='#1f77b4', linewidth=2, marker='o')
    ax2.plot(history.history['val_loss'], label='Val Loss', color='#ff7f0e', linewidth=2, marker='x')
    ax2.set_title('ANN Model Loss Curve', fontsize=12, fontweight='bold', pad=10)
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Loss (Binary Crossentropy)')
    ax2.legend(loc='upper right', frameon=True, facecolor='white')
    ax2.grid(True, linestyle='--', alpha=0.6)
    
    plt.suptitle("Deep Learning ANN Model Training Performance", fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    curves_path = os.path.join(output_dir, "training_curves.png")
    plt.savefig(curves_path, dpi=150)
    plt.close()
    print(f" - Saved Training Curves to: {curves_path}")
    
    # ==================== Graph 2: Comparison Metrics (Confusion Matrices and ROC Curve) ====================
    fig = plt.figure(figsize=(15, 5))
    
    # 1. Confusion Matrix: Logistic Regression
    ax_cm1 = fig.add_subplot(1, 3, 1)
    im1 = ax_cm1.imshow(lr_cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax_cm1.set_title("LR Confusion Matrix", fontsize=11, fontweight='bold')
    fig.colorbar(im1, ax=ax_cm1, fraction=0.046, pad=0.04)
    tick_marks = np.arange(2)
    ax_cm1.set_xticks(tick_marks)
    ax_cm1.set_xticklabels(['Ham', 'Spam'])
    ax_cm1.set_yticks(tick_marks)
    ax_cm1.set_yticklabels(['Ham', 'Spam'])
    ax_cm1.set_ylabel('Actual Label', fontweight='bold')
    ax_cm1.set_xlabel('Predicted Label', fontweight='bold')
    
    # Annotate values inside the matrix
    thresh1 = lr_cm.max() / 2.
    for i in range(2):
        for j in range(2):
            ax_cm1.text(j, i, format(lr_cm[i, j], 'd'),
                     ha="center", va="center",
                     color="white" if lr_cm[i, j] > thresh1 else "black",
                     fontweight='bold', fontsize=12)
                     
    # 2. Confusion Matrix: Artificial Neural Network
    ax_cm2 = fig.add_subplot(1, 3, 2)
    im2 = ax_cm2.imshow(ann_cm, interpolation='nearest', cmap=plt.cm.Oranges)
    ax_cm2.set_title("ANN Confusion Matrix", fontsize=11, fontweight='bold')
    fig.colorbar(im2, ax=ax_cm2, fraction=0.046, pad=0.04)
    ax_cm2.set_xticks(tick_marks)
    ax_cm2.set_xticklabels(['Ham', 'Spam'])
    ax_cm2.set_yticks(tick_marks)
    ax_cm2.set_yticklabels(['Ham', 'Spam'])
    ax_cm2.set_ylabel('Actual Label', fontweight='bold')
    ax_cm2.set_xlabel('Predicted Label', fontweight='bold')
    
    # Annotate values inside the matrix
    thresh2 = ann_cm.max() / 2.
    for i in range(2):
        for j in range(2):
            ax_cm2.text(j, i, format(ann_cm[i, j], 'd'),
                     ha="center", va="center",
                     color="white" if ann_cm[i, j] > thresh2 else "black",
                     fontweight='bold', fontsize=12)
                     
    # 3. ROC Curve comparison
    ax_roc = fig.add_subplot(1, 3, 3)
    
    # Compute ROC curve and AUC
    lr_fpr, lr_tpr, _ = roc_curve(y_test, lr_probs)
    lr_auc = roc_auc_score(y_test, lr_probs)
    
    ann_fpr, ann_tpr, _ = roc_curve(y_test, ann_probs)
    ann_auc = roc_auc_score(y_test, ann_probs)
    
    ax_roc.plot(lr_fpr, lr_tpr, color='#1f77b4', lw=2, label=f'Logistic Regression (AUC = {lr_auc:.4f})')
    ax_roc.plot(ann_fpr, ann_tpr, color='#ff7f0e', lw=2, label=f'Neural Network ANN (AUC = {ann_auc:.4f})')
    ax_roc.plot([0, 1], [0, 1], color='navy', lw=1.5, linestyle='--', label='Random Classifier')
    
    ax_roc.set_xlim([0.0, 1.0])
    ax_roc.set_ylim([0.0, 1.05])
    ax_roc.set_xlabel('False Positive Rate (1 - Specificity)')
    ax_roc.set_ylabel('True Positive Rate (Sensitivity)')
    ax_roc.set_title('Receiver Operating Characteristic (ROC)', fontsize=11, fontweight='bold')
    ax_roc.legend(loc="lower right", frameon=True, facecolor='white', fontsize=8)
    ax_roc.grid(True, linestyle='--', alpha=0.5)
    
    plt.suptitle("Model Evaluation & Comparison Graphs", fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    metrics_plot_path = os.path.join(output_dir, "model_metrics_plot.png")
    plt.savefig(metrics_plot_path, dpi=150)
    plt.close()
    print(f" - Saved Evaluation Comparison plots to: {metrics_plot_path}")


def print_performance_comparison(lr_metrics, ann_metrics):
    """
    Renders a side-by-side comparison table of classification metrics for both models.
    """
    print_section_header("Step 9: Machine Learning vs Deep Learning Comparison")
    
    headers = ["Evaluation Metric", "Logistic Regression (ML)", "Neural Network (DL)"]
    row_format = "| {:<28} | {:<24.4f} | {:<19.4f} |"
    divider = "-" * 81
    
    print(divider)
    print(f"| {headers[0]:<28} | {headers[1]:<24} | {headers[2]:<19} |")
    print(divider)
    print(row_format.format("Accuracy", lr_metrics['accuracy'], ann_metrics['accuracy']))
    print(row_format.format("Precision", lr_metrics['precision'], ann_metrics['precision']))
    print(row_format.format("Recall", lr_metrics['recall'], ann_metrics['recall']))
    print(row_format.format("F1 Score", lr_metrics['f1'], ann_metrics['f1']))
    print(row_format.format("ROC-AUC Score", lr_metrics['roc_auc'], ann_metrics['roc_auc']))
    print(divider)
    
    # Brief comparative analysis text
    print("\n--- Quantitative Summary Analysis ---")
    if lr_metrics['accuracy'] > ann_metrics['accuracy']:
        print("Logistic Regression has a slightly higher overall Accuracy on the test dataset.")
    elif lr_metrics['accuracy'] < ann_metrics['accuracy']:
        print("The Deep Learning ANN model outperformed Logistic Regression in terms of overall Accuracy.")
    else:
        print("Both models achieved identical overall Accuracy scores.")
        
    if lr_metrics['f1'] > ann_metrics['f1']:
        print("For balanced precision-recall optimization, Logistic Regression is the preferred model.")
    else:
        print("For balanced precision-recall optimization, the Artificial Neural Network is the preferred model.")


def predict_message(message, vectorizer, lr_model, ann_model, stemmer, stop_words):
    """
    Cleans a custom user message, converts it into numerical features, 
    and predicts whether it is Spam or Ham using both trained models.
    """
    # 1. Clean the input text using the trained vocabulary parameters
    cleaned = clean_text(message, stemmer, stop_words)
    
    # 2. Transform the text via the saved TF-IDF vectorizer
    vectorized = vectorizer.transform([cleaned]).toarray()
    
    # 3. Predict using Logistic Regression
    lr_pred = lr_model.predict(vectorized)[0]
    lr_prob = lr_model.predict_proba(vectorized)[0][1]
    lr_label = "SPAM" if lr_pred == 1 else "HAM (Not Spam)"
    
    # 4. Predict using Deep Learning ANN
    ann_prob = ann_model.predict(vectorized)[0][0]
    ann_pred = 1 if ann_prob >= 0.5 else 0
    ann_label = "SPAM" if ann_pred == 1 else "HAM (Not Spam)"
    
    return {
        'cleaned': cleaned,
        'lr': {'label': lr_label, 'prob': lr_prob},
        'ann': {'label': ann_label, 'prob': ann_prob}
    }


def start_interactive_predictor(vectorizer, lr_model, ann_model, stemmer, stop_words):
    """
    Runs an interactive terminal loop that prompts users to input message texts 
    to obtain real-time classification reports.
    """
    print_section_header("Step 10: Interactive Real-Time Prediction Console")
    print("Welcome to the Spam Detection Console!")
    print("Enter any message text below to test the trained models.")
    print("Type 'exit' or 'quit' to terminate the loop.\n")
    
    # Check if stdin is a standard terminal device (TTY) or automated test environment
    # If not interactive, run a default test set to avoid locking executions.
    if not sys.stdin.isatty():
        print("[INFO] Non-interactive environment detected. Running pre-defined verification test messages...")
        demo_messages = [
            "Urgently urgent! Please call 09061104282 from landline. £5000 cash balance waiting! Claim code: K42. Valid 12 hrs only.",
            "Hey buddy! Are you free to play football tonight at 7 PM? Let me know.",
            "FREE Ringtone! Text FREE to 80123 to get your free content now. T&Cs apply."
        ]
        for msg in demo_messages:
            print(f"\nMessage: \"{msg}\"")
            res = predict_message(msg, vectorizer, lr_model, ann_model, stemmer, stop_words)
            print(f" Cleaned words: '{res['cleaned']}'")
            print(f" [ML - Logistic Regression] Prediction: {res['lr']['label']} (Spam Confidence: {res['lr']['prob']:.4%})")
            print(f" [DL - Neural Network]     Prediction: {res['ann']['label']} (Spam Probability: {res['ann']['prob']:.4%})")
            # Final verdict: if either model says SPAM, flag it as SPAM (conservative approach)
            is_spam = (res['lr']['prob'] >= 0.5) or (res['ann']['prob'] >= 0.5)
            verdict = "🔴 SPAM" if is_spam else "🟢 NOT SPAM"
            print(f" {'=' * 50}")
            print(f" >>>  FINAL VERDICT:  {verdict}  <<<")
            print(f" {'=' * 50}")
        print("\nAll demo tests completed successfully.")
        return
        
    while True:
        try:
            user_input = input("\nEnter your message: ").strip()
            if not user_input:
                continue
            if user_input.lower() in ['exit', 'quit']:
                print("\nExiting prediction console. Thank you for using the Spam Detector!")
                break
                
            results = predict_message(user_input, vectorizer, lr_model, ann_model, stemmer, stop_words)
            
            # Determine final verdict: conservative approach — flag as SPAM if either model predicts spam
            is_spam = (results['lr']['prob'] >= 0.5) or (results['ann']['prob'] >= 0.5)
            verdict = "SPAM" if is_spam else "NOT SPAM"
            verdict_icon = "🔴" if is_spam else "🟢"
            
            print("\n" + "=" * 60)
            print(f"  INPUT MESSAGE: \"{user_input}\"")
            print(f"  PREPROCESSED:  '{results['cleaned']}'")
            print("=" * 60)
            print(f"  1. Machine Learning (Logistic Regression)")
            print(f"     => Result:     {results['lr']['label']}")
            print(f"     => Confidence: {results['lr']['prob']:.4%} spam probability")
            print(f"  2. Deep Learning (Artificial Neural Network)")
            print(f"     => Result:     {results['ann']['label']}")
            print(f"     => Confidence: {results['ann']['prob']:.4%} spam probability")
            print("=" * 60)
            # Display the final unified verdict prominently
            print(f"  {verdict_icon}  FINAL VERDICT:  *** {verdict} ***")
            print("=" * 60)
            
        except KeyboardInterrupt:
            print("\n\nInteractive console interrupted. Shutting down...")
            break
        except Exception as e:
            print(f"An error occurred: {e}")


def main():
    """
    Main orchestrator function that triggers the preprocessing, training, 
    evaluating, saving, and interaction pipelines.
    """
    print("================================================================================")
    print("          AI-POWERED SMS & EMAIL SPAM DETECTION SYSTEM PIPELINE                  ")
    print("================================================================================")
    
    # Define primary directories
    dataset_file = "spam.csv"
    outputs_dir = "outputs"
    
    # Step 1: Initialize NLTK
    download_nltk_dependencies()
    
    # Step 2: Load and Preprocess Dataset
    df = load_and_preprocess_data(dataset_file)
    
    # Step 3: Run EDA (Exploratory Data Analysis)
    perform_eda(df, output_dir=outputs_dir)
    
    # Step 4: Text Cleaning
    df_preprocessed, stemmer, stop_words = preprocess_text_dataset(df)
    
    # Step 5: TF-IDF Vectorization & Train/Test Splitting (80:20)
    # The saved files tfidf_vectorizer.joblib and spam_ann_model.keras will be written to the root directory
    X_train, X_test, y_train, y_test, vectorizer = vectorize_and_split(df_preprocessed, output_dir=".")
    
    # Step 6: Train Machine Learning (Logistic Regression)
    lr_model, lr_metrics, lr_preds, lr_probs = train_logistic_regression(X_train, y_train, X_test, y_test)
    
    # Step 7: Train Deep Learning (ANN with TensorFlow/Keras)
    ann_model, ann_history, ann_metrics, ann_preds, ann_probs = train_ann_model(
        X_train, y_train, X_test, y_test, 
        input_dim=X_train.shape[1], 
        output_dir="."
    )
    
    # Step 8: Plot and save performance metrics
    plot_performance_graphs(
        ann_history, y_test, lr_probs, ann_probs, lr_preds, ann_preds,
        lr_metrics['confusion_matrix'], ann_metrics['confusion_matrix'],
        output_dir=outputs_dir
    )
    
    # Step 9: Print comparative summary metrics
    print_performance_comparison(lr_metrics, ann_metrics)
    
    # Step 10: Launch Interactive Predictor Loop
    start_interactive_predictor(vectorizer, lr_model, ann_model, stemmer, stop_words)


if __name__ == "__main__":
    main()
