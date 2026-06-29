# AI-Powered Email Spam Detection using Machine Learning and Deep Learning

An end-to-end Python-based mini-project designed to classify email and SMS messages into **Spam** or **Ham (Not Spam)**. This project implements and compares a classical Machine Learning model (**Logistic Regression**) against a Deep Learning architecture (**Artificial Neural Network - ANN** using TensorFlow/Keras).

---

## 📌 Project Overview
Spam email detection is a crucial aspect of cybersecurity and email service quality. This project demonstrates how text mining, Natural Language Processing (NLP), and artificial intelligence can be applied to classify messages. It utilizes the **Kaggle SMS Spam Collection dataset**, processes raw texts, extracts TF-IDF numerical features, and compares Logistic Regression (ML) with an Artificial Neural Network (DL) on multiple classification criteria: **Accuracy, Precision, Recall, F1 Score, and ROC-AUC**.

### Key Features
*   **Automatic Dependency Solver**: Programmatically checks and downloads required NLTK resources (`punkt`, `punkt_tab`, and `stopwords`) at runtime.
*   **EDA Data Visualization**: Generates distribution statistics and plots (bar, pie, word count frequency, character length histogram) saved automatically in the `outputs/` folder.
*   **Robust Text Preprocessing Pipeline**: Tokenizes text, removes punctuation/HTML tags/special characters/numbers, filters English stopwords, and stems words using the Porter Stemmer.
*   **TF-IDF Feature Representation**: Converts text into numerical vectors using TF-IDF (4000 max features).
*   **Dual Model Architecture**:
    *   **Logistic Regression (ML)**: Fast baseline classifier optimized with the liblinear solver.
    *   **Artificial Neural Network (DL)**: Keras Sequential model with Input, two Dense hidden layers with ReLU activation, Dropout regularizers to prevent overfitting, and a Sigmoid output layer.
*   **Performance Comparison Metrics**: Builds a comparison dashboard table and saves evaluation graphs (Learning Curves, Confusion Matrices, ROC curves).
*   **Interactive CLI Console**: A command-line program that predicts the category (Spam/Ham) and confidence percentage of any text entered by the user in real time.
*   **Model Persistence**: Saves the trained Keras model (`spam_ann_model.keras`) and TF-IDF vectorizer (`tfidf_vectorizer.joblib`) to disk for instant deployment.

---

## 📂 Folder Structure
The project directory is structured as follows:
```
Email_Spam_Project/
├── spam.csv                      # Source SMS Spam Collection dataset (Kaggle)
├── spam_detection.py             # Core Python script containing the pipeline
├── requirements.txt              # Python library dependencies list
├── README.md                     # Comprehensive setup and run instructions
├── project_report.md             # Detailed academic project report
├── sample_output.txt             # Simulated console output log of training
├── folder_structure.txt          # Visual directory hierarchy overview
└── outputs/                      # Saved plots and figures
    ├── bar_chart.png             # Class distribution: Bar chart
    ├── pie_chart.png             # Class distribution: Pie chart
    ├── msg_length_dist.png       # Message character length distribution
    ├── word_count_dist.png       # Message word count distribution
    ├── training_curves.png       # ANN model loss and accuracy epoch history
    └── model_metrics_plot.png    # ROC-AUC curve & side-by-side Confusion Matrices
```

---

## 🛠️ Technology Stack
*   **Programming Language**: Python 3.10+
*   **Data Science Utilities**: NumPy, Pandas
*   **Natural Language Processing**: NLTK (Natural Language Toolkit)
*   **Data Visualization**: Matplotlib
*   **Machine Learning framework**: Scikit-Learn
*   **Deep Learning framework**: TensorFlow 2.10+ / Keras
*   **Model Serialization**: Joblib

---

## ⚙️ Installation and Setup

### 1. Clone or Move to the Project Directory
Navigate to the directory containing the project:
```bash
cd Email_Spam_Project
```

### 2. Install Required Python Libraries
Use pip to install all dependencies listed in the requirements file:
```bash
pip3 install -r requirements.txt
```
*(Dependencies: pandas, numpy, matplotlib, scikit-learn, tensorflow, joblib, nltk)*

### 3. Place the Dataset File
Ensure that `spam.csv` (the SMS Spam Collection dataset) is in the root directory.

---

## 🚀 How to Run the Project
To run the end-to-end preprocessing, training, evaluation, plot generation, and interactive predictor, execute the main script:
```bash
python3 spam_detection.py
```

### Script Execution Workflow
1.  **Dependency Check**: Downloads NLTK corpora automatically.
2.  **Dataset Preprocessing**: Loads `spam.csv`, drops empty columns, maps labels to `0`/`1`, and displays missing value audits.
3.  **EDA Plotting**: Generates and saves four plots in the `outputs/` folder.
4.  **Text Clean & Split**: Applies cleaning, tokenization, stemming, extracts TF-IDF, and splits training and testing sets (80:20).
5.  **ML Training**: Trains Logistic Regression, displaying precision, recall, and a confusion matrix.
6.  **DL Training**: Compiles and fits the Keras ANN model for 15 epochs, saving epoch loss and accuracy outputs.
7.  **Plotting Graphs**: Saves ANN loss/accuracy curves and comparative ROC curves/confusion matrices to `outputs/`.
8.  **Model Comparison**: Displays a Markdown table comparing model performance.
9.  **Interactive Predictor**:
    *   *If run in an interactive shell*: Prompts you with `Enter your message: ` to write custom messages and view class probability predictions from both classifiers.
    *   *If run in a non-interactive shell*: Executes a series of pre-defined testing emails automatically to verify functional correctness.

---

## 📊 Summary of Model Performance
Typical model results on the test split:

| Evaluation Metric | Logistic Regression (ML) | Neural Network (DL) |
| :--- | :---: | :---: |
| **Accuracy** | ~97.04% | **~98.30%** |
| **Precision** | **~99.15%** | ~98.51% |
| **Recall** | ~78.52% | **~88.59%** |
| **F1 Score** | ~87.64% | **~93.29%** |
| **ROC-AUC Score** | **~98.47%** | ~98.31% |

*Note: Results may vary slightly depending on weight initializations during ANN training.*

---

## 🧠 Interactive Console Example
```text
Enter your message: Urgently urgent! Please call 09061104282 from landline. £5000 cash balance waiting! Claim code: K42. Valid 12 hrs only.

------------------------------------------------------------
INPUT MSG: "Urgently urgent! Please call 09061104282 from landline. £5000 cash balance waiting! Claim code: K42. Valid 12 hrs only."
PREPROCESSED TEXT: 'urgent urgent pleas call landlin cash balanc wait claim code valid hr'
------------------------------------------------------------
1. Machine Learning (Logistic Regression):
   => Label:      SPAM
   => Confidence: 89.8788% probability of being spam
2. Deep Learning (Artificial Neural Network):
   => Label:      SPAM
   => Probability: 99.9998% likelihood of being spam
------------------------------------------------------------
```
