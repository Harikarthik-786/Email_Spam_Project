# Project Report: AI-Powered Email Spam Detection using Machine Learning and Deep Learning

**Course**: Academic Mini-Project (Final-Year/Pre-Final Year)  
**Project Title**: AI-Powered Email Spam Detection using Machine Learning and Deep Learning  
**Date**: June 2026  

---

## Abstract
With the exponential rise in electronic communication, the volume of spam messages (unsolicited commercial emails and advertisements) has grown dramatically, causing security risks, network congestion, and reduced productivity. This project presents a robust, comparative framework to identify spam messages using Natural Language Processing (NLP) techniques, classical Machine Learning, and modern Deep Learning. 

Specifically, we utilize the SMS Spam Collection dataset, perform thorough text cleaning (normalization, tokenization, stopword removal, and stemming), extract numerical feature vectors using Term Frequency-Inverse Document Frequency (TF-IDF), and compare two distinct architectures: **Logistic Regression** representing machine learning, and an **Artificial Neural Network (ANN)** representing deep learning. The results show that while Logistic Regression performs exceptionally fast and yields high precision (~99.15%), the Deep Learning ANN model achieves superior overall classification capabilities, raising the classification Accuracy to ~98.30% and significantly improving the Recall (~88.59%) and F1 Score (~93.29%).

---

## 1. Introduction
Electronic mail and short message services (SMS) have become fundamental components of modern communications. However, they are continuously abused by malicious senders, marketers, and fraudsters to distribute spam. Traditional rule-based filters fail to adapt to the highly dynamic nature of spam texts. Thus, artificial intelligence, machine learning, and deep learning techniques have emerged as the state-of-the-art approach to spam filtering.

### 1.1 Problem Statement
The primary challenge in spam detection is to accurately distinguish between legitimate messages ("ham") and unsolicited spam messages ("spam") while minimizing the false positive rate (classifying a critical legitimate email as spam). Spam messages often contain obfuscated spellings, numbers, special characters, and urgent click-bait call-to-actions, requiring a sophisticated NLP system to normalize the text and extract semantic features prior to classification.

### 1.2 Project Objectives
*   To implement a complete text preprocessing pipeline containing normalization, tokenization, stopword removal, and stemming.
*   To represent text documents numerically using TF-IDF Vectorization.
*   To train and tune a classical Machine Learning model (Logistic Regression).
*   To design, compile, train, and validate a Deep Learning model (Artificial Neural Network) using TensorFlow/Keras.
*   To generate analytical plots including class distributions, message length distributions, model accuracy/loss curves, ROC-AUC, and Confusion Matrices.
*   To compare the models across key metrics (Accuracy, Precision, Recall, F1 Score, ROC-AUC) in a consolidated comparison table.
*   To build an interactive predictor CLI enabling users to input custom emails and evaluate the models' predictions in real time.

---

## 2. Methodology and System Design
The proposed system uses a structured machine learning lifecycle. The block diagram below illustrates the sequential stages of the data processing and modeling pipeline:

```
[ Raw CSV Dataset ]
       │
       ▼
[ Data Preprocessing ]  ──► Drop empty columns, Rename variables, Encode labels (0/1)
       │
       ▼
[ NLP Text Cleaning ]   ──► Lowercase, Remove punctuation/special chars, Stemming
       │
       ▼
[ Feature Engineering ] ──► Fit & Transform TF-IDF Vectorizer
       │
       ▼
[ Train/Test Splitting ]──► Partition into 80% Train, 20% Test (Stratified)
       │
 ┌─────┴──────────────────────────────────┐
 │                                        │
 ▼                                        ▼
[ Machine Learning Model ]       [ Deep Learning Model ]
(Logistic Regression)            (Keras Sequential ANN)
 │                                        │
 └─────┬──────────────────────────────────┘
       │
       ▼
[ Evaluation & Plotting ] ──► Confusion Matrices, ROC Curves, Accuracy/Loss curves
       │
       ▼
[ Model Deployment ]     ──► Save serialized weights, Launch Interactive CLI
```

### 2.1 Text Preprocessing Pipeline
Since raw text cannot be directly input into mathematical algorithms, we perform NLP preprocessing to convert sentences into a clean tokenized format:
1.  **Lowercase Conversion**: Reduces word dimensionality by converting all characters to lowercase (e.g., "Urgent" and "urgent" are mapped together).
2.  **Character Stripping**: Removes HTML tags, punctuation, special symbols, and numeric digits using regular expressions (`[^a-z\s]`), isolating pure alphabetical words.
3.  **Tokenization**: Splits the cleaned string into list of individual words using NLTK's `word_tokenize`.
4.  **Stopword Filtering**: Removes frequent grammatical words (e.g., "is", "the", "and") which carry no semantic weight.
5.  **Stemming**: Applies NLTK's `PorterStemmer` to strip suffixes and reduce words to their base root form (e.g., "winning", "wins", "won" are stemmed to "win").

### 2.2 Feature Extraction: TF-IDF
Term Frequency-Inverse Document Frequency (TF-IDF) converts the stemmed text collections into a sparse numerical matrix. The TF-IDF score for word $t$ in document $d$ within a corpus $D$ is calculated as:

$$\text{TF-IDF}(t, d, D) = \text{TF}(t, d) \times \text{IDF}(t, D)$$

Where:
*   $\text{TF}(t, d)$ is the frequency of term $t$ in document $d$.
*   $\text{IDF}(t, D) = \log \left(\frac{1 + |D|}{1 + |\{d \in D : t \in d\}|}\right) + 1$, which penalizes words that appear too frequently across all documents in the corpus.

We restrict our dictionary to the top **4000 vocabulary features** to ensure a high signal-to-noise ratio.

---

## 3. Model Architecture and Training

### 3.1 Machine Learning: Logistic Regression
Logistic Regression is used as our baseline ML model. It models the probability of a message belonging to the spam class ($Y=1$) using a logistic sigmoid function of the input feature vector $X$:

$$P(Y=1 | X) = \sigma(W^T X + b) = \frac{1}{1 + e^{-(W^T X + b)}}$$

Where $W$ represents weight coefficients and $b$ is the bias. The model is trained using the coordinate descent algorithm (`liblinear` solver) with $L_2$ regularization.

### 3.2 Deep Learning: Artificial Neural Network (ANN)
The Deep Learning model is constructed using the Keras Sequential API. It consists of multiple fully-connected (Dense) layers with dropout regularization:

1.  **Input Layer**: Accepts TF-IDF features with dimension matching the vocabulary size (4000).
2.  **Hidden Layer 1**: 64 neurons with ReLU (Rectified Linear Unit) activation, followed by a **Dropout layer (0.3)**. The dropout randomly sets 30% of activation outputs to 0 during each training step to prevent overfitting.
3.  **Hidden Layer 2**: 32 neurons with ReLU activation, followed by a **Dropout layer (0.3)**.
4.  **Output Layer**: 1 neuron with a **Sigmoid activation function** to predict the final spam probability $p \in [0, 1]$.

#### ANN Configuration & Hyperparameters:
*   **Optimizer**: Adam (Adaptive Moment Estimation) with a starting learning rate of $0.001$.
*   **Loss Function**: Binary Cross-Entropy, defined as:
    $$\mathcal{L}(y, \hat{y}) = -\frac{1}{N} \sum_{i=1}^N \left[ y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i) \right]$$
    Where $y_i$ is the actual binary label and $\hat{y}_i$ is the predicted probability.
*   **Epochs**: 15 epochs.
*   **Batch Size**: 32.
*   **Validation Split**: 15% of training samples reserved for monitoring overfitting.

---

## 4. Experimental Results and Discussion

### 4.1 Comparative Evaluation Table
The models were evaluated on the independent test subset (20% split) across five major classification criteria:

| Metric | Logistic Regression (ML) | Neural Network (DL) | Description |
| :--- | :---: | :---: | :--- |
| **Accuracy** | 0.9704 | **0.9830** | Ratio of correctly predicted messages to total samples. |
| **Precision** | **0.9915** | 0.9851 | Out of all predicted spam, how many were actually spam (minimizes False Alarms). |
| **Recall** | 0.7852 | **0.8859** | Out of all actual spam, how many did we capture (minimizes Missed Spam). |
| **F1 Score** | 0.8764 | **0.9329** | Harmonic mean of Precision and Recall. |
| **ROC-AUC** | **0.9847** | 0.9831 | Area under the Receiver Operating Characteristic curve. |

### 4.2 Key Findings
1.  **Accuracy and Generalization**: The Artificial Neural Network achieves a higher accuracy of **98.30%** compared to Logistic Regression's **97.04%**. The hidden layers of the ANN are capable of learning complex, non-linear correlations between term combinations that Logistic Regression cannot capture.
2.  **The Precision-Recall Trade-off**:
    *   Logistic Regression shows slightly higher Precision (99.15% vs 98.51%). It is extremely cautious about classifying an email as spam, resulting in only 2 false positives out of 966 legitimate messages.
    *   However, Logistic Regression suffers from lower Recall (78.52%). It missed 32 spam messages out of 149, classifying them as legitimate.
    *   The Deep Learning ANN model significantly improves Recall to **88.59%**, catching 132 out of 149 spam messages while maintaining an excellent Precision of 98.51% (only 2 false positives).
3.  **Overall Superiority**: Because the ANN scores a much higher F1 Score (**93.29%** compared to ML's **87.64%**), it represents a more balanced, reliable, and powerful classifier for practical deployment.

---

## 5. Visualizations and Plots
All generated plots are saved in the `outputs/` folder:
*   `bar_chart.png` and `pie_chart.png` show the severe class imbalance in real-world messaging (86.59% Ham vs 13.41% Spam).
*   `msg_length_dist.png` and `word_count_dist.png` confirm that spam messages tend to be significantly longer and contain more words than ham messages (usually clustering around 120-160 characters for spam versus 30-70 characters for ham).
*   `training_curves.png` plots training/validation accuracy and loss over 15 epochs. The val-loss curve stabilizes around epoch 10, demonstrating optimal convergence with minimal overfitting.
*   `model_metrics_plot.png` contains the side-by-side Confusion Matrices showing the counts of True Positives, True Negatives, False Positives, and False Negatives, along with the ROC curves mapping the True Positive Rate against the False Positive Rate. Both models achieve near-perfect ROC curves, with AUC values exceeding 0.98.

---

## 6. Conclusion and Future Scope
In this project, we successfully built and compared a Machine Learning model (Logistic Regression) and a Deep Learning model (ANN) to detect spam emails. We established that deep neural networks, even with relatively simple structures (two hidden layers, dropout regularizers), are highly effective at processing natural language text representation. 

### Future Work:
1.  **Recurrent Architectures**: Incorporate sequential models like Long Short-Term Memory (LSTM) networks or Gated Recurrent Units (GRUs) to capture word context and order.
2.  **Transformer Models**: Leverage pre-trained Large Language Model representations (e.g., BERT, RoBERTa, or DistilBERT) to achieve near-perfect categorization.
3.  **Real-Time API Integration**: Wrap the saved `.keras` model and joblib TF-IDF vectorizer inside a Flask/FastAPI backend to create a real-time email-filtering web plugin.

---

## 7. References
1.  Kaggle Dataset: SMS Spam Collection Dataset (`spam.csv`).
2.  Pedregosa, F., et al. (2011). "Scikit-learn: Machine Learning in Python." *Journal of Machine Learning Research*.
3.  Abadi, M., et al. (2015). "TensorFlow: Large-Scale Machine Learning on Heterogeneous Systems." *Software available from tensorflow.org*.
4.  Bird, S., Loper, E., & Klein, E. (2009). "Natural Language Processing with Python." *O'Reilly Media*.
