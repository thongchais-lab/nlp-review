# 🐦 Twitter Sentiment Analysis with NLP

This repository contains a **Natural Language Processing (NLP)** project focused on classifying the sentiment of tweets related to various entities such as **brands, products, and games**.  
The project walks through the **entire data science pipeline**  from raw data cleaning to high-performance machine learning classification.

---

## 📊 Project Overview

Using a dataset of approximately **75,000 tweets**, this project classifies social media mentions into **four sentiment categories**:

- ✅ **Positive**
- ❌ **Negative**
- 😐 **Neutral**
- 🚫 **Irrelevant**

This analysis is especially valuable for **brand sentiment monitoring**, **customer feedback analysis**, and **market research**.

---

## 🛠️ Features

### 🔹 Data Cleaning
- Handling missing values  
- Removing duplicate tweets  

### 🔹 Text Preprocessing
- Lowercasing
- Removing URLs, mentions (@), hashtags (#)
- Removing non-alphabetic characters
- Tokenization and lemmatization using ```WordNetLemmatizer```
- Stopword removal and filtering short words (< 2 characters)

### 🔹 Exploratory Data Analysis (EDA)
- Visualization of sentiment distribution  
- Entity frequency analysis  

### 🔹 Modeling
- Implementation and comparison of multiple classifiers  

### 🔹 Performance Evaluation
- Accuracy  
- F1-Score  
- Confusion Matrices  

---

## 🧪 Tech Stack

- **Language:** Python 🐍  
- **Data Analysis:** Pandas, NumPy  
- **Visualization:** Matplotlib, Seaborn  
- **NLP:** NLTK (Natural Language Toolkit)  
- **Machine Learning:** Scikit-learn  
  - TF-IDF Vectorization  
  - Random Forest  
  - Logistic Regression  
  - Multinomial Naive Bayes  

---

## 📈 Model Performance

Multiple models were evaluated using a train-test split.  
The **Random Forest classifier** achieved the best performance.

| Model | Accuracy |
|------|----------|
| 🌲 Random Forest | ~86% |
| 📊 Logistic Regression | ~68% |
| 📉 Multinomial Naive Bayes | ~63% |

---

## 📁 Dataset

The project uses the `twitter_training.csv` dataset, which includes:

- **Tweet ID**  
- **Entity** – Brand or game being discussed  
- **Sentiment** – Target label  
- **Tweet Content**  

---

## 🚀 Installation & Usage

### 1. Clone the repository
```bash
git clone <repository-url>
cd twitter-sentiment-analysis
```

### 2. Install dependencies
```text
please see the requirement in the file <3
```

### 3. Download NLTK data
```python
import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
```
### 4. Run the analysis
Open and execute ```nlp-twitter.ipynb``` in Jupyter Notebook or JupyterLab.

---

## 📊 Key Insights
### 1. Data Distribution  
The dataset contains tweets about various entities including games (Borderlands, Call Of Duty), tech companies (Nvidia, Microsoft), and social media platforms.

### 2.Model Performance  
Random Forest significantly outperformed other models, suggesting that ensemble methods work well for this sentiment classification task.

### 3.Sentiment Challenges  
The model performs best on Negative and Neutral classifications, while Irrelevant tweets have the highest precision but lower recall.

---
## 🔍 Future Improvements
### 1. Advanced Text Processing:
- Implement word embeddings (Word2Vec, GloVe)
- Try BERT or other transformer models

### 2.Model Enhancement:
- Hyperparameter tuning for better performance
- Ensemble methods combining multiple classifiers
- Deep learning approaches (LSTM, CNN for text)

---
## 📝 Note
The project demonstrates a complete NLP pipeline from data cleaning to model deployment. The Random Forest model with TF-IDF features provides a robust baseline for sentiment classification on Twitter data.
