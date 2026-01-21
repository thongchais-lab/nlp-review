Twitter Sentiment Analysis - NLP Project
========================================

📌 Project Overview
-------------------

This project performs sentiment analysis on Twitter data using natural language processing (NLP) techniques and machine learning algorithms. The goal is to classify tweets into different sentiment categories and compare the performance of various classification models.

📊 Dataset
----------

*   **Source**: twitter\_training.csv
    
*   **Original Columns**:
    
    *   Tweet\_ID
        
    *   Entity (e.g., Borderlands, Nvidia, Amazon, etc.)
        
    *   Sentiment (Positive, Negative, Neutral, Irrelevant)
        
    *   Text (Tweet content)
        
*   **Processed Data**: 73,995 tweets after cleaning
    

🔧 Key Features
---------------

### 1\. Data Preprocessing

*   **Column Renaming**: Assigned meaningful names to columns
    
*   **Missing Values**: Removed 686 rows with missing text
    
*   **Text Cleaning**:
    
    *   Converted to lowercase
        
    *   Removed URLs, mentions (@), and hashtags (#)
        
    *   Removed special characters
        
    *   Tokenization and lemmatization
        
    *   Stopword removal
        
    *   Filtered short words (< 2 characters)
        

### 2\. Exploratory Data Analysis

Created visualizations for:

*   **Sentiment Distribution** (Positive, Negative, Neutral, Irrelevant)
    
*   **Top 10 Most Discussed Entities**
    
*   **Text Length Distribution** (with mean length ~45 characters)
    

### 3\. Feature Engineering

*   **TF-IDF Vectorization**: Extracted 5000 features using unigrams and bigrams
    
*   **Text Length Feature**: Added character count of cleaned content
    

### 4\. Model Comparison

Implemented and compared multiple machine learning models:

**Model Accuracy**

| Model | Accuracy |

|------|----------|

| 🌲 Random Forest | ~86% |

| 📊 Logistic Regression | ~68% |

| 📉 Multinomial Naive Bayes | ~63% |

### 5\. Best Performing Model

**Random Forest Classifier** achieved the best results:

*   **Overall Accuracy**: 86.5%
    
*   **Precision/Recall by class**:
    
    *   Irrelevant: 0.93 precision, 0.79 recall
        
    *   Negative: 0.88 precision, 0.89 recall
        
    *   Neutral: 0.89 precision, 0.84 recall
        
    *   Positive: 0.80 precision, 0.91 recall
        

📁 Project Structure
--------------------

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   twitter-sentiment-analysis/  ├── nlp-twitter.ipynb          # Main Jupyter notebook  ├── twitter_training.csv       # Dataset  ├── README.md                  # This file  └── requirements.txt           # Dependencies   `

🛠️ Technical Stack
-------------------

*   **Programming Language**: Python
    
*   **Libraries**:
    
    *   pandas, numpy: Data manipulation
        
    *   matplotlib: Data visualization
        
    *   scikit-learn: Machine learning models and evaluation
        
    *   nltk: Natural language processing
        
    *   re: Regular expressions
        

🚀 Installation & Usage
-----------------------

### 1\. Clone the repository

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   git clone   cd twitter-sentiment-analysis   `

### 2\. Install dependencies

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   pip install -r requirements.txt   `

### 3\. Download NLTK data

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   import nltk  nltk.download('punkt')  nltk.download('wordnet')  nltk.download('stopwords')   `

### 4\. Run the analysis

Open and execute nlp-twitter.ipynb in Jupyter Notebook or JupyterLab

📈 Key Insights
---------------

1.  **Data Distribution**: The dataset contains tweets about various entities including games (Borderlands, CallOfDuty), tech companies (Nvidia, Microsoft), and social media platforms.
    
2.  **Model Performance**: Random Forest significantly outperformed other models, suggesting that ensemble methods work well for this sentiment classification task.
    
3.  **Sentiment Challenges**: The model performs best on Negative and Neutral classifications, while Irrelevant tweets have the highest precision but lower recall.
    

🔍 Future Improvements
----------------------

1.  **Advanced Text Processing**:
    
    *   Implement word embeddings (Word2Vec, GloVe)
        
    *   Try BERT or other transformer models
        
    *   Handle emojis and slang more effectively
        
2.  **Model Enhancement**:
    
    *   Hyperparameter tuning for better performance
        
    *   Ensemble methods combining multiple classifiers
        
    *   Deep learning approaches (LSTM, CNN for text)
        
3.  **Features**:
    
    *   Add sentiment lexicons
        
    *   Include meta-features (presence of exclamation marks, capital letters)
        
    *   Time-based analysis of sentiment trends
        

📝 Note
-------

The project demonstrates a complete NLP pipeline from data cleaning to model deployment. The Random Forest model with TF-IDF features provides a robust baseline for sentiment classification on Twitter data.

👤 Author
---------

Created as an educational project for demonstrating NLP and machine learning techniques on social media data.