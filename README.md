# VARS-NLP
# VARS-NLP-Project

## Quality Control Report Analysis  
**Course:** NLP (Semester 6) - Pillai College of Engineering  

---

## Project Overview  
This project focuses on classifying manufacturing quality reports into predefined categories such as **compliant, minor defects,** or **major issues**. By leveraging machine learning and deep learning models, the system enables quality assurance teams to prioritize product batches, identify critical defects faster, and enhance overall production efficiency.

---

## Acknowledgements  
We would like to express our sincere gratitude to the following individuals:

### **Theory Faculty:**  
- Dhiraj Amin  
- Sharvari Govilkar  

### **Lab Faculty:**  
- Dhiraj Amin  
- Neha Ashok  
- Shubhangi Chavan  

Their guidance and support have been invaluable throughout this project.

---

## Project Abstract  
Quality assurance plays a vital role in maintaining product standards and reducing production defects in the manufacturing industry. This project leverages **natural language processing (NLP)** to analyze textual data from quality reports and classify them into categories: **compliant, minor defects, and major issues**.  

The dataset consists of two key columns:
- **Report Text** - The content of the quality report.  
- **Category** - The assigned quality status.  

Through preprocessing techniques such as **tokenization, stopword removal, and lemmatization**, raw text is transformed into a structured format. The classification is performed using **machine learning, deep learning, and large language models (LLMs)** to enhance automation, improve defect detection efficiency, and minimize production waste.

---

## Algorithms Used  
### **Machine Learning Algorithms**  
- Logistic Regression  
- Support Vector Machine (SVM)  
- Random Forest Classifier  

### **Deep Learning Algorithms**  
- Convolutional Neural Networks (CNN)  
- Bidirectional Long Short-Term Memory (BILSTM)  
- Long Short-Term Memory (LSTM)  
- Gated Recurrent Unit (GRU)  
- Multilayer Perceptron (MLP)  

### **Language Models**  
- ROBERTA (Robustly Optimized BERT Approach)  
- BERT (Bidirectional Encoder Representations from Transformers)  

---

## Comparative Analysis  
Below is a comparative analysis of the different models used in the project:

| Model | Notes/Predictions |
|--------|-----------------|
| **Logistic Regression** | High accuracy for TF-IDF, struggles with imbalanced data. |
| **SVM** | Performs well with TF-IDF, sensitive to feature scaling. |
| **Random Forest** | High accuracy, handles imbalanced data better. |
| **CNN** | Strong with word embeddings but overfits on small datasets. |
| **BILSTM** | Slightly better with embeddings, but still weak. |
| **GRU** | Performs similarly to LSTM but slightly better in efficiency. |
| **MLP** | Works well on structured data but lacks deep contextual understanding. |
| **Transformer** | Struggles with small datasets, requires large training data. |
| **BERT** | Predictions: (Rejected, Approved, Rejected) for three input texts. |
| **ROBERTA** | Predictions: (Rejected, Rejected, Approved) for three input texts. |

---
# Model Performance Comparison

## **1️⃣ Machine Learning Models (BoW, TF-IDF, FastText, Combined Features)**

| Model            | BoW Features | TF-IDF Features | FastText Features | All Combined Features |
|-----------------|-------------|----------------|------------------|----------------------|
| SVM            | 0.8752      | 0.8752         | 0.8451           | 0.8193               |
| Random Forest  | 0.5601      | 0.4232         | 0.4860           | 0.8677               |
| Logistic Regression | 0.8430 | 0.8602         | 0.8193           | 0.8795               |
| KNN            | 0.8225      | 0.8322         | 0.8258           | 0.8387               |
| Decision Tree  | 0.7741      | 0.7677         | 0.6548           | 0.7612               |
| Naïve Bayes    | 0.8645      | 0.8709         | 0.8709           | 0.8677               |

---

## **2️⃣ Deep Learning Models (CNN, LSTM, CNN-BiLSTM)**

| Model        | Accuracy | Recall | F1-Score | Support |
|-------------|----------|--------|----------|---------|
| CNN         | 0.30322  | 0.00   | 0.00     | 42      |
| LSTM        | 0.3624   | 0.053  | 0.061    | 43      |
| CNN-BiLSTM  | 0.3014   | 0.012  | 0.035    | 42      |

---

## **3️⃣ Transformer Models (BERT, RoBERTa)**
| Model  | Accuracy | MCC   |
|--------|---------|--------|
| BERT   | 0.8806  | 0.7623 |
| RoBERTa| 0.8839  | 0.8279 |

---

## Insights
- **Machine Learning Models:** Logistic Regression and SVM performed the best with high accuracy across all feature sets.
- **Deep Learning Models:** LSTM performed slightly better than CNN and CNN-BiLSTM but was still significantly weaker than traditional ML models.
- **Transformer Models:** RoBERTa outperformed BERT in both Accuracy and MCC, making it the best among deep learning approaches.

---


## Conclusion  
This project effectively utilizes **machine learning and NLP** techniques to classify manufacturing quality reports into predefined categories. The application of **tokenization, stopword removal, and lemmatization** improves the model's accuracy by converting raw textual data into a structured format. 

By automating the classification process, this system allows **quality assurance teams to prioritize product batches, quickly identify major defects, and improve overall production quality**. This approach not only minimizes manual effort but also helps reduce waste and inefficiencies, leading to a more streamlined and optimized manufacturing process.

---

## Learn More  
[Pillai College of Engineering](https://www.pce.ac.in/)
