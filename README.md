# Text-Classification-NLP 
Ensemble-based Emotion Detection using ML &amp; CNN with Anvil UI
 

## Overview  
This project is an emotion classification system that predicts the underlying emotion in a given text.  
It combines traditional machine learning models (Logistic Regression, Naive Bayes) with a Neural Network and uses ensemble soft-voting for final predictions.  

The system is also connected with Anvil to allow deployment in a simple web interface.  

---

## Dataset  
- Total Samples: 422,746  
- Number of Classes: 6  
  - joy → 143,067  
  - sad → 121,187  
  - anger → 59,317  
  - fear → 49,649  
  - love → 34,554  
  - surprise → 14,972  

---

## Features  
- Data preprocessing with TF-IDF and Tokenization  
- Models used:  
  - Logistic Regression  
  - Naive Bayes  
  - Neural Network (Embedding → Dense layers with Dropout)  
- Ensemble Learning (soft-voting) for improved accuracy  
- Visualizations:  
  - Class distribution  
  - Word Cloud  
  - Confusion matrices for individual and ensemble models  
- Deployable with Anvil server  

---

## Results  

### Individual Model Accuracies
- Logistic Regression: 89.71%  
- Naive Bayes: 85.24%  
- Neural Network: 91.60%  

### Ensemble Accuracy  
- 91.34%  

### Classification Report (Ensemble)  
<img width="589" height="337" alt="image" src="https://github.com/user-attachments/assets/03aea86b-a7e5-4094-9edb-83505aa471cf" />

## Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/Nidhish36/text-classification-nlp.git
cd text-classification-nlp
pip install -r requirements.txt


