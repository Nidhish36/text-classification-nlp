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
          precision    recall  f1-score   support

   anger       0.92      0.91      0.92     11863
    fear       0.88      0.86      0.87      9930
     joy       0.91      0.96      0.94     28614
    love       0.90      0.71      0.80      6911
     sad       0.94      0.96      0.95     24238
 surprise     0.80      0.72      0.76      2994

accuracy                           0.91     84550


