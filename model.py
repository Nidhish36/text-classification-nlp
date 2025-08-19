# UI
import anvil.server

# Data handling
import numpy as np
import pandas as pd

# Visualization
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# Preprocessing & ML
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Deep Learning
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential

# SERVER CONNECTION
anvil.server.connect('server_RAIK4MK2PUAL6TJ5G5OMJEP7-LMSKAUGN3O7UN364')


df = pd.read_csv('C:\\Users\\nidhi\\Downloads\\combined_emotion.csv')

print(df.isnull().sum().sum())
print(df.shape)
print(df['emotion'].value_counts())


plt.figure(figsize=(7, 4))
sns.countplot(x=df['emotion'], palette="coolwarm")
plt.title("Class Distribution")
plt.xlabel("Labels")
plt.ylabel("Count")
plt.show()

# Word Cloud
text = " ".join(df["sentence"])
wordcloud = WordCloud(width=800, height=400, background_color="white").generate(text)

plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.title("Most Frequent Words in the Dataset")
plt.show()

# LABEL ENCODING 
label_encoder = LabelEncoder()
df["label"] = label_encoder.fit_transform(df["emotion"])
num_classes = len(label_encoder.classes_)

# TRAIN TEST SPLIT 
X_train, X_test, y_train, y_test = train_test_split(
    df["sentence"], df["label"],
    test_size=0.2, random_state=100, stratify=df["label"]
)

# TF-IDF 
tfidf = TfidfVectorizer(lowercase=True, stop_words="english", max_features=5000)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# LOGISTIC REGRESSION
lr = LogisticRegression(multi_class="auto", max_iter=1000)
lr.fit(X_train_tfidf, y_train)

# NAIVE BAYES 
nb = MultinomialNB()
nb.fit(X_train_tfidf, y_train)

# CNN PREPROCESSING
max_words = 10000
max_length = 100

tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
tokenizer.fit_on_texts(X_train)

train_sequences = tokenizer.texts_to_sequences(X_train)
test_sequences = tokenizer.texts_to_sequences(X_test)

X_train_pad = pad_sequences(train_sequences, maxlen=max_length, padding="post", truncating="post")
X_test_pad = pad_sequences(test_sequences, maxlen=max_length, padding="post", truncating="post")

y_train_oh = tf.keras.utils.to_categorical(y_train, num_classes=num_classes)
y_test_oh = tf.keras.utils.to_categorical(y_test, num_classes=num_classes)

# CNN MODEL 
embedding_dim = 64
nn_model = Sequential([
    Embedding(input_dim=max_words, output_dim=embedding_dim, input_length=max_length),
    Flatten(),
    Dense(128, activation="relu"),
    Dense(64, activation="relu"),
    Dense(64, activation="relu"),
    Dropout(0.5),
    Dense(num_classes, activation="softmax")
])

nn_model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
early_stop = EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)

nn_model.fit(
    X_train_pad, y_train_oh,
    validation_split=0.1,
    epochs=10,
    batch_size=32,
    callbacks=[early_stop],
    verbose=1
)

# MODEL EVALUATION
lr_probs = lr.predict_proba(X_test_tfidf)
nb_probs = nb.predict_proba(X_test_tfidf)
nn_probs = nn_model.predict(X_test_pad)

train_acc_nn = nn_model.evaluate(X_train_pad, y_train_oh)[1]
test_acc_nn = nn_model.evaluate(X_test_pad, y_test_oh)[1]

print(f"NN Train Accuracy: {train_acc_nn:.2f}")
print(f"NN Test Accuracy: {test_acc_nn:.2f}")

# Ensemble via soft-voting
ensemble_probs = (lr_probs + nb_probs + nn_probs) / 3.0
ensemble_preds = np.argmax(ensemble_probs, axis=1)

accuracy = accuracy_score(y_test, ensemble_preds)
print("Ensemble Accuracy: {:.2f}%".format(accuracy * 100))
print("\nClassification Report (Ensemble):")
print(classification_report(y_test, ensemble_preds, target_names=label_encoder.classes_))

lr_pred = np.argmax(lr_probs, axis=1)
nb_pred = np.argmax(nb_probs, axis=1)
nn_pred = np.argmax(nn_probs, axis=1)

print("\nIndividual Model Accuracies:")
print("Logistic Regression: {:.2f}%".format(accuracy_score(y_test, lr_pred) * 100))
print("Naive Bayes:         {:.2f}%".format(accuracy_score(y_test, nb_pred) * 100))
print("NN:                  {:.2f}%".format(accuracy_score(y_test, nn_pred) * 100))

# PREDICTION FUNCTION 
@anvil.server.callable
def predict_emotion(sentence):
    sentence_tfidf = tfidf.transform([sentence])
    lr_prob = lr.predict_proba(sentence_tfidf)
    nb_prob = nb.predict_proba(sentence_tfidf)

    sequence = tokenizer.texts_to_sequences([sentence])
    padded = pad_sequences(sequence, maxlen=max_length, padding="post", truncating="post")
    cnn_prob = nn_model.predict(padded)

    avg_prob = (lr_prob + nb_prob + cnn_prob) / 3.0
    pred_index = np.argmax(avg_prob, axis=1)[0]
    emotion = label_encoder.inverse_transform([pred_index])[0]
    return emotion

# Test prediction
sample_sentence = "hello bad morning"
predicted_emotion = predict_emotion(sample_sentence)
print("Predicted Emotion:", predicted_emotion)

# CONFUSION MATRIX 
def plot_confusion_matrix(ax, y_true, y_pred, model_name):
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                ax=ax, xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title(model_name)

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
plot_confusion_matrix(axes[0, 1], y_test, lr_pred, "Logistic Regression")
plot_confusion_matrix(axes[1, 0], y_test, nb_pred, "Naive Bayes")
plot_confusion_matrix(axes[1, 1], y_test, nn_pred, "NN")
plot_confusion_matrix(axes[0, 0], y_test, ensemble_preds, "Ensemble")

plt.tight_layout()
plt.show()
