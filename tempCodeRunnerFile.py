import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score
import nltk
from nltk.corpus import stopwords
import string

nltk.download('stopwords') # stopwords, palavras naturais sem influência no programa 

def preprocess_text(text):
    # transforma texto em minúsculo
    text = text.lower()
    # remove pontuação
    text = text.translate(str.maketrans('', '', string.punctuation))
    # remove stopwords
    stop_words = set(stopwords.words('english'))
    words = text.split()
    words = [word for word in words if word not in stop_words] # ['expression' for 'item' in 'iterable' if 'condition']
    return ' '.join(words)

#print(preprocess_text("Hello! This is a sample text. It includes punctuation, stopwords, and Mixed CASE."))
dados = pd.read_csv('LLM.csv')

# Pré-processar textos
dados['Text'] = dados['Text'].apply(preprocess_text) # o método apply aplica pré-processamento a cada linha na coluna 'Text'

textos = dados['Text']
labels = dados['Label']

# Textos em vetores
vectorizer = TfidfVectorizer()
X_tfidf = vectorizer.fit_transform(textos) # transforma os textos em vetores TF-IDF, no formato '(i, j) valor'

# Divisão dos dados em treinamento e teste
xTrain, xTest, yTrain, yTest = train_test_split(X_tfidf, labels, test_size=0.3, random_state=42) # 30% dos dados para teste, 70% para treino

 # Treinar o classificador k-NN
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(xTrain, yTrain) 

# Previsões nos dados de teste
predicao = knn.predict(xTest)

# Avaliar o desempenho do modelo
print("Accuracy:", accuracy_score(yTest, predicao))
print(classification_report(yTest, predicao))