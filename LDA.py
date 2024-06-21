"""
Código desenvolvido baseado no exemplo constante no curso da USP - SME0822 
Análise Multivariada e Aprendizado Não-Supervisionado, disponível em 
https://edisciplinas.usp.br/course/view.php?id=78145 e 
https://github.com/cibelerusso/AnaliseMultivariadaEAprendizadoNaoSupervisionado:

Referência:
Russo, C. M. (2023). cibelerusso/AnaliseMultivariadaEAprendizadoNaoSupervisionado: 
Análise Multivariada e Aprendizado Não Supervisionado (Version v0.0.0). 
https://doi.org/10.5281/zenodo.10203429.
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import classification_report, accuracy_score, roc_curve, auc
from sklearn.preprocessing import LabelBinarizer
import nltk
from nltk.corpus import stopwords
import string

#<-- PROCESSSAMENTO DO TEXTO -->#

nltk.download('stopwords') # stopwords, palavras naturais sem influência no programa

def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    stop_words = set(stopwords.words('english'))
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return ' '.join(words)

dados = pd.read_csv('LLM.csv')
dados.dropna(subset=['Label'], inplace=True)
dados['Text'] = dados['Text'].apply(preprocess_text)

textos = dados['Text']
labels = dados['Label']

                            #=== LDA ===#

#<-- Vetorização dos textos e divisão dos dados em conjuntos de treinamento e teste -->#

vectorizer = TfidfVectorizer()
X_tfidf = vectorizer.fit_transform(textos)

sss = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=42)

for train_index, test_index in sss.split(X_tfidf, labels):
    xTrain, xTest = X_tfidf[train_index], X_tfidf[test_index]
    yTrain, yTest = labels.iloc[train_index], labels.iloc[test_index]

#<-- Treinamento do classificador LDA e previsões nos dados de testes, com a biblioteca sklearn-->#

lda = LDA()
lda.fit(xTrain, yTrain)

predicao = lda.predict(xTest)

print("Accuracy:", accuracy_score(yTest, predicao))
print(classification_report(yTest, predicao))
