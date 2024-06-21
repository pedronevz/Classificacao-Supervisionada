import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import label_binarize
import nltk
from nltk.corpus import stopwords
import string

nltk.download('stopwords') # stopwords, palavras naturais sem influência no programa 


def preprocess_text(text):
    text = text.lower()  # transforma texto em minúsculo
    text = text.translate(str.maketrans('', '', string.punctuation))  # remove pontuação
    # remove stopwords
    stop_words = set(stopwords.words('english'))
    words = text.split() 
    words = [word for word in words if word not in stop_words] 
    return ' '.join(words)

dados = pd.read_csv('LLM.csv')

# remover linhas com NaN
dados.dropna(subset=['Label'], inplace=True)

# pré-processar textos
dados['Text'] = dados['Text'].apply(preprocess_text)

textos = dados['Text']
labels = dados['Label']

        #== SVM ==#
# textos em vetores
vectorizer = TfidfVectorizer()
X_tfidf = vectorizer.fit_transform(textos) # transforma os textos em vetores TF-IDF, no formato '(i, j) valor'

sss = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=42) # 30% dos dados para teste, 70% para treino

# binarizar labels
labels_bin = labels.map({'ai': 0, 'student': 1})

# divisão dos dados em treinamento e teste, com os binarios
for train_index, test_index in sss.split(X_tfidf, labels_bin):
    xTrain, xTest = X_tfidf[train_index], X_tfidf[test_index]
    yTrain, yTest = labels_bin.iloc[train_index], labels_bin.iloc[test_index]

# treinar o classificador SVM
svm = SVC(probability=True)
svm.fit(xTrain, yTrain)

# previsões nos dados de teste
predicao = svm.predict(xTest)

# avaliar o desempenho do modelo
print("Accuracy:", accuracy_score(yTest, predicao))
print(classification_report(yTest, predicao))

""" 
### caso queiramos ver as frases e as probabilidades de previsão de cada uma
    original_texts = vectorizer.inverse_transform(xTest)
    original_texts = [" ".join(text) for text in original_texts]

    for text, prob in zip(original_texts, yProb):
        print(f"Frase: {text}")
        print(f"Probabilidades: {prob}")
        print()
"""

## Gráficos
# calcular probabilidade prevista
probPredicao= svm.predict_proba(xTest)[:, 1]

# taxas de falso positivo e verdadeiro positivo
fp, tp, _ = roc_curve(yTest, probPredicao, pos_label=1)

# ROC e AUC
roc_auc = auc(fp, tp)

# plotar a curva ROC
plt.figure()
plt.plot(fp, tp, color='red', lw=2, label=f'Curva ROC para SVM (área = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Taxa de Falsos Positivos')
plt.ylabel('Taxa de Verdadeiros Positivos')
plt.title('Curva ROC para SVM')
plt.legend(loc="lower right")
plt.show()

# calcular a matriz de confusão
cm = confusion_matrix(yTest, predicao)

# plotar a matriz de confusão
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=svm.classes_)
disp.plot(cmap=plt.cm.Blues)
plt.show()
