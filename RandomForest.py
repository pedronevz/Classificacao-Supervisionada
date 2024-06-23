import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit, train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score, roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import label_binarize
import nltk
from nltk.corpus import stopwords
import string
from sklearn.decomposition import PCA

nltk.download('stopwords') # stopwords, palavras naturais sem influência no programa 

def preprocess_text(text):
    text = text.lower()
    # remove pontuação
    text = text.translate(str.maketrans('', '', string.punctuation))
    # remove stopwords
    stop_words = set(stopwords.words('english'))
    words = text.split()
    words = [word for word in words if word not in stop_words] # ['expression' for 'item' in 'iterable' if 'condition']
    return ' '.join(words)
    # transforma texto em minúsculo

#print(preprocess_text("Hello! This is a sample text. It includes punctuation, stopwords, and Mixed CASE."))
dados = pd.read_csv('LLM.csv')

# remover linhas com NaN
dados.dropna(subset=['Label'], inplace=True)

# pré-processar textos
dados['Text'] = dados['Text'].apply(preprocess_text) # o método apply aplica pré-processamento a cada linha na coluna 'Text'

textos = dados['Text']
labels = dados['Label']


        #== Random Forest ==#
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

# treinar o classificador RF
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(xTrain, yTrain)

# previsões nos dados de teste
predicao = rf.predict(xTest)

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
plt.figure(figsize=(10, 5))

# plotar gráfico PCA (padrão)
pca = PCA(n_components=2)
xTest_2D = pca.fit_transform(xTest.toarray())

plt.subplot(1, 2, 1)
for i, class_label in enumerate(['ai', 'student']):
    plt.scatter(xTest_2D[yTest == i, 0], xTest_2D[yTest == i, 1], 
                label=class_label, alpha=0.5)

plt.title('PCA')
plt.xlabel('Componente Principal 1')
plt.ylabel('Componente Principal 2')
plt.legend()

# ROC e AUC
# prob de predicao de cada classe
probPredicao = rf.predict_proba(xTest)[:, 1]

# taxas de falso positivo e verdadeiro positivo
fp, tp, _ = roc_curve(yTest, probPredicao, pos_label=1)

roc_auc = auc(fp, tp)

# plotar gráfico ROC
plt.subplot(1, 2, 2)
plt.plot(fp, tp, color='red', lw=2, label=f'Curva ROC para Random Forest (área = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Taxa de Falsos Positivos')
plt.ylabel('Taxa de Verdadeiros Positivos')
plt.title('Curva ROC para Random Forest')
plt.legend(loc="lower right")
plt.grid(True)

plt.tight_layout()
plt.show()


# calcular a matriz de confusão
cm = confusion_matrix(yTest, predicao)

# plotar a matriz de confusão
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=rf.classes_)
disp.plot(cmap=plt.cm.Blues)
plt.show()