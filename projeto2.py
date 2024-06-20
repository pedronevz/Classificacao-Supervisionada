import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit, train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score, roc_curve, auc
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


        #== KNN ==#
# textos em vetores
vectorizer = TfidfVectorizer()
X_tfidf = vectorizer.fit_transform(textos) # transforma os textos em vetores TF-IDF, no formato '(i, j) valor'

sss = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=42) # 30% dos dados para teste, 70% para treino

# divisão dos dados em treinamento e teste
for train_index, test_index in sss.split(X_tfidf, labels):
    xTrain, xTest = X_tfidf[train_index], X_tfidf[test_index]
    yTrain, yTest = labels.iloc[train_index], labels.iloc[test_index]

# treinar o classificador k-NN
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(xTrain, yTrain)

# previsões nos dados de teste
predicao = knn.predict(xTest)

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
plt.subplot(1, 2, 1)
pca = PCA(n_components=2)
xTest_2D = pca.fit_transform(xTest.toarray())

for class_label, color in zip(['ai', 'student'], ['blue', 'orange']):
    plt.scatter(xTest_2D[yTest == class_label, 0], xTest_2D[yTest == class_label, 1],
                label=class_label, alpha=0.5, color=color)

plt.title('PCA')
plt.legend()

# ROC e AUC
aiIdx = 0  # 'ai' classe 0
studentIdx = 1  #'student' classe 1

# labels binarizadas
yTestBinAi = (yTest == 'ai').astype(int)
yTestBinStudent = (yTest == 'student').astype(int)

# prob de predicao de cada classe
probPredicao = knn.predict_proba(xTest)
probPredAi = probPredicao[:, aiIdx]
probPredStudent = probPredicao[:, studentIdx]

# ROC para 'ai'
fpAi, tpAi, _ = roc_curve(yTestBinAi, probPredAi) # false positive e true positive
rocAi = auc(fpAi, tpAi)

# ROC para 'student'
fpStudent, tpStudent, _ = roc_curve(yTestBinStudent, probPredStudent) # false positive e true positive
rocStudent = auc(fpStudent, tpStudent)

# plotar gráfico ROC
plt.subplot(1, 2, 2)
plt.plot(fpAi, tpAi, color='blue', lw=2, label='ROC curve - Class "ai" (AUC = {:.2f})'.format(rocAi))
plt.plot(fpStudent, tpStudent, color='orange', lw=2, label='ROC curve - Class "student" (AUC = {:.2f})'.format(rocStudent))
plt.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Taxa de Falsos Positivos')
plt.ylabel('Taxa de Verdadeiros Positivos')
plt.title('Curvas ROC para Classes "ai" e "student"')
plt.legend(loc="lower right")
plt.grid(True)

plt.tight_layout()
plt.show()



