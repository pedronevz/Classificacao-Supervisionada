import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
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

# remover linhas com NaN
dados.dropna(subset=['Label'], inplace=True)

# pré-processar textos
dados['Text'] = dados['Text'].apply(preprocess_text) # o método apply aplica pré-processamento a cada linha na coluna 'Text'

textos = dados['Text']
labels = dados['Label']

# textos em vetores
vectorizer = TfidfVectorizer()
X_tfidf = vectorizer.fit_transform(textos) # transforma os textos em vetores TF-IDF, no formato '(i, j) valor'

# divisão dos dados em treinamento e teste
xTrain, xTest, yTrain, yTest = train_test_split(X_tfidf, labels, test_size=0.3, random_state=42) # 30% dos dados para teste, 70% para treino

# treinar o classificador k-NN
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(xTrain, yTrain) 

# previsões nos dados de teste
predicao = knn.predict(xTest)

# avaliar o desempenho do modelo
print("Accuracy:", accuracy_score(yTest, predicao))
print(classification_report(yTest, predicao))


## Gráficos
# plotar gráfico PCA (padrão)
plt.figure(figsize=(10, 5))
pca = PCA(n_components=2)
xTest_2D = pca.fit_transform(xTest.toarray())

plt.subplot(1, 2, 1)
for i, class_label in enumerate(['ai', 'student']):
    plt.scatter(xTest_2D[yTest == class_label, 0], xTest_2D[yTest == class_label, 1], 
                label=class_label, alpha=0.5)

plt.scatter(xTest_2D[yTest == 'ai', 0], xTest_2D[yTest == 'ai', 1], color='blue', alpha=0.5, label='AI')
plt.scatter(xTest_2D[yTest == 'student', 0], xTest_2D[yTest == 'student', 1], color='orange', alpha=0.5, label='Student')
plt.title('PCA')
plt.xlabel('1')
plt.ylabel('2')

# ROC e AUC
# probabilidades de previsão
yProb = knn.predict_proba(xTest)

""" 
### caso queiramos ver as frases e as probabilidades de previsão de cada uma
    original_texts = vectorizer.inverse_transform(xTest)
    original_texts = [" ".join(text) for text in original_texts]

    for text, prob in zip(original_texts, yProb):
        print(f"Frase: {text}")
        print(f"Probabilidades: {prob}")
        print()
"""    

# binarizar as labels
yTestBin = label_binarize(yTest, classes=['ai', 'student'])
n_classes = yTestBin.shape[1]

# inicializar listas para ROC e AUC
fp = dict()
tp = dict()
roc_auc = dict()

for i in range(n_classes):
    fp[i], tp[i], _ = roc_curve(yTestBin[:, i], yProb[:, i])
    roc_auc[i] = auc(fp[i], tp[i])

# calcular AUC
all_fp = np.unique(np.concatenate([fp[i] for i in range(n_classes)]))
mean_tp = np.zeros_like(all_fp)

for i in range(n_classes):
    mean_tp += np.interp(all_fp, fp[i], tp[i])

mean_tp /= n_classes

fp["macro"] = all_fp
tp["macro"] = mean_tp
roc_auc["macro"] = auc(fp["macro"], tp["macro"])

# plotar grafico ROC
plt.subplot(1, 2, 2)

colors = ['blue', 'red']
for i, color in zip(range(n_classes), colors):
    plt.plot(fp[i], tp[i], color=color, lw=2,
             label='ROC curve of class {0} (area = {1:0.2f})'
                   ''.format(['ai', 'student'][i], roc_auc[i]))

plt.plot(fp["macro"], tp["macro"], color='red', linestyle='--',
         label='macro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["macro"]))

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")

plt.tight_layout()
plt.show()

