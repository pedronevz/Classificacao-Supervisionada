import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from sklearn.metrics import classification_report, accuracy_score, roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay
import nltk
from nltk.corpus import stopwords
import string
from sklearn.decomposition import PCA

#<-- processamento do texto-->#

nltk.download('stopwords') 

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

                                    #=== QDA ===#

#<-- vetorização dos textos e divisão dos dados em conjuntos de treinamento e teste -->#

vectorizer = TfidfVectorizer()
X_tfidf = vectorizer.fit_transform(textos)

sss = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=42)

# binarização dos classes
labels_bin = labels.map({'ai': 0, 'student': 1})

for train_index, test_index in sss.split(X_tfidf, labels_bin):
    xTrain, xTest = X_tfidf[train_index], X_tfidf[test_index]
    yTrain, yTest = labels_bin.iloc[train_index], labels_bin.iloc[test_index]

#<-- treinamento do classificador QDA e previsões nos dados de testes -->#

qda = QDA()
qda.fit(xTrain.toarray(), yTrain)  

predicao = qda.predict(xTest.toarray())

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

#<-- Gráficos -->#

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

#<-- ROC e AUC -->#
# probabilidade de predicao de cada classe
probPredicao = qda.predict_proba(xTest.toarray())[:, 1]

# taxas de falso positivo e verdadeiro positivo
fp, tp, _ = roc_curve(yTest, probPredicao, pos_label=1)
roc_auc = auc(fp, tp)

# plotar gráfico ROC
plt.subplot(1, 2, 2)
plt.plot(fp, tp, color='red', lw=2, label=f'Curva ROC para QDA (área = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Taxa de Falsos Positivos')
plt.ylabel('Taxa de Verdadeiros Positivos')
plt.title('Curva ROC para QDA')
plt.legend(loc="lower right")
plt.grid(True)

plt.tight_layout()
plt.show()

# calcular e plotar a matriz de confusão
cm = confusion_matrix(yTest, predicao)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['ai', 'student'])
disp.plot(cmap=plt.cm.Blues)
plt.show()
