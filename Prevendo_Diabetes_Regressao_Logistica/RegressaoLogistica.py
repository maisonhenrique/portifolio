# Importando as Bibliotecas
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from imblearn.over_sampling import SMOTE
from sklearn.metrics import confusion_matrix, classification_report
import warnings
warnings.filterwarnings('ignore')

# Base de Dados
df = pd.read_csv('diabetes.csv', sep=',')

# Renomeando as Colunas
df.rename(columns={
    'Pregnancies': 'Número_Gestações',
    'Glucose': 'Glicose',
    'BloodPressure': 'Pressao_sanguinea',
    'SkinThickness': 'Espessura_Pele',
    'Insulin': 'Insulina',
    'BMI': 'IMC',
    'DiabetesPedigreeFunction': 'Funcao_HistoricoFamiliar',
    'Age': 'Idade',
    'Outcome': 'Resultado'}, inplace=True)

print(df)
df.info()

check = df['Resultado'].value_counts(normalize=True) * 100
print(check)

# Criando o Modelo - Scikit-learn

# Matriz de Variáveis
X = df.drop(columns='Resultado')
y = df['Resultado']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)

print('Dados de Treino: ', len(X_train))
print('Dados de Teste: ', len(X_test))

# Regressão Logistica
logistic_model = LogisticRegression()
logistic_model.fit(X_train, y_train)

# Avaliação do Modelo
Previsao = logistic_model.predict(X_test)
print('Matriz Confusão: \n', confusion_matrix(y_test, Previsao), '\n')

f, ax = plt.subplots()
ax = sns.heatmap(confusion_matrix(y_test, Previsao), annot=True)
ax.set_title('Matriz Confusão', fontsize=10, loc='left', pad=13)
plt.tight_layout()
plt.show()

# Métricas de Classificação - Relatório de Clasfficação
print('Relatório de Clasfficação - Regressão Logistica: \n\n', classification_report(y_test, Previsao))

# Previsão Balanceada
print('Score (Treino): ', round(logistic_model.score(X_train, y_train), 2))
print('Score (Teste): ', round(logistic_model.score(X_test, y_test), 2))

# Validação Cruzada
Validacao_Cruzada = cross_val_score(logistic_model, X, y, cv=5)
print(Validacao_Cruzada)

# Random Forest Classifier
forest_model = RandomForestClassifier(max_depth=3)
forest_model.fit(X_train, y_train)

Previsao_forest = forest_model.predict(X_test)
print('Relatório de Clasfficação - Random Forest: \n\n', classification_report(y_test, Previsao_forest))

# SVM
svm_model = svm.SVC()
svm_model.fit(X_train, y_train)

Previsao_svm = svm_model.predict(X_test)
print('Relatório de Clasfficação - SVM: \n\n', classification_report(y_test, Previsao_svm))


# Hiperparamentros
xtrain2, xtest2, ytrain2, ytest2 = train_test_split(X, y, test_size=0.2, random_state=10)

smote = SMOTE()
xtrainsmote, ytrainsmote = smote.fit_resample(xtrain2, ytrain2)

param_grid = [
    {
    'C': np.logspace(-4, 4, 20),
    'max_iter': [100, 1000, 2500, 5000],
    }
]

logModel = LogisticRegression(solver='liblinear')
clf = GridSearchCV(logModel, param_grid=param_grid, cv=10, verbose=True, n_jobs=-1)
best_clf = clf.fit(xtrainsmote, ytrainsmote)

print(best_clf.best_estimator_)
print(best_clf.best_params_)

finallr = LogisticRegression(C=1.623776739188721, max_iter=100, solver='liblinear')

lrfinalmod = finallr.fit(xtrainsmote, ytrainsmote)
preds = lrfinalmod.predict(xtest2)

print('Relatório GridSearchCV: \n\n', classification_report(ytest2, finallr.predict(xtest2)))
