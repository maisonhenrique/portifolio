# Importar as Bibliotecas
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import dataframe_image as dfi
from sklearn.pipeline import make_pipeline
from sklearn import svm
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, RobustScaler, LabelEncoder
from imblearn.over_sampling import ADASYN
from sklearn.model_selection import RandomizedSearchCV
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report

import warnings
warnings.filterwarnings('ignore')

from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)

pd.set_option('display.float_format', lambda x: '%.2f' % x)

#Base de Dados
df = pd.read_csv('Telecom-Customer-Churn.csv')
print(df)
print("O conjunto de dados contém {} linhas e {} colunas".format(df.shape[0], df.shape[1]))

#Verificando dataset

df1 = df.drop(columns=['customerID', 'TotalCharges'])
print(df1)

check_dataset = pd.DataFrame({
                    'type': df1.dtypes,
                    'missing': df1.isna().sum(),
                    'size': df1.shape[0],
                    'unique': df1.nunique()})
check_dataset['percentual'] = round(check_dataset['missing'] / check_dataset['size'], 2)
print(check_dataset)


#Análise Exploratória
df2 = df1.copy()
df2 = df2.drop(columns=['Churn'])

df_columns = df2.select_dtypes(include='object').columns.tolist()

# Gráfico das variáveis categóricas
rc_params = {'axes.spines.top': False,
             'axes.spines.right': False,
             'legend.fontsize': 8,
             'legend.title_fontsize': 8,
             'legend.loc': 'upper right',
             'legend.fancybox': False,
             'axes.titleweight': 'bold',
             'axes.titlesize': 12,
             'axes.titlepad': 12}
sns.set_theme(style='ticks', rc=rc_params)
sns.set_color_codes('muted')

num_plots = len(df_columns)
total_cols = 3
total_rows = 5
fig1, axs = plt.subplots(nrows=total_rows, ncols=total_cols, figsize=(12, 9), constrained_layout=True)

for i, col in enumerate(df_columns):
  row = i//total_cols
  pos = i%total_cols
  sns.countplot(x=col, data=df1, hue='Churn', ax=axs[row][pos])

plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# Porcentagem dos Contratos
contratos = (df[df.Contract == 'Month-to-month'].value_counts().shape[0]/df.shape[0])*100

# Porcentagem dos Contratos Month-to-Month dos clientes churn
contratos_churn = (df[(df.Churn == 'Yes') & (df.Contract == 'Month-to-month')].shape[0]/df[df.Churn == 'Yes'].shape[0])*100

# Porcentagem dos Contratos Month-to-Month dos clientes no churn
contratos_nochurn = (df[(df.Churn == 'No') & (df.Contract == 'Month-to-month')].shape[0]/df[df.Churn == 'No'].shape[0])*100

print('Porcentagem de contratos mensais: {:.2f}%'.format(contratos))
print('Porcentagem de contratos mensais de clientes que não cancelaram: {:.2f}%'.format(contratos_nochurn))
print('Porcentagem de contratos mensais de clientes que cancelaram: {:.2f}%'.format(contratos_churn))

# Quantidade de Churn e No Churn
print(df.Churn.value_counts())
print('\nPorcentagem de Churn {:.2f}% no dataset.\n'.format((df[df.Churn == "Yes"].shape[0] / df.shape[0]) * 100))

sns.countplot(x=df.Churn)
plt.title('Quantidade de Churn e No Churn', fontsize=12, loc='left', pad=10)
plt.tight_layout()
plt.show()

# Comparação dos Contratos
df_churn = df[df.Churn == 'Yes']
df_nochurn = df[df.Churn == 'No']

fig3, ax = plt.subplots(nrows=1, ncols=2, figsize=(15, 6))

sns.countplot(x='Contract', data=df, ax=ax[0])
ax[0].set_title('General Contracts', fontsize=12, loc='left', pad=10)

sns.countplot(x='Contract', data=df_churn, order=['Month-to-month', 'One year', 'Two year'], ax=ax[1])
ax[1].set_title('Churn Contracts', fontsize=12, loc='left', pad=10)

plt.show()


# Limpeza dos Dados
df3 = df.copy()
df3.drop('customerID', axis=1, inplace=True)

df3.replace(['No internet service', 'No phone service'], 'No', inplace=True)
df3.replace(['Bank transfer (automatic)', 'Credit card (automatic)'], 'Automatic', inplace=True)

# Convertendo a coluna TotalCharges para float
def converter_str_float(column):
  try:
    return float(column)
  except ValueError:
    return np.nan

df3['TotalCharges'] = df3['TotalCharges'].apply(converter_str_float)
print('Total Charges nan values: {}'.format(df3['TotalCharges'].isnull().sum()))

# Alterando os valores nulos com a Mediana
df3["TotalCharges"].fillna(df3.TotalCharges.median(), inplace=True)

# Estatistica descritiva
describe = df3.describe()
print(describe)
dfi.export(describe, 'describe.png')

# Plotar gráficos boxplot para verificar a presença de outliers
plt.style.use('seaborn-dark')
fig4, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 4), constrained_layout=True)

sns.boxplot(x=df3['MonthlyCharges'], ax=ax[0])
sns.boxplot(x=df3['TotalCharges'], ax=ax[1])

plt.tight_layout()
plt.show()


# Histograma das variáveis
custom_params = {'axes.spines.right': False,
                 'axes.spines.top': False,
                 'legend.fontsize': 9,
                 'legend.title_fontsize': 10,
                 'legend.loc': 'upper right'}
sns.set_theme(style='ticks', rc=custom_params)
sns.set_color_codes('muted')

fig2, ax = plt.subplots(nrows=3, ncols=1, figsize=(10, 8), constrained_layout=True)

sns.histplot(data=df3, x='tenure', hue='Churn', multiple='stack', ax=ax[0])
ax[0].set_title('Tempo de permanência na operadora', fontsize=12, loc='left', pad=10)

sns.histplot(data=df3, x='MonthlyCharges', hue='Churn', multiple='stack', ax=ax[1])
ax[1].set_title('Valor de pagamento mensal', fontsize=12, loc='left', pad=10)

sns.histplot(data=df3, x='TotalCharges', hue='Churn', multiple='stack', ax=ax[2])
ax[2].set_title('Valor Total pago durante o tempo de contrato', fontsize=12, loc='left', pad=10)

plt.tight_layout()
plt.show()


# Pré-Processamento
var_bi = df3.nunique()[df3.nunique() == 2].keys().tolist()
var_num = [col for col in df3.select_dtypes(['int', 'float']).columns.tolist() if col not in var_bi]
var_cat = [col for col in df3.columns.tolist() if col not in var_bi + var_num]

#Aplicando o LabelEncoder no dataframe para variaveis binárias
df4 = df3.copy()

le = LabelEncoder()
for i in var_bi:
    df4[i] = le.fit_transform(df4[i])

df4 = pd.get_dummies(df4, columns=var_cat)
print(df4)

# Preparação dos Dados
# Separar dados entre Treino e Teste
X = df4.drop('Churn', axis=1)
y = df4['Churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Função de validação dos modelos
def val_model(X, y, model, quite=False):
    X = np.array(X)
    y = np.array(y)

    pipeline1 = make_pipeline(StandardScaler(), model)
    pipeline2 = make_pipeline(RobustScaler(), model)
    scores1 = cross_val_score(pipeline1, X, y, scoring='recall')
    scores2 = cross_val_score(pipeline2, X, y, scoring='recall')

    if quite == False:
        print("Recall StandardScaler: {:.4f} (+/- {:.4f})".format(scores1.mean(), scores1.std() * 2))
        print('Recall RobustScaler: {:.4f} (+/- {:.4f})'.format(scores2.mean(), scores2.std() * 2))
    return scores1.mean()

# Importar Modelos a serem avaliados
rf = RandomForestClassifier()
svc = SVC()
lr = LogisticRegression()
knn = KNeighborsClassifier()
xgb = XGBClassifier()

# Desempenho dos modelos com os dados padronizados
print('\nDesempenho dos modelos com os dados padronizados')
print('Cross-validation RF:')
score_teste1 = val_model(X_train, y_train, rf)
print('\nCross-validation SVC:')
score_teste2 = val_model(X_train, y_train, svc)
print('\nCross-validation LR:')
score_teste3 = val_model(X_train, y_train, lr)
print('\nCross-validation KNN:')
score_teste4 = val_model(X_train, y_train, knn)
print('\nCross-validation XGB:')
score_teste5 = val_model(X_train, y_train, xgb)


#Padronização e Balanceamento dos Dados
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

Rob_scaler = RobustScaler()
X_train_Rscaled = Rob_scaler.fit_transform(X_train)

# Balanceamento RUS
rus = RandomUnderSampler()
X_rus_scaled, y_train_rus_scaled = rus.fit_resample(X_train_scaled, y_train)
X_rus_Rscaled, y_train_rus_Rscaled = rus.fit_resample(X_train_Rscaled, y_train)

# Checando o balanceamento das classes
print(pd.Series(y_train_rus_scaled).value_counts())

sns.countplot(x=y_train_rus_scaled)
plt.title('Balanceamento RUS', fontsize=12, loc='left', pad=10)
plt.tight_layout()
plt.show()

# Balanceamento ADASYN
ada = ADASYN()
X_ada_scaled, y_train_ada_scaled = ada.fit_resample(X_train_scaled, y_train)
X_ada_Rscaled, y_train_ada_Rscaled = ada.fit_resample(X_train_Rscaled, y_train)

# Checando o balanceamento das classes
print(pd.Series(y_train_ada_scaled).value_counts())

plt.title('Balanceamento ADASYN', fontsize=12, loc='left', pad=10)
sns.countplot(x=y_train_ada_scaled)
plt.tight_layout()
plt.show()


# Avaliandos Modelos com Cross-Validation (Dados Balanceados e Padronizados)

# Definindo funçao de validação com dados balanceados
def val_model_balanced(X, y, model, quite=False):
  X = np.array(X)
  y = np.array(y)

  scores = cross_val_score(model, X, y, scoring='recall')

  if quite == False:
    print('Recall: {:.4f} (+/- {:.4f})'.format(scores.mean(), scores.std() * 2))
  return scores.mean()


# Cross-Validation com Balanceamento RUS e StandardScaler
print('\nCross-Validation com Balanceamento RUS e StandardScaler')
print('Cross-validation RF:')
score_teste1 = val_model_balanced(X_rus_scaled, y_train_rus_scaled, rf)
print('\nCross-validation SVC:')
score_teste2 = val_model_balanced(X_rus_scaled, y_train_rus_scaled, svc)
print('\nCross-validation LR:')
score_teste3 = val_model_balanced(X_rus_scaled, y_train_rus_scaled, lr)
print('\nCross-validation KNN:')
score_teste4 = val_model_balanced(X_rus_scaled, y_train_rus_scaled, knn)
print('\nCross-validation XGB:')
score_teste5 = val_model_balanced(X_rus_scaled, y_train_rus_scaled, xgb)


# Cross-Validation com Balanceamento RUS e RobustScaler
print('\nCross-Validation com Balanceamento RUS e RobustScaler')
print('Cross-validation RF:')
score_teste1 = val_model_balanced(X_rus_Rscaled, y_train_rus_Rscaled, rf)
print('\nCross-validation SVC:')
score_teste2 = val_model_balanced(X_rus_Rscaled, y_train_rus_Rscaled, svc)
print('\nCross-validation LR:')
score_teste3 = val_model_balanced(X_rus_Rscaled, y_train_rus_Rscaled, lr)
print('\nCross-validation KNN:')
score_teste4 = val_model_balanced(X_rus_Rscaled, y_train_rus_Rscaled, knn)
print('\nCross-validation XGB:')
score_teste5 = val_model_balanced(X_rus_Rscaled, y_train_rus_Rscaled, xgb)


# Cross-Validation com Balanceamento ADASYN e StandardScaler
print('\nCross-Validation com Balanceamento ADASYN e StandardScaler')
print('Cross-validation RF:')
score_teste1 = val_model_balanced(X_ada_scaled, y_train_ada_scaled, rf)
print('\nCross-validation SVC:')
score_teste2 = val_model_balanced(X_ada_scaled, y_train_ada_scaled, svc)
print('\nCross-validation LR:')
score_teste3 = val_model_balanced(X_ada_scaled, y_train_ada_scaled, lr)
print('\nCross-validation KNN:')
score_teste4 = val_model_balanced(X_ada_scaled, y_train_ada_scaled, knn)
print('\nCross-validation XGB:')
score_teste5 = val_model_balanced(X_ada_scaled, y_train_ada_scaled, xgb)


# Cross-Validation com Balanceamento ADASYN e RobustScaler
print('\nCross-Validation com Balanceamento ADASYN e RobustScaler')
print('Cross-validation RF:')
score_teste1 = val_model_balanced(X_ada_Rscaled, y_train_ada_Rscaled, rf)
print('\nCross-validation SVC:')
score_teste2 = val_model_balanced(X_ada_Rscaled, y_train_ada_Rscaled, svc)
print('\nCross-validation LR:')
score_teste3 = val_model_balanced(X_ada_Rscaled, y_train_ada_Rscaled, lr)
print('\nCross-validation KNN:')
score_teste4 = val_model_balanced(X_ada_Rscaled, y_train_ada_Rscaled, knn)
print('\nCross-validation XGB:')
score_teste5 = val_model_balanced(X_ada_Rscaled, y_train_ada_Rscaled, xgb)


# Imprimindo Tabela do Modelos escolhido
model = []
recall = []

model.append('Random Forest Classifier')
recall.append(val_model_balanced(X_ada_Rscaled, y_train_ada_Rscaled, rf, quite=True))
model.append('SVC')
recall.append(val_model_balanced(X_ada_Rscaled, y_train_ada_Rscaled, svc, quite=True))
model.append('Logistic Regression')
recall.append(val_model_balanced(X_ada_Rscaled, y_train_ada_Rscaled, lr, quite=True))
model.append('KNeighbors Classifier')
recall.append(val_model_balanced(X_ada_Rscaled, y_train_ada_Rscaled, knn, quite=True))
model.append('XGBClassifier')
recall.append(val_model_balanced(X_ada_Rscaled, y_train_ada_Rscaled, xgb, quite=True))

recall_model = pd.DataFrame(data=recall, index=model, columns=['Recall'])
print(recall_model)
dfi.export(recall_model, 'modelo.png')


# Hiperparâmetros
#RandomForestClassifier
param_grid = {
            'max_depth': [5, 10, 20, 30, 40, 50, None],
            'n_estimators': [10, 100, 1000],
            'max_features': ['auto', 'sqrt', 'log2']}

kfold = StratifiedKFold(n_splits=10, shuffle=True)
grid_search = GridSearchCV(rf, param_grid=param_grid, scoring='recall', n_jobs=-1, cv=kfold)
grid_result = grid_search.fit(X_ada_Rscaled, y_train_ada_Rscaled)
print(f'Melhor: {grid_result.best_score_} para {grid_result.best_params_}')


#SVC
kfold = StratifiedKFold(n_splits=10, shuffle=True)
clf_svc = RandomizedSearchCV(svm.SVC(gamma='auto'), {
    'C': [100, 10, 20, 1.0, 0.1, 0.001],
    'kernel': ['rbf', 'linear', 'poly', 'sigmoid']}, n_jobs=-1, scoring='recall', n_iter=15, cv=kfold)

clf_svc.fit(X_ada_Rscaled, y_train_ada_Rscaled)
print(clf_svc.best_params_)


# Regressão Logistica
param_grid = {
    'penalty': ['l1', 'l2', 'elasticnet', 'none'],
    'C': np.logspace(-4, 4, 20),
    'solver': ['lbfgs', 'newton-cg', 'libkinear', 'sag', 'saga'],
    'max_iter': [100, 1000, 2500, 5000]}

grid_search = GridSearchCV(lr, param_grid=param_grid, cv=5, return_train_score=False, scoring='recall')
grid_result = grid_search.fit(X_ada_Rscaled, y_train_ada_Rscaled)
print(f'Melhor: {grid_result.best_score_} para {grid_result.best_params_}')


# KNeighborsClassifier
param_grid = {'n_neighbors': [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21],
              'metric': ['euclidean', 'manhattan', 'minkowski'],
              'weights': ['uniform', 'distance']}

kfold = StratifiedKFold(n_splits=10, shuffle=True)
grid_search = GridSearchCV(knn, param_grid=param_grid, cv=kfold, n_jobs=-1, scoring='recall')
grid_result = grid_search.fit(X_ada_Rscaled, y_train_ada_Rscaled)
print(f'Melhor: {grid_result.best_score_} para {grid_result.best_params_}')


# XGBClassifier
xgb_hiper = XGBClassifier(learning_rate=0.01)
param_grid = {
    'n_estimators': [100, 200, 1000],
    'max_depth': [1, 3, 6],
    'min_child_weight': [1, 5, 10],
    'gamma': [0, 1, 5]}

kfold = StratifiedKFold(n_splits=10, shuffle=True)
grid_search = GridSearchCV(xgb_hiper, param_grid, scoring='recall', n_jobs=-1, cv=kfold)
grid_result = grid_search.fit(X_ada_Rscaled, y_train_ada_Rscaled)
print(f'Melhor: {grid_result.best_score_} para {grid_result.best_params_}')


# Definindo o melhor modelo

#RandomForestClassifier
model_rf = RandomForestClassifier(max_depth=10, max_features='log2', n_estimators=1000)
model_rf.fit(X_ada_Rscaled, y_train_ada_Rscaled)

X_test_Rscaled = Rob_scaler.transform(X_test)
y_pred_rf = model_rf.predict(X_test_Rscaled)

print('Relatório de Clasfficação - Random Forest Classifier: \n\n', classification_report(y_test, y_pred_rf))

# Imprimir a área sob curva
print('AUC: {:.4f}\n'.format(roc_auc_score(y_test, y_pred_rf)))

# Gráfico Matriz Confusão
f, ax = plt.subplots()
ax = sns.heatmap(confusion_matrix(y_test, y_pred_rf), fmt='g', cmap='Blues', square=True, annot=True, cbar=False)
ax.set_xlabel('Predicted Values')
ax.set_ylabel('Actual Values')
ax.set_title('Confusion Matrix - Random Forest Classifier', fontsize=12, loc='left', pad=10)
plt.tight_layout()
plt.show()


# SVC
model_svc = svm.SVC(kernel='poly', C=0.1)
model_svc.fit(X_ada_Rscaled, y_train_ada_Rscaled)

y_pred_svc = model_svc.predict(X_test_Rscaled)

print('Relatório de Clasfficação - SVC: \n\n', classification_report(y_test, y_pred_svc))

# Imprimir a área sob curva
print('AUC: {:.4f}\n'.format(roc_auc_score(y_test, y_pred_svc)))

# Gráfico Matriz Confusão
f, ax = plt.subplots()
ax = sns.heatmap(confusion_matrix(y_test, y_pred_svc), fmt='g', cmap='Blues', square=True, annot=True, cbar=False)
ax.set_xlabel('Predicted Values')
ax.set_ylabel('Actual Values')
ax.set_title('Confusion Matrix - SVC', fontsize=12, loc='left', pad=10)
plt.tight_layout()
plt.show()

# Regressão Logistica
model_rl = LogisticRegression(C=0.0018329807108324356, max_iter=100, penalty='l1', solver='saga')
model_rl.fit(X_ada_Rscaled, y_train_ada_Rscaled)

y_pred_rl = model_rl.predict(X_test_Rscaled)

print('Relatório de Clasfficação - Logistic Regression: \n\n', classification_report(y_test, y_pred_rl))

# Imprimir a área sob curva
print('AUC: {:.4f}\n'.format(roc_auc_score(y_test, y_pred_rl)))

# Gráfico Matriz Confusão
f, ax = plt.subplots()
ax = sns.heatmap(confusion_matrix(y_test, y_pred_rl), fmt='g', cmap='Blues', square=True, annot=True, cbar=False)
ax.set_xlabel('Predicted Values')
ax.set_ylabel('Actual Values')
ax.set_title('Confusion Matrix - Logistic Regression', fontsize=12, loc='left', pad=10)
plt.tight_layout()
plt.show()


#KNeighborsClassifier
model_knn = KNeighborsClassifier(metric='euclidean', n_neighbors=11, weights='distance')
model_knn.fit(X_ada_Rscaled, y_train_ada_Rscaled)

y_pred_knn = model_knn.predict(X_test_Rscaled)

print('Relatório de Clasfficação - KNeighbors Classifier: \n\n', classification_report(y_test, y_pred_knn))

# Imprimir a área sob curva
print('AUC: {:.4f}\n'.format(roc_auc_score(y_test, y_pred_knn)))

# Gráfico Matriz Confusão
f, ax = plt.subplots()
ax = sns.heatmap(confusion_matrix(y_test, y_pred_knn), fmt='g', cmap='Blues', square=True, annot=True, cbar=False)
ax.set_xlabel('Predicted Values')
ax.set_ylabel('Actual Values')
ax.set_title('Confusion Matrix - KNeighbors Classifier', fontsize=12, loc='left', pad=10)
plt.tight_layout()
plt.show()

# XGBClassifier
model_xgb = XGBClassifier(gamma=0, max_depth=3, min_child_weight=5, n_estimators=100)
model_xgb.fit(X_ada_Rscaled, y_train_ada_Rscaled)

y_pred_xgb = model_xgb.predict(X_test_Rscaled)

print('Relatório de Clasfficação - XGBClassifier: \n\n', classification_report(y_test, y_pred_xgb))

# Imprimir a área sob curva
print('AUC: {:.4f}\n'.format(roc_auc_score(y_test, y_pred_xgb)))

# Gráfico Matriz Confusão
f, ax = plt.subplots()
ax = sns.heatmap(confusion_matrix(y_test, y_pred_xgb), fmt='g', cmap='Blues', square=True, annot=True, cbar=False)
ax.set_xlabel('Predicted Values')
ax.set_ylabel('Actual Values')
ax.set_title('Confusion Matrix - XGBClassifier', fontsize=12, loc='left', pad=10)
plt.tight_layout()
plt.show()
