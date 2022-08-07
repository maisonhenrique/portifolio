# Importando as Bibliotecas
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from sklearn import metrics
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, auc, roc_curve
import warnings
warnings.filterwarnings('ignore')

# Base de Dados
df = pd.read_csv('creditcard.csv', sep=',', header=0)
print(df)

# Análises Estatísticas Preliminares
describe = df.describe().round(2)
print(describe)

# Verificando Valores Ausentes
ausentes = df.isnull().sum()
print(ausentes)

# Porecetagem de Fraudes no Dataset
print('Percentual de não fraudes:', round(df['Class'].value_counts()[0]/len(df) * 100, 2), '% no dataset')
print('Percentual de fraudes:', round(df['Class'].value_counts()[1]/len(df) * 100, 2), '% no dataset')

# Distribuição Classes
plt.style.use('seaborn-darkgrid')
sns.countplot('Class', data=df)
plt.title('Distribuição das Classes \n0: Not Fraud || 1: Fraud', fontsize=12, loc='left', pad=10)
plt.tight_layout()
plt.show()


# Estatisticas para fraudes (1) e nao fraudes (0)
df_fraud = pd.DataFrame(df[df['Class'] == 1]['Amount'].describe())
df_fraud.columns = ['Valores de Fraude']
df_fraud['Valores sem Fraude'] = pd.Series(df[df['Class'] == 0]['Amount'].describe())
print(df_fraud)

# Boxplot
fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(12, 6), constrained_layout=True)
fig.suptitle('Quantidade de Transações', fontsize=14)
plt.style.use('seaborn-darkgrid')

sns.boxplot(ax=ax1, x='Class', y='Amount', data=df, showfliers=True)
sns.boxplot(ax=ax2, x='Class', y='Amount', data=df, showfliers=False)

plt.tight_layout()
plt.show()


#Função para alterar legenda do gráfico
def SI(x, pos):
    if x == 0:
        return x
    bins = [1000000000000.0, 1000000000.0, 1000000.0, 1000.0, 1, 0.001, 0.000001, 0.000000001]
    abbrevs = ['T', 'G', 'M', 'k', '', 'm', 'u', 'n']
    label = x
    for i in range(len(bins)):
        if abs(x) >= bins[i]:
            label = '{1:.{0}f}'.format(0, x / bins[i]) + abbrevs[i]
            break

    return label

# Quantidade de Transações Fraudulentas
plt.style.use('seaborn-darkgrid')
df_fraud = df[df['Class'] == 1]
f, ax = plt.subplots(figsize=(12, 5))
ax = sns.scatterplot(df_fraud['Time'], df_fraud['Amount'])
ax.xaxis.set_major_formatter(FuncFormatter(SI))
plt.title('Quantidade de Transações Fraudulentas por Segundos', fontsize=12, loc='left', pad=10)
plt.xlabel('Time(s)')
plt.ylabel('Amount')
plt.xlim([0, 175000])
plt.ylim([0, 2500])
plt.show()


# Preparação dos Dados

# Separar dados entre Treino e Teste
X = df.drop('Class', axis=1)
y = df['Class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Balanceamento dos Dados - SMOTE
smote = SMOTE(sampling_strategy='auto', random_state=42, n_jobs=-1)
X_train_smt, y_train_smt = smote.fit_resample(X_train, y_train)
print(pd.Series(y_train_smt).value_counts())

plt.style.use('seaborn-darkgrid')
sns.countplot(y_train_smt)
plt.title('SMOTE \n0: Not Fraud || 1: Fraud', fontsize=12, loc='left', pad=10)
plt.tight_layout()
plt.show()


# Matriz de Correlação
corr = X_train.corr()
corr_smt = pd.DataFrame(X_train_smt).corr()

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(18, 8))
fig.suptitle('Matriz de Correlação', fontsize=14)

sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns, linewidths=.1, cmap='coolwarm', ax=ax[0])
ax[0].set_title('Dados Desbalanceado', fontsize=12, loc='left', pad=10)

sns.heatmap(corr_smt, xticklabels=corr.columns, yticklabels=corr.columns, linewidths=.1, cmap='coolwarm', ax=ax[1])
ax[1].set_title('Dados Balanceado com SMOTE', fontsize=12, loc='left', pad=10)

plt.tight_layout()
plt.show()


# Construção do modelo
# Random Forest Classifier - Dados Desbalanceados
clf = RandomForestClassifier(n_estimators=600, max_depth=6, random_state=10)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

print('Relatório de Clasfficação \nRandom Forest - Dados Desbalanceados: \n\n', classification_report(y_test, y_pred))

# Imprimir a área sob curva
print('AUC: {:.4f}\n'.format(roc_auc_score(y_test, y_pred)))

# Gráfico Matriz Confusão
f, ax = plt.subplots()
ax = sns.heatmap(confusion_matrix(y_test, y_pred), cmap='Blues', fmt='g', annot=True)
ax.set_xlabel('Predicted Values')
ax.set_ylabel('Actual Values')
ax.set_title('Matriz Confusão \nRandom Forest Classifier - Dados Desbalanceados', fontsize=12, loc='left', pad=10)
ax.xaxis.set_ticklabels(['Not Fraud', 'Fraud'])
ax.yaxis.set_ticklabels(['Not Fraud', 'Fraud'])
plt.tight_layout()
plt.show()


# Random Forest Classifier - SMOTE
clf_smt = RandomForestClassifier(n_estimators=600, max_depth=6, random_state=12, criterion='gini')
clf_smt.fit(X_train_smt, y_train_smt)

y_pred_smt = clf_smt.predict(X_test)

print('Relatório de Clasfficação \nRandom Forest - SMOTE: \n\n', classification_report(y_test, y_pred_smt))

# Imprimir a área sob curva
print('AUC: {:.4f}\n'.format(roc_auc_score(y_test, y_pred_smt)))

# Gráfico Matriz Confusão
f, ax = plt.subplots()
ax = sns.heatmap(confusion_matrix(y_test, y_pred_smt), cmap='Blues', fmt='g', annot=True)
ax.set_xlabel('Predicted Values')
ax.set_ylabel('Actual Values')
ax.set_title('Matriz Confusão \nRandom Forest Classifier - SMOTE', fontsize=12, loc='left', pad=10)
ax.xaxis.set_ticklabels(['Not Fraud', 'Fraud'])
ax.yaxis.set_ticklabels(['Not Fraud', 'Fraud'])
plt.tight_layout()
plt.show()


# Logistic Regression - Dados Desbalanceados
model_logistic = LogisticRegression(max_iter=100, solver='liblinear')
model_logistic.fit(X_train, y_train)

y_pred_log = model_logistic.predict(X_test)

print('Relatório de Clasfficação \nLogistic Regression - Dados Desbalanceados: \n\n', classification_report(y_test, y_pred_log))

# Imprimir a área sob curva
print('AUC: {:.4f}\n'.format(roc_auc_score(y_test, y_pred_log)))

# Gráfico Matriz Confusão
f, ax = plt.subplots()
ax = sns.heatmap(confusion_matrix(y_test, y_pred_log), cmap='Blues', fmt='g', annot=True)
ax.set_xlabel('Predicted Values')
ax.set_ylabel('Actual Values')
ax.set_title('Matriz Confusão \nLogistic Regression - Dados Desbalanceados', fontsize=12, loc='left', pad=10)
ax.xaxis.set_ticklabels(['Not Fraud', 'Fraud'])
ax.yaxis.set_ticklabels(['Not Fraud', 'Fraud'])
plt.tight_layout()
plt.show()


# Logistic Regression - SMOTE
model_logistic_smt = LogisticRegression(max_iter=100, solver='liblinear')
model_logistic_smt.fit(X_train_smt, y_train_smt)

y_pred_log_smt = model_logistic_smt.predict(X_test)

print('Relatório de Clasfficação \nLogistic Regression - SMOTE: \n\n', classification_report(y_test, y_pred_log_smt))

# Imprimir a área sob curva
print('AUC: {:.4f}\n'.format(roc_auc_score(y_test, y_pred_log_smt)))

# Gráfico Matriz Confusão
f, ax = plt.subplots()
ax = sns.heatmap(confusion_matrix(y_test, y_pred_log_smt), cmap='Blues', fmt='g', annot=True)
ax.set_xlabel('Predicted Values')
ax.set_ylabel('Actual Values')
ax.set_title('Matriz Confusão \nLogistic Regression - SMOTE', fontsize=12, loc='left', pad=10)
ax.xaxis.set_ticklabels(['Not Fraud', 'Fraud'])
ax.yaxis.set_ticklabels(['Not Fraud', 'Fraud'])
plt.tight_layout()
plt.show()


# XGBClassifier - Dados Desbalanceados
XGB = XGBClassifier(n_estimators=400, learning_rate=0.1, max_depth=4, min_child_weight=1, subsample=0.8, colsample_bytree=1, objective='binary:logistic')
XGB.fit(X_train, y_train)

y_pred_XGB = XGB.predict(X_test)

print('Relatório de Clasfficação \nXGBClassifier - Dados Desbalanceados: \n\n', classification_report(y_test, y_pred_XGB))

# Imprimir a área sob curva
print('AUC: {:.4f}\n'.format(roc_auc_score(y_test, y_pred_XGB)))

# Gráfico Matriz Confusão
f, ax = plt.subplots()
ax = sns.heatmap(confusion_matrix(y_test, y_pred_XGB), cmap='Blues', fmt='g', annot=True)
ax.set_xlabel('Predicted Values')
ax.set_ylabel('Actual Values')
ax.set_title('Matriz Confusão \nXGBClassifier - Dados Desbalanceados', fontsize=12, loc='left', pad=10)
ax.xaxis.set_ticklabels(['Not Fraud', 'Fraud'])
ax.yaxis.set_ticklabels(['Not Fraud', 'Fraud'])
plt.tight_layout()
plt.show()


# XGBClassifier - SMOTE
XGB_smt = XGBClassifier(n_estimators=400, learning_rate=0.1, max_depth=4, min_child_weight=1, subsample=0.8, colsample_bytree=1, objective='binary:logistic')
XGB_smt.fit(X_train_smt, y_train_smt)

y_pred_XGB_smt = XGB_smt.predict(X_test)

print('Relatório de Clasfficação \nXGBClassifier - SMOTE: \n\n', classification_report(y_test, y_pred_XGB_smt))

# Imprimir a área sob curva
print('AUC: {:.4f}\n'.format(roc_auc_score(y_test, y_pred_XGB_smt)))

# Gráfico Matriz Confusão
f, ax = plt.subplots()
ax = sns.heatmap(confusion_matrix(y_test, y_pred_XGB_smt), cmap='Blues', fmt='g', annot=True)
ax.set_xlabel('Predicted Values')
ax.set_ylabel('Actual Values')
ax.set_title('Matriz Confusão \nXGBClassifier - SMOTE', fontsize=12, loc='left', pad=10)
ax.xaxis.set_ticklabels(['Not Fraud', 'Fraud'])
ax.yaxis.set_ticklabels(['Not Fraud', 'Fraud'])
plt.tight_layout()
plt.show()

# ROC Curve - SMOTE
y_pred_prob_clf = clf_smt.predict_proba(X_test)[:, 1]
y_pred_prob_lg = model_logistic_smt.predict_proba(X_test)[:, 1]
y_pred_prob_XGB = XGB_smt.predict_proba(X_test)[:, 1]

fpr_smt, tpr_smt, _ = roc_curve(y_test, y_pred_smt)
roc_auc_smt = auc(fpr_smt, tpr_smt)

fpr_log_smt, tpr_log_smt, _ = roc_curve(y_test, y_pred_log_smt)
roc_auc_log_smt = auc(fpr_log_smt, tpr_log_smt)

fpr_XGB_smt, tpr_XGB_smt, _ = roc_curve(y_test, y_pred_XGB_smt)
roc_auc_XGB_smt = auc(fpr_XGB_smt, tpr_XGB_smt)


fig = plt.figure(figsize=(10, 8))
plt.plot(fpr_smt, tpr_smt, label='Random Forest Classifier = %0.2f' % roc_auc_smt, color='blue')
plt.plot(fpr_log_smt, tpr_log_smt, label='Logistic Regression = %0.2f' % roc_auc_log_smt, color='orange')
plt.plot(fpr_XGB_smt, tpr_XGB_smt, label='XGBClassifier = %0.2f' % roc_auc_XGB_smt, color='green')
plt.plot([0, 1], [0, 1], 'r--')
plt.title('ROC Curve - SMOTE')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.legend(loc=4)
plt.show()
