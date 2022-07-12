#Importando as Bibliotecas
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import statsmodels.formula.api as smf
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVR
import warnings
warnings.filterwarnings('ignore')

#Base de Dados
df = pd.read_csv('consumo_cerveja.csv', decimal=',')

df = df.dropna()

print("O conjunto de dados contém {} linhas e {} colunas".format(df.shape[0], df.shape[1]))

df['Consumo de cerveja (litros)'] = df['Consumo de cerveja (litros)'].astype(str)
df['Consumo de cerveja (litros)'] = df['Consumo de cerveja (litros)'].str.replace('.', '')

#Verificando valores ausentes
df.isnull().sum()

df = df.rename(columns={'Temperatura Media (C)': 'Temperatura_mean', 'Temperatura Minima (C)': 'Temperatura_min', 'Temperatura Maxima (C)': 'Temperatura_max',
                        'Precipitacao (mm)': 'Precipitacao', 'Final de Semana': 'Final_semana', 'Consumo de cerveja (litros)': 'Consumo_cerveja'})

df['Consumo_cerveja'] = pd.to_numeric(df['Consumo_cerveja'])

#Convertendo coluna Data para Datetime
df['Data'] = pd.to_datetime(df['Data'], format='%Y-%m-%d')
df['Mes'] = df['Data'].apply(lambda x: x.strftime('%B'))
df['Dia'] = df['Data'].apply(lambda x: x.strftime('%A'))

#Renomeando os dias da semana e os meses
df['Mes'] = df['Mes'].map({
    'January': 'Janeiro',
    'February': 'Fevereiro',
    'March': 'Março',
    'April': 'Abril',
    'May': 'Maio',
    'June': 'Junho',
    'July': 'Julho',
    'August': 'Agosto',
    'September': 'Setembro',
    'October': 'Outubro',
    'November': 'Novembro',
    'December': 'Dezembro'
})

df['Dia'] = df['Dia'].map({
    'Sunday': 'Domingo',
    'Monday': 'Segunda-feira',
    'Tuesday': 'Terça-feira',
    'Wednesday': 'Quarta-feira',
    'Thursday': 'Quinta-feira',
    'Friday': 'Sexta-feira',
    'Saturday': 'Sábado'
})

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

#Análises Estatísticas Preliminares
describe = df.describe().round(2)
print(describe)

# Analise de Correlação - Matriz de Correlação
corr = df.corr().round(4)
print(corr)

#Grafico de Correlação
correlation_matrix = df.corr()
mask = np.zeros_like(correlation_matrix)
mask[np.triu_indices_from(mask)] = True

f, ax1 = plt.subplots()
ax1 = sns.heatmap(correlation_matrix, annot=True, square=True, cmap='Blues')
ax1.set_title('Gráfico de Correlação', fontsize=10, loc='left', pad=13)
plt.tight_layout()
plt.show()

#Grafico de Dispersão
pp = sns.pairplot(df, y_vars='Consumo_cerveja', x_vars=['Temperatura_mean', 'Temperatura_min', 'Temperatura_max', 'Precipitacao'], kind='reg',
                 plot_kws={'line_kws': {'color': 'red'}})
plt.show()

#Grafico Boxplot - Semana
sns.set_theme(palette='pastel')
f, ax2 = plt.subplots(figsize=(12, 5))
ax2 = sns.boxplot(data=df, x='Dia', y='Consumo_cerveja',
                  order=['Segunda-feira', 'Terça-feira', 'Quarta-feira', 'Quinta-feira', 'Sexta-feira', 'Sábado', 'Domingo'])
ax2.set_title('Consumo de Cerveja por Dia da Semana', fontsize=12, loc='left', pad=13)
ax2.yaxis.set_major_formatter(FuncFormatter(SI))
plt.xlabel('')
plt.ylabel('')
plt.show()

#Grafico Boxplot - Mês
sns.set_theme(palette='pastel')
f, ax3 = plt.subplots(figsize=(12, 5))
ax3 = sns.boxplot(x='Mes', y='Consumo_cerveja', data=df)
ax3.set_title('Consumo de Cerveja por Mês durante 2015', fontsize=12, loc='left', pad=13)
ax3.yaxis.set_major_formatter(FuncFormatter(SI))
plt.xlabel('')
plt.ylabel('')
plt.show()


#Criando o Modelo de Regressão Linear

#Utilizando Statsmodels
model_1 = smf.ols(formula="""Consumo_cerveja~Temperatura_mean+Temperatura_min+Temperatura_max+Precipitacao+Final_semana""", data=df).fit()
print(model_1.summary(title='Modelo 1'))

#Utilizando scikit-learn

#Matriz de variáveis explicativas - X = Variáveis explicativas, y = Variável dependente

X = df.drop(columns=['Data', 'Mes', 'Dia', 'Consumo_cerveja'])
y = df.Consumo_cerveja

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=99)

#Coeficiente de determinação (R²)
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

r1 = linear_model.score(X_train, y_train).round(2)
print('Coeficiente de determinação R²:', r1)


#Coeficiente de determinação (R²) para as previsões do nosso modelo
y_pred = linear_model.predict(X_test)
r2 = metrics.r2_score(y_test, y_pred).round(2)
print('Coeficiente de determinação R²:', r2)


# Gerando previsão pontual
# Utilizando uma linha qualquer da base de teste para gerar uma previsão de consumo

entrada = X_test[0:1]
print(entrada)

model = linear_model.predict(entrada)[0].round(2)
print('{0:.2f} Litros'.format(model))


# Coeficientes do modelo
index = ['Intercepto', 'Temperatura_mean', 'Temperatura_min', 'Temperatura_max', 'Precipitacao', 'Final_semana']
previsao = pd.DataFrame(data=np.append(linear_model.intercept_, linear_model.coef_), index=index, columns=['Parâmetros'])
print(previsao)

# Gerando as previsões do modelo para os dados de TREINO

y_previsto = linear_model.predict(X_train)
sns.set_theme()
f, ax = plt.subplots(figsize=(12, 5))
ax = sns.regplot(x=y_previsto, y=y_train, line_kws={"color": "red"})
plt.title('Previsão x Real', fontsize=12, loc='left', pad=13)
plt.xlabel('Consumo de Cerveja (litros) - Previsão')
plt.ylabel('Consumo de Cerveja (litros) - Real')
ax.xaxis.set_major_formatter(FuncFormatter(SI))
ax.yaxis.set_major_formatter(FuncFormatter(SI))
plt.rcParams['xtick.labelsize'] = 6
plt.rcParams['ytick.labelsize'] = 6
plt.show()

#Verificação com outros modelos
models = [('KNN', KNeighborsRegressor()), ('Tree', DecisionTreeRegressor()), ('LR', LinearRegression()), ('SVR', SVR())]

result_models = {}
for nome_modelo, modelo in models:
    model = modelo
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    result_models[nome_modelo] = mse

print(result_models)

#Validação Cruzada
linear_regression = LinearRegression()
linear_regression_cross = cross_val_score(linear_regression, X, y, cv=10, verbose=1)
print(linear_regression_cross)

tree = DecisionTreeRegressor(max_depth=5)
tree.fit(X_train, y_train)

y_pred = tree.predict(X_test)

mse_tree = mean_squared_error(y_test, y_pred)
print('MSE:', mse_tree)

tree_model = tree.score(X_train, y_train)
print('Coeficiente de determinação R²:', tree_model)
