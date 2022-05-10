#Importando as Bibliotecas
import pandas as pd
import numpy as np
import math
from mplfinance.original_flavor import candlestick_ohlc
import matplotlib.dates as mpdates
from matplotlib.ticker import FuncFormatter
from matplotlib.dates import DateFormatter
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, LSTM
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

#Base de Dados
df = pd.read_csv('ITUB4.csv', header=0)
print(df)

#Convertendo coluna Date para Datetime
df['Date'] = pd.to_datetime(df['Date'], infer_datetime_format=True)
df.info()

#Excluindo valores nulos
df = df.dropna()

Dados = df.set_index('Date')
print(Dados)

#Gráfico Fechamento
plt.style.use('seaborn-darkgrid')
ax = plt.figure(figsize=(18, 6))
plt.title('Histórico de Preço - Fechamento', fontsize=15, loc='left')
plt.plot(Dados.index, Dados['Close'])
plt.ylabel('Preço (R$)')
plt.tight_layout()
#plt.show()

#Gráfico de Candlestick
plt.style.use('seaborn-darkgrid')
df = df[['Date', 'Open', 'High', 'Low', 'Close']]
df['Date'] = df['Date'].map(mpdates.date2num)

fig, ax = plt.subplots(figsize=(18, 6))
candlestick_ohlc(ax, df.values, width=0.6, colorup='green', colordown='red', alpha=0.8)
ax.set_ylabel('Preço (R$)')
plt.title('Histórico de Preço - Gráfico de Candlestick', fontsize=15, loc='left')
date_format = mpdates.DateFormatter('%Y-%b')
ax.xaxis.set_major_formatter(date_format)
plt.tight_layout()
#plt.show()

#Filtrando a Base de Dados
dados_fechamento = Dados.filter(['Close'])
dados_fechamento_valor = dados_fechamento.values
dados_fechamento_valor_tamanho = math.ceil(len(dados_fechamento_valor) * .8)

#Escalonamento dos Dados
funcao_escalonamento = MinMaxScaler()
dados_escalonados_fechamento = funcao_escalonamento.fit_transform(dados_fechamento_valor)
#print(dados_escalonados_fechamento)

#Dados de Treinamento
dados_treino = dados_escalonados_fechamento
x_treinamento = []
y_treinamento = []

#Separar dados treino e teste
for Loop in range(60, len(dados_treino)):
    amostra_treinamento_x = dados_treino[Loop - 60: Loop, 0]
    x_treinamento.append(amostra_treinamento_x)

    amostra_treinamento_y = dados_treino[Loop, 0]
    y_treinamento.append(amostra_treinamento_y)

#Trasnformando lista em array
x_treinamento, y_treinamento = np.array(x_treinamento), np.array(y_treinamento)
x_treinamento = np.reshape(x_treinamento, (x_treinamento.shape[0], x_treinamento.shape[1], 1))
x_treinamento.shape

#Treinar o modelo da rede neural recorrente
modelo = Sequential()
modelo.add(LSTM(50, return_sequences=True, input_shape=(x_treinamento.shape[1], 1)))
modelo.add(LSTM(50, return_sequences=False))

#Adicionando camadas na rede neural
modelo.add(Dense(25))
modelo.add(Dense(1))
modelo.compile(optimizer='adam', loss='mean_squared_error')
modelo.fit(x_treinamento, y_treinamento, batch_size=1, epochs=1)

#Definindo amostra para teste
dados_teste = dados_escalonados_fechamento[dados_fechamento_valor_tamanho - 60:, :]
x_teste = []
y_teste = dados_fechamento_valor[dados_fechamento_valor_tamanho:, :]

for Loop in range(60, len(dados_teste)):
    x_teste.append(dados_teste[Loop - 60: Loop, 0])

#Transformando dados em array
x_teste = np.array(x_teste)
x_teste = np.reshape(x_teste, (x_teste.shape[0], x_teste.shape[1], 1))


#Aplicando as previsões
previsoes = modelo.predict(x_teste)
previsoes = funcao_escalonamento.inverse_transform(previsoes)

rsme = np.sqrt(np.mean(previsoes - y_teste) ** 2)
print('Erro quadrático médio:', rsme)

Validação = dados_fechamento[dados_fechamento_valor_tamanho:]
Validação['Previsões'] = previsoes
Validação[['Close', 'Previsões']].head()
print(Validação)

#Gráfico Realizado vs Modelo
plt.style.use('seaborn-darkgrid')
fig, ax = plt.subplots(figsize=(18, 6))

date_form = DateFormatter('%Y-%b')
ax.xaxis.set_major_formatter(date_form)
ax.xaxis.set_major_locator(mpdates.WeekdayLocator(interval=3))

plt.title('Realizado vs Modelo', fontsize=14, loc='left')
plt.plot(Validação.index, Validação['Close'], label='Fechamento')
plt.plot(Validação.index, Validação['Previsões'], label='Previsão')
ax.legend()
plt.ylabel('Preço (R$)')
plt.tight_layout()
plt.show()

#Plotagem Final
fig = plt.figure(figsize=(18, 10))
plt.style.use('seaborn-darkgrid')

#Titulo
plt.suptitle('Prevendo o preço das Ações ITAÚ - ITUB4  \n Com Redes Neurais Recorrentes', fontsize=18, color='#404040', fontweight=600)

#Gráfico Fechamento
ax1 = plt.subplot(2, 1, 1)
df = df[['Date', 'Open', 'High', 'Low', 'Close']]
def dec (x, pos):
    return f'{x:.0f}'
formatter = FuncFormatter(dec)
ax1.yaxis.set_major_formatter(formatter)
date_format = mpdates.DateFormatter('%Y-%b')
ax1.xaxis.set_major_formatter(date_format)
ax1.xaxis.set_major_locator(mpdates.MonthLocator(interval=6))
ax1 = candlestick_ohlc(ax1, df.values, width=0.6, colorup='green', colordown='red', alpha=0.8)
plt.ylabel('Preço (R$)')
plt.title('Histórico de Preço - Gráfico de Candlestick', fontsize=13, loc='left')

#Gráfico Realizado vs Modelo
ax2 = plt.subplot(2, 1, 2)
date_form = DateFormatter('%Y-%b')
ax2.xaxis.set_major_formatter(date_form)
ax2.xaxis.set_major_locator(mpdates.WeekdayLocator(interval=4))
plt.title('Preço de Fechamento vs Previsão', fontsize=13, loc='left')
ax2 = plt.plot(Validação.index, Validação['Close'], label='Fechamento')
ax2 = plt.plot(Validação.index, Validação['Previsões'], label='Previsão')
plt.legend()
plt.ylabel('Preço (R$)')

fig.tight_layout()

plt.subplots_adjust(bottom=0.125, top=0.88, wspace=0.35, hspace=0.35)

rodape = '''
'Esse relatório foi elaborado somente para estudos e não para recomendar ações.'
https://github.com/maisonhenrique
Maison Henrique
'''
fig.text(0.5, -0.01, rodape, ha='center', va='bottom', size=12, color='#938ca1')

plt.show()
