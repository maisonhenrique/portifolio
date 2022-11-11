import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# Tabelas com as ações
df_acoes = pd.read_csv('base_dados.csv')

# Ajustar data
date_start = datetime.today() - timedelta(days=1865)
date_end = datetime.today()

intervals = ['5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo']

def format_date(dt, format='%Y-%m-%d'):
    return dt.strftime(format)

df = yf.download(tickers = 'ITSA4.SA', start=date_start, end=date_end, interval='1d').reset_index()
print(df)

#Filtrando a Base de Dados
df1 = df[['Date', 'Close']].set_index('Date')
print(df1)

#Gráfico Fechamento
plt.style.use('seaborn-darkgrid')
ax = plt.figure(figsize=(18, 6))
plt.title('Preço de Fechamento', fontsize=15, loc='left', pad=10)
plt.plot(df1.index, df1['Close'])
plt.ylabel('Preço (R$)')
plt.tight_layout()
plt.show()

# Separação dos dados Treino e Teste 
close_price = df1['Close']
step = 15

train_size = int(len(close_price) * 0.8)
test_size = len(close_price) - train_size
train_data, input_data = np.array(close_price[0:train_size]), np.array(close_price[train_size - step:])
test_data = np.array(close_price[train_size:])

# Normalizar os dados
scaler = MinMaxScaler(feature_range=(0,1))
train_data_norm = scaler.fit_transform(np.array(train_data).reshape(-1,1))
test_data_norm = scaler.transform(np.array(input_data).reshape(-1,1))

# Pré-processamento
X_train, y_train = [], []
for i in range(step, len(train_data)):
    X_train.append(train_data_norm[i-step:i])
    y_train.append(train_data_norm[i])
    
X_test, y_test = [], []
for i in range(step, step + len(test_data)):
    X_test.append(test_data_norm[i-step:i])
    y_test.append(test_data_norm[i])
    
# Trasnformando em array
X_train = np.array(X_train)
y_train = np.array(y_train)
X_test = np.array(X_test)
y_test = np.array(y_test)

# Montando a Rede Neural
model = Sequential()
model.add(LSTM(35, return_sequences=True, activation='relu', input_shape=(step, 1)))
model.add(Dropout(0.2))
model.add(LSTM(35, return_sequences=False))
model.add(Dropout(0.2))


#Adicionando camadas na rede neural
model.add(Dense(25))
model.add(Dense(1))
model.compile(optimizer ='adam', loss= 'mse')
model.summary()

#Treinamento do modelo
results=model.fit(X_train, y_train, validation_data = (X_test, y_test), epochs=270, batch_size=180, verbose=2)

#model.save('model_app.h5')

plt.plot(results.history["loss"], label='Training loss')
plt.plot(results.history["val_loss"], label='Validation loss')
plt.legend()
plt.show()

# Fazendo a previsão
predict = model.predict(X_test)
predict = scaler.inverse_transform(predict)

rsme = np.sqrt(np.mean(predict - y_test) ** 2)
print('Erro quadrático médio:', rsme)

mse = mean_squared_error(test_data, predict)
print(mse)

df_predict = df1.filter(['Close'])[train_size:]
df_predict['Predict'] = predict
df_predict[['Close', 'Predict']]
print(df_predict) 


# Gráfico Realizado vs Modelo
plt.figure(figsize = (18,9))
plt.plot(df_predict.index, df_predict['Close'], color = 'green', label = 'Real')
plt.plot(df_predict.index, df_predict['Predict'], color = 'red', label = 'Previsão')
plt.title('Realizado vs Modelo', fontsize=14, loc='left', pad=10)
plt.ylabel('Preço (R$)')
plt.tight_layout()
plt.legend()
plt.show()


# Prevendo o preço para os próximos 10 dias
lenght_test = len(test_data_norm)

days_steps = lenght_test - step

input_steps = test_data_norm[days_steps:]
input_steps = np.array(input_steps).reshape(1, -1)

list_output_steps = list(input_steps)
list_output_steps = list_output_steps[0].tolist()

pred_output=[]
i=0
n_future=10
while(i<n_future):
    
    if(len(list_output_steps) > step):
        
        input_steps = np.array(list_output_steps[1:])
        input_steps = input_steps.reshape(1, -1)
        input_steps = input_steps.reshape((1, step, 1))

        pred = model.predict(input_steps, verbose=0)
        list_output_steps.extend(pred[0].tolist())
        list_output_steps=list_output_steps[1:]
  
        pred_output.extend(pred.tolist())
        i=i+1
    else:
        input_steps = input_steps.reshape((1, step,1))
        pred = model.predict(input_steps, verbose=0)
        list_output_steps.extend(pred[0].tolist())
        i=i+1
print(pred_output)

prev = scaler.inverse_transform(pred_output)
prev = np.array(prev).reshape(1,-1)
list_output_prev = list(prev)
list_output_prev = prev[0].tolist()

dates = df1.index
predict_dates = pd.date_range(list(dates)[-1] + pd.DateOffset(1), periods=9, freq='b').tolist()

forecast_dates = []
for i in predict_dates:
    forecast_dates.append(i.date())

df_forecast = pd.DataFrame({'Date': np.array(forecast_dates), 'Predict': list_output_prev})

df_forecast=df_forecast.set_index(pd.DatetimeIndex(df_forecast['Date'].values))
df_forecast.drop('Date', axis=1, inplace=True)
print(df_forecast)