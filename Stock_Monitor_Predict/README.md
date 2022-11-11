# Stock Monitor Predict - Análise e Previsão de Preços de Ações

Este projeto tem o objetivo fazer um aplicativo no Streamlit. Para deixar o projeto mais robusta inclui a previsão de preçoes com Long Short Term Memory (LSTM).

Mercado de ações é um assunto que eu gosto muito, foi um projeto bem interessante e prazeroso particularmente fiquei muito satisfeito com o resultado.


Fiz o modelo de previsão em um arquivo separado[StockMonitorPredict.py]() e adicionei em outro arquivo [app.py]() que contém todas as funções do aplicativo com as análises e previsões.



## Base de Dados

Eu utilizei a biblioteca Yahoo Finance (yfinance) para obter as informações das ações.


```shell
df_acoes = pd.read_csv('base_dados.csv')

def consultar_acao(tickers, start, end, interval):
    for empresa in df_acoes['Empresas']:
        df = yf.download(tickers = tickers, start=start, end=end, interval=interval)
        return df
```


## Preparação dos dados

Antes de mais nada como todo modelo de machine learning é preciso separar os dados em treino e teste, em seguida para normalizar os dados optei por utilizar o MinMaxScaler.


```shell
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
```shell


## Long Short Term Memory (LSTM)
Já havia feito um projeto sobre Previsão de Preços com LSTM fiz algumas alterações e tentei melhorar as previsões, utilizei a base desse projeto para incluir no Stock Monitor Predict.

Para entender o que é LSTM segue a descrição abaixo, gosto muito da definição do Aurélien Géron:

"Em resumo, uma célula LSTM pode aprender a reconhecer uma entrada importante (que é o papel do input gate), armazená-la no estado de longo prazo, aprender a preservá-la pelo tempo necessário (esse é o papel do forget gate) e aprender a extraí-la sempre que for preciso. Isso explica por que elas têm sido surpreendentemente bem-sucedidas em capturar padrões de longo prazo em séries temporais." 


```shell
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
results=model.fit(X_train, y_train, validation_data = (X_test, y_test), epochs=50, batch_size=130, verbose=2)
```


Com o histórico de treinamento do modelo é possível diagnoticar o desempenho a apartir do gráfico. O ideal é a perda de treino e validação diminuir e estabilizar em torno do mesmo ponto, mas isso depende do conjunto de dados.

<p align="center">
  <img src="">
</p>


**Observação:** Foi feito vários testes para encontrar a melhor combinação dos parâmetros.


# Aplicando as previsões

Com o modelo pronto e todos os parâmetros ajustados ai podemos aplicar as previsões. 

```shell
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
```


No gráfico Realizado vs Modelo é possível verificar que os valores das previsões estão bem próximos dos valores reais de fechamento.


<p align="center">
  <img src="">
</p>


# Aplicativo no Streamlit

Para uma melhor visualização e entendimento abaixo as imagens do aplicativo. O código completo está no arquivo [app.py]() que é possivel verificar no detalhe cada função. 

<p align="center">
  <img src="">
</p>

<p align="center">
  <img src="">
</p>


## Considerações Finais

Este projeto foi elaborado somente para fins de estudos e não para recomendar ações. Para escolha e decisão sobre seus investimentos faça com responsabilidade e verificando sempre todos os critérios em torno do ativo escolhido.


É importante ressaltar que o projeto tem margem para melhoria e estou aberto para sugestões.