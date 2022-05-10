# Prevendo o Preço das Ações ITAÚ - Com Redes Neurais Recorrentes

Nesse projeto explico como utilizar Aprendizado de Máquina para prever os preços das ações do Itáu (ITUB4).

**Importante:** Esse artigo foi elaborado somente para estudos e não para recomendar ações.


## Machine Learning

Aprendizado de Máquina é a ciência (e a arte) da programação de computadores para que eles possam aprender com os dados. Aurélien Géron

Utilizei conjuntos de treinamentos que são sistema para o aprendizado. Com isso separamos os dados para manipulação da seguinte forma: Dados das amostras, Dados de treinamento, Dados de teste. Dessa forma, vamos utilizar os dados do passado para avaliar as predições do "futuro".

A Figura abaixo demonstra o fluxo de trabalho no Machine Learning.

<img src = "https://github.com/maisonhenrique/portifolio/blob/13bea3565de0ff6844453708daeee6bdd429ccd4/Prevendo_Preco_Acoes/Processo%20ML.png" />


## Base de Dados

Os dados histórico (04/2017 à 04/22) que eu utilizei nesse projeto está disponível no [Yahoo Finance](https://br.financas.yahoo.com/quote/ITUB4.SA/history?period1=1493856000&period2=1651622400&interval=1d&filter=history&frequency=1d&includeAdjustedClose=true). 

<img src = "https://github.com/maisonhenrique/portifolio/blob/bf420ad85ff44172a77c88a0b06077ca9182efa4/Prevendo_Preco_Acoes/Figure_1.png" />


## Rede Neural Recorrente (RNN)

Uma rede neural recorrente (RNN) é uma classe de redes neurais que inclui conexões ponderadas dentro de uma camada 
(em comparação com as redes de feed-forward tradicionais, onde conecta alimentação apenas para camadas subsequentes). 
Como as RNNs incluem loops, elas podem armazenar informações ao processar novas entradas. Essa memória os torna ideais para tarefas de processamento 
onde as entradas anteriores devem ser consideradas (como dados da série temporal). M. Tim Jones

<img src = "https://github.com/maisonhenrique/portifolio/blob/13bea3565de0ff6844453708daeee6bdd429ccd4/Prevendo_Preco_Acoes/Rede%20Neural%20Recorrente.png" />


## Long Short Term Memory (LSTM)

Em resumo, uma célula LSTM pode aprender a reconhecer uma entrada importante (que é o papel do input gate), armazená-la no estado de longo prazo, aprender a preservá-la
pelo tempo necessário (esse é o papel do forget gate) e aprender a extraí-la sempre que for preciso. Isso explica por que elas têm sido surpreendentemente bem-sucedidas em
capturar padrões de longo prazo em séries temporais. Aurélien Géron

Uma Rede Neural é o processamento de dados em camadas. Para criar estas camadas vamos utilizar o tf.keras, que é uma api para construir e treinar modelos no TensorFlow.

Treinando o modelo:

```shell
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
```

Após todas as etapas anteriores de separação das amostras, treino e teste etc. Aplicamos as previsões.


## Aplicando as Previsões

```shell
previsoes = modelo.predict(x_teste)
previsoes = funcao_escalonamento.inverse_transform(previsoes)

rsme = np.sqrt(np.mean(previsoes - y_teste)**2)
print('erro quadrático médio:', rsme)

Validação = dados_fechamento[dados_fechamento_valor_tamanho:]
Validação['Previsões'] = previsoes
Validação[['Close', 'Previsões']].head()
print(Validação)
```

Abaixo é possível verificar o gráfico Realizado vs Modelo, os valores das previsões estão bem próximos dos valores reais de fechamento.

<img src = "https://github.com/maisonhenrique/portifolio/blob/bf420ad85ff44172a77c88a0b06077ca9182efa4/Prevendo_Preco_Acoes/Figure_3.png" />

## Considerações Finais

Ao utilizar o método de aprendizado de máquina tivemos resultados satisfatórios ao tentar prevermos os preços das ações do Banco Itaú. 
A rede neural LSTM performa muito bem, claro que depende de outros fatores como: notícias, eventos, demonstrações de resultados etc. 

Reforçando o que disse acima, esse artigo é para fins de estudos e não para recomendar preço das ações.

Abaixo uma plotagem final com um comparativo dos Preços de Fechamento vs Previsão.

<img src = "https://github.com/maisonhenrique/portifolio/blob/bf420ad85ff44172a77c88a0b06077ca9182efa4/Prevendo_Preco_Acoes/Figure_4.png" />

## Referências:

[Um mergulho profundo nas redes neurais recorrentes - M. Tim Jones](https://imasters.com.br/data/um-mergulho-profundo-nas-redes-neurais-recorrentes#:~:text=Uma%20rede%20neural%20recorrente%20(RNN,alimenta%C3%A7%C3%A3o%20apenas%20para%20camadas%20subsequentes))

[Arquitetura de Redes Neurais Artificiais - Vinicius Almir Weiss](https://ateliware.com/blog/redes-neurais-artificiais)

Mãos à Obra: Aprendizado de Máquina com Scikit-Learn & TensorFlow - Aurélien Géron
