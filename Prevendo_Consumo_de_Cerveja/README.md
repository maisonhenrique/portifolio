# Prevendo o Consumo de Cerveja - Regressão Linear

![Captura de tela 2022-07-12 150218](https://user-images.githubusercontent.com/99361817/178568491-ff1d36ac-65ea-4426-aa09-f32ddf379ae8.png)

O ojetivo desse artigo é estimar o consumo médio de cerveja utilizando um modelo de Machine Learning e técnica de Regressão Linear. Os dados foram coletados em São Paulo no período de Janeiro/2015 a Dezembro/2015 em uma área universitária, a faixa etária dos participantes foi de 18 a 28 anos de idade. O dataset está disponivel no [Kaggle](https://www.kaggle.com/datasets/dongeorge/beer-consumption-sao-paulo).


## Análise Exploratória

Antes de tudo foi feito uma exploração, limpeza e tratamento dos dados removendo valores nulos, convertendo algumas colunas para variáveis numéricas entre outras.

Conhecendo o nosso dataset:

* Data - Data 
* Temperatura_mean - Temperatura Média (°C)
* Temperatura_min - Temperatura Mínima (°C)
* Temperatura_max - Temperatura Máxima (°C)
* Precipitacao - Precipitação (mm) Chuva
* Final_semana - Final de Semana (1 = Sim; 0 = Não)
* Consumo_cerveja - Consumo de Cerveja (litros)


![Dados](https://user-images.githubusercontent.com/99361817/178568803-5f4d2831-1326-4f0e-a2a3-b2270f4a2328.png)


**Estatistica descritiva:** O objetivo de fazer uma análise descritiva é verificar se há dados discrepantes, por exemplo: erro de digitação, falha no sistema ou outro motivo. 


![Describe](https://user-images.githubusercontent.com/99361817/178569192-a191ddc5-f1ed-4c0f-93b4-2f543deaae5a.png)


Conforme gráfico abaixo, durante toda a semana a maior parte do consumo de cerveja foi no sábado, seguido do domingo. Podemos concluir que, em média, o consumo é nos finais de semana.


![Figure_3](https://user-images.githubusercontent.com/99361817/178569452-5f646a30-aa03-4b9f-a091-73ecf6b35f60.png)


Abaixo podemos observar o consumo de cerveja ao longo do ano de 2015.


![Figure_4](https://user-images.githubusercontent.com/99361817/178569555-8830e89d-48e1-49c2-bf1d-7b7c2ff0554c.png)


**Visualizando a Relação entre as Variáveis**

As análises de correlação e regressão podem ser usadas para determinar se há uma relação significativa entre duas variáveis. Quando há, você pode usar uma das variáveis para prever o valor da outra variável. (Ron Larson, Betsy Farber)


**Mapa de Calor - Correlação:** Valores próximos a 0 indicam baixa ou nenhuma correlação e valores próximos a 1 ou -1 refletem alta correlação.


![Figure_1](https://user-images.githubusercontent.com/99361817/178569627-ba3b42c8-b0a1-46e2-829d-0e2dca32dde7.png)


**Dispersão entre as Variáveis (Variável Dependente x Variáveis Explicativas):** O gráfico de dispersão nos permite identificar se duas variáveis apresentam uma relação linear entre elas e a direção dessa relação.


![Figure_2](https://user-images.githubusercontent.com/99361817/178569664-c720b247-d734-4801-8e5b-942921f88c89.png)


## Criando o Modelo de Regressão Linear

A análise de regressão múltipla é uma técnica estatística que pode ser usada para analisar a relação entre uma única variável dependente (critério) e várias variáveis independentes (preditoras). O objetivo da análise de regressão múltipla é usar as variáveis independentes cujos valores são conhecidos para prever os valores da variável dependente selecionada pelo pesquisador. ((Joseph F Hair Jr)


Equação da reta de regressão linear estimada:


![equação](https://user-images.githubusercontent.com/99361817/178570430-059cc51c-1fd2-470a-ba93-a953dba8acfe.png))


**Utilizando scikit-learn**

**Defininindo a matriz de variáveis explicativas:**

* X = Variáveis explicativas
* y = Variável dependente


```shell
X = df.drop(columns=['Data', 'Mes', 'Dia', 'Consumo_cerveja'])
y = df.Consumo_cerveja

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=99)
```


**Coeficiente de Determinação - R²:** Medida da proporção da variância da variável dependente em torno de sua média que é explicada pelas variáveis independentes ou preditoras. O coeficiente pode variar entre 0 e 1. Se o modelo de regressão é propriamente aplicado e estimado, o pesquisador pode assumir que quanto maior o valor de R2, maior o poder de explicação da equação de regressão e, portanto, melhor a previsão da variável dependente. (Joseph F Hair Jr)


![R2](https://user-images.githubusercontent.com/99361817/178570400-834ce8e6-f86f-4edf-8428-d764f7acf5be.png)


```shell
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

r1 = linear_model.score(X_train, y_train).round(2)
print('Coeficiente de determinação R²:', r1)
```
[out] **Coeficiente de determinação R²: 0.71**


﻿**Previsão pontual:﻿** Utilizando uma linha qualquer de nossa base de teste para gerar uma previsão de consumo.

```shell
entrada = X_test[0:1]
print(entrada)

model = linear_model.predict(entrada)[0].round(2)
print('{0:.2f} Litros'.format(model))
```
[out] **25269.69 Litros**


﻿**Coeficientes do modelo:﻿** O intercepto representa o efeito médio em Y (Consumo_cerveja) tendo todas as variáveis explicativas excluídas do modelo.

```shell
index = ['Intercepto', 'Temperatura_mean', 'Temperatura_min', 'Temperatura_max', 'Precipitacao', 'Final_semana']
previsao = pd.DataFrame(data=np.append(linear_model.intercept_, linear_model.coef_), index=index, columns=['Parâmetros'])
print(previsao)
```
[out]

![saida](https://user-images.githubusercontent.com/99361817/178570672-81d9572f-e34a-4f90-8f10-089b1b75c130.png)


**Intercepto -** Excluindo o efeito das variáveis explicativas o efeito médio no Consumo de Cerveja seria de **6654,15 litros**.

**Final de Semana -** O Final de Semana gera uma variação média no Consumo de Cerveja de **4923,26 litros**.


**Validação Cruzada:** As métricas clássicas de regressão estatística (R2, estatísticas F e valores p) são todas métricas “na amostra” — são aplicadas nos mesmos dados que foram usados para ajustar o modelo. Intuitivamente, pode-se ver que faria mais sentido deixar de lado uma parte dos dados originais, não usá-los para ajustar o modelo, e então aplicar o modelo aos dados reservados (retenção) para ver como se comportam. Normalmente usaríamos a maioria dos dados para ajustar o modelo, e uma proporção menor para testá-lo. ( Peter Bruce,  Andrew Bruce)

Em alguns casos a métrica foi muito ruim, onde o  R² deu 0.50760468 e 0.58148058 conforme a saída abaixo:

```shell
linear_regression = LinearRegression()
linear_regression_cross = cross_val_score(linear_regression, X, y, cv=10, verbose=1)
print(linear_regression_cross)
```

[out] [0.58839753 0.50760468 0.58148058 0.63480466 0.8095751 0.75974495 0.71076483 0.59478944 0.54005214 0.54649093]


Por isso é muito importante fazer a validação cruzada e também comparar com outros modelos.


**Utilizando Statsmodels**

![StatsModels](https://user-images.githubusercontent.com/99361817/178570734-8d3f1b2e-85da-43f7-ac74-31b9b41af026.png)


**Intercepto -** Excluindo o efeito das variáveis explicativas o efeito médio no Consumo de Cerveja seria de 6444,69 litros.

**Final de Semana -** O Final de Semana gera uma variação média no Consumo de Cerveja de 5183,18 litros.


**Análise Gráfica - Previsto x Real**


![Figure_5](https://user-images.githubusercontent.com/99361817/178570784-2780eea4-5201-4836-ac6f-bc404331ac10.png)


## Considerações Finais

Independentemente das nossas variáveis explicativas (Temperatura_mean, Temperatura_min, Temperatura_max, Precipitacao, Final_semana) haverá um consumo médio de 
**6444,69 litros** de cerveja. Além disso, chover contribui para um menor consumo de cerveja e sendo final de semana ou não, impacta consideravelmente o consumo de cerveja (uma diferença de **5183,18 litros**).


## Referências

Análise multivariada de dados [recurso eletrônico] / Joseph F Hair Jr ... [et al.] ; tradução Adonai Schlup Sant’Anna. – 6. ed. – Dados eletrônicos. – Porto Alegre : Bookman, 2009.

Estatística Aplicada / Ron Larson, Betsy Farber ; tradução José Fernando Pereira Gonçalves ; revisão técnica Manoel Henrique Salgado. - São Paulo : Pearson Education do Brasil, 2015.

Estatística Básica/Pedro A. Morettin, Wilton O. Bussab. – 6. ed. – São Paulo : Saraiva, 2010.

Estatística Prática para Cientistas de Dados - 50 Conceitos Essenciais - Peter Bruce,  Andrew Bruce