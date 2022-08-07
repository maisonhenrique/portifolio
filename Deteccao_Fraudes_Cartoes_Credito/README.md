# Detecção de Fraudes em Cartões de Crédito

<p align="center">
  <img src="https://user-images.githubusercontent.com/99361817/183304700-734961b1-7fa7-428c-8d40-ef0943999817.png">
</p>


Neste projeto iremos abordar sobre as fraudes nos cartões de crédito, uma das principais preocupações das instituições financeiras.

Segundo o Indicador de Tentativas de Fraude da [Serasa Experian](https://www.serasaexperian.com.br/sala-de-imprensa/analise-de-dados/tentativas-de-fraude-atingem-326-mil-brasileiros-em-fevereiro-afirma-serasa-experian/), em fevereiro de 2022 houve 326.290 tentativas de fraudes. Isso quer dizer que a cada 7 segundos alguém foi alvo de golpistas.

Ainda, conforme o indicador o setor que mais teve tentativas de fraudes foi o segmento de Bancos e Cartões, onde foram registrados 181,739 casos.

O objetivo desse estudo é usar modelos de Machine Learning para evitar essas transações fraudulentas. Sabemos que as instituições financeiras, já usam Aprendizado de Máquina para prever e evitar esse tipo de ação. 



## Base de Dados

O conjunto de dados contém transações feitas por cartões de crédito em setembro de 2013 por titulares de cartões europeus. Este conjunto de dados apresenta transações que ocorreram em dois dias, onde temos 492 fraudes em 284.807 transações. O conjunto de dados é altamente desequilibrado, a classe positiva (fraudes) responde por 0,172% de todas as transações.

As variáveis passaram por uma transformação conhecida como Análise de Componentes Principais (Principal Component Analysis - PCA).

A PCA permite a redução da dimensionalidade enquanto mantém o maior número possível de informações. Esses componentes são em número menor ou igual às variáveis originais.

Acesse o [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) para fazer download do dataset e verificar mais informações.


## Análise Exploratória

No gráfico abaixo é possível perceber a discrepância entre os dados. Dessa forma, será necessário fazer um balanceamento dos dados para que o modelo não seja prejudicado ao treinar com dados desbalanceados.

Para efeito de estudos fiz os modelos considerando dados desbalanceados e balanceados. 

<p align="center">
  <img src="https://user-images.githubusercontent.com/99361817/183304393-e1e12a88-5ee3-4463-9105-65eafed3b5a4.png">
</p>


Percentual de não fraudes: 99.83 % no dataset

Percentual de fraudes: 0.17 % no dataset


**Análises Preliminares**

A variável alvo está localizada na coluna **Class**, onde: 

* 1 é uma transação fraudulenta;
* 0 é uma transação comum.

Abaixo uma análise estátistica descritiva da variável Class:


<p align="center">
  <img src="https://user-images.githubusercontent.com/99361817/183304401-95c2cabf-c204-4cf7-ae6e-69a304699892.png">
</p>



As transações fraudulentas têm uma distribuição mais uniforme do que as transações válidas, são igualmente distribuídas no tempo incluindo os baixos tempos reais de transação.

<p align="center">
  <img src="https://user-images.githubusercontent.com/99361817/183304422-9845f4e5-797b-4c41-85a3-31fe99a9a2bc.png">
</p>


Uma representação gráfica de boxplot para entender a diferença no padrão das transações em relação ao seu valor (Amount). É possível perceber que há uma distribuição diferente para as duas classes.

<p align="center">
  <img src="https://user-images.githubusercontent.com/99361817/183304429-db59f245-c24a-434f-8a66-3bedafe14e2b.png)">
</p>



## Preparação dos Dados


**Balanceamento dos Dados**

Devido a discrepância dos dados de Fraude e Não Fraude é preciso fazer o balanceamento para ter um melhor desempenho no momento de treinar os modelos.

**Synthetic Minority Over-sampling Technique (SMOTE) -** é uma técnica estatística para aumentar o número de casos em seu conjunto de um modo equilibrado. O componente funciona gerando novas instâncias de casos minoritários existentes que você fornece como entrada. 

**Dados Balanceados:** 0 = 199008 / 1 = 199008


<p align="center">
  <img src="https://user-images.githubusercontent.com/99361817/183304451-d11ad165-9c88-4a06-8f55-d6aa13a83ef9.png">
</p>



**Matriz de Correlação**

Antes do balanceamento podemos perceber que no primeiro gráfico não era possível obter informações relevantes da matriz de correlação. Após o balanceamento com o método SMOTE podemos verificar como cada variável se comporta em função de outra.


<p align="center">
  <img src="https://user-images.githubusercontent.com/99361817/183304453-03cb098f-c9c3-4fb7-ac0e-ff45941ca0f9.png">
</p>



## Construção do Modelo

Neste projeto trabalharemos com modelos supervisionados que são usados quando queremos explicar ou prever dados, isso é feito com a ajuda de dados antigos que serão destinados ao treino do modelo e assim ele será capaz de prever dados de saída para novas entradas.

Para a construção dos modelos utilizei três tipos de algoritmos:


* **Random Forest:** O algoritmo Random Forest (Floresta Aleatória em português) é um algoritmo de aprendizado de máquina utilizado para realizar predições. Resumidamente, o algoritmo cria de forma aleatória várias Árvores de Decisão (Decision Trees) e combina o resultado de todas elas para chegar no resultado final.

* **Logistic Regression:** É um método usado para prever a probabilidade de um resultado e é popular especialmente por tarefas de classificação. O algoritmo prevê a probabilidade de ocorrência de um evento ajustando dados para uma função logística.

* **XGBClassifier:**  É um algoritmo de aprendizado de máquina, baseado em árvore de decisão e que utiliza uma estrutura de Gradient boosting.


Em cada modelo para fins de comparação vamos plotar um **relatório de classificação, área sob a curva (AUC) e a matriz de confusão**. Isso nos ajudará a avaliar o desempenho do modelo, de forma que poderemos comparar e dizer qual possui um melhor desempenho na detecção de fraudes.



**Random Forest Classifier - Dados Desbalanceados**

Abaixo podemos ver que:

* O modelo possui uma grande quantidade de falsos negativos **(33 transações fraudulentas classificadas como comuns)**, o que não é bom, pois o banco ou o cliente irá arcar com os custos.
* Houve um bom desempenho para prever as transações normais, mas um desempenho abaixo para prever fraudes **(103)**.
* O AUC deste modelo foi **0.8786**.


```shell
clf = RandomForestClassifier(n_estimators=600, max_depth=6, random_state=10)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

print('Relatório de Clasfficação \nRandom Forest - Dados Desbalanceados: \n\n', classification_report(y_test, y_pred))

# Imprimir a área sob curva
print('AUC: {:.4f}\n'.format(roc_auc_score(y_test, y_pred)))
```
**[out]** AUC: 0.8786


<p align="center">
  <img src="https://user-images.githubusercontent.com/99361817/183304483-08f97558-9a78-4c58-91e9-29ae9a1c98ca.png">
</p>



**Random Forest Classifier - SMOTE**

Considerando os dados balanceados, podemos ver que:

* A quantidade de falsos negativos diminuiu consideravelmente **(12)**, o que é um ponto positivo.
* Houve uma melhora na previsão de fraudes **(124)**.
* Esse modelo teve um excelente **AUC: 0.9534**


```shell
clf_smt = RandomForestClassifier(n_estimators=600, max_depth=6, random_state=12, criterion='gini')
clf_smt.fit(X_train_smt, y_train_smt)

y_pred_smt = clf_smt.predict(X_test)

print('Relatório de Clasfficação \nRandom Forest - SMOTE: \n\n', classification_report(y_test, y_pred_smt))

# Imprimir a área sob curva
print('AUC: {:.4f}\n'.format(roc_auc_score(y_test, y_pred_smt)))
```
**[out]** AUC: 0.9534


<p align="center">
  <img src="https://user-images.githubusercontent.com/99361817/183304491-2c8fb4fe-edfe-4756-b51b-4929c91bbe30.png">
</p>


**Logistic Regression - Dados Desbalanceados**

Avaliando o modelo de Regressão Logística com dados desbalanceados:

* A quantidade de falsos negativos aumentou consideravelmente **(54)**, quando comparado com o modelo Randon Forest.
* A quantidade de fraudes previstas caiu muito **(82)**.
* O **AUC** deste modelo foi **0.8013**

```shell
model_logistic = LogisticRegression(max_iter=100, solver='liblinear')
model_logistic.fit(X_train, y_train)

y_pred_log = model_logistic.predict(X_test)

print('Relatório de Clasfficação \nLogistic Regression - Dados Desbalanceados: \n\n', classification_report(y_test, y_pred_log))

# Imprimir a área sob curva
print('AUC: {:.4f}\n'.format(roc_auc_score(y_test, y_pred_log)))
```
**[out]** AUC: 0.8013


<p align="center">
  <img src="https://user-images.githubusercontent.com/99361817/183304498-c9053939-4b4a-49c9-b4e8-404d98911a94.png">
</p>


**Logistic Regression - SMOTE**

Avaliando o modelo de Regressão Logística com dados balanceados pelo método SMOTE: 

* A quantidade de falsos negativos foi equilibrada com **14 falsos negativos**.
* A quantidade de fraudes também se manteve equilibrada, foram **122 fraudes previstas**, enquanto o modelo Random Forest foi capaz de realizar **124 previsões**.
* Esse modelo teve o** AUC: 0.9411**


```shell
model_logistic_smt = LogisticRegression(max_iter=100, solver='liblinear')
model_logistic_smt.fit(X_train_smt, y_train_smt)

y_pred_log_smt = model_logistic_smt.predict(X_test)

print('Relatório de Clasfficação \nLogistic Regression - SMOTE: \n\n', classification_report(y_test, y_pred_log_smt))

# Imprimir a área sob curva
print('AUC: {:.4f}\n'.format(roc_auc_score(y_test, y_pred_log_smt)))
```
**[out]** AUC: 0.9411


<p align="center">
  <img src="https://user-images.githubusercontent.com/99361817/183304504-295b5129-ff4c-41e7-b367-fe4266a16d18.png">
</p>



**XGBClassifier - Dados Desbalanceados**

Avaliando o desempenho do último modelo:

* A quantidade de falsos negativos foram **25**.
* A quantidade de fraudes previstas foram **111**.
* O **AUC** deste modelo foi **0.9081**

```shell
XGB = XGBClassifier(n_estimators=400, learning_rate=0.1, max_depth=4, min_child_weight=1, subsample=0.8, colsample_bytree=1, objective='binary:logistic')
XGB.fit(X_train, y_train)

y_pred_XGB = XGB.predict(X_test)

print('Relatório de Clasfficação \nXGBClassifier - Dados Desbalanceados: \n\n', classification_report(y_test, y_pred_XGB))

# Imprimir a área sob curva
print('AUC: {:.4f}\n'.format(roc_auc_score(y_test, y_pred_XGB)))
```
**[out]** AUC: 0.9081


<p align="center">
  <img src="https://user-images.githubusercontent.com/99361817/183304506-111c82a1-73f9-49c2-99fa-c28173d48f5e.png">
</p>



**XGBClassifier - SMOTE**

Com este modelo, podemos ver que:

* A quantidade de falsos negativos foi a maior entre os modelos com dados balanceados, com **15 falsos negativos**.
* A quantidade de fraudes se manteve equilibrada mais uma vez, foram **121 fraudes previstas**.
* Esse modelo teve **AUC: 0.9446**.


```shell
XGB_smt = XGBClassifier(n_estimators=400, learning_rate=0.1, max_depth=4, min_child_weight=1, subsample=0.8, colsample_bytree=1, objective='binary:logistic')
XGB_smt.fit(X_train_smt, y_train_smt)

y_pred_XGB_smt = XGB_smt.predict(X_test)

print('Relatório de Clasfficação \nXGBClassifier - SMOTE: \n\n', classification_report(y_test, y_pred_XGB_smt))

# Imprimir a área sob curva
print('AUC: {:.4f}\n'.format(roc_auc_score(y_test, y_pred_XGB_smt)))
```
**[out]** AUC: 0.9446


<p align="center">
  <img src="https://user-images.githubusercontent.com/99361817/183304510-c85c3e11-a13a-4f04-8c86-6f4a4e6ee07b.png">
</p>



## Considerações Finais

É possível notar que todos os modelos tiveram resultados ruins com dados desbalanceados, por isso a necessidade de balancear os dados e fazer uma boa análise antes de treinar os modelos. É possível concluir que:

* O algoritmo que melhor conseguiu prever fraudes foi o modelo de **Random Forest Classifier** que treinou com dados balanceados pelo método **SMOTE**.
* O modelo de **Random Forest Classifie**r teve a melhor área sob a curva **(AUC)** com **0,95**. 
* Com isso a solução ideal é aquela que melhor atende a instituição financeira, podendo ser a com o maior **AUC ou o modelo com o maior número de **detecção de fraudes**.

<p align="center">
  <img src="https://user-images.githubusercontent.com/99361817/183304515-6d3ba382-d422-41f2-a4ca-1c8405da9d09.png">
</p>


## Referências

Mãos à Obra: Aprendizado de Máquina com Scikit-Learn & TensorFlow - Aurélien Géron

Machine Learning – Guia de Referência Rápida Trabalhando com dados estruturados em Python - Matt Harrison