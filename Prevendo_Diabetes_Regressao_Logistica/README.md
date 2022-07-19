# Previsão de Diabetes - Regressão Logística

Esse projeto faz parte de uma sequência de estudos relacionado a Machine Learning. Para esse projeto foi utilizado Regressão Logística.


## Base de Dados

Esse conjunto de dados é utilizado para prever diagnosticamente se um paciente tem ou não diabetes com base nos parâmetros de entrada, como Idade, Glicose, Pressão arterial, 
Insulina, IMC, entre outros. Todos os pacientes aqui são mulheres com pelo menos 21 anos de ascendência indígena Pima. O dataset está disponível no 
[Kaggle](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database).

Conhecendo o dataset:

* Pregnancies: Número de Gestações
* Glucose: Glicose
* Blood Pressure: Pressão Arterial
* Skin Thickness: Espessura da pele
* Insulin: Insulina
* BMI: IMC
* Diabetes Pedigree Function: Função que pontua a probabilidade de diabetes com base no histórico familiar
* Age: Idade
* Outcome: Resultado (1= Sim, 0= Não)


## Criando o modelo com Scikit-learn

**Matriz de Variáveis**

```shell
X = df.drop(columns='Resultado')
y = df['Resultado']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)
```

**Regressão Logística**

```shell
logistic_model = LogisticRegression()
logistic_model.fit(X_train, y_train)

# Avaliação do Modelo

Previsao = logistic_model.predict(X_test)
print('Matriz Confusão: \n', confusion_matrix(y_test, Previsao), '\n')
```

iamgem

**Métricas de Classificação - Relatório de Classificação**

```shell
print('Relatório de Clasfficação - Regressão Logistica: \n', classification_report(y_test, Previsao)
```

iamgem


**Previsão Balanceada:** Calcular a precisão balanceada evita estimativas de desempenho inflados em conjunto de dados desequilibrados.

```shell
print('Score (Treino): ', round(logistic_model.score(X_train, y_train), 2))
print('Score (Teste): ', round(logistic_model.score(X_test, y_test), 2))
```

**[out]**
Score (Treino): 0.77 
Score (Teste): 0.77


Portanto se utilizar os dados de treino e teste no final teremos uma precisão balanceada. 


**Validação cruzada:** A técnica de validação cruzada para avaliar validade externa é feita com múltiplos subconjuntos da amostra total. A abordagem mais amplamente usada é o 
método jackknife. Validação cruzada é baseada no princípio do “deixe um de fora”. O uso mais comum desse método é estimar k – 1 amostras, eliminando-se uma observação por 
vez a partir de uma amostra de k casos. Uma função discriminante é calculada para cada subamostra, e em seguida a pertinência a grupo prevista da observação eliminada é 
feita com a função discriminante estimada sobre os demais casos. Depois que todas as previsões de pertinência a grupo foram feitas, uma por vez, uma matriz de classificação 
é construída e a razão de sucesso é calculada.

No entanto, validação cruzada pode representar a única técnica de validação possível em casos em que a amostra original é muito pequena para dividir em amostras de análise e 
de teste, mas ainda excede as orientações já discutidas. (Peter Bruce, Andrew Bruce)

```shell
Validacao_Cruzada = cross_val_score(logistic_model, X, y, cv=5)
print(Validacao_Cruzada)
```

**[out]** [0.77272727 0.74675325 0.75974026 0.81699346 0.75163399]


A validação cruzada é importante para identificar as proporções ruins no modelo quando temos valores discrepantes nos resultados. Verificando na saída [out] acima, 
temos uma média de 0,77 isso é bom quando comparado com a **Previsão Balanceada**, que também tivemos uma média de 0,77. 


## Comparação com outros Modelos

**Random Forest**

```shell
forest_model = RandomForestClassifier(max_depth=3)
forest_model.fit(X_train, y_train)

Previsao_forest = forest_model.predict(X_test)
print('Relatório de Clasfficação - Random Forest: \n', classification_report(y_test, Previsao_forest))
```

imagem


**SVM**

```shell
svm_model = svm.SVC()
svm_model.fit(X_train, y_train)
Previsao_svm = svm_model.predict(X_test)

print('Relatório de Clasfficação - SVM: \n', classification_report(y_test, Previsao_svm))
```

iamgem


## Considerações Finais

Considerando os modelos testados a Regressão Logística apresentou a melhor acurácia com cerca de 77%. Quando ampliamos a análise verificamos que o Recall apresentou valores 
de 0,93 para 0 = Não e 0,51 para 1= Sim. Isso significa que o modelo conseguiu identificar mais a Classe 0 que a classe 1. No modelo em questão parece promissoras se tratando 
de previsão. Além disso, antes de finalizar um diagnóstico de situação de saúde com base em modelos de Machine Learning, é essencial colocar um foco maior na interpretação da 
matriz de confusão como falsos positivos – falsos negativos podem ser arriscados.

**Observação:** Para esse estudo não foi utilizado hiper parâmetro. 


## Referências

Análise multivariada de dados [recurso eletrônico] / Joseph F Hair Jr ... [et al.] ; tradução Adonai Schlup Sant’Anna. – 6. ed. – Dados eletrônicos. – Porto Alegre : Bookman, 2009.

Estatística Prática para Cientistas de Dados - 50 Conceitos Essenciais - Peter Bruce,  Andrew Bruce