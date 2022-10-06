# Churn Prediction - Empresa de Telecomunicações

**O que é Churn?**
Podemos explicar como o número de clientes que cancelam um serviço em um determinado período de tempo. 

Podemos extrair vários insights de uma análise de Churn. Com isso, uma empresa terá números suficientes para a tomada de decisão onde o objetivo é manter esses clientes por mais tempo em seu portifólio.

Todas as empresas possuem informações valiosíssimas de seus clientes em sua base de dados. Assim, com acesso a informações sobre esse público é possível analisar com mais cuidado para evitar Churns. 

Um bom exemplo é entender o comportamento de uso do produto ou serviço pelo cliente. Assim, com todas essas informações disponíveis é possível entender como o produto ou serviço está sendo utilizado. Se está correto, se há muito tempo sem acesso ou se está usando somente por utilizar.


Por meio da ferramenta de gestão de relacionamento com o cliente é possível verificar:

* registros de reclamações;
* tempo de resposta para as demandas do cliente;
* indisponibilidade ou oscilações nos serviços contratados;
* quantidade de reclamações recorrentes, entre outras.

Dessa forma a empresa pode identificar os motivos que levam a desistência do cliente de usar ou comprar um produto ou serviço.



## Base de Dados
Os dados utilizados neste projeto foram originalmente disponibilizados na plataforma de ensino da IBM Developer e atualmente pode ser encontrado [neste link](https://www.kaggle.com/datasets/blastchar/telco-customer-churn?resource=download) no Kaggle.

Conhecendo as variáveis do dataset:

* customerID - Código de identificação do consumidor;
* gender - Gênero do consumidor;
* SeniorCitizen -  Indica se o cliente tem 65 anos ou mais;
* Partner - Indica se o cliente é casado;
* tenure - Quantos meses a pessoa é cliente da empresa;
* PhoneService - Possui serviço telefônico;
* MultipleLines - Possui múltiplas linhas telefônicas;
* InternetService - Qual provedor de serviço de internet;
* OnlineSecurity - Possui serviço de segurança online;
* OnlineBackup - Possui serviço de backup online ativado;
* DeviceProtection - Cliente possui alguma proteção de sistema;
* TechSupport - Possui serviço de suporte técnico ativado;
* StreamingTV - Possui streaming de TV ativado;
* StreamingMovies - Possui serviço de streaming de filmes ativado;
* Contract - Tipo do contrato do consumidor;
* PaperlessBilling - Cliente utiliza faturamento sem papel;
* PaymentMethod - Método de pagamento;
* MonthlyCharges - Pagamento mensal atual;
* TotalCharges - Valor total que o cliente pagou pelos serviços durante todo o tempo;
* Churn - Cliente abandonou o serviço.


Algumas observações sobre o dataset do estudo:

* A coluna customerID trata-se de um código de identificação dos clientes, por isso foi excluída do dataset.
* A variável alvo é a coluna Churn, ela indica se o cliente cancelou ou não o serviço.
* A coluna TotalCharges assim como a coluna MonthlyCharges, apresenta valores flutuantes, mas está como tipo string uma vez que seu tipo de dado deveria ser float.
* Ao tentar converter a coluna TotalCharges para tipo float apresentou um erro porque os dados ausentes foram preenchidos com " ". Foi criada uma função para lidar com o erro e em seguida uma imputação de valores utilizando a mediana.


## Análise Exploratória dos Dados (EDA)
Principais insights das variáveis categóricas em relação a variável que iremos prever (Churn):

* Clientes que não possuem dependentes e que não são casados tem maior probabilidade de abandonar os serviços;
* Como podermos observar, cliente que não possuem internet tendem a permanecer mais tempo com a empresa;
* A qualidade do serviço deve ser considerada na estratégia de reter o cliente, pois podemos observar uma taxa de Churn considerável para os consumidores que possuem streaming e internet de fibra ótica.

<p align="center">
  <img src=".png">
</p>


**Contratos**

A maioria dos contratos é do tipo Month-to-month, esses clientes tendem cancelar mais os serviços. Dessa forma, a empresa precisa pensar em uma forma de reter esses clientes com um contrato maior e com melhores serviços de internet, visto que, os clientes que mais abandonam a empresa são os que possuem internet.

* Porcentagem de contratos mensais: **55.02%**
* Porcentagem de contratos mensais de clientes que não cancelaram: **42.91%**
* Porcentagem de contratos mensais de clientes que cancelaram: **88.55%**

<p align="center">
  <img src=".png">
</p>



Com base no histograma abaixo é possível entender que:

* As variáveis TotalCharges e tenure estão relacionadas indiretamente, ou seja, quanto mais tempo de permanência na operadora maior a taxa de retenção do cliente;
* Quanto maior o preço da mensalidade do serviço, maior a chance de perder o cliente.

<p align="center">
  <img src=".png">
</p>


**Churn**
É perceptível a diferença entre as classes No e Yes da variável Churn. Para obter os melhores resultados com o modelo de Machine Learning , é preciso fazer o balanceamento das classes.

Porcentagem de Churn é **26.54%** no dataset.

<p align="center">
  <img src=".png">
</p>

Conforme o gráfico acima, os dados relacionados com a variável alvo Churn está desbalanceada, onde apenas 26,54% dos dados totais estão relacionados com os clientes que cancelaram o serviço. 


**Estatística Descritiva**
Conforme método describe abaixo é possível notar que não há presença de outliers.

<p align="center">
  <img src=".png">
</p>

Assim como, nos gráficos de boxplot.

<p align="center">
  <img src=".png">
</p>


## Preparação dos Dados

Nesta etapa é necessário fazer um pré-processamento dos dados. O primeiro passo é transformar as variáveis categóricas em valores numéricos. Para isso, será utilizado o LabelEncoder para transformar as variáveis binárias em valores de 0 e 1.

Em seguida, as variáveis categóricas serão tratadas com o getDummies, que faz a transformação de forma direta das variáveis categóricas, assim poderão ser utilizadas em todos os modelos.


**Padronização e Balanceamento**
Para a padronização será usado **StandardScale**r e **RobustScaler**, para ter uma melhor comparação entre os métodos. Em seguida os dados padronizados serão combinados com o balanceamento **Random UnderSampling** e **ADASYN**. 


**Random UnderSampling**
Como os dados da variável alvo Churn encontra-se desbalanceado. O método Random Under Sampling é simples e envolve a exclusão aleatória de algumas instâncias da classe majoritária.

<p align="center">
  <img src=".png">
</p>


**ADASYN**
No balanceamento ADASYN são adicionadas entradas e tenta fazer uma diferenciação das entradas já existentes, leva em consideração a densidade de distribuição para distribuir os pontos de dados uniformemente

<p align="center">
  <img src=".png">
</p>


**Avaliando os Modelos com Cross-Validation (Dados Balanceados e Padronizados)**
Após realizar a validação de todos os modelos, o balanceamento **ADASYN** com padronização **RobustScaler** obtiveram melhores desempenhos.

<p align="center">
  <img src=".png">
</p>


**Hiperparâmetros**
A fim de obter resultados mais consistentes foi realizado um processo de otimização de hiperparâmetros utilizando o algoritmo **GridSearchCV** do **Scikit-Learn**, que visa escolher o melhor conjunto de hiperparâmetros a partir de uma validação cruzada.

<p align="center">
  <img src=".png">
</p>


O modelo que obteve os melhores resultados foi definitivamente o modelo SVC que foi treinado com os dados balanceados no método ADASYN e com padronização RobustScaler.

<p align="center">
  <img src=".png">
</p>

Esse modelo obteve:

531 previsões de Churn;
43 previsões de falsos negativos;
715 previsões de falsos positivos.



## Considerações Finais

Churn é algo preocupante para todas as empresas e por meio de modelos de Machine Learning é possível reduzir esse número com churn prediction. 

O modelo escolhido nesse estudo se baseou na métrica recall com o objetivo de acertar o máximo de Churn em clientes que realmente cancelaram o serviço.

Dessa forma, é possível prever dentre os novos dados de clientes quais podem cancelar o serviço, assim, buscar maneiras que antecipem e contribuam para reter e fidelizar o cliente.

É importante ressaltar que o projeto tem margem para melhoria e estou aberto para sugestões.