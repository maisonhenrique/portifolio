# Análise de Internações Hospitalares do SUS

Este é um projeto de análise e insights da base de dados de internação do SUS no período: 12/2017 à 07/2019, com uso de séries temporais para previsão de novas internações e os custos.

**Observação:** Acesse o arquivo [pdf](https://github.com/maisonhenrique/portifolio/blob/0d28024f7a2df9667f3cdbacfe001c0c8ec81f38/Internacoes_SUS/Analise%20Final.pdf) com a análise final.


## Fonte de Dados

http://tabnet.datasus.gov.br/cgi/sih/sxdescr.htm


## Tratamento de Dados

Limpeza dos dados, exclusão dos somatórios para não influenciar nas etapas posteriores. Renomeação dos Estados (Eliminação de caracteres especiais), separação de Estados e Regiões, renomeação das colunas. 


## Missing

Durante o tratamento de dados tinha períodos faltantes em 2018: Janeiro, Fevereiro, Junho e Outubro e em 2019: Março e Maio. Ao verificar os dados novamente, no mês de Janeiro de 2019 não tinha o Estado do Acre, onde foi feito a inserção manualmente por se tratar apenas de uma linha. 

Para a imputação de dados poderia ter utilizado, Mediana, Média, método de Missing Forest e KNNimputer. Como é o primeiro modelo para o projeto, foi utilizado Média Móvel para imputação dos dados ausentes. 


## Feature Engineering

Conforme o desenvolvimento do projeto foi preciso criar algumas features para trazer mais segurança nos valores finais do modelo. Criação das features de Total de Complemento Federal e Gestor, Prorrogações, Região, Número de Leitos Ocupados.


## Análise Exploratória dos Dados (EDA)

Essa etapa é muito importante para extração de insights para solucionar os problemas relacionados as internações hospitalares do SUS. Durante o projeto foi possível obter mais insights sobre os dados, trazendo uma separação por Estado, Região e  durante os períodos analisados. 
Foi utilizado a biblioteca Pandas para manipulação e análise de dados e Matplotlib e Seaborn para visualização de dados e criação de gráficos.


## Modelagem de Dados

No modelo de previsão foi feito teste de estacionaridade de Dickey-Fuller e para as previsões futuras foi utilizado Auto ARIMA.


## Avaliações Finais

A avaliação final do modelo foi feita comparando os valores previstos dos reais. Conforme as previsões dos Valores Totais de Internações, Número de Óbitos e o Valor Médio AIH, foi possível estimar os valores gastos para os próximos seis meses.
