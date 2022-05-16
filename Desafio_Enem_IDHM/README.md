# Desafio Enem

Esse projeto faz parte de um desafio para uma oportunidade de Analista de Dados, que por sinal foi muito prazeiroso. Pude explorar muitos conhecimentos e 
particurlamente gostei muito do resultado.

**Objetivo:** Investir em 100 escolas de municípios com IDHM de nível Baixo e Muito Baixo. 

De acordo com os critérios de investimento fiz uma análise de dados considerando os resultados da prova do ENEM 2020 e o IDHM.


## Tratamento de Dados

Coleta, limpeza, exploração e modelagem dos dados. Utilizei o framework Pandas.

* Eliminação de dados desnecessários,
* Renomeação dos municípios que eram divergentes entre as bases de dados;
* Eliminação dos participantes ausentes.


## Missing

Durante o tratamento de dados percebi muitas informações nulas referente aos dados da escola dos participantes (pode ser que no momento da inscrição não era obrigatório), eliminei esses dados. Para as análises utilizei os dados do local de prova. 
Por isso é possível identificar vários dados nulos nos resultados das provas, questionário socioeconômico entre outros.


## Número de Inscritos

A quantidade de participantes ausentes no exame em 2020 foi maior que os presentes, já que no site não tem os anos anteriores para fazer uma comparação. 


## Fonte de Dados

[IDHM](http://www.atlasbrasil.org.br), [ENEM 2020](https://www.gov.br/inep/pt-br/acesso-a-informacao/dados-abertos/microdados/enem) e 
[Cotação Dólar](https://br.investing.com/currencies/usd-brl-historical-data)


## Resultado

O arquivo .pbix ficou muito extenso e não foi possível adicionar no GitHub. Para verificar o resultado final acesse o [PDF]() extraido do Power BI.