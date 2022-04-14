# Web Scraping Ações do Índice Brasil 50

Web Scraping em tradução livre siginifica raspagem de rede. É utilizado para fazer coleta de dados estruturados da web de forma automatizada.

No dia a dia deparamos com situações diversas que requer tempo para coleta de dados ou até mesmo execução de atividades importantes para o trabalho ou uso pessoal.
Dessa forma, podemos criar soluções que otimizem o nosso tempo com recursos como, Web Sraping. 


"Modelar um código para coleta de dados utilizando Web Scraping é você imaginar os passos que faria manualmente para obter os dados que precisa de forma automática."

**Exemplo:** Passo a passo para obter os dados das Ações do Índice Brasil 50 no site [Investing](https://br.investing.com/markets/brazil):

* Abrir o navegador;
* Acessar o site;
* Clicar no Marcador: Brasil;
* Selecionar a aba Ações;
* Clicar na caixa de seleção;
* Acessar: Índice Brasil 50.


Vejamos abaixo os passos que foram necessários para obtenção, modelagem e ao final uma plotagem dos dados com informação do valor do preço de fechamento das ações.

* Coleta de Dados com a biblioteca com um script Python e biblioteca Selenium com o webdriver do Chrome
* Limpeza, Exploração e Modelagem de dados utilizei a biblioteca Pandas
* Visualização dos Dados utilizei Matplotlib

# Acessando os Dados

Para o processo de coleta utilizei o Selenium com o webdriver do Chrome. Se você utiliza outro navegador é preciso fazer o 
download do arquivo executável específico para seu navegador. [Link para download](https://www.selenium.dev/downloads)


# Coletando e Salvando os Dados
Informe o local para salvar os dados coletados em extensão .csv.


# Fechando as Janelas
Logo quando acessamos o site algumas janelas aparecem e com isso dificulta o processo de coleta dos dados. 

A primeira janela é para aceitar os cookies da página. Uma das formas de fechá-la é através do XPATH utilizando o find_element. Para acessar essa informação é preciso clicar com botão 
direito sobre o elemento e inspecionar ao abrir você verá uma página como essa abaixo:

Aqui você tem todas as informações dos elementos que precisa para modelar seu código.


Abaixo o código para fechar essa janela:


Nesse processo tive dificuldade, pois, em determinado momento não aparecia essa janela e o meu código era encerrado. Mas quando fiz os testes novamente apareceu todas as vezes. 
Caso aconteça algum erro nessa parte do código observe se está aparecendo essa janela e monitore.


# Acessando a Aba Ações

Como eu disse acima, para esse processo você tem que informar todos os passos que faria manualmente para torná-lo automático. Como descrito no código abaixo: 


# Modelagem e Tratamento de Dados

Com os dados em mãos podemos começar as análises e o tratamento das informações. Converter os dados de string para float, datetime etc. 

Para ter uma boa manipulação desses dados precisamos investir muito tempo nessa etapa de tratamento e identificando possíveis erros nos dados extraídos.


# Visualização dos Dados

Para a visualização dos dados utilizei a biblioteca Matplotlib. Abaixo o gráfico do Valor de Fechamento das Ações de cada Empresa.





