# Web Scraping das Ações do Índice Brasil 50

Web Scraping em tradução livre siginifica raspagem de rede. É utilizado para fazer coleta de dados estruturados da web de forma automatizada.

"Modelar um código para coleta de dados utilizando Web Scraping é você imaginar os passos que faria manualmente para obter os dados que precisa de forma automática."

**Exemplo:** Passo a passo para obter os dados das Ações do Índice Brasil 50 no site [Investing](https://br.investing.com/markets/brazil):

* Abrir o navegador;
* Acessar o site;
* Clicar no Marcador: Brasil;
* Selecionar a aba Ações;
* Clicar na caixa de seleção;
* Acessar: Índice Brasil 50.

Vejamos abaixo as etapas necessárias para coleta, limpeza, modelagem e ao final uma plotagem dos dados com informação do valor do preço de fechamento das ações:

* Coleta de Dados com a biblioteca com um script Python e biblioteca Selenium com o webdriver do Chrome;
* Limpeza, Exploração e Modelagem de dados utilizei a biblioteca Pandas;
* Visualização dos Dados utilizei Matplotlib.


## Acessando os Dados

Para o processo de coleta utilizei o Selenium com o webdriver do Chrome. Se você utiliza outro navegador é preciso fazer o 
download do arquivo executável específico para seu navegador. [Link para download](https://www.selenium.dev/downloads)

```shell
#Importar as Bibliotecas
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from time import sleep
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

webdriver.Chrome(executable_path='C:/Users/maison/Documents/DataScience/Portifolio/WebScraping/chromedriver.exe')
url = 'https://br.investing.com/markets/brazil'
driver = webdriver.Chrome()
driver.get(url)
sleep(8)
```


## Coletando e Salvando os Dados
Informe o local para salvar os dados coletados em extensão csv.

```shell
arquivo = open('C://table.csv', 'w', encoding='utf-8')
```


## Fechando as Janelas

Logo quando acessamos o site algumas janelas aparecem e com isso dificulta o processo de coleta dos dados. 

A primeira janela é para aceitar os cookies da página. Uma das formas de fechá-la é através do XPATH utilizando o find_element. Para acessar essa informação é preciso clicar com botão direito sobre o elemento e inspecionar ao abrir você verá uma página como essa abaixo:

Aqui você tem todas as informações dos elementos que precisa para modelar seu código.

![paginapng](https://user-images.githubusercontent.com/99361817/163293680-be51ec1a-a473-48fa-a30c-6b9c9928f6f1.png)

Abaixo o código para fechar essa janela:

```shell
#Permitir os cookies da página
button_accept = driver.find_element(By.XPATH, '//*[@id="onetrust-accept-btn-handler"]')
button_accept.click()
sleep(3)
```

A outra janela é para fazer login no site. Para fechar essa janela é semelhante a anterior. Clicar sobre o X da janela e inspecionar para pegar o XPath.

![logn](https://user-images.githubusercontent.com/99361817/163293996-21ebcb23-50d1-4170-9d65-44e376275475.png)


Abaixo o código para fechar a janela de login:

```shell
#Fechar aviso de login
button_close = driver.find_element(By.XPATH, '//*[@id="PromoteSignUpPopUp"]/div[2]/i')
button_close.click()
sleep(3)
```

Nesse processo tive dificuldade, pois, em determinado momento não aparecia essa janela e o meu código era encerrado. Mas quando fiz os testes novamente apareceu todas as vezes. Caso aconteça algum erro nessa parte do código observe se está aparecendo essa janela e monitore.


## Acessando a Aba Ações e Tabela

Como eu disse acima, para esse processo você tem que informar todos os passos que faria manualmente para torná-lo automático. Como descrito no código abaixo: 

```shell
#Acessar a aba de ações
button_acoes = driver.find_element(By.ID, 'stocks')
button_acoes.click()
sleep(5)

#Acessar Índice Brasil 50
button_indice = driver.find_element(By.XPATH, '/html/body/div[5]/section/div[7]/div[1]/select/option[4]')
button_indice.click()
sleep(5)

#Acessar a Tabela
table_body = driver.find_element(By.XPATH, '//*[@id="cross_rate_markets_stocks_1"]')
entries = table_body.find_elements(By.TAG_NAME, 'tr')
headers = entries[0].find_elements(By.TAG_NAME, 'th')

table_header = ''
for i in range(len(headers)):
    header = headers[i].text
    if i == len(headers) - 1:
        table_header = table_header + header + '\n'
    else:
        table_header = table_header + header + ';'
arquivo.write(table_header)

for i in range(0, len(entries)):
    cols = entries[i].find_elements(By.TAG_NAME, 'td')
    table_row = ''
    for j in range(len(cols)):
        col = cols[j].text
        if j == len(cols) - 1:
            table_row = table_row + col + '\n'
        else:
            table_row = table_row + col + ';'
    arquivo.write(table_row)

driver.close()
arquivo.close()
```


## Modelagem e Tratamento de Dados

Com os dados em mãos podemos começar as análises e o tratamento das informações. Converter os dados de string para float, datetime etc. 

Para ter uma boa manipulação desses dados precisamos investir muito tempo nessa etapa de tratamento e identificando possíveis erros nos dados extraídos.

```shell
#Importar Base de Dados
Base_Dados = pd.read_csv('table.csv', sep=';')

#Tratamento de Dados
colunas = ['Nome', 'Último', 'Máxima', 'Mínima', 'Variação', 'Var%', 'Vol.', 'Hora']
Base_Dados = pd.DataFrame(Base_Dados, columns=colunas).sort_values(by='Nome', ascending=False)

Base_Dados['Var%'] = Base_Dados['Var%'].apply(lambda x: float(x.replace('%', '').replace(',', '.')))
Base_Dados['Último'] = Base_Dados['Último'].apply(lambda x: float(x.replace(',', '.')))
Base_Dados['Máxima'] = Base_Dados['Máxima'].apply(lambda x: float(x.replace(',', '.')))
Base_Dados['Mínima'] = Base_Dados['Mínima'].apply(lambda x: float(x.replace(',', '.')))
Base_Dados['Variação'] = Base_Dados['Variação'].apply(lambda x: float(x.replace(',', '.')))
Base_Dados['Hora'] = pd.to_datetime(Base_Dados['Hora'], infer_datetime_format=True)
Base_Dados['Nome'] = Base_Dados['Nome'].apply(lambda x: str(x.replace(' ON', '',).replace(' PN', '',))).str.title()
Base_Dados['Vol.'] = Base_Dados['Vol.'].apply(lambda x: float(x.replace('M', '').replace(',', '.')))
Base_Dados.info()
print(Base_Dados)
```


## Visualização dos Dados

Para a visualização dos dados utilizei a biblioteca Matplotlib. Abaixo o gráfico do Valor de Fechamento das Ações de cada Empresa.

```shell
#Gráfico
#Função par formartar eixo em R$
def dec (x, pos):
    return f'R$ {x:.0f}'
formatter = FuncFormatter(dec)
plt.style.use('seaborn-darkgrid')
fig, ax = plt.subplots(figsize=(6, 10))
ax.xaxis.set_major_formatter(formatter)
plt.barh(Base_Dados['Nome'], Base_Dados['Último'])
plt.title('Índice Brasil 50 (IBrX 50 B3)', fontsize=11, loc='left')
plt.xlabel('Preço de Fechamento')
plt.ylabel('Empresa')
labels = ax.get_xticklabels()
plt.rcParams['xtick.labelsize'] = 6
plt.rcParams['ytick.labelsize'] = 6
plt.setp(labels, horizontalalignment='center')
plt.tight_layout()
plt.show()
```

<img src = "https://github.com/maisonhenrique/portifolio/blob/cf3fd424e366c5e34b0870e77868cc5482c4896b/Web_Scraping_Investing/Grafico.png" />


## Considerações Finais
No dia a dia deparamos com situações diversas que requer tempo para coleta de dados ou até mesmo execução de atividades importantes para o trabalho ou uso pessoal. Dessa forma, podemos criar soluções que otimizem o nosso tempo com recursos como, Web Sraping.

Nesse projeto percebemos o quanto essa ferramenta é útil. No exemplo mostrado podemos obter os dados das Ações e fazer analises como, por exemplo, Volume negociado no dia, Valor de Fechamento, Valor Mínimo e Máximo etc.


