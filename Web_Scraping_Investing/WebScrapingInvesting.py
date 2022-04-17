#Importar as Bibliotecas
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from time import sleep
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

webdriver.Chrome(executable_path='C:/chromedriver.exe')
url = 'https://br.investing.com/markets/brazil'
driver = webdriver.Chrome()
driver.get(url)
sleep(8)

#Local para salvar o arquivo em .csv
arquivo = open('C:/table.csv', 'w', encoding='utf-8')

#Permitir os cookies da página
button_accept = driver.find_element(By.XPATH, '//*[@id="onetrust-accept-btn-handler"]')
button_accept.click()
sleep(3)

#Fechar aviso de login
button_close = driver.find_element(By.XPATH, '//*[@id="PromoteSignUpPopUp"]/div[2]/i')
button_close.click()
sleep(3)

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
