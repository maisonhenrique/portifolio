# Importando as Bibliotecas
import pandas as pd
import numpy as np
from datetime import date
import dataframe_image as dfi
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from pmdarima.arima import auto_arima
import warnings
warnings.filterwarnings('ignore')

pd.set_option('display.float_format', lambda x: '%.2f' % x)

# Base de Dados
arquivo = pd.read_excel('case_internacao_SUS.xls', sheet_name=None)
Base_Dados = pd.DataFrame()

for key in arquivo.keys():
    dados = arquivo[key]
    dados['data'] = key
    Base_Dados = pd.concat([Base_Dados, dados], axis=0)
print(Base_Dados)

# Renomeando as colunas
df = Base_Dados.copy()
df.columns = ['uf', 'internacoes', 'aih_aprovadas',
       'valor_total', 'valor_servicos_hosp',
       'val_serv_hosp_compl_federal', 'val_serv_hosp_compl_gestor',
       'valor_serviços_profissionais', 'val_serv_prof_compl_federal',
       'val_serv_prof_compl_gestor', 'valor_medio_aih', 'valor_medio_internacao',
       'dias_permanencia', 'media_permanencia', 'obitos', 'taxa_mortalidade',
       'data']


# Excluindo a linha Total
df.drop(df.loc[df['uf'] == 'Total'].index, inplace=True)

#Inserindo o Estado Acre no mês Julho/19
df = df.append({'uf': '.. Acre', 'data': 'jul19'}, ignore_index=True)

df = df.dropna()

# Filtrando os Estados não nulos
df = df[df['uf'].notnull()]

# Alterando os nomes dos Estados
df = df[df['uf'].str.contains('.', regex=False)]
df['uf'] = df['uf'].apply(lambda x: x.replace('.. ', ''))

# Renomear os Estados
estados = {
    'AC': 'Acre',
    'AL': 'Alagoas',
    'AP': 'Amapá',
    'AM': 'Amazonas',
    'BA': 'Bahia',
    'CE': 'Ceará',
    'DF': 'Distrito Federal',
    'ES': 'Espírito Santo',
    'GO': 'Goiás',
    'MA': 'Maranhão',
    'MT': 'Mato Grosso',
    'MS': 'Mato Grosso do Sul',
    'MG': 'Minas Gerais',
    'PA': 'Pará',
    'PB': 'Paraíba',
    'PR': 'Paraná',
    'PE': 'Pernambuco',
    'PI': 'Piauí',
    'RJ': 'Rio de Janeiro',
    'RN': 'Rio Grande do Norte',
    'RS': 'Rio Grande do Sul',
    'RO': 'Rondônia',
    'RR': 'Roraima',
    'SC': 'Santa Catarina',
    'SP': 'São Paulo',
    'SE': 'Sergipe',
    'TO': 'Tocantins'
}

renomear_uf = {v: k for k, v in estados.items()}
df['uf'] = df['uf'].map(renomear_uf)

# Preencher dados faltantes por 0
df.fillna(0, inplace=True)

# Substituir - por NaN
for i in df.columns:
    df[df[i] == '-'] = df[df[i] == '-'].apply(lambda x: x.replace('-', np.NaN))

# Tratamento de Dados na coluna Data
df['mes'] = df['data'].apply(lambda x: x[0:3])
df['ano'] = df['data'].apply(lambda x: x[-2:]).apply(lambda x: '20'+x)

mes = {'jan': '1', 'fev': '2', 'mar': '3', 'abr': '4', 'mai': '5', 'jun': '6', 'jul': '7', 'ago': '8', 'set': '9', 'out': '10', 'nov': '11', 'dez': '12'}
df['mes'] = df['mes'].replace(mes)

df['data'] = df['ano']+'-'+df['mes']

# Converter para Datetime
df['data'] = pd.to_datetime(df['data'], format='%Y-%m')

# Adicionar Meses faltantes
# Meses faltantes
# 2018 - Janeiro, Fevereiro, Junho, Outubro
# 2019 - Março, Maio

add = pd.DataFrame(columns=df.columns)
meses_faltantes = [date(2018, 1, 1), date(2018, 2, 1), date(2018, 6, 1), date(2018, 10, 1), date(2019, 3, 1), date(2019, 5, 1)]
for i in range(0, len(df['uf'].unique())):
    for j in range(0, len(meses_faltantes)):
        add1 = pd.DataFrame([[df['uf'].unique()[i], np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN,
                              meses_faltantes[j], np.NaN, np.NaN]], columns=df.columns)
        add = pd.concat([add, add1])

print(add)

# Criando a coluna Mês e Ano
add['mes'] = pd.DatetimeIndex(add['data']).month
add['ano'] = pd.DatetimeIndex(add['data']).year
add['mes'] = add['mes'].apply(lambda x: str(x))
add['ano'] = add['ano'].apply(lambda x: str(x))

df = pd.concat([df, add])
df = df.sort_values(by='data')
df = df.reset_index(drop=True)
df['data'] = pd.to_datetime(df['data'], format='%Y-%m')


# Criar Coluna Região
norte = ['AM', 'PA', 'AP', 'AC', 'RR', 'RO', 'TO']
nordeste = ['MA', 'PI', 'CE', 'RN', 'PB', 'PE', 'SE', 'BA', 'AL']
centro_oeste = ['MT', 'MS', 'GO', 'DF']
sudeste = ['MG', 'ES', 'RJ', 'SP']
sul = ['PR', 'SC', 'RS']

df['regiao'] = df['uf'].apply(lambda x: 'Norte' if x in norte else 'Nordeste' if x in nordeste else 'Centro_Oeste' if x in centro_oeste else 'Sudeste' if x in sudeste else 'Sul' if x in sul else np.NaN)
print(df)

# Ajustando o Dataset
# Numero de Leitos Ocupados
df['numero_leitos_ocupados'] = (df['aih_aprovadas'] * df['media_permanencia']) / 30

# Prorrogações
df['prorrogacoes'] = df['aih_aprovadas'] - df['internacoes']
df['complemento_federal'] = df['val_serv_hosp_compl_federal'] + df['val_serv_prof_compl_federal']
df['complemento_gestor'] = df['val_serv_hosp_compl_gestor'] + df['val_serv_prof_compl_gestor']


# Analise Estatística descritiva
describe = df.describe().T
print(describe)

# Filtrando valores para o Dataset
df1 = df[['uf', 'regiao', 'data', 'mes', 'ano', 'prorrogacoes', 'internacoes', 'aih_aprovadas', 'valor_total', 'complemento_federal', 'complemento_gestor',
           'valor_serviços_profissionais', 'valor_medio_internacao', 'valor_medio_aih', 'numero_leitos_ocupados', 'obitos', 'taxa_mortalidade', 'media_permanencia']]


# Analise Exploratória dos Dados
df2 = df1.copy()
df2 = df2[df2['internacoes'].notnull()]
print(df2)

# Configuração de plotagem
plt.style.use('seaborn-darkgrid')
sns.set_theme()
sns.set_color_codes('muted')

# Valor Total de Pagamentos de Internações
fig1, ax1 = plt.subplots(nrows=2, ncols=2, figsize=(18, 8), constrained_layout=True)

plt.subplot(2, 2, (1,3))
aux = df2[['ano', 'uf', 'valor_total']].groupby('uf').sum().sort_values(by='valor_total', ascending=False).reset_index()
sns.barplot(x='valor_total', y='uf', data=aux, color='b')
plt.title('Valor Total para Pagamentos de Internações por Estado no ano de 2018 (R$ bilhões)', fontsize=12, loc='left', pad=10)
plt.xlabel('')
plt.ylabel('')

plt.subplot(2, 2, 2)
aux = df2[['regiao', 'valor_total']].groupby('regiao').sum().sort_values(by='valor_total', ascending=False).reset_index()
sns.barplot(x='valor_total', y='regiao', data=aux, color='c')
plt.title('Valor Total para Pagamentos de Internações por Região (R$ bilhões)', fontsize=12, loc='left', pad=10)
plt.xlabel('')
plt.ylabel('')

plt.subplot(2, 2, 4)
aux = df2[['data', 'uf', 'valor_total']].groupby(['data', 'uf']).sum().sort_values(by='valor_total', ascending=False).reset_index()
aux = aux[aux['uf'].isin(['SP', 'MG', 'PR', 'RS', 'RJ'])]
ax = sns.lineplot(x='data', y='valor_total', hue='uf', data=aux)
plt.title('Valor Pago para Internações Mensalmente pelos 5 maiores Estado (R$ bilhões)', fontsize=12, loc='left', pad=10)
plt.legend(bbox_to_anchor=(1, 1), loc='upper left', borderaxespad=0.)
plt.ylim(0, 3.5*10**8)
plt.xlabel('')
plt.ylabel('')

plt.tight_layout()
plt.show()


# Analise de Porcentagem
analise = df2[df2['ano'] =='2018'][['uf', 'regiao', 'valor_total']].groupby(['regiao', 'uf']).sum().reset_index()

porcentagem_sudeste = analise[analise['uf'] == 'SP']['valor_total'].sum()/analise[analise['regiao'] == 'Sudeste']['valor_total'].sum()*100
print('Porcentagem paga para o estado de São Paulo representa {}% do valor total pago em internações na região Sudeste em 2018'.format(round(porcentagem_sudeste, 2)))


# Valor Total Pago em Complemento Federal
fig2, ax2 = plt.subplots(nrows=2, ncols=2, figsize=(18, 8), constrained_layout=True)

plt.subplot(2, 2, (1,3))
aux = df2[['uf', 'complemento_federal']].groupby('uf').sum().sort_values(by='complemento_federal', ascending=False).reset_index()
sns.barplot(x='complemento_federal', y='uf', data=aux, color='b')
plt.title('Valor Total Pago em Complemento Federal por Estado (R$ milhões)', fontsize=12, loc='left', pad=10)
plt.xlabel('')
plt.ylabel('')

plt.subplot(2, 2, 2)
aux = df2[['regiao', 'complemento_federal']].groupby('regiao').sum().sort_values(by='complemento_federal', ascending=False).reset_index()
sns.barplot(x='complemento_federal', y='regiao', data=aux, color='c')
plt.title('Valor Total Pago em Complemento Federal por Região (R$ milhões)', fontsize=12, loc='left', pad=10)
plt.xlabel('')
plt.ylabel('')

plt.subplot(2, 2, 4)
aux = df2[['data', 'valor_total', 'complemento_federal', 'regiao']].groupby(['data', 'regiao']).sum().sort_values(by='complemento_federal', ascending=False).reset_index()
sns.lineplot(x='data', y='complemento_federal', hue='regiao', data=aux)
plt.title('Valor Total Pago em Complemento Federal ao Longo do Tempo (R$ milhões)', fontsize=12, loc='left', pad=10)
plt.legend(bbox_to_anchor=(1, 1), loc='upper left', borderaxespad=0.)
plt.xlabel('')
plt.ylabel('')

plt.tight_layout()
plt.show()


# Valor Total Pago em Complemento Gestor
fig3, ax3 = plt.subplots(nrows=2, ncols=2, figsize=(18, 8), constrained_layout=True)

plt.subplot(2, 2, (1,3))
aux = df2[['uf', 'complemento_gestor']].groupby('uf').sum().sort_values(by='complemento_gestor', ascending=False).reset_index()
sns.barplot(x='complemento_gestor', y='uf', data=aux, color='b')
plt.title('Valor Total Pago em Complemento Gestor por Estado (R$ milhões)', fontsize=12, loc='left', pad=10)
plt.xlabel('')
plt.ylabel('')

plt.subplot(2, 2, 2)
aux = df2[['regiao', 'complemento_gestor']].groupby('regiao').sum().sort_values(by='complemento_gestor', ascending=False).reset_index()
sns.barplot(x='complemento_gestor', y='regiao', data=aux, color='c')
plt.title('Valor Total Pago em Complemento Gestor por Região (R$ milhões)', fontsize=12, loc='left', pad=10)
plt.xlabel('')
plt.ylabel('')

plt.subplot(2, 2, 4)
aux = df2[['data', 'valor_total', 'complemento_gestor', 'regiao']].groupby(['data', 'regiao']).sum().sort_values(by='complemento_gestor', ascending=False).reset_index()
sns.lineplot(x='data', y='complemento_gestor', hue='regiao', data=aux)
plt.title('Valor Total Pago em Complemento Gestor ao Longo do Tempo (R$ milhões)', fontsize=12, loc='left', pad=10)
plt.legend(bbox_to_anchor=(1, 1), loc='upper left', borderaxespad=0.)
plt.xlabel('')
plt.ylabel('')

plt.tight_layout()
plt.show()


# Valor Médio de Internações
fig4, ax4 = plt.subplots(nrows=2, ncols=2, figsize=(18, 8), constrained_layout=True)

plt.subplot(2, 2, (1,3))
aux = df2[['valor_medio_internacao', 'uf']].groupby('uf').mean().sort_values(by='valor_medio_internacao', ascending=False).reset_index()
sns.barplot(y='uf', x='valor_medio_internacao', data=aux, color='b')
plt.title('Valor Médio de Internações por Estado (R$)', fontsize=12, loc='left', pad=10)
plt.xlabel('')
plt.ylabel('')

plt.subplot(2, 2, 2)
aux = df2[['valor_medio_internacao', 'regiao']].groupby('regiao').mean().sort_values(by='valor_medio_internacao', ascending=False).reset_index()
sns.barplot(x='valor_medio_internacao', y='regiao', data=aux, color='c')
plt.title('Valor Médio de Internações por Região (R$)', fontsize=12, loc='left', pad=10)
plt.xlabel('')
plt.ylabel('')

plt.subplot(2, 2, 4)
aux = df2[['data', 'valor_medio_internacao', 'uf']].groupby(['data', 'uf']).mean().sort_values(by='valor_medio_internacao', ascending=False).reset_index()
aux = aux[aux['uf'].isin(['PR', 'RS', 'SC', 'MG', 'PE'])]
sns.lineplot(x='data', y='valor_medio_internacao', hue='uf', data=aux)
plt.title('Valor Médio de Internações por 5 maiores Estados ao Longo do Tempo (R$)', fontsize=12, loc='left', pad=10)
plt.legend(bbox_to_anchor=(1, 1), loc='upper left', borderaxespad=0.)
plt.xlabel('')
plt.ylabel('')

plt.tight_layout()
plt.show()

# Valor Médio dos Serviços Profissionais
fig5, ax5 = plt.subplots(nrows=2, ncols=2, figsize=(18, 8), constrained_layout=True)

plt.subplot(2, 2, (1,3))
aux = df2[['uf', 'valor_serviços_profissionais', 'internacoes']]
aux['valor_serv_prof_internacao'] = aux['valor_serviços_profissionais'] / aux['internacoes']
aux = aux.groupby('uf').mean().sort_values(by='valor_serv_prof_internacao', ascending=False).reset_index()
sns.barplot(x='valor_serv_prof_internacao', y='uf', data=aux, color='b')
plt.title('Valor Médio dos Serviços Profissionais por Estado (R$)', fontsize=12, loc='left', pad=10)
plt.xlabel('')
plt.ylabel('')

plt.subplot(2, 2, 2)
aux = df2[['regiao', 'valor_serviços_profissionais', 'internacoes']]
aux['valor_serv_prof_internacao'] = aux['valor_serviços_profissionais'] / aux['internacoes']
aux = aux.groupby('regiao').mean().sort_values(by='valor_serv_prof_internacao', ascending=False).reset_index()
sns.barplot(x='valor_serv_prof_internacao', y='regiao', data=aux, color='c')
plt.title('Valor Médio dos Serviços Profissionais por Região (R$)', fontsize=12, loc='left', pad=10)
plt.xlabel('')
plt.ylabel('')

plt.subplot(2, 2, 4)
aux = df2[['data', 'regiao', 'valor_serviços_profissionais', 'internacoes']]
aux['valor_serv_prof_internacao'] = aux['valor_serviços_profissionais'] / aux['internacoes']
aux = aux.groupby(['data', 'regiao']).mean().sort_values(by='valor_serv_prof_internacao', ascending=False).reset_index()
sns.lineplot(x='data', y='valor_serv_prof_internacao', hue='regiao', data=aux)
plt.title('Valor Médio dos Serviços Profissionais ao Longo do Tempo por Região (R$)', fontsize=12, loc='left', pad=10)
plt.legend(bbox_to_anchor=(1, 1), loc='upper left', borderaxespad=0.)
plt.xlabel('')
plt.ylabel('')

plt.tight_layout()
plt.show()


# Média de Leitos ocupados
fig6, ax6 = plt.subplots(nrows=2, ncols=2, figsize=(18, 8), constrained_layout=True)

plt.subplot(2, 2, (1,3))
aux = df2[['regiao', 'uf', 'numero_leitos_ocupados']].groupby('uf').mean().sort_values(by='numero_leitos_ocupados', ascending=False).reset_index()
sns.barplot(x='numero_leitos_ocupados', y='uf', data=aux, color='b')
plt.title('Média do Número de Leitos Ocupados por Estado', fontsize=12, loc='left', pad=10)
plt.xlabel('')
plt.ylabel('')

plt.subplot(2, 2, 2)
aux = df2[['regiao', 'uf', 'numero_leitos_ocupados']].groupby('regiao').mean().sort_values(by='numero_leitos_ocupados', ascending=False).reset_index()
sns.barplot(x='numero_leitos_ocupados', y='regiao', data=aux, color='c')
plt.title('Média do Número de Leitos Ocupados por Região', fontsize=12, loc='left', pad=10)
plt.xlabel('')
plt.ylabel('')

plt.subplot(2, 2, 4)
aux = df2[['data', 'regiao', 'numero_leitos_ocupados']].groupby(['data', 'regiao']).mean().sort_values(by='numero_leitos_ocupados', ascending=False).reset_index()
sns.lineplot(x='data', y='numero_leitos_ocupados', hue='regiao', data=aux)
plt.title('Média do Número de Leitos Ocupados por Região ao Longo do Tempo', fontsize=12, loc='left', pad=10)
plt.legend(bbox_to_anchor=(1, 1), loc='upper left', borderaxespad=0.)
plt.xlabel('')
plt.ylabel('')

plt.tight_layout()
plt.show()


# Média de Permanencia
fig7, ax7 = plt.subplots(nrows=2, ncols=2, figsize=(18, 8), constrained_layout=True)

plt.subplot(2, 2, (1,3))
aux = df2[['uf', 'media_permanencia']].groupby('uf').mean().sort_values(by='media_permanencia', ascending=False).reset_index()
sns.barplot(x='media_permanencia', y='uf', data=aux, color='b')
plt.title('Média de Permanência em Internação por Estado (Dias)', fontsize=12, loc='left', pad=10)
plt.xlabel('')
plt.ylabel('')

plt.subplot(2, 2, 2)
aux = df2[['regiao', 'media_permanencia']].groupby('regiao').mean().sort_values(by='media_permanencia', ascending=False).reset_index()
sns.barplot(x='media_permanencia', y='regiao', data=aux, color='c')
plt.title('Média de Permanência em Internação por Região (Dias)', fontsize=12, loc='left', pad=10)
plt.xlabel('')
plt.ylabel('')

plt.subplot(2, 2, 4)
aux = df2[['data', 'regiao', 'media_permanencia']].groupby(['data', 'regiao']).mean().sort_values(by='media_permanencia', ascending=False).reset_index()
sns.lineplot(x='data', y='media_permanencia', hue='regiao', data=aux)
plt.title('Média de Permanência em Internação por Região ao Longo do Tempo (Dias)', fontsize=12, loc='left', pad=10)
plt.legend(bbox_to_anchor=(1, 1), loc='upper left', borderaxespad=0.)
plt.xlabel('')
plt.ylabel('')

plt.tight_layout()
plt.show()


# Taxa de Mortalidade
fig8, ax8 = plt.subplots(nrows=2, ncols=2, figsize=(18, 8), constrained_layout=True)

plt.subplot(2, 2, (1,3))
aux = df2[['uf', 'taxa_mortalidade']].groupby('uf').mean().sort_values(by='taxa_mortalidade', ascending=False).reset_index()
sns.barplot(x='taxa_mortalidade', y='uf', data=aux, color='b')
plt.title('Taxa de Mortalidade Média em Internações por Estado (%)', fontsize=12, loc='left', pad=10)
plt.xlabel('')
plt.ylabel('')

plt.subplot(2, 2, 2)
aux = df2[['regiao', 'taxa_mortalidade']].groupby('regiao').mean().sort_values(by='taxa_mortalidade', ascending=False).reset_index()
sns.barplot(x='taxa_mortalidade', y='regiao', data=aux, color='c')
plt.title('Taxa de Mortalidade Média em Internações por Região (%)', fontsize=12, loc='left', pad=10)
plt.xlabel('')
plt.ylabel('')

plt.subplot(2, 2, 4)
aux = df2[['data', 'regiao', 'taxa_mortalidade']].groupby(['data', 'regiao']).mean().sort_values(by='taxa_mortalidade', ascending=False).reset_index()
sns.lineplot(x='data', y='taxa_mortalidade', hue='regiao', data=aux)
plt.title('Taxa de Mortalidade Média em Internações por Região ao Longo do Tempo (%)', fontsize=12, loc='left', pad=10)
plt.legend(bbox_to_anchor=(1, 1), loc='upper left', borderaxespad=0.)
plt.xlabel('')
plt.ylabel('')

plt.tight_layout()
plt.show()


# Media de Internações nos estados
fig9, ax9 = plt.subplots(figsize=(16, 8), constrained_layout=True)
aux = df2[['data', 'internacoes', 'uf']].groupby(['data', 'uf']).mean().sort_values(by='internacoes', ascending=False).reset_index()
sns.lineplot(x='data', y='internacoes', hue='uf', data=aux)
plt.title('Media de Internação por Estado ao Longo do Tempo', fontsize=12, loc='left', pad=10)
plt.legend(bbox_to_anchor=(1, 1), loc='upper left', borderaxespad=0.)
plt.yticks(np.arange(0, 250000, 50000))
plt.xlabel('')
plt.ylabel('')
plt.tight_layout()
plt.show()


# Imputação de Dados Faltantes
df3 = df1.copy()
df_mv = pd.DataFrame(columns=df3.columns)
uf = list(df3['uf'].unique())

for i in range(0, len(uf)):
    df_uf = df3.loc[df3['uf'] == uf[i]]
    df_uf = df_uf.fillna(df_uf[['internacoes', 'valor_total', 'valor_medio_aih', 'obitos']].rolling(6, min_periods=0).mean())
    df_mv = pd.concat([df_mv, df_uf])

df4 = df_mv[['data', 'internacoes', 'valor_total', 'valor_medio_aih', 'obitos']].groupby('data').sum().reset_index()

# Estimação do número de Internações
internacoes = df4[['data', 'internacoes']]
internacoes.index = internacoes['data']
internacoes = internacoes.drop(columns=['data'])

# Teste de Estacionariedade
test_adfuller = adfuller(internacoes, autolag='AIC')

stepwise = auto_arima(internacoes, start_p=1, start_q=1, max_q=8, m=6, seasonal=True, d=1, D=1, trace=True,
                      error_action='ignore', suppress_warnings=True, stepwise=True)

print(stepwise.summary())

train = internacoes.loc['2017-12-01':'2019-01-01']
test = internacoes.loc['2019-02-01':]

stepwise.fit(train)
future_forecast = stepwise.predict(n_periods=6)

future_forecast = pd.DataFrame(future_forecast, index=test.index, columns=['internacoes'])
previsao_internacoes = pd.concat([test, future_forecast], axis=1)

previsao_internacoes.columns = ['Real', 'Previsto']
previsao_internacoes['Erro Percentual'] = (previsao_internacoes['Previsto'] - previsao_internacoes['Real']) / previsao_internacoes['Real'] * 100
print(previsao_internacoes)
dfi.export(previsao_internacoes, 'previsao_internacoes.png')

# Estimação do Valor Total
valor_total = df4[['data', 'valor_total']]
valor_total.index = valor_total['data']
valor_total = valor_total.drop(columns=['data'])

# Teste de Estacionariedade
df_test = adfuller(valor_total, autolag='AIC')

stepwise = auto_arima(valor_total, start_p=1, start_q=1, max_q=8, m=6, seasonal=True, d=1, D=1, trace=True,
                      error_action='ignore', suppress_warnings=True, stepwise=True)

print(stepwise.summary())

train = valor_total.loc['2017-12-01':'2019-01-01']
test = valor_total.loc['2019-02-01':]

stepwise.fit(train)
future_forecast = stepwise.predict(n_periods=6)

future_forecast = pd.DataFrame(future_forecast, index=test.index, columns=['valor_total'])
previsao_valor_total = pd.concat([test, future_forecast], axis=1)

previsao_valor_total.columns = ['Real', 'Previsto']
previsao_valor_total['Erro Percentual'] = (previsao_valor_total['Previsto'] - previsao_valor_total['Real']) / previsao_valor_total['Real'] * 100
print(previsao_valor_total)
dfi.export(previsao_valor_total, 'previsao_valor_total.png')


# Previsões Finais
df5 = df_mv[['data', 'internacoes', 'obitos']].groupby('data').sum().reset_index()
meses = ['2019-08-01', '2019-09-01', '2019-10-01', '2019-11-01', '2019-12-01', '2020-01-01']

# Internações
prev_final_internacao = df5[['data', 'internacoes']]
prev_final_internacao.index = prev_final_internacao['data']
prev_final_internacao = prev_final_internacao.drop(columns=['data'])

# Teste de Estacionariedade
df_test = adfuller(prev_final_internacao, autolag='AIC')

stepwise = auto_arima(prev_final_internacao, start_p=1, start_q=1, max_q=8, m=6, seasonal=True, d=1, D=1, trace=True,
                      error_action='ignore', suppress_warnings=True, stepwise=True)

print(stepwise.summary())

stepwise.fit(prev_final_internacao)
forecast_internacao = stepwise.predict(n_periods=6)

forecast_internacao = pd.DataFrame(forecast_internacao, index=meses, columns=['Previsao Internação'])


# Número de Óbitos
previsao_obitos = df5[['data', 'obitos']]
previsao_obitos.index = previsao_obitos['data']
previsao_obitos = previsao_obitos.drop(columns=['data'])

# Teste de Estacionariedade
df_test = adfuller(previsao_obitos, autolag='AIC')

stepwise = auto_arima(previsao_obitos, start_p=1, start_q=1, max_q=8, m=6, seasonal=True, d=1, D=1, trace=True,
                      error_action='ignore', suppress_warnings=True, stepwise=True)

print(stepwise.summary())

stepwise.fit(previsao_obitos)
forecast_obito = stepwise.predict(n_periods=6)

forecast_obito = pd.DataFrame(forecast_obito, index=meses, columns=['Previsao Óbitos'])


# Valor Médio AIH
prev_valor_medio_aih = df_mv[['data', 'valor_medio_aih']].groupby('data').mean()

# Teste de Estacionariedade
df_test = adfuller(prev_valor_medio_aih, autolag='AIC')

stepwise = auto_arima(prev_valor_medio_aih, start_p=1, start_q=1, max_q=8, m=6, seasonal=True, d=1, D=1, trace=True,
                      error_action='ignore', suppress_warnings=True, stepwise=True)

print(stepwise.summary())

stepwise.fit(prev_valor_medio_aih)
forecast_aih = stepwise.predict(n_periods=6)

forecast_aih = pd.DataFrame(forecast_aih, index=meses, columns=['Previsao Valor Médio AIH'])

# Tabela - Internações, Número de Óbitos e Valor Médio AIH
tabela_forecast = pd.concat([forecast_internacao, forecast_obito, forecast_aih, ], axis=1)
print('Previsão para os próximos 6 meses\nInternações, Número de Óbitos e Valor Médio AIH \n\n', tabela_forecast)
dfi.export(tabela_forecast, 'tabela_forecast.png')

# Previsão Final
previsao_final = pd.concat([forecast_internacao, forecast_aih], axis=1)
previsao_final['Total - Bilhões de R$'] = previsao_final['Previsao Internação'] * previsao_final['Previsao Valor Médio AIH'] / 10**9
print('Previsão Final \n\n', previsao_final)
dfi.export(previsao_final, 'previsao_final.png')
