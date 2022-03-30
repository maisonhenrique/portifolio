#Importanto as Bibliotecas para o projeto
import pandas as pd
import plotly.express as px
import warnings

warnings.filterwarnings('ignore')

#Importando a Base Dados - Site:https://www.kaggle.com/datasets/danielfesalbon/covid-19-global-reports-early-march-2022
Base = pd.read_csv('covid_19_clean_complete_2022.csv')
Base_Dados = Base.copy()

#Tratamento de Dados
Base_Dados.info()

Base_Dados.isna().sum()
Base_Dados = Base_Dados.iloc[:, [1, 4, 5, 6, 7, 8, 9]]
Base_Dados.head()

Base_Dados['WHO Region'].isna()
cond = Base_Dados['WHO Region'].dropna().index
Base_Dados = Base_Dados.iloc[cond, :]

Base_Dados['Country/Region'].unique()
Base_Dados['WHO Region'].unique()

Base_Dados.rename(columns={'Country/Region': 'Country', 'WHO Region': 'Region'}, inplace=True)
Base_Dados.isna().sum()

#Convertendo a Coluna Date para Datetime
Base_Dados['Date'] = pd.to_datetime(Base_Dados['Date'], infer_datetime_format=True)
Base_Dados.info()

#Gráfico - Caso confirmado por País
Base[~Base['Province/State'].isna()]['Country/Region'].unique()
country = Base_Dados.groupby(['Country', 'Date']).sum().reset_index()

Grafico1 = px.line(country, x='Date', y='Confirmed', color='Country')
Grafico1.update_layout(title_text='Confirmed Cases by Countries', title_x=0.5)
Grafico1.show()

#Gráfico - Número de mortes por País
Grafico2 = px.line(country, x='Date', y='Deaths', color='Country')
Grafico2.update_layout(title_text='Deaths by Countries', title_x=0.5)
Grafico2.show()

#Gráfico - Casos confirmado por Continente
region = Base_Dados.groupby(by=['Region']).sum().reset_index()
Grafico3 = px.bar(region, x='Region', y='Confirmed', color='Region', text_auto='.2s', hover_name='Region')
Grafico3.update_traces(textfont_size=12, textangle=0, textposition="outside")
Grafico3.update_layout(title_text='Confirmed Cases by Regions', title_x=0.5)
Grafico3.show()

#Gráfico - Número de mortes por Continente
Grafico4 = px.bar(region, x='Region', y='Deaths', color='Region', text_auto='.2s', hover_name='Region')
Grafico4.update_traces(textfont_size=12, textangle=0, textposition="outside")
Grafico4.update_layout(title_text='Deaths by Regions', title_x=0.5)
Grafico4.show()

#Gráfico - Casos Atuais
regionpie= Base_Dados.groupby(['Region', 'Date']).sum().reset_index()
active = regionpie[regionpie['Date'] == regionpie['Date'].max()]

Grafico5 = px.pie(active, values='Active', names='Region')
Grafico5.update_traces(textposition='inside', textinfo='percent+label')
Grafico5.update_layout(title_text='Current Active Cases by Region (March, 2022)', title_x=0.5)
Grafico5.show()
