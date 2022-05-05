#Importando as Bibliotecas
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

#Base de Dados
df = pd.read_csv('MICRODADOS_ENEM_2020.csv', encoding='ISO-8859-1', sep=';')

#Renomeando nomes das Cidades divergentes com a tabela de IDHM
df['NO_MUNICIPIO_PROVA'] = \
df['NO_MUNICIPIO_PROVA'].replace(['São João del Rei', 'Embu das Artes', "Itaporanga d'Ajuda", 'Abreu e Lima', "Santa Bárbara d'Oeste", 'Santa Izabel do Pará', "Dias d'Ávila",
                                  'Eldorado do Carajás', "Mirassol d'Oeste", "Olho d'Água das Flores", 'Pontes e Lacerda'],
                                ['São João Del Rei', 'Embu', "Itaporanga D'Ajuda", 'Abreu E Lima', "Santa Bárbara D'Oeste", 'Santa Isabel do Pará', "Dias D'Ávila",
                                 'Eldorado dos Carajás', "Mirassol D'Oeste", "Olho D'Água das Flores", 'Pontes E Lacerda'])

#Eliminando Participantes Ausentes
df = df.loc[(df['TP_PRESENCA_CN'] != 0) & (df['TP_PRESENCA_CH'] != 0) & (df['TP_PRESENCA_LC'] != 0) & (df['TP_PRESENCA_MT'] != 0)]

df2 = df.filter(items=['NO_MUNICIPIO_PROVA', 'SG_UF_PROVA', 'NU_NOTA_CN', 'NU_NOTA_CH', 'NU_NOTA_LC', 'NU_NOTA_MT', 'NU_NOTA_REDACAO'])

#Média das provas pos Município
media = df2.groupby(['NO_MUNICIPIO_PROVA', 'SG_UF_PROVA'], as_index=False).mean()
print(media.sort_values(by=['NU_NOTA_CN', 'NU_NOTA_CH', 'NU_NOTA_LC', 'NU_NOTA_MT', 'NU_NOTA_REDACAO']))

df2 = df2.rename(columns={'NU_NOTA_CN': 'CN', 'NU_NOTA_CH': 'CH', 'NU_NOTA_LC': 'LC', 'NU_NOTA_MT': 'MT', 'NU_NOTA_REDACAO': 'REDACAO'})

#Grafico Boxplot
sns.set_theme(palette='pastel')
f, ax = plt.subplots()
ax = sns.boxplot(data=df2.loc[:, ['CN', 'CH', 'LC', 'MT', 'REDACAO']])
plt.show()

#Exportar arquivo CSV
media.to_csv('Media_Notas_Municipios.csv')
