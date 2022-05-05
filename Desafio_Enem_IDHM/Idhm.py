#Importando as Bibliotecas
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

#Base de Dados
df = pd.read_excel('dados_idhm.xlsx')

#Eliminando Dados desnecessarios
df = df.drop(columns=['Posição IDHM Longevidade', 'IDHM Longevidade', 'Posição IDHM Renda', 'IDHM Renda', 'Posição IDHM Educação', 'Posição IDHM', 'IDHM Educação'])

df['UF'] = df['Municipio'].str.extract(r'([A-Z]{2})')

df['Municipio'] = df['Municipio'].apply(lambda x: x[:-5])
df['Municipio'] = df['Municipio'].replace(['Itapagé', 'Pedro Ii', 'Pio Xii', 'Pio Ix', 'Poxoréo', 'Seridó'], ['Itapajé', 'Pedro II', 'Pio XII', 'Pio IX', 'Poxoréu', 'São Vicente do Seridó'])

#Definação do Indice
def valor_faixa(i):
    if 0.000 <= i <= 0.499:
        return 'Muito Baixo'
    elif 0.500 <= i <= 0.599:
        return 'Baixo'
    elif 0.600 <= i <= 0.699:
        return 'Médio'
    elif 0.700 <= i <= 0.799:
        return 'Alto'
    elif 0.800 <= i <= 1.000:
        return 'Muito Alto'

df['Faixa IDHM'] = df['IDHM'].apply(valor_faixa)
print(df)

Filtro = df['IDHM'].describe()
print(Filtro)

#Gráfico Boxplot
sns.set_theme(palette='pastel')
f, ax = plt.subplots(figsize=(7, 6))
ax = sns.boxplot(y=df['IDHM'])
plt.rcParams['ytick.labelsize'] = 8
plt.show()

#Exportar arquivo CSV
df.to_csv('IDHM_Municipios.csv')
