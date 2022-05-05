#Importando as Bibliotecas
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

#Base de Dados
df = pd.read_csv('MICRODADOS_ENEM_2020.csv', encoding='ISO-8859-1', sep=';')

columns = (['NU_INSCRICAO', 'TP_PRESENCA_CN', 'TP_PRESENCA_CH', 'TP_PRESENCA_LC', 'TP_PRESENCA_MT'])
df2 = pd.DataFrame(df, columns=columns)

def presenca(i):
    if i == 0:
        return 'Ausente'
    elif 1 <= i <= 2:
        return 'Presente'

df2['TP_PRESENCA_CN'] = df2['TP_PRESENCA_CN'].apply(presenca)
df2['TP_PRESENCA_CH'] = df2['TP_PRESENCA_CH'].apply(presenca)
df2['TP_PRESENCA_LC'] = df2['TP_PRESENCA_LC'].apply(presenca)
df2['TP_PRESENCA_MT'] = df2['TP_PRESENCA_MT'].apply(presenca)
print(df2)

#Exportar arquivo CSV
df2.to_csv('Dados_Presenca.csv')
