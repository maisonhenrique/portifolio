#Importando as Bibliotecas
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

#Base de Dados
df = pd.read_csv('MICRODADOS_ENEM_2020.csv', encoding='ISO-8859-1', sep=';')

#Renomeando alguns nomes de Municipios que não estava de acordo com a tabela de IDHM
df['NO_MUNICIPIO_PROVA'] = df['NO_MUNICIPIO_PROVA'].replace(['São João del Rei', 'Embu das Artes', "Itaporanga d'Ajuda", 'Abreu e Lima', "Santa Bárbara d'Oeste", 'Santa Izabel do Pará', "Dias d'Ávila",
                                  'Eldorado do Carajás', "Mirassol d'Oeste", "Olho d'Água das Flores", 'Pontes e Lacerda'],
                                ['São João Del Rei', 'Embu', "Itaporanga D'Ajuda", 'Abreu E Lima', "Santa Bárbara D'Oeste", 'Santa Isabel do Pará', "Dias D'Ávila",
                                 'Eldorado dos Carajás', "Mirassol D'Oeste", "Olho D'Água das Flores", 'Pontes E Lacerda'])

#Eliminando Dados desnecessarios
df = df.drop(columns=['NU_ANO', 'TP_ESTADO_CIVIL', 'TP_NACIONALIDADE', 'TP_ANO_CONCLUIU', 'TP_ENSINO', 'CO_MUNICIPIO_ESC', 'NO_MUNICIPIO_ESC', 'SG_UF_ESC', 'CO_UF_ESC', 'TP_SIT_FUNC_ESC', 'TP_DEPENDENCIA_ADM_ESC', 'TP_LOCALIZACAO_ESC',
                       'CO_UF_PROVA', 'CO_PROVA_CN', 'CO_PROVA_CH', 'CO_PROVA_LC', 'CO_PROVA_MT', 'TX_RESPOSTAS_CN', 'TX_RESPOSTAS_CH', 'TX_RESPOSTAS_LC', 'TX_RESPOSTAS_MT',
                       'TX_GABARITO_CN', 'TX_GABARITO_CH', 'TX_GABARITO_LC', 'TX_GABARITO_MT', 'TP_STATUS_REDACAO', 'NU_NOTA_COMP1', 'NU_NOTA_COMP2', 'NU_NOTA_COMP3',
                       'NU_NOTA_COMP4', 'NU_NOTA_COMP5', 'TP_LINGUA', 'Q001', 'Q002', 'Q003', 'Q004', 'Q005', 'Q006', 'Q007', 'Q008', 'Q009', 'Q010', 'Q011', 'Q012', 'Q013', 'Q014',
                      'Q015', 'Q016', 'Q017', 'Q018', 'Q019', 'Q020', 'Q021', 'Q022', 'Q023', 'Q024', 'Q025'])

df = df.loc[(df['TP_PRESENCA_CN'] != 0) & (df['TP_PRESENCA_CH'] != 0) & (df['TP_PRESENCA_LC'] != 0) & (df['TP_PRESENCA_MT'] != 0)]

def valor_faixa_etaria(i):
    if i == 1:
        return 'Menor de 17 anos'
    elif i == 2:
        return '17 anos'
    elif i == 3:
        return '18 anos'
    elif i == 4:
        return '19 anos'
    elif i == 5:
        return '20 anos'
    elif i == 6:
        return '21 anos'
    elif i == 7:
        return '22 anos'
    elif i == 8:
        return '23 anos'
    elif i == 9:
        return '24 anos'
    elif i == 10:
        return '25 anos'
    elif i == 11:
        return 'Entre 26 e 30 anos'
    elif i == 12:
        return 'Entre 31 e 35 anos'
    elif i == 13:
        return 'Entre 36 e 40 anos'
    elif i == 14:
        return 'Entre 41 e 45 anos'
    elif i == 15:
        return 'Entre 46 e 50 anos'
    elif i == 16:
        return 'Entre 51 e 55 anos'
    elif i == 17:
        return 'Entre 56 e 60 anos'
    elif i == 18:
        return 'Entre 61 e 65 anos'
    elif i == 19:
        return 'Entre 66 e 70 anos'
    elif i == 20:
        return 'Maior de 70 anos'

df['TP_FAIXA_ETARIA'] = df['TP_FAIXA_ETARIA'].apply(valor_faixa_etaria)


def cor_raca(i):
    if i == 0:
        return 'Não declarado'
    elif i == 1:
        return 'Branca'
    elif i == 2:
        return 'Preta'
    elif i == 3:
        return 'Parda'
    elif i == 4:
        return 'Amarela'
    elif i == 5:
        return 'Indígena'

df['TP_COR_RACA'] = df['TP_COR_RACA'].apply(cor_raca)


def tp_escola(i):
    if i == 1:
        return 'Não Respondeu'
    elif i == 2:
        return 'Pública'
    elif i == 3:
        return 'Privada'
    elif i == 4:
        return 'Exterior'

df['TP_ESCOLA'] = df['TP_ESCOLA'].apply(tp_escola)


def situacao_conclusao(i):
    if i == 1:
        return 'Já concluí o Ensino Médio'
    elif i == 2:
        return 'Estou cursando e concluirei o Ensino Médio em 2020'
    elif i == 3:
        return 'Estou cursando e concluirei o Ensino Médio após 2020'
    elif i == 4:
        return 'Não concluí e não estou cursando o Ensino Médio'

df['TP_ST_CONCLUSAO'] = df['TP_ST_CONCLUSAO'].apply(situacao_conclusao)


def treineiro(i):
    if i == 0:
        return 'Não'
    elif i == 1:
        return 'Sim'

df['IN_TREINEIRO'] = df['IN_TREINEIRO'].apply(treineiro)

#Exportar arquivo CSV
df.to_csv('Base_dados.csv')
