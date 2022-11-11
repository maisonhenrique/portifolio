import pandas as pd
import numpy as np
import streamlit as st
import yfinance as yf
from yahooquery import Ticker
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

pd.set_option('display.float_format', lambda x: '%.2f' % x)
st.set_page_config(layout='wide')

# Base de Dados
df_acoes = pd.read_csv('StockMonitorPredict/base_dados.csv')

# Ajustar data
date_start = datetime.today() - timedelta(days=30)
date_end = datetime.today()

intervals = ['1d', '5d', '1wk', '1mo', '3mo']

def format_date(dt, format='%Y-%m-%d'):
    return dt.strftime(format)

# Consultar Ação
@st.cache(allow_output_mutation=True)
def consultar_acao(tickers, start, end, interval):
    for empresa in df_acoes['Empresas']:
        df = yf.download(tickers = tickers, start=start, end=end, interval=interval)
        return df

# Menu lateral
menu_lateral = st.sidebar.empty()
acao = df_acoes
stock_select = st.sidebar.selectbox("Selecione o ativo:", acao)
from_date = st.sidebar.date_input('De:', date_start)
to_date = st.sidebar.date_input('Para:', date_end)
interval_select = st.sidebar.selectbox("Selecione o intervalo:", intervals)
carregar_dados = st.sidebar.checkbox('Carregar dados')

grafico_candle = st.empty()
grafico_line = st.empty()

def predict(df):
    df = consultar_acao(stock_select, from_date, to_date, interval_select)

    # Separação dos dados Treino e Teste 
    close_price = df['Close']
    step = 15
    train_size = int(len(close_price) * .8)
    test_size = len(close_price) - train_size
    train_data, input_data = np.array(close_price[0:train_size]), np.array(close_price[train_size - step :])
    test_data = np.array(close_price[train_size:])

    # Normalizar os dados
    scaler = MinMaxScaler(feature_range=(0,1))
    train_data_norm = scaler.fit_transform(np.array(train_data).reshape(-1,1))
    test_data_norm = scaler.transform(np.array(input_data).reshape(-1,1))
  
    # Pré-processamento
    X_train, y_train = [], []
    for i in range(step, len(train_data)):
        X_train.append(train_data_norm[i-step:i])
        y_train.append(train_data_norm[i])
    
    X_test, y_test = [], []
    for i in range(step, step + len(test_data)):
        X_test.append(test_data_norm[i-step:i])
        y_test.append(test_data_norm[i])
    
    # Trasnformando em array
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)
    y_test = np.array(y_test)

    #Loading Model
    model = load_model('StockMonitorPredict/model_app.h5')

    # Fazendo a previsão
    predict = model.predict(X_test)
    predict = scaler.inverse_transform(predict)

    df_predict = df.filter(['Close'])[train_size:]
    df_predict['Predict'] = predict
    df_predict[['Close', 'Predict']]
    return df_predict

def predict_10days(df):
    # Prevendo o preço para os próximos 10 dias
    close_price = df['Close']
    step = 15
    train_size = int(len(close_price) * .8)
    test_size = len(close_price) - train_size
    train_data, input_data = np.array(close_price[0:train_size]), np.array(close_price[train_size - step :])
    test_data = np.array(close_price[train_size:])

    # Normalizar os dados
    scaler = MinMaxScaler(feature_range=(0,1))
    train_data_norm = scaler.fit_transform(np.array(train_data).reshape(-1,1))
    test_data_norm = scaler.transform(np.array(input_data).reshape(-1,1))

    model = load_model('StockMonitorPredict/model_app.h5')

    lenght_test = len(test_data_norm)

    days_input_steps = lenght_test - step

    input_steps = test_data_norm[days_input_steps:]
    input_steps = np.array(input_steps).reshape(1, -1)

    list_output_steps = list(input_steps)
    list_output_steps = list_output_steps[0].tolist()

    pred_output=[]
    i=0
    n_future=10
    while(i<n_future):
        
        if(len(list_output_steps) > step):
            input_steps = np.array(list_output_steps[1:])
            input_steps = input_steps.reshape(1, -1)
            input_steps = input_steps.reshape((1, step, 1))

            pred = model.predict(input_steps, verbose=0)
            list_output_steps.extend(pred[0].tolist())
            list_output_steps=list_output_steps[1:]
    
            pred_output.extend(pred.tolist())
            i=i+1
        else:
            input_steps = input_steps.reshape((1, step,1))
            pred = model.predict(input_steps, verbose=0)
            list_output_steps.extend(pred[0].tolist())
            i=i+1

    prev = scaler.inverse_transform(pred_output)
    prev = np.array(prev).reshape(1,-1)
    list_output_prev = list(prev)
    list_output_prev = prev[0].tolist()

    dates = df.index
    predict_dates = pd.date_range(list(dates)[-1] + pd.DateOffset(1), periods=9, freq='b').tolist()

    forecast_dates = []
    for i in predict_dates:
        forecast_dates.append(i.date())

    df_forecast = pd.DataFrame({'Date': np.array(forecast_dates), 'Predict': list_output_prev})

    df_forecast=df_forecast.set_index(pd.DatetimeIndex(df_forecast['Date'].values))
    df_forecast.drop('Date', axis=1, inplace=True)
    return df_forecast

# Configuração titulo
st.title(f'Stock Monitor - {stock_select}')

tab1, tab2, tab3 = st.tabs(['📈 Gráfico', '🗃 Dados Históricos', '📑 Demonstração de Resultados'])

# Gráfico de CandleStick
def plotCandleStick(df):
    fig1 = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_width=[0.2, 0.7])

    fig1.add_trace(go.Candlestick(x=df.index, open=df.Open, high=df.High, low=df.Low, close=df.Close, showlegend=False), row=1, col=1)
    fig1.add_trace(go.Bar(x=df.index, y=df.Volume, showlegend=False, marker_color='blue', opacity=0.6), row=2, col=1)

    fig1.update(layout_xaxis_rangeslider_visible=False)
    fig1.update_layout(autosize = False, width = 800, height=600)
    fig1.update_yaxes(showgrid=True, color='gray', griddash='dot', gridcolor='gray', zerolinecolor='gray')
    fig1.update_xaxes(showgrid=False)
    return fig1

# Gráfico de linha
def plotGraficoLinha(df):
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x = df.index, y = df.Close, mode='lines', name='Fechamento', marker_color = '#FF7F0E', showlegend=True))
    fig2.update_layout(autosize = False, width = 900, height=500)
    fig2.update_yaxes(showgrid=True, color='gray', griddash='dot', gridcolor='gray', zerolinecolor='gray')
    fig2.update_xaxes(showgrid=False)
    return fig2

# Média Móvel
def moving_avarage(df):
    MA7 = df.Close.rolling(7).mean()
    MA21 = df.Close.rolling(21).mean()
    MA50 = df.Close.rolling(50).mean()
    MA100 = df.Close.rolling(100).mean()
    MA200 = df.Close.rolling(200).mean()
    return MA7, MA21, MA50, MA100, MA200

# Dados Financeiros
def dadosfinanceiros():
    df_dados = Ticker(stock_select)
    df_dados = df_dados.income_statement()
    df_dados = df_dados.transpose()  
    df_dados.columns = df_dados.iloc[0,:]
    df_dados = df_dados.iloc[2:,1:]
    return (df_dados)

# Visualização
def main():
    if from_date > to_date:
        st.sidebar.error('Data de ínicio maior que a data final!')
    else:
        df = consultar_acao(stock_select, format_date(from_date), format_date(to_date), interval_select)
        try:
            # Gráfico de CandleStick
            with tab1:
                fig1 = plotCandleStick(df)
                st.write('\n\n')
                options = st.multiselect('Selecione a Média Móvel:', ['MA7', 'MA21', 'MA50', 'MA100', 'MA200'])
                MA7, MA21, MA50, MA100, MA200 = moving_avarage(df)
                if 'MA7' in options:
                    fig1.add_trace(go.Scatter(x = MA7.index, y = MA7.values, mode='lines', name='MA7', marker_color = '#AFFC41', showlegend=True))
                if 'MA21' in options:
                    fig1.add_trace(go.Scatter(x = MA21.index, y = MA21.values, mode='lines', name='MA21', marker_color = '#A3CEF1', showlegend=True))
                if 'MA50' in options:
                    fig1.add_trace(go.Scatter(x = MA50.index, y = MA50.values, mode='lines', name='MA50', marker_color = '#F72585', showlegend=True))
                if 'MA100' in options:
                    fig1.add_trace(go.Scatter(x = MA100.index, y = MA100.values, mode='lines', name='MA100', marker_color = '#FF9F1C', showlegend=True))
                if 'MA200' in options:
                    fig1.add_trace(go.Scatter(x = MA200.index, y = MA200.values, mode='lines', name='MA200', marker_color = '#4361EE', showlegend=True))
            st.plotly_chart(fig1, use_container_width=True)
            
            # Preço de Fechamento x Previsão
            st.subheader('Preço de Fechamento')
            fig2 = plotGraficoLinha(df)
            if st.checkbox('Previsão'):
                st.write('''Para as previsões dos preços foi utilizado um modelo de Rede Neural Recorrente, a LSTM que é o processamento de dados em camadas. 
                Conforme Aurélien Géron, uma célula LSTM pode aprender a reconhecer uma entrada importante (input gate), armazená-la no estado de longo prazo, 
                aprender a preservá-la pelo tempo necessário (forget gate) e aprender a extraí-la sempre que for preciso.
                Todas as informações sobre o modelo estão disponíveis no [GitHub](https://github.com/maisonhenrique).''')
                st.write('\n\n')
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown('Preço de Fechamento x Previsão')
                    df_predict = predict(df)
                fig2.add_trace(go.Scatter(x = df_predict.index, y = df_predict.Predict, mode='lines', name='Previsão', marker_color = '#2CA02C', showlegend=True))
                with col2:
                    df_forecast = predict_10days(df)
                    st.markdown(f'Previsão de preços da {stock_select} para os próximos 10 dias')
                    st.write(df_forecast)
            st.plotly_chart(fig2, use_container_width=True)
            # Dados Históricos
            with tab2:
                st.dataframe(df, use_container_width=True)
            
            with tab3:
                df_dados = dadosfinanceiros()
                st.dataframe(df_dados, use_container_width=True)

            # Rodapé
            st.caption('''<h4 style='text-align: center' >Este projeto foi elaborado somente para fins de estudos e não para recomendar ações. 
            Para escolha e decisão sobre seus investimentos faça com responsabilidade e verificando sempre todos os critérios em torno do ativo escolhido.</h4>''', unsafe_allow_html=True)
            
            st.write('\n\n')
            st.write('\n\n')
            st.write('\n\n')

            st.markdown('''<h6 style='text-align: center' >Maison Henrique \n\n[![Foo](https://img.icons8.com/color/40/null/linkedin.png)](https://www.linkedin.com/in/maison-henrique) [![Foo](https://img.icons8.com/material-outlined/40/000000/github.png)](https://github.com/maisonhenrique) </h6>''', unsafe_allow_html=True)
        except Exception as e:
            st.error(e)
main()
