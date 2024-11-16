#!/usr/bin/env python
# coding: utf-8

# In[1]:


# In[1]:


import pandas as pd
import yfinance as yf
import numpy as np
import math
import datetime as dt
from sklearn.metrics import mean_squared_error, mean_absolute_error, explained_variance_score, r2_score 
from sklearn.metrics import mean_poisson_deviance, mean_gamma_deviance, accuracy_score
from sklearn.preprocessing import MinMaxScaler

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM, GRU

from itertools import cycle

# ! pip install plotly
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.io as pio
import json



# In[2]:

def fetchStockData(symbol, time):
    stock_symbol = symbol
    period = time
    # Fetch stock data from Yahoo Finance
    stock_data = yf.download(stock_symbol, period=period, progress=False)

    # Save the stock data to a CSV file
    csv_file_path = f"/Users/koushikgovardhanam/Documents/Major/project/data/{stock_symbol}.csv"
    stock_data.to_csv(csv_file_path)

    # Read the CSV file into a pandas DataFrame
    bist100 = pd.read_csv(csv_file_path)

    # Rename columns
    bist100.rename(columns={"Date": "date", "Open": "open", "High": "high", "Low": "low", "Close": "close"}, inplace=True)

    # Display the first few rows of the DataFrame
    #print(bist100.head())


    # In[3]:


    # Rename columns
    bist100.rename(columns={"Date":"date","Open":"open","High":"high","Low":"low","Close":"close"}, inplace= True)
    bist100.head()


    # In[4]:


    # Checking null value
    bist100.isnull().sum()


    # In[5]:


    # Checking na value
    bist100.isna().any()


    # In[6]:


    bist100.dropna(inplace=True)
    bist100.isna().any()


    # In[7]:


    # Checking Data type of each column
    """
    print("Date column data type: ", type(bist100['date'][0]))
    print("Open column data type: ", type(bist100['open'][0]))
    print("Close column data type: ", type(bist100['close'][0]))
    print("High column data type: ", type(bist100['high'][0]))
    print("Low column data type: ", type(bist100['low'][0]))
    """


    # In[8]:


    # convert date field from string to Date format and make it index
    bist100['date'] = pd.to_datetime(bist100.date)
    bist100.head()


    # In[9]:


    bist100.sort_values(by='date', inplace=True)
    bist100.head()


    # In[10]:


    bist100.shape
    return bist100


# In[11]:


#EDA - Exploratory Data Analysis
#Get the duration of dataset
# def print_date_duration(bist100):
#     start_date = bist100.iloc[0][0]
#     end_date = bist100.iloc[-1][0]
#     duration = end_date - start_date
    
#     print("Starting date:", start_date)
#     print("Ending date:", end_date)
#     print("Duration:", duration)



# In[12]:





# In[13]:

def monthwise_comparison(bist100):
    monthvise= bist100.groupby(bist100['date'].dt.strftime('%B'))[['open','close']].mean().sort_values(by='close')
    monthvise.head()
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=monthvise.index,
        y=monthvise['open'],
        name='Stock Open Price',
        marker_color='crimson'
    ))
    
    fig.add_trace(go.Bar(
        x=monthvise.index,
        y=monthvise['close'],
        name='Stock Close Price',
        marker_color='lightsalmon'
    ))
    
    fig.update_layout(barmode='group', xaxis_tickangle=-45, title='Monthwise Comparison between Stock Actual, Open and Close Price')
    # print("figure",fig)
    # print("figure.show",fig.show)
    # fig.show("notebook")
    return fig

# # In[14]:


# bist100.groupby(bist100['date'].dt.strftime('%B'))['low'].min()


# # In[15]:


# monthvise_high= bist100.groupby(bist100['date'].dt.strftime('%B'))['high'].max()
# monthvise_low= bist100.groupby(bist100['date'].dt.strftime('%B'))['low'].min()


# # In[16]:

def monthwise_high_low(data):
    monthvise_high= data.groupby(data['date'].dt.strftime('%B'))['high'].max()
    monthvise_low= data.groupby(data['date'].dt.strftime('%B'))['low'].min()

    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=monthvise_high.index,
        y=monthvise_high,
        name='Stock High Price',
        marker_color='rgb(0, 153, 204)'
    ))
    
    fig.add_trace(go.Bar(
        x=monthvise_low.index,
        y=monthvise_low,
        name='Stock Low Price',
        marker_color='rgb(255, 128, 0)'
    ))
    
    fig.update_layout(barmode='group', title='Monthwise High and Low Stock Price')
    return fig


# # In[17]:


# #Trend comparision between stock price, open price, close price, high price, low price
def stock_analysis_chart(data):
    names = cycle(['Stock Open Price','Stock Close Price','Stock High Price','Stock Low Price'])

    fig = px.line(data, x=data.date, y=[data['open'], data['close'], 
                                              data['high'], data['low']],
                 labels={'date': 'Date', 'value': 'Stock value'})
    fig.update_layout(title_text='Stock analysis chart', font_size=15, font_color='black', legend_title_text='Stock Parameters')
    fig.for_each_trace(lambda t: t.update(name=next(names)))
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)

    return fig


# # In[18]:
# #Make separate dataframe with close price
# closedf = bist100[['date','close']]
# #print("Shape of close dataframe:", closedf.shape)


# # In[19]:
    
# fig = px.line(closedf, x=closedf.date, y=closedf.close,labels={'date':'Date','close':'Close Stock'})
# fig.update_traces(marker_line_width=2, opacity=0.6)
# fig.update_layout(title_text='Stock close price chart', plot_bgcolor='white', font_size=15, font_color='black')
# fig.update_xaxes(showgrid=False)
# fig.update_yaxes(showgrid=False)
# fig.show()

# #Normalizing / scaling close value between 0 to 1
# close_stock = closedf.copy()
# del closedf['date']
# scaler=MinMaxScaler(feature_range=(0,1))
# closedf=scaler.fit_transform(np.array(closedf).reshape(-1,1))
#print(closedf.shape)
# # In[21]:
# #Split data for training and testing
# #Ratio for training and testing data is 70:30
# training_size=int(len(closedf)*0.70)
# test_size=len(closedf)-training_size
# train_data,test_data=closedf[0:training_size,:],closedf[training_size:len(closedf),:1]
# """
# print("train_data: ", train_data.shape)
# print("test_data: ", test_data.shape)
# """
# # In[22]:
# # convert an array of values into a dataset matrix
# def create_dataset(dataset, time_step=1):
#     dataX, dataY = [], []
#     for i in range(len(dataset)-time_step-1):
#         a = dataset[i:(i+time_step), 0]   ###i=0, 0,1,2,3-----99   100 
#         dataX.append(a)
#         dataY.append(dataset[i + time_step, 0])
#     return np.array(dataX), np.array(dataY)

# In[23]:
# reshape into X=t,t+1,t+2,t+3 and Y=t+4
# time_step = 15
# X_train, y_train = create_dataset(train_data, time_step)
# X_test, y_test = create_dataset(test_data, time_step)
# """
# print("X_train: ", X_train.shape)
# print("y_train: ", y_train.shape)
# print("X_test: ", X_test.shape)
# print("y_test", y_test.shape)
# """

# # In[20]:

def closedf_prepare(data):
    closedf = data[['date','close']]
    fig = px.line(closedf, x=closedf.date, y=closedf.close,labels={'date':'Date','close':'Close Stock'})
    fig.update_traces(marker_line_width=2, opacity=0.6)
    fig.update_layout(title_text='Stock close price chart', plot_bgcolor='white', font_size=15, font_color='black')
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)
    #return fig
# #Normalizing / scaling close value between 0 to 1
    close_stock = closedf.copy()
    del closedf['date']
    scaler=MinMaxScaler(feature_range=(0,1))
    closedf=scaler.fit_transform(np.array(closedf).reshape(-1,1))
# # In[21]:
# #Split data for training and testing
# #Ratio for training and testing data is 70:30
    training_size=int(len(closedf)*0.70)
    test_size=len(closedf)-training_size
    train_data,test_data=closedf[0:training_size,:],closedf[training_size:len(closedf),:1]
    # return train_data,test_data,scaler,close_stock
# # In[22]:
# # convert an array of values into a dataset matrix
    def create_dataset(dataset, time_step=1):
        dataX, dataY = [], []
        for i in range(len(dataset)-time_step-1):
            a = dataset[i:(i+time_step), 0]   ###i=0, 0,1,2,3-----99   100 
            dataX.append(a)
            dataY.append(dataset[i + time_step, 0])
        return np.array(dataX), np.array(dataY)
# In[23]:
# reshape into X=t,t+1,t+2,t+3 and Y=t+4
    time_step = 15
    X_train, y_train = create_dataset(train_data, time_step)
    X_test, y_test = create_dataset(test_data, time_step)
    return X_train,X_test,y_train,y_test,time_step,closedf,close_stock,scaler,train_data,test_data
# # In[2]:


# #Algorithms
#--------------------------------------------------------
#SVR(MINE)
# #Algorithms
# #Super vector regression - SVR

from sklearn.svm import SVR
def svr_data(X_train,X_test,y_train,y_test,time_step,closedf,close_stock,scaler,train_data,test_data,final=False):
    figures = []
    #print('xtrain:',X_train,'ytrain:',y_train,'xtest:',X_test,'ytest:',y_test,'cdf:',closedf,'ts:',time_step,'cs:',close_stock,'sclr:',scaler)
   # from sklearn.svm import SVR
    svr_rbf = SVR(kernel= 'rbf', C= 1e2, gamma= 0.1)
    svr_rbf.fit(X_train, y_train)
# Lets Do the prediction 
    train_predict=svr_rbf.predict(X_train)
    test_predict=svr_rbf.predict(X_test)
    train_predict = train_predict.reshape(-1,1)
    test_predict = test_predict.reshape(-1,1)
# Transform back to original form
    train_predict = scaler.inverse_transform(train_predict)
    test_predict = scaler.inverse_transform(test_predict)
    original_ytrain = scaler.inverse_transform(y_train.reshape(-1,1)) 
    original_ytest = scaler.inverse_transform(y_test.reshape(-1,1)) 
# # shift train predictions for plotting
    look_back=time_step
    trainPredictPlot = np.empty_like(closedf)
    trainPredictPlot[:, :] = np.nan
    trainPredictPlot[look_back:len(train_predict)+look_back, :] = train_predict
# # shift test predictions for plotting
    testPredictPlot = np.empty_like(closedf)
    testPredictPlot[:, :] = np.nan
    testPredictPlot[len(train_predict)+(look_back*2)+1:len(closedf)-1, :] = test_predict
    names = cycle(['Original close price', 'Train predicted close price', 'Test predicted close price'])

    plotdf = pd.DataFrame({'date': close_stock['date'],
                        'original_close': close_stock['close'],
                        'train_predicted_close': trainPredictPlot.reshape(1, -1)[0].tolist(),
                        'test_predicted_close': testPredictPlot.reshape(1, -1)[0].tolist()})

    fig = px.line(plotdf, x=plotdf['date'], y=[plotdf['original_close'], plotdf['train_predicted_close'],
                                                plotdf['test_predicted_close']],
                    labels={'value': 'Stock price', 'date': 'Date'})
    fig.update_layout(title_text='Comparison between original close price vs predicted close price',
                        plot_bgcolor='white', font_size=15, font_color='black', legend_title_text='Close Price')
    fig.for_each_trace(lambda t: t.update(name=next(names)))

    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)
    figures.append(fig)
    #################################return fig
    x_input=test_data[len(test_data)-time_step:].reshape(1,-1)
    temp_input=list(x_input)
    temp_input=temp_input[0].tolist()

    from numpy import array

    lst_output=[]
    n_steps=time_step
    i=0
    pred_days = 20
    while(i<pred_days): 
        if(len(temp_input)>time_step):       
            x_input=np.array(temp_input[1:])
            x_input=x_input.reshape(1,-1)      
            yhat = svr_rbf.predict(x_input)
            temp_input.extend(yhat.tolist())
            temp_input=temp_input[1:]
       
            lst_output.extend(yhat.tolist())
            i=i+1
        
        else:
            yhat = svr_rbf.predict(x_input)
        
            temp_input.extend(yhat.tolist())
            lst_output.extend(yhat.tolist())
        
            i=i+1
#Plotting last 15 days and next predicted 10 days
    last_days=np.arange(1,time_step+1)
    day_pred=np.arange(time_step+1,time_step+pred_days+1)
    temp_mat = np.empty((len(last_days)+pred_days+1,1))
    temp_mat[:] = np.nan
    temp_mat = temp_mat.reshape(1,-1).tolist()[0]

    last_original_days_value = temp_mat
    next_predicted_days_value = temp_mat

    last_original_days_value[0:time_step+1] = scaler.inverse_transform(closedf[len(closedf)-time_step:]).reshape(1,-1).tolist()[0]
    next_predicted_days_value[time_step+1:] = scaler.inverse_transform(np.array(lst_output).reshape(-1,1)).reshape(1,-1).tolist()[0]

    new_pred_plot = pd.DataFrame({
    'last_original_days_value':last_original_days_value,
    'next_predicted_days_value':next_predicted_days_value
    })
    names = cycle(['Last 15 days close price', 'Predicted next 20 days close price'])

    fig = px.line(new_pred_plot, x=new_pred_plot.index, y=[new_pred_plot['last_original_days_value'],
                                                            new_pred_plot['next_predicted_days_value']],
                  labels={'value': 'Stock price', 'index': 'Timestamp'})
    fig.update_layout(title_text='Compare last 15 days vs next 10 days',
                      plot_bgcolor='white', font_size=15, font_color='black', legend_title_text='Close Price')
    fig.for_each_trace(lambda t: t.update(name=next(names)))

    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)
    figures.append(fig)
    ########return fig
    svrdf=closedf.tolist()
    svrdf.extend((np.array(lst_output).reshape(-1,1)).tolist())
    svrdf=scaler.inverse_transform(svrdf).reshape(1,-1).tolist()[0]
    names = cycle(['Close Price'])

    fig = px.line(svrdf, labels={'value': 'Stock price', 'index': 'Timestamp'})
    fig.update_layout(title_text='Plotting whole closing stock price with prediction',
                      plot_bgcolor='white', font_size=15, font_color='black', legend_title_text='Stock')
    fig.for_each_trace(lambda t: t.update(name=next(names)))

    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)
    figures.append(fig)
    if final == True:
        return svrdf
    else:
        return figures
    # svr_dict = {'figures':figures,'data':svrdf}
    # ##return fig
    # return svr_dict

#End SVR
#-----------------------------------------------------------------------

#------------------------------------
# #Random Forest Regressor - RF(mine)
from sklearn.ensemble import RandomForestRegressor
def rf_data(X_train,X_test,y_train,y_test,time_step,closedf,close_stock,scaler,train_data,test_data,final=False):
    figures = []
    regressor = RandomForestRegressor(n_estimators = 100, random_state = 0)
    regressor.fit(X_train, y_train)
# Lets Do the prediction 
    train_predict=regressor.predict(X_train)
    test_predict=regressor.predict(X_test)
    train_predict = train_predict.reshape(-1,1)
    test_predict = test_predict.reshape(-1,1)
# Transform back to original form
    train_predict = scaler.inverse_transform(train_predict)
    test_predict = scaler.inverse_transform(test_predict)
    original_ytrain = scaler.inverse_transform(y_train.reshape(-1,1)) 
    original_ytest = scaler.inverse_transform(y_test.reshape(-1,1))
# shift train predictions for plotting
    look_back=time_step
    trainPredictPlot = np.empty_like(closedf)
    trainPredictPlot[:, :] = np.nan
    trainPredictPlot[look_back:len(train_predict)+look_back, :] = train_predict
# shift test predictions for plotting
    testPredictPlot = np.empty_like(closedf)
    testPredictPlot[:, :] = np.nan
    testPredictPlot[len(train_predict)+(look_back*2)+1:len(closedf)-1, :] = test_predict
    names = cycle(['Original close price','Train predicted close price','Test predicted close price'])
    plotdf = pd.DataFrame({'date': close_stock['date'],
                           'original_close': close_stock['close'],
                           'train_predicted_close': trainPredictPlot.reshape(1,-1)[0].tolist(),
                           'test_predicted_close': testPredictPlot.reshape(1,-1)[0].tolist()})
    
    fig = px.line(plotdf, x=plotdf['date'], y=[plotdf['original_close'], plotdf['train_predicted_close'],
                                                plotdf['test_predicted_close']],
                  labels={'value':'Stock price', 'date': 'Date'})
    fig.update_layout(title_text='Comparison between original close price vs predicted close price',
                      plot_bgcolor='white', font_size=15, font_color='black', legend_title_text='Close Price')
    fig.for_each_trace(lambda t: t.update(name=next(names)))
    
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)
    #######################return fig
    figures.append(fig)
#Predicting next 10 days
    x_input=test_data[len(test_data)-time_step:].reshape(1,-1)
    temp_input=list(x_input)
    temp_input=temp_input[0].tolist()
    from numpy import array
    lst_output=[]
    n_steps=time_step
    i=0
    pred_days = 20
    while(i<pred_days):     
        if(len(temp_input)>time_step):
            x_input=np.array(temp_input[1:])
            #print("{} day input {}".format(i,x_input))
            x_input=x_input.reshape(1,-1)
            
            yhat = regressor.predict(x_input)
            #print("{} day output {}".format(i,yhat))
            temp_input.extend(yhat.tolist())
            temp_input=temp_input[1:]
        
            lst_output.extend(yhat.tolist())
            i=i+1
            
        else:
            yhat = regressor.predict(x_input)
            
            temp_input.extend(yhat.tolist())
            lst_output.extend(yhat.tolist())
            
            i=i+1
#Plotting last 15 days and next predicted 10 days
    last_days=np.arange(1,time_step+1)
    day_pred=np.arange(time_step+1,time_step+pred_days+1)
    temp_mat = np.empty((len(last_days)+pred_days+1,1))
    temp_mat[:] = np.nan
    temp_mat = temp_mat.reshape(1,-1).tolist()[0]
    last_original_days_value = temp_mat
    next_predicted_days_value = temp_mat
    last_original_days_value[0:time_step+1] = scaler.inverse_transform(closedf[len(closedf)-time_step:]).reshape(1,-1).tolist()[0]
    next_predicted_days_value[time_step+1:] = scaler.inverse_transform(np.array(lst_output).reshape(-1,1)).reshape(1,-1).tolist()[0]

    names = cycle(['Last 15 days close price','Predicted next 20 days close price'])   
    new_pred_plot = pd.DataFrame({
        'last_original_days_value': last_original_days_value,
        'next_predicted_days_value': next_predicted_days_value
    })
    
    fig = px.line(new_pred_plot, x=new_pred_plot.index, y=[new_pred_plot['last_original_days_value'],
                                                           new_pred_plot['next_predicted_days_value']],
                  labels={'value': 'Stock price', 'index': 'Timestamp'})
    fig.update_layout(title_text='Compare last 15 days vs next 20 days',
                      plot_bgcolor='white', font_size=15, font_color='black', legend_title_text='Close Price')
    fig.for_each_trace(lambda t: t.update(name=next(names)))
    
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)
    ##################################return fig
    figures.append(fig)
    rfdf=closedf.tolist()
    rfdf.extend((np.array(lst_output).reshape(-1,1)).tolist())
    rfdf=scaler.inverse_transform(rfdf).reshape(1,-1).tolist()[0]

    names = cycle(['Close price'])
    
    fig = px.line(rfdf, labels={'value': 'Stock price', 'index': 'Timestamp'})
    fig.update_layout(title_text='Plotting whole closing stock price with prediction',
                      plot_bgcolor='white', font_size=15, font_color='black', legend_title_text='Stock')
    fig.for_each_trace(lambda t: t.update(name=next(names)))
    
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)
    ##############return fig
    figures.append(fig)
    if final == True:
        return rfdf
    else:
        return figures
    # rf_dict = {'figures':figures,'data':rfdf}
    # return rf_dict
#end rf
#----------------------------------------------------------------------------------
# #K-nearest neighgbour - KNN(mine)
from sklearn import neighbors
def knn_data(X_train,X_test,y_train,y_test,time_step,closedf,close_stock,scaler,train_data,test_data,final=False):
    figures = []
    K = time_step
    neighbor = neighbors.KNeighborsRegressor(n_neighbors = K)
    neighbor.fit(X_train, y_train)
# Lets Do the prediction 
    train_predict=neighbor.predict(X_train)
    test_predict=neighbor.predict(X_test)
    train_predict = train_predict.reshape(-1,1)
    test_predict = test_predict.reshape(-1,1)
# Transform back to original form
    train_predict = scaler.inverse_transform(train_predict)
    test_predict = scaler.inverse_transform(test_predict)
    original_ytrain = scaler.inverse_transform(y_train.reshape(-1,1)) 
    original_ytest = scaler.inverse_transform(y_test.reshape(-1,1)) 
# shift train predictions for plotting
    look_back=time_step
    trainPredictPlot = np.empty_like(closedf)
    trainPredictPlot[:, :] = np.nan
    trainPredictPlot[look_back:len(train_predict)+look_back, :] = train_predict
# shift test predictions for plotting
    testPredictPlot = np.empty_like(closedf)
    testPredictPlot[:, :] = np.nan
    testPredictPlot[len(train_predict)+(look_back*2)+1:len(closedf)-1, :] = test_predict
    names = cycle(['Original close price','Train predicted close price','Test predicted close price']) 
    plotdf = pd.DataFrame({'date': close_stock['date'],
                           'original_close': close_stock['close'],
                           'train_predicted_close': trainPredictPlot.reshape(1,-1)[0].tolist(),
                           'test_predicted_close': testPredictPlot.reshape(1,-1)[0].tolist()})
    
    fig = px.line(plotdf, x=plotdf['date'], y=[plotdf['original_close'], plotdf['train_predicted_close'],
                                                plotdf['test_predicted_close']],
                  labels={'value':'Stock price', 'date': 'Date'})
    fig.update_layout(title_text='Comparison between original close price vs predicted close price',
                      plot_bgcolor='white', font_size=15, font_color='black', legend_title_text='Close Price')
    fig.for_each_trace(lambda t: t.update(name=next(names)))
    
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)
    ###################return fig
    figures.append(fig)
#Predicting next 10 days
    x_input=test_data[len(test_data)-time_step:].reshape(1,-1)
    temp_input=list(x_input)
    temp_input=temp_input[0].tolist()
    from numpy import array
    lst_output=[]
    n_steps=time_step
    i=0
    pred_days = 20
    while(i<pred_days):      
        if(len(temp_input)>time_step):
            x_input=np.array(temp_input[1:])
            x_input=x_input.reshape(1,-1)
            yhat = neighbor.predict(x_input)
            temp_input.extend(yhat.tolist())
            temp_input=temp_input[1:]
            lst_output.extend(yhat.tolist())
            i=i+1
        else:
            yhat = neighbor.predict(x_input)
            temp_input.extend(yhat.tolist())
            lst_output.extend(yhat.tolist())
            i=i+1
#Plotting last 15 days and next predicted 10 days
    last_days=np.arange(1,time_step+1)
    day_pred=np.arange(time_step+1,time_step+pred_days+1)
    temp_mat = np.empty((len(last_days)+pred_days+1,1))
    temp_mat[:] = np.nan
    temp_mat = temp_mat.reshape(1,-1).tolist()[0]
    last_original_days_value = temp_mat
    next_predicted_days_value = temp_mat
    last_original_days_value[0:time_step+1] = scaler.inverse_transform(closedf[len(closedf)-time_step:]).reshape(1,-1).tolist()[0]
    next_predicted_days_value[time_step+1:] = scaler.inverse_transform(np.array(lst_output).reshape(-1,1)).reshape(1,-1).tolist()[0]
    new_pred_plot = pd.DataFrame({
        'last_original_days_value':last_original_days_value,
        'next_predicted_days_value':next_predicted_days_value
    })
    names = cycle(['Last 15 days close price','Predicted next 20 days close price'])
    fig = px.line(new_pred_plot, x=new_pred_plot.index, y=[new_pred_plot['last_original_days_value'],
                                                           new_pred_plot['next_predicted_days_value']],
                  labels={'value': 'Stock price', 'index': 'Timestamp'})
    fig.update_layout(title_text='Compare last 15 days vs next 10 days',
                      plot_bgcolor='white', font_size=15, font_color='black', legend_title_text='Close Price')
    fig.for_each_trace(lambda t: t.update(name=next(names)))
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)
    ###########return fig
    figures.append(fig)
    knndf=closedf.tolist()
    knndf.extend((np.array(lst_output).reshape(-1,1)).tolist())
    knndf=scaler.inverse_transform(knndf).reshape(1,-1).tolist()[0]
    names = cycle(['Close price'])  
    fig = px.line(knndf, labels={'value': 'Stock price', 'index': 'Timestamp'})
    fig.update_layout(title_text='Plotting whole closing stock price with prediction',
                        plot_bgcolor='white', font_size=15, font_color='black', legend_title_text='Stock')
    fig.for_each_trace(lambda t: t.update(name=next(names)))    
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)
    #############3return fig
    figures.append(fig)
    if final == True:
        return knndf
    else:
        return figures
    # knn_dict = {'figures':figures,'data':knndf}
    # return knn_dict
#end knn
#---------------------------------------------------------------------
# #LSTM(mine)
# # reshape input to be [samples, time steps, features] which is required for LSTM
def lstm_data(X_train,X_test,y_train,y_test,time_step,closedf,close_stock,scaler,train_data,test_data,final=False):
    figures = []
    X_train =X_train.reshape(X_train.shape[0],X_train.shape[1] , 1)
    X_test = X_test.reshape(X_test.shape[0],X_test.shape[1] , 1)
#LSTM model structure
    tf.keras.backend.clear_session()
    model=Sequential()
    model.add(LSTM(32,return_sequences=True,input_shape=(time_step,1)))
    model.add(LSTM(32,return_sequences=True))
    model.add(LSTM(32))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error',optimizer='adam')
    model.summary()
    model.fit(X_train,y_train,validation_data=(X_test,y_test),epochs=200,batch_size=5,verbose=1)
### Lets Do the prediction and check performance metrics
    train_predict=model.predict(X_train)
    test_predict=model.predict(X_test)
    train_predict.shape, test_predict.shape
# Transform back to original form
    train_predict = scaler.inverse_transform(train_predict)
    test_predict = scaler.inverse_transform(test_predict)
    original_ytrain = scaler.inverse_transform(y_train.reshape(-1,1)) 
    original_ytest = scaler.inverse_transform(y_test.reshape(-1,1))
# shift train predictions for plotting
    look_back=time_step
    trainPredictPlot = np.empty_like(closedf)
    trainPredictPlot[:, :] = np.nan
    trainPredictPlot[look_back:len(train_predict)+look_back, :] = train_predict
# shift test predictions for plotting
    testPredictPlot = np.empty_like(closedf)
    testPredictPlot[:, :] = np.nan
    testPredictPlot[len(train_predict)+(look_back*2)+1:len(closedf)-1, :] = test_predict
    names = cycle(['Original close price','Train predicted close price','Test predicted close price'])  
    plotdf = pd.DataFrame({'date': close_stock['date'],
                           'original_close': close_stock['close'],
                           'train_predicted_close': trainPredictPlot.reshape(1,-1)[0].tolist(),
                           'test_predicted_close': testPredictPlot.reshape(1,-1)[0].tolist()})
    
    fig = px.line(plotdf, x=plotdf['date'], y=[plotdf['original_close'], plotdf['train_predicted_close'],
                                                plotdf['test_predicted_close']],
                  labels={'value': 'Stock price', 'date': 'Date'})
    fig.update_layout(title_text='Comparision between original close price vs predicted close price',
                      plot_bgcolor='white', font_size=15, font_color='black', legend_title_text='Close Price')
    fig.for_each_trace(lambda t: t.update(name=next(names)))
    
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)
    ############return fig
    figures.append(fig)
#Predicting next 10 days
    x_input=test_data[len(test_data)-time_step:].reshape(1,-1)
    temp_input=list(x_input)
    temp_input=temp_input[0].tolist()
    from numpy import array
    lst_output=[]
    n_steps=time_step
    i=0
    pred_days = 20
    while(i<pred_days): 
        if(len(temp_input)>time_step): 
            x_input=np.array(temp_input[1:])
            x_input = x_input.reshape(1,-1)
            x_input = x_input.reshape((1, n_steps, 1))      
            yhat = model.predict(x_input, verbose=0)
            temp_input.extend(yhat[0].tolist())
            temp_input=temp_input[1:]    
            lst_output.extend(yhat.tolist())
            i=i+1     
        else:     
            x_input = x_input.reshape((1, n_steps,1))
            yhat = model.predict(x_input, verbose=0)
            temp_input.extend(yhat[0].tolist())
            lst_output.extend(yhat.tolist())
            i=i+1
# #Plotting last 15 days and next predicted 10 days
    last_days=np.arange(1,time_step+1)
    day_pred=np.arange(time_step+1,time_step+pred_days+1)
    temp_mat = np.empty((len(last_days)+pred_days+1,1))
    temp_mat[:] = np.nan
    temp_mat = temp_mat.reshape(1,-1).tolist()[0]
    last_original_days_value = temp_mat
    next_predicted_days_value = temp_mat
    last_original_days_value[0:time_step+1] = scaler.inverse_transform(closedf[len(closedf)-time_step:]).reshape(1,-1).tolist()[0]
    next_predicted_days_value[time_step+1:] = scaler.inverse_transform(np.array(lst_output).reshape(-1,1)).reshape(1,-1).tolist()[0]
    new_pred_plot = pd.DataFrame({
        'last_original_days_value':last_original_days_value,
        'next_predicted_days_value':next_predicted_days_value
    })
    names = cycle(['Last 15 days close price', 'Predicted next 20 days close price'])
    fig = px.line(new_pred_plot, x=new_pred_plot.index, y=[new_pred_plot['last_original_days_value'],
                                                            new_pred_plot['next_predicted_days_value']],
                  labels={'value': 'Stock price', 'index': 'Timestamp'})
    fig.update_layout(title_text='Compare last 15 days vs next 10 days',
                      plot_bgcolor='white', font_size=15, font_color='black', legend_title_text='Close Price')

    fig.for_each_trace(lambda t: t.update(name=next(names)))
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)
    ##################return fig
    figures.append(fig)
    lstmdf=closedf.tolist()
    lstmdf.extend((np.array(lst_output).reshape(-1,1)).tolist())
    lstmdf=scaler.inverse_transform(lstmdf).reshape(1,-1).tolist()[0]
    names = cycle(['Close price'])      
    fig = px.line(lstmdf, labels={'value': 'Stock price', 'index': 'Timestamp'})
    fig.update_layout(title_text='Plotting whole closing stock price with prediction',
                        plot_bgcolor='white', font_size=15, font_color='black', legend_title_text='Stock')    
    fig.for_each_trace(lambda t: t.update(name=next(names)))
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)
    ################# return fig
    figures.append(fig)
    if final == True:
        return lstmdf
    else:
        return figures
    # lstm_dict = {'figures':figures,'data':lstmdf}
    # return lstm_dict
#end lstm
#-----------------------------------------------------------
# #GRU(mine)
# # reshape input to be [samples, time steps, features] which is required for LSTM
def gru_data(X_train,X_test,y_train,y_test,time_step,closedf,close_stock,scaler,train_data,test_data,final=False):
    figures = []
    X_train =X_train.reshape(X_train.shape[0],X_train.shape[1] , 1)
    X_test = X_test.reshape(X_test.shape[0],X_test.shape[1] , 1)
#GRU model structure
    tf.keras.backend.clear_session()
    model=Sequential()
    model.add(GRU(32,return_sequences=True,input_shape=(time_step,1)))
    model.add(GRU(32,return_sequences=True))
    model.add(GRU(32,return_sequences=True))
    model.add(GRU(32))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error',optimizer='adam')
    model.summary()
    model.fit(X_train,y_train,validation_data=(X_test,y_test),epochs=200,batch_size=5,verbose=1)
### Lets Do the prediction and check performance metrics
    train_predict=model.predict(X_train)
    test_predict=model.predict(X_test)
    train_predict.shape, test_predict.shape
# Transform back to original form
    train_predict = scaler.inverse_transform(train_predict)
    test_predict = scaler.inverse_transform(test_predict)
    original_ytrain = scaler.inverse_transform(y_train.reshape(-1,1)) 
    original_ytest = scaler.inverse_transform(y_test.reshape(-1,1))
# shift train predictions for plotting
    look_back=time_step
    trainPredictPlot = np.empty_like(closedf)
    trainPredictPlot[:, :] = np.nan
    trainPredictPlot[look_back:len(train_predict)+look_back, :] = train_predict
# shift test predictions for plotting
    testPredictPlot = np.empty_like(closedf)
    testPredictPlot[:, :] = np.nan
    testPredictPlot[len(train_predict)+(look_back*2)+1:len(closedf)-1, :] = test_predict
    names = cycle(['Original close price', 'Train predicted close price', 'Test predicted close price'])  
    plotdf = pd.DataFrame({
        'date': close_stock['date'],
        'original_close': close_stock['close'],
        'train_predicted_close': trainPredictPlot.reshape(1,-1)[0].tolist(),
        'test_predicted_close': testPredictPlot.reshape(1,-1)[0].tolist()
    })
    fig = px.line(plotdf, x=plotdf['date'], y=[plotdf['original_close'], plotdf['train_predicted_close'],
                                                plotdf['test_predicted_close']],
                  labels={'value': 'Stock price', 'date': 'Date'})
    fig.update_layout(title_text='Comparison between original close price vs predicted close price',
                      plot_bgcolor='white', font_size=15, font_color='black', legend_title_text='Close Price')
    fig.for_each_trace(lambda t: t.update(name=next(names)))

    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)
    #####################3return fig
    figures.append(fig)
#Predicting next 10 days
    x_input=test_data[len(test_data)-time_step:].reshape(1,-1)
    temp_input=list(x_input)
    temp_input=temp_input[0].tolist()
    from numpy import array
    lst_output=[]
    n_steps=time_step
    i=0
    pred_days = 20
    while(i<pred_days):     
        if(len(temp_input)>time_step):         
            x_input=np.array(temp_input[1:])
            x_input = x_input.reshape(1,-1)
            x_input = x_input.reshape((1, n_steps, 1))          
            yhat = model.predict(x_input, verbose=0)
            temp_input.extend(yhat[0].tolist())
            temp_input=temp_input[1:]
            lst_output.extend(yhat.tolist())
            i=i+1          
        else:          
            x_input = x_input.reshape((1, n_steps,1))
            yhat = model.predict(x_input, verbose=0)
            temp_input.extend(yhat[0].tolist())           
            lst_output.extend(yhat.tolist())
            i=i+1
#Plotting last 15 days and next predicted 10 days
    last_days=np.arange(1,time_step+1)
    day_pred=np.arange(time_step+1,time_step+pred_days+1)
    temp_mat = np.empty((len(last_days)+pred_days+1,1))
    temp_mat[:] = np.nan
    temp_mat = temp_mat.reshape(1,-1).tolist()[0]
    last_original_days_value = temp_mat
    next_predicted_days_value = temp_mat
    last_original_days_value[0:time_step+1] = scaler.inverse_transform(closedf[len(closedf)-time_step:]).reshape(1,-1).tolist()[0]
    next_predicted_days_value[time_step+1:] = scaler.inverse_transform(np.array(lst_output).reshape(-1,1)).reshape(1,-1).tolist()[0]
    new_pred_plot = pd.DataFrame({
        'last_original_days_value':last_original_days_value,
        'next_predicted_days_value':next_predicted_days_value
    })
    names = cycle(['Last 15 days close price', 'Predicted next 20 days close price'])

    fig = px.line(new_pred_plot, x=new_pred_plot.index, y=[new_pred_plot['last_original_days_value'],
                                                            new_pred_plot['next_predicted_days_value']],
                  labels={'value': 'Stock price', 'index': 'Timestamp'})
    fig.update_layout(title_text='Compare last 15 days vs next 10 days',
                      plot_bgcolor='white', font_size=15, font_color='black', legend_title_text='Close Price')
    fig.for_each_trace(lambda t: t.update(name=next(names)))

    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)
    ##############return fig
    figures.append(fig)
    grudf=closedf.tolist()
    grudf.extend((np.array(lst_output).reshape(-1,1)).tolist())
    grudf=scaler.inverse_transform(grudf).reshape(1,-1).tolist()[0]
    names = cycle(['Close price'])
    fig = px.line(grudf, labels={'value': 'Stock price', 'index': 'Timestamp'})
    fig.update_layout(title_text='Plotting whole closing stock price with prediction',
                      plot_bgcolor='white', font_size=15, font_color='black', legend_title_text='Stock')
    fig.for_each_trace(lambda t: t.update(name=next(names)))
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)
    ############return fig   
    figures.append(fig)
    if final == True:
        return grudf
    else:
        return figures
    # gru_dict = {'figures':figures,'data':grudf}
    # return gru_dict
#end gru
#--------------------------------------------------------------
 #LSTM + GRU

def lstmgru_data(X_train,X_test,y_train,y_test,time_step,closedf,close_stock,scaler,train_data,test_data,final=False):
    figures = []
# reshape input to be [samples, time steps, features] which is required for LSTM
    X_train =X_train.reshape(X_train.shape[0],X_train.shape[1] , 1)
    X_test = X_test.reshape(X_test.shape[0],X_test.shape[1] , 1)
#Model structure
    tf.keras.backend.clear_session()
    model=Sequential()
    model.add(LSTM(32,return_sequences=True,input_shape=(time_step,1)))
    model.add(LSTM(32,return_sequences=True))
    model.add(GRU(32,return_sequences=True))
    model.add(GRU(32))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error',optimizer='adam')
    model.summary()
    model.fit(X_train,y_train,validation_data=(X_test,y_test),epochs=200,batch_size=5,verbose=1)
# Lets Do the prediction and check performance metrics
    train_predict=model.predict(X_train)
    test_predict=model.predict(X_test)
    train_predict.shape, test_predict.shape
# Transform back to original form
    train_predict = scaler.inverse_transform(train_predict)
    test_predict = scaler.inverse_transform(test_predict)
    original_ytrain = scaler.inverse_transform(y_train.reshape(-1,1)) 
    original_ytest = scaler.inverse_transform(y_test.reshape(-1,1))
# shift train predictions for plotting
    look_back=time_step
    trainPredictPlot = np.empty_like(closedf)
    trainPredictPlot[:, :] = np.nan
    trainPredictPlot[look_back:len(train_predict)+look_back, :] = train_predict
# shift test predictions for plotting
    testPredictPlot = np.empty_like(closedf)
    testPredictPlot[:, :] = np.nan
    testPredictPlot[len(train_predict)+(look_back*2)+1:len(closedf)-1, :] = test_predict
    names = cycle(['Original close price', 'Train predicted close price', 'Test predicted close price'])
    plotdf = pd.DataFrame({'date': close_stock['date'],
                           'original_close': close_stock['close'],
                           'train_predicted_close': trainPredictPlot.reshape(1, -1)[0].tolist(),
                           'test_predicted_close': testPredictPlot.reshape(1, -1)[0].tolist()})
    fig = px.line(plotdf, x=plotdf['date'], y=[plotdf['original_close'], plotdf['train_predicted_close'],
                                                plotdf['test_predicted_close']],
                  labels={'value': 'Stock price', 'date': 'Date'})
    fig.update_layout(title_text='Comparison between original close price vs predicted close price',
                      plot_bgcolor='white', font_size=15, font_color='black', legend_title_text='Close Price')
    fig.for_each_trace(lambda t: t.update(name=next(names)))
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)
    #####################return fig
    figures.append(fig)
#Predicting next 10 days
    x_input=test_data[len(test_data)-time_step:].reshape(1,-1)
    temp_input=list(x_input)
    temp_input=temp_input[0].tolist()
    from numpy import array
    lst_output=[]
    n_steps=time_step
    i=0
    pred_days = 20
    while(i<pred_days):     
        if(len(temp_input)>time_step):           
            x_input=np.array(temp_input[1:])
            x_input = x_input.reshape(1,-1)
            x_input = x_input.reshape((1, n_steps, 1))         
            yhat = model.predict(x_input, verbose=0)
            temp_input.extend(yhat[0].tolist())
            temp_input=temp_input[1:]
            lst_output.extend(yhat.tolist())
            i=i+1          
        else:          
            x_input = x_input.reshape((1, n_steps,1))
            yhat = model.predict(x_input, verbose=0)
            temp_input.extend(yhat[0].tolist())       
            lst_output.extend(yhat.tolist())
            i=i+1
#Plotting last 15 days and next predicted 10 days
    last_days=np.arange(1,time_step+1)
    day_pred=np.arange(time_step+1,time_step+pred_days+1)
    temp_mat = np.empty((len(last_days)+pred_days+1,1))
    temp_mat[:] = np.nan
    temp_mat = temp_mat.reshape(1,-1).tolist()[0]
    last_original_days_value = temp_mat
    next_predicted_days_value = temp_mat
    last_original_days_value[0:time_step+1] = scaler.inverse_transform(closedf[len(closedf)-time_step:]).reshape(1,-1).tolist()[0]
    next_predicted_days_value[time_step+1:] = scaler.inverse_transform(np.array(lst_output).reshape(-1,1)).reshape(1,-1).tolist()[0]
    new_pred_plot = pd.DataFrame({
        'last_original_days_value':last_original_days_value,
        'next_predicted_days_value':next_predicted_days_value
    })
    names = cycle(['Last 15 days close price', 'Predicted next 20 days close price'])

    fig = px.line(new_pred_plot, x=new_pred_plot.index, y=[new_pred_plot['last_original_days_value'],
                                                            new_pred_plot['next_predicted_days_value']],
                  labels={'value': 'Stock price', 'index': 'Timestamp'})
    fig.update_layout(title_text='Compare last 15 days vs next 10 days',
                      plot_bgcolor='white', font_size=15, font_color='black', legend_title_text='Close Price')
    fig.for_each_trace(lambda t: t.update(name=next(names)))

    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)
    #######return fig
    figures.append(fig)
    lstmgrudf=closedf.tolist()
    lstmgrudf.extend((np.array(lst_output).reshape(-1,1)).tolist())
    lstmgrudf=scaler.inverse_transform(lstmgrudf).reshape(1,-1).tolist()[0]
    names = cycle(['Close price'])
    fig = px.line(lstmgrudf, labels={'value': 'Stock price', 'index': 'Timestamp'})
    fig.update_layout(title_text='Plotting whole closing stock price with prediction',
                      plot_bgcolor='white', font_size=15, font_color='black', legend_title_text='Stock')
    fig.for_each_trace(lambda t: t.update(name=next(names)))
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)
    figures.append(fig)
    if final == True:
        return lstmgrudf
    else:
        return figures
    # lstmgru_dict = {'figures':figures,'data':lstmgrudf}
    # return lstmgru_dict
 #end lstm+gru
#-------------------------------
'''
def final(X_train,X_test,y_train,y_test,time_step,closedf,close_stock,scaler,train_data,test_data):
    figures = []
    svrdf = svr_data(X_train,X_test,y_train,y_test,time_step,closedf,close_stock,scaler,train_data,test_data,final=True)
    rfdf = rf_data(X_train,X_test,y_train,y_test,time_step,closedf,close_stock,scaler,train_data,test_data,final=True)
    knndf = knn_data(X_train,X_test,y_train,y_test,time_step,closedf,close_stock,scaler,train_data,test_data,final=True)
    lstmdf = lstm_data(X_train,X_test,y_train,y_test,time_step,closedf,close_stock,scaler,train_data,test_data,final=True)
    grudf = gru_data(X_train,X_test,y_train,y_test,time_step,closedf,close_stock,scaler,train_data,test_data,final=True)
    lstmgrudf = lstmgru_data(X_train,X_test,y_train,y_test,time_step,closedf,close_stock,scaler,train_data,test_data,final=True)
    finaldf = pd.DataFrame({
        'svr':svrdf,
        'rf':rfdf,
        'knn':knndf,
        'lstm':lstmdf,
        'gru':grudf,
        'lstm_gru':lstmgrudf,
    })
    #finaldf.head()
    finaldf.to_csv('/Users/koushikgovardhanam/Documents/Major/project/data/final_prediction.csv', index=False)
    names = cycle(['SVR', 'RF', 'KNN', 'LSTM', 'GRU', 'LSTM + GRU'])
    fig = px.line(finaldf[:], x=finaldf.index[:], y=[finaldf['svr'][:], finaldf['rf'][:], finaldf['knn'][:], 
                                                          finaldf['lstm'][:], finaldf['gru'][:], finaldf['lstm_gru'][:]],
                 labels={'x': 'Timestamp', 'value': 'Stock close price'})
    fig.update_layout(title_text='Final stock analysis chart', font_size=15, font_color='black', legend_title_text='Algorithms')
    fig.for_each_trace(lambda t: t.update(name=next(names)))
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)
    figures.append(fig)
    return figures
    
'''
    

def final(X_train, X_test, y_train, y_test, time_step, closedf, close_stock, scaler, train_data, test_data):
    figures = []
    
    # Get prediction data
    svrdf = svr_data(X_train, X_test, y_train, y_test, time_step, closedf, close_stock, scaler, train_data, test_data, final=True)
    rfdf = rf_data(X_train, X_test, y_train, y_test, time_step, closedf, close_stock, scaler, train_data, test_data, final=True)
    knndf = knn_data(X_train, X_test, y_train, y_test, time_step, closedf, close_stock, scaler, train_data, test_data, final=True)
    lstmdf = lstm_data(X_train, X_test, y_train, y_test, time_step, closedf, close_stock, scaler, train_data, test_data, final=True)
    grudf = gru_data(X_train, X_test, y_train, y_test, time_step, closedf, close_stock, scaler, train_data, test_data, final=True)
    lstmgrudf = lstmgru_data(X_train, X_test, y_train, y_test, time_step, closedf, close_stock, scaler, train_data, test_data, final=True)
    
    # Create a DataFrame with predictions
    finaldf = pd.DataFrame({
        'day': range(1, len(svrdf) + 1),  # Adding serial number starting from 1
        'svr': svrdf,
        'rf': rfdf,
        'knn': knndf,
        'lstm': lstmdf,
        'gru': grudf,
        'lstm_gru': lstmgrudf,
    })
    
    # Save to CSV
    finaldf.to_csv('/Users/koushikgovardhanam/Documents/Major/project/data/final_prediction.csv', index=False)
    
    # Prepare plot
    names = cycle(['SVR', 'RF', 'KNN', 'LSTM', 'GRU', 'LSTM + GRU'])
    fig = px.line(finaldf, x='day', y=['svr', 'rf', 'knn', 'lstm', 'gru', 'lstm_gru'],
                  labels={'x': 'Day', 'value': 'Stock close price'})
    fig.update_layout(title_text='Final stock analysis chart', font_size=15, font_color='black', legend_title_text='Algorithms')
    fig.for_each_trace(lambda t: t.update(name=next(names)))
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)
    
    figures.append(fig)
    return figures




#finaldf = pd.DataFrame({
#    
#     'svr':svrdf,
#     'rf':rfdf,
#     'knn':knndf,
#     'lstm':lstmdf,
#       'gru':grudf,
#    'lstm_gru':lstmgrudf,
# })



# In[67]:


#def final(finaldf):
#     names = cycle(['SVR', 'RF', 'KNN', 'LSTM', 'GRU', 'LSTM + GRU'])
#
#     fig = px.line(finaldf[440:], x=finaldf.index[440:], y=[finaldf['svr'][440:], finaldf['rf'][440:], finaldf['knn'][440:]], 
#                  labels={'x': 'Timestamp', 'value': 'Stock close price'})
#     fig.update_layout(title_text='Final stock analysis chart', font_size=15, font_color='black', legend_title_text='Algorithms')
#     fig.for_each_trace(lambda t: t.update(name=next(names)))
#     fig.update_xaxes(showgrid=False)
#     fig.update_yaxes(showgrid=False)

#     fig.show()
#finaldf['lstm'][440:], finaldf['gru'][440:], finaldf['lstm_gru'][440:]]

# # In[70]:


# #Calling functions

# #Raw data
# print_date_duration(bist100)
# monthwise_comparison(monthvise)
# monthwise_high_low(monthvise_high, monthvise_low)
# stock_analysis_chart(bist100)


# # In[71]:


# #Calling functions

# #svr
# svr_data(train_predict, test_predict, original_ytrain, original_ytest)
# svr_1(close_stock, trainPredictPlot, testPredictPlot)
# svr_2(new_pred_plot)
# svr_3(svrdf)

# #rf
# rf_data(train_predict, test_predict, original_ytrain, original_ytest)
# rf_1(close_stock, trainPredictPlot, testPredictPlot)
# rf_2(last_original_days_value, next_predicted_days_value)
# rf_3(rfdf)

# #knn
# knn_data(train_predict, test_predict, original_ytrain, original_ytest)
# knn_1(close_stock, trainPredictPlot, testPredictPlot)
# knn_2(last_original_days_value, next_predicted_days_value)
# knn_3(knndf)

# #lstm
# lstm_data(X_train, X_test, original_ytrain, original_ytest, train_predict, test_predict)
# lstm_1(close_stock, trainPredictPlot, testPredictPlot)
# lstm_2(new_pred_plot)
# lstm_3(lstmdf)

# #gru
# gru_data(X_train, X_test, original_ytrain, original_ytest, train_predict, test_predict)
# gru_1(close_stock, trainPredictPlot, testPredictPlot)
# gru_2(new_pred_plot)
# gru_3(grudf)

# #lstm+gru
# lg_data(X_train, X_test, original_ytrain, train_predict, original_ytest, test_predict)
# lg_1(close_stock, trainPredictPlot, testPredictPlot)
# lg_2(new_pred_plot)
# lg_3(lstmgrudf)

# #final graph
# final(finaldf)


# # In[ ]:




