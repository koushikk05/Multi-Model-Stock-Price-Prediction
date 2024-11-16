from flask import Flask, redirect, url_for, render_template, request, session
from datetime import timedelta
import json
import plotly
import pandas as pd
import plotly.express as px
import plotly.io as pio

pio.renderers.default = 'browser'


from Final import fetchStockData, monthwise_comparison , monthwise_high_low , stock_analysis_chart , closedf_prepare , svr_data , rf_data , knn_data , lstm_data , gru_data , lstmgru_data , final

app = Flask(__name__)
app.secret_key = 'stockpredictorkey'
app.permanent_session_lifetime = timedelta(minutes=30)

user_details = {
    'sandeep@gmail.com':'sandeep', 'saiteja@gmail.com':'saiteja', 'vaibhav@gmail.com':'vaibhav'
}

@app.route('/')
def root():
    return redirect(url_for('login'))

@app.route('/login', methods=["POST", 'GET'])
def login():
    if request.method == 'POST':
        form_details = request.form.to_dict()
        try:
            if user_details.get(form_details['email'],None) == form_details['password']:
                session.permanent = True
                session['email'] = form_details['email']
                return redirect(url_for('home'))
            else:
                return render_template('login.html',error = 'Invalid Credentials')
        except Exception as err:
            if 'email' in session:
                session.pop('email',None)
            return render_template('login.html')
    else:
        if 'email' in session:
            return redirect(url_for('home'))
        return render_template('login.html')


@app.route('/logout')
def logout():
    session.pop('email',None)
    return redirect(url_for('login'))

@app.route('/home', methods = ["POST","GET"])
def home():
    if 'email' in session:
        if request.method == "POST":
            form_details = request.form.to_dict()
            try:
                symbol = form_details['symbol'].strip()
                time = form_details['period'].strip()
                algorithm = form_details['algorithm'].strip()
                print(form_details,symbol,time)
                stock_data = fetchStockData(symbol, time)
                fig = monthwise_comparison(stock_data)
                fig = monthwise_high_low(stock_data)
                fig = stock_analysis_chart(stock_data)
                X_train,X_test,y_train,y_test,time_step,closedf,close_stock,scaler,train_data,test_data = closedf_prepare(stock_data)
                #print('xtrain:',X_train,'ytrain:',y_train,'xtest:',X_test,'ytest:',y_test,'cdf:',closedf,'ts:',time_step,'cs:',close_stock,'sclr:',scaler)
                if algorithm == 'Support Vector Regression(SVR)':
                    fig = svr_data(X_train,X_test,y_train,y_test,time_step,closedf,close_stock,scaler,train_data,test_data)
                elif algorithm  == 'Random Forest Regression(RF)':
                    fig = rf_data(X_train,X_test,y_train,y_test,time_step,closedf,close_stock,scaler,train_data,test_data)
                elif algorithm == 'K-nearest neighgbour(KNN)':
                    fig = knn_data(X_train,X_test,y_train,y_test,time_step,closedf,close_stock,scaler,train_data,test_data)
                elif algorithm == 'Long short-term memory (LSTM)':
                    fig = lstm_data(X_train,X_test,y_train,y_test,time_step,closedf,close_stock,scaler,train_data,test_data)
                elif algorithm == 'Gated recurrent unit(GRU)':
                    fig = gru_data(X_train,X_test,y_train,y_test,time_step,closedf,close_stock,scaler,train_data,test_data)
                elif algorithm == 'LSTM + GRU':
                    fig = lstmgru_data(X_train,X_test,y_train,y_test,time_step,closedf,close_stock,scaler,train_data,test_data)
                else:
                    fig = final(X_train,X_test,y_train,y_test,time_step,closedf,close_stock,scaler,train_data,test_data)

                graphJSON = {index:json.dumps(figure, cls=plotly.utils.PlotlyJSONEncoder) for index,figure in enumerate(fig)}
                return render_template('home.html', isGraph = True, graphJSON=graphJSON, btntext = 'Logout')
            except KeyError as err:
                    return render_template('home.html', isGraph = False, btntext = 'Logout', error = f'Symbol {form_details["symbol"].strip()} is not found.')
            except Exception as err:
                return render_template('home.html', isGraph = False, error = str(err), btntext = 'Logout')
        else:
            return render_template('home.html', isGraph = False, btntext = 'Logout')
    else:
        return redirect(url_for('logout'))


if __name__ == "__main__":
    app.run(debug=True)
