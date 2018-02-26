import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
import math

from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler

from keras.optimizers import Adam
from keras.layers.core import Dense, Activation
from keras.models import Sequential
from keras import metrics


def date_to_colunm(stock):
    stock['Date'] = stock.index  # sensex
    stock.index = stock['Date']
    stock.index = pd.to_datetime(stock.index)
    stock.drop('Date', axis=1, inplace=True)


def get_data(stock):
    try:
        df = pd.read_csv(stock + '.csv')
        date_to_colunm(df)
        return df
    except IOError:
        print('Não existe o arquivo ')
        return


def n_moving_avg(stock_colunm, n=10):
    return stock_colunm.rolling(n).mean()


def w_moving_avg(values, n=10):
    weight = np.exp(np.linspace(-1., 0., n))
    weight /= weight.sum()
    wma = np.convolve(values, weight)[:len(values)]
    wma[:n - 1] = np.nan

    return wma

def rsi(stock, n=14):
    delta = stock['Close'].diff()

    dUp, dDown = delta.copy(), delta.copy()
    dUp[dUp < 0] = 0
    dDown[dDown > 0] = 0
    dDown = dDown.abs()

    RolUp = dUp.ewm(com=n).mean()[n - 1:]
    RolDown = dDown.ewm(com=n).mean()[n - 1:]

    RS = RolUp / RolDown

    rsi = 100.0 - (100.0 / (1.0 + RS))
    stock['RSI'] = rsi


def momentum(stock, n=10):
    stock['Momentum'] = stock['Close'].diff(n - 1)


def stc_k_d(stock, n=14):
    stock['Stc_k'] = ((stock['Close'] - \
                       stock['Low'].rolling(window=n).min()) / \
                      (stock['High'].rolling(window=n).max() - \
                       stock['Low'].rolling(window=n).min())) * 100.0
    stock['Stc_d'] = stock['Stc_k'].rolling(3).mean()

def macd_f(stock, x=12, y=26, z=9):
    macd = stock['Close'].ewm(span=x).mean()[x - 1:] - \
           stock['Close'].ewm(span=y).mean()[y - 1:]
    stock['MACD'] = macd.rolling(z).mean()


def acc_dist(stock):
    ar = np.zeros(stock.shape[0])
    i = 1
    while i < stock.shape[0]:
        ar[i] = (stock['High'][i] - stock['Close'][i - 1]) / \
                (stock['High'][i] - stock['Low'][i])
        i += 1
    stock['Acc_Dist'] = ar


def cci(stock, n=20):
    typ_price = (stock['High'] + stock['Low'] + stock['Close']) / 3
    sma_tp_20 = typ_price.rolling(n).mean()
    i = n
    MDar = []
    cci = np.zeros_like(sma_tp_20)
    while i <= len(typ_price):
        considtp = typ_price[i - n:i]
        consid_smatp = sma_tp_20[i - 1]
        Mds = 0
        j = 0
        while j < len(considtp):
            curMd = abs(considtp[j] - consid_smatp)
            Mds += curMd
            j += 1
        Mds = Mds / n
        MDar.append(Mds)
        i += 1

    ii = n
    while ii < len(sma_tp_20):
        cci[ii] = (typ_price[ii - 1] - sma_tp_20[ii - 1]) / (0.015 * MDar[ii - n])
        ii += 1
    cci[cci == 0] = np.nan
    stock['CCI'] = cci


def lw_r(stock, n=14):
    value = (stock['High'].rolling(n).max() - stock['Close'][n:]) / \
            (stock['High'].rolling(n).max() - stock['Low'].rolling(n).min()) * \
            -100.0
    stock['R%'] = value


def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean((np.abs(y_true - y_pred) / y_true)) * 100

df = get_data('^BVSP')
df = df.drop('Adj Close', axis=1)
df = df.drop('Volume', axis=1)

## preprocessing step
df = df[df['High'] != df['Low']]
df = df[df['Open'] != 'null']
df.Open = pd.to_numeric(df.Open)
df.High = pd.to_numeric(df.High)
df.Low = pd.to_numeric(df.Low)
df.Close = pd.to_numeric(df.Close)

print(df.head())
rsi(df)
df['SMA-10'] = n_moving_avg(df['Close'], n=10)
df['WMA'] = w_moving_avg(np.array(df['Close']))
momentum(df)
stc_k_d(df)
macd_f(df)
acc_dist(df)
cci(df, n=30)
lw_r(df)
df.dropna(axis=0, how='any', inplace=True)

X = df.values[:, 4:]
y = df.values[:, 3]

n_days_ahead = 1  # predict n days ahead
tuning_set = int(X.shape[0] * 0.2)
tuning_train_set = int(tuning_set * 0.8)

# Scaling the data
y = y.reshape(-1, 1)
sc = StandardScaler()
X = sc.fit_transform(X, y)
y = sc.fit_transform(y, X)

# 80% train, 20% test
X_train, X_test = X[tuning_set - n_days_ahead:X.shape[0] -tuning_set], \
                  X[X.shape[0] - tuning_set:-n_days_ahead]  #
y_train, y_test = y[tuning_set:(X.shape[0] - tuning_set) + n_days_ahead], \
                  y[(X.shape[0] - tuning_set) + n_days_ahead:]

attr = {'C': [1, 2, 5, 10, 25, 50, 100],
        'gamma': [1e-2, 1e-3],
        'epsilon': [1e-8]
        }

# ----------- FIRST STAGE -----------#
test_attr = []
for i in range(X.shape[1]):
    svr_model = SVR()
    svr_cv = GridSearchCV(svr_model, attr, cv=5)
    svr_cv.fit(X[tuning_set - n_days_ahead:X.shape[0] - tuning_set, i].reshape(-1, 1),
               X[tuning_set:(X.shape[0] - tuning_set) + n_days_ahead, i])
    test_attr.append(svr_cv.predict(X[X.shape[0] - tuning_set:-n_days_ahead, i].reshape(-1, 1)))

test_attr = np.array(test_attr)
test_attr = test_attr.transpose()

# ----------- SECOND STAGE ----------- #
my_opt = Adam()
model = Sequential()
model.add(Dense(30, activation='tanh', input_shape=(X.shape[1],)))
model.add(Dense(1))
model.compile(optimizer=my_opt, loss='mse')
model.fit(X[tuning_set:X.shape[0]-tuning_set], y[tuning_set:X.shape[0]-tuning_set], epochs=10)

y_pred = model.predict(test_attr)

print('Prediction - '+str(n_days_ahead)+' days ahead \n' +
      ' MSE = '+str(mean_squared_error(y_pred, y_test)) +
      ' MAE = '+str(mean_absolute_error(y_pred, y_test)) +
      ' MAPE = '+str(mean_absolute_percentage_error(y_pred, y_test)) + '\n')

