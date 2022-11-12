import numpy as np
from numpy import array, ndarray
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler,  Normalizer
import pandas as pd

from statsmodels.tsa.holtwinters import ExponentialSmoothing


def StSc(data):
    scaler = StandardScaler()
    scaled = scaler.fit_transform(data)
    print(scaled)

def MinMaxSc(data):
    MinMax = MinMaxScaler()
    scaled = MinMax.fit_transform(data)
    print(scaled)

def RobSc(data):
    robSc = RobustScaler()
    scaled = robSc.fit_transform(data)
    print(scaled)

def Normzer(data):
    norm = Normalizer()
    scaled = norm.fit_transform(data)
    print(scaled)
    
    
def absScaler(data)


from sklearn import linear_model

    # x = df[[dataFr['wheat'], dataFr['barley'], dataFr['corn'], dataFr['rice'], dataFr['legumes']]]
    # x = dataFr[['barley'],['wheat'],['corn'],['rice'],['legumes'],]
    x = dataFr[['wheat', 'barley', 'corn', 'rice', 'legumes']]
    # x = np.array([dataFr['wheat'], dataFr['barley'], dataFr['corn'], dataFr['rice'], dataFr['legumes']])
    y = dataFr['Y']

    regr = linear_model.LinearRegression()
    regr.fit(x, y)

    print('Intercept: \n', regr.intercept_)
    print('Coefficients: \n', regr.coef_)

    # with statsmodels
    x = sm.add_constant(x)  # adding a constant

    model = sm.OLS(y, x).fit()
    predictions = model.predict(x)

    print_model = model.summary()
    print(print_model)

if __name__ == "__main__":

    dataFr = pd.read_csv('C:/Users/user/Desktop/ACM/OOZ2.csv', encoding='ISO-8859-1')
    #data = array([dataFr['bug'], dataFr['ar'], dataFr['mak'], dataFr['sholi'], dataFr['duk'], dataFr['Y']])
    data = array([dataFr['bug'], dataFr['ar'], dataFr['mak'], dataFr['sholi'], dataFr['duk']])
    #data = array([dataFr['bug'], dataFr['ar'], dataFr['mak'], dataFr['sholi'], dataFr['duk']])
    #StSc(data)
    RobSc(data)

