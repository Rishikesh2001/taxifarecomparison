import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler


def funUber(distance, cabType):
    df = pd.read_csv("uber_rides.csv")
    df.dropna(axis=1, inplace=True)
    var = pd.get_dummies(df['cab_type'])

    x = df.drop(columns=["fare", "cab_type"])
    y = df['fare']
    x = pd.concat([x, var], axis=1)
    scaler = MinMaxScaler()
    scaler.fit_transform(x)
    x = pd.DataFrame(scaler.fit_transform(x))

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, random_state=42, test_size=0.15)
    linreg = LinearRegression().fit(x_train, y_train)

    if(cabType == 'UberGo'):
        data = scaler.transform([[distance, 0, 0, 0, 0, 1, 0]])
    elif(cabType == 'UberAuto'):
        data = scaler.transform([[distance, 0, 0, 0, 1, 0, 0]])
    elif(cabType == 'Premier'):
        data = scaler.transform([[distance, 0, 0, 1, 0, 0, 0]])
    elif(cabType == 'POOL'):
        data = scaler.transform([[distance, 0, 1, 0, 0, 0, 0]])
    elif(cabType == 'Moto'):
        data = scaler.transform([[distance, 1, 0, 0, 0, 0, 0]])
    elif(cabType == 'UberXL'):
        data = scaler.transform([[distance, 0, 0, 0, 0, 0, 1]])

    prediction = linreg.predict(data)
    answer = {
        'prediction': int(prediction[0]),
        'status': 'OK'
    }
    return answer

# print(funUber(5.686, 'POOL'))


def funOla(distance, cabType):
    df = pd.read_csv("ola_rides.csv")
    df.dropna(axis=1, inplace=True)
    var = pd.get_dummies(df['cab_type'])

    x = df.drop(columns=["fare", "cab_type"])
    y = df['fare']
    x = pd.concat([x, var], axis=1)
    scaler = MinMaxScaler()
    scaler.fit_transform(x)
    x = pd.DataFrame(scaler.fit_transform(x))

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, random_state=42, test_size=0.15)
    linreg = LinearRegression().fit(x_train, y_train)

    if(cabType == 'Mini'):
        data = scaler.transform([[distance, 1, 0, 0, 0, 0]])
    elif(cabType == 'Moto'):
        data = scaler.transform([[distance, 0, 1, 0, 0, 0]])
    elif(cabType == 'OlaAuto'):
        data = scaler.transform([[distance, 0, 0, 1, 0, 0]])
    elif(cabType == 'Sedan'):
        data = scaler.transform([[distance, 0, 0, 0, 0, 1]])
    elif(cabType == 'SUV'):
        data = scaler.transform([[distance, 0, 0, 0, 1, 0]])

    prediction = linreg.predict(data)
    answer = {
        'prediction': prediction[0],
        'status': 'OK'
    }
    return answer
    