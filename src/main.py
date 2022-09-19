# Requirement: install anaconda on your os
# start the application using streamlit run (path to main.py)
# recommended: enable auto reload in streamlit settings
# data source: yahoo finance
# To gain better understanding, please read the comments and visualize the data, graphs using the jupyter notebook
# ----------------------------------------------------------

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pandas_datareader as data
from keras.saving.save import load_model
from pandas_datareader._utils import RemoteDataError
from sklearn.preprocessing import MinMaxScaler
import datetime as dt

start_date = '2012-01-01'
end_date = dt.datetime.now()
st.markdown("<h1 style='text-align: center; color: #00008B;'>Stock Prediction App</h1>", unsafe_allow_html=True)
st.markdown("<div style='text-align: center; color: blue;'>"
            "This app predicts the <b>closing price</b> of a stock using the <b>LSTM</b> model!"
            "</div> <br>", unsafe_allow_html=True)
userInput = st.text_input("Enter the stock ticker symbol of the company you would like to predict:")


def predictStockPrice(dataframe, inputData, model, scalingFactor):
    arr = dataframe.index.array
    latestDate = arr[len(arr) - 1].to_pydatetime()
    predictionDate = (latestDate + dt.timedelta(days=1)).strftime("%d.%m.%Y")
    inputData = [inputData[len(inputData) - 50 + 0: len(inputData), 0]]  # prediction using 50 days data
    inputData = np.array(inputData)
    inputData = np.reshape(inputData, (inputData.shape[0], inputData.shape[1], 1))
    # future prediction:
    prediction = model.predict(inputData)
    prediction *= scalingFactor
    st.success(f"prediction for next stock market day from {predictionDate} onwards: {prediction[0][0]}")
    st.balloons()


try:
    dataframe = data.DataReader(userInput, 'yahoo', start_date, end_date)
    st.success("You have entered " + format(userInput))
    dataframe.rename(columns={'Adj Close': 'Adjusted_Close'}, inplace=True)

    st.subheader("Data available between 2002 and " + str(end_date.year) + ":")
    st.write(dataframe)
    predictionDays = st.slider("Select the simple moving average days: ", 1, 200, 50)

    sma = dataframe.Adjusted_Close.rolling(50).mean()
    graph = plt.figure(figsize=(14, 9))
    plt.plot(sma, color='blue')
    plt.xlabel('Date')
    plt.ylabel('Adjusted Closing Price')
    plt.title('Simple Moving Average (SMA) for ' + str(predictionDays) + ' days')
    plt.legend()
    st.pyplot(graph)

    trainingData = pd.DataFrame(dataframe['Adjusted_Close'][0: int(0.8 * len(dataframe))])
    testingData = pd.DataFrame(dataframe['Adjusted_Close'][int(0.2 * len(dataframe)): int(len(dataframe))])
    minMaxScaler = MinMaxScaler(feature_range=(0, 1))

    # predicting the stock price:
    previousNDays = trainingData.tail(50)  # n may be adapted
    concatenatedDataframe = pd.concat((previousNDays, testingData), axis=0)
    # normalize the concatenated data frame
    inputData = minMaxScaler.fit_transform(concatenatedDataframe)
    scalingFactor = 1 / minMaxScaler.scale_[0]  # to undo normalization scale it back
    model = load_model('adjustedClosingPredictionModel.h5')

    predictStockPrice(dataframe, inputData, model, scalingFactor)
    # ----------------------------------------------------------
    # Now the accuracy of graph is analysed:
    xTest = []
    yTest = []

    n = 0  # n is herby the amount of days in the future being predicted
    for i in range(50, len(inputData) + n):
        xTest.append(inputData[i - 50: i])
        yTest.append(inputData[i, 0])

    # a numpy array for testing
    xTest = np.array(xTest)
    yTest = np.array(yTest)
    # Now the accuracy will be verified:
    yPrediction = model.predict(xTest)
    yTest = scalingFactor * yTest
    yPrediction = scalingFactor * yPrediction

    st.subheader("The actual vs predicted price of the stock for testing set: ")
    graph2 = plt.figure(figsize=(14, 9))
    plt.plot(yTest, color="blue", label="Actual Price")
    plt.plot(yPrediction, color="green", label="Expectation Price")
    plt.xlabel('Time')
    plt.ylabel("Stock Price")
    plt.legend()
    st.pyplot(graph2)

except RemoteDataError:
    st.error("Please enter a valid stock ticker symbol!")

#except Exception:
#    st.error("Unknown problem occurred. Please try again later!")