import pandas_datareader as dtr
import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf
import numpy as np
import tensorflow as tf

#from keras_model import load_data
import streamlit as st


start='2012-01-01'
end= '2022-12-30'

st.title("Stock Price Predictor ")

user_input = st.text_input("Enter stock ticker", "AAPL")

# data = dtr.DataReader(user_input,'yahoo', start, end)
data = yf.download('AAPL',start,end)

# visulasizations

st.subheader("Closing price V/S Time ")
fig = plt.figure(figsize=(12,6))
plt.title("Close price history", fontsize=18)
plt.xlabel('Time', fontsize = 14)
plt.ylabel('Price', fontsize = 14)
plt.plot(data.Close)
st.pyplot(fig)


st.subheader("Closing price With 100MA and 200MA V/S Time ")

ma200 = data.Close.rolling(200).mean()
ma100 = data.Close.rolling(100).mean()

fig = plt.figure(figsize=(12,6))

plt.plot(data.Close)
plt.plot(ma100, 'r' , label= "100MA")
plt.plot(ma200, 'g', label= '200MA')
plt.title("Close price with MA100 and MA200", fontsize=18)
plt.xlabel('Day_elapsed', fontsize = 14)
plt.ylabel('Price', fontsize = 14)
plt.legend()
st.pyplot(fig)



# Spliting data into training and test set

data_train = pd.DataFrame(data.Close[0:int(len(data)* .7)])
data_test = pd.DataFrame(data.Close[int(len(data)* .7):len(data)])


# Doing feature scaling as we will apply LSTM

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))
train_data = scaler.fit_transform(data_train)

# load model
model = tf.keras.load_model('share_model.h5')


# Testing part
past_100_days= data_train.tail(100)
final_data_test = past_100_days.append(data_test , ignore_index = True)
test_data = scaler.transform(final_data_test)
x_test = []
y_test = []
for i in range(100,test_data.shape[0]):
  x_test.append(test_data[i-100:i])
  y_test.append(test_data[i,0])

x_test , y_test = np.array(x_test), np.array(y_test)  
predictions= model.predict(x_test)
predictions = scaler.inverse_transform(predictions)



# plot the data
st.subheader("Prediction V/S Original ")
fig2= plt.figure(figsize=(12,6))
plt.plot(y_test,  'b', label='Original_price')
plt.plot(predictions, 'r', label='predicted_price')
plt.xlabel('day')
plt.ylabel('price')
plt.legend()
st.pyplot(fig2)


