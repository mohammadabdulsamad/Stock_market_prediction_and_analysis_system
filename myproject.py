import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as data
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
import streamlit as st

#Creating variable for starting and ending time of data set
start = '2011-01-01'
end = '2021-12-31'
st.title('Stock Market Prediction And Analysis')
#scrapping the data from the site
input = st.text_input('Enter Stock Ticker')
df = data.DataReader(input,'yahoo',start,end)

#Describe Data
st.subheader('Data from 2011 - 2021')
st.write(df.describe())

#Visualizations
st.subheader('Closing Price vs Time chart')
graph = plt.figure(figsize = (12,6))
plt.plot(df.Close)
st.pyplot(graph)

st.subheader('Closing Price vs Time chart with Hundred Moving Avg')
ma100 = df.Close.rolling(100).mean()
graph = plt.figure(figsize = (12,6))
plt.plot(ma100)
plt.plot(df.Close)
st.pyplot(graph)

st.subheader('Closing Price vs Time chart with Hundred and Two Hundred Moving Avg')
ma100 = df.Close.rolling(100).mean()
ma200 = df.Close.rolling(200).mean()
graph = plt.figure(figsize = (12,6))
plt.plot(ma100,'r')
plt.plot(ma200,'g')
plt.plot(df.Close,'b')
st.pyplot(graph)

D_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
D_testing = pd.DataFrame(df['Close'][int(len(df)*0.70):int(len(df))])

scaler = MinMaxScaler(feature_range=(0,1))
D_training_array = scaler.fit_transform(D_training)

#Predict the upcoming value based on previous or historical data
x_train = []
y_train = []
for i in range(100,D_training_array.shape[0]):
    x_train.append(D_training_array[i-100: i])
    y_train.append(D_training_array[i, 0])

x_train,y_train = np.array(x_train), np.array(y_train)
#Load my model
model = load_model('keras_model.h5')

#Testing part
past_100_days = D_training.tail(100)
final_df = past_100_days.append(D_testing, ignore_index=True)

input_data = scaler.fit_transform(final_df)
x_test = []
y_test = []

for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i-100: i])
    y_test.append(input_data[i, 0])
x_test, y_test = np.array(x_test), np.array(y_test)

y_predicted = model.predict(x_test)
scaler = scaler.scale_
scale_factor = 1/scaler[0]
y_predicted = y_predicted * scale_factor
y_test = y_test * scale_factor

#Final Graph
st.subheader('Prediction vs Original')
graph2 = plt.figure(figsize = (12,6))
plt.plot(y_test, 'b' , label = 'Original Price')
plt.plot(y_predicted, 'r', label = 'Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(graph2)

