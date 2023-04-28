# Stock Market Prediction And Forecasting Using Stacked LSTM

## Objective
The objective of this project is to predict the closing stock price of Tata Global Beverages Ltd. for the next 28 days using a stacked LSTM neural network.

## Dataset
The dataset used for this project is taken from Kaggle. The dataset consists of daily stock price data for Tata Global Beverages Ltd. from 2013 to 2018. The dataset contains several columns such as Open, High, Low, Last, Close, Total Trade Quantity, and Turnover (Lacs). We will only use the 'Date' and 'Close' columns for this project.

Dataset link: https://raw.githubusercontent.com/mwitiderrick/stockprice/master/NSE-TATAGLOBAL.csv

## Approach
We will use a stacked LSTM neural network to predict the closing stock price for the next 28 days. We will preprocess the data by scaling it between 0 and 1 and splitting it into training and testing datasets. We will then reshape the data to be 3-dimensional in order to feed it to the stacked LSTM model. We will train the model on the training data and validate it on the testing data. Finally, we will make predictions on the future 28 days.

## Results
We will evaluate the performance of the model using mean squared error (MSE) and mean absolute error (MAE). We will compare the predicted stock prices with the actual stock prices for the testing data. We will plot the predicted and actual stock prices on a graph to visualize the performance of the model.

## Conclusion
In this project, we used a stacked LSTM neural network to predict the closing stock price of Tata Global Beverages Ltd. for the next 28 days. The model was trained on the stock price data from 2013 to 2018 and was tested on data from 2018 to 2019. The model's performance was evaluated using mean squared error (MSE) and mean absolute error (MAE). The predicted and actual stock prices were plotted on a graph to visualize the performance of the model.
