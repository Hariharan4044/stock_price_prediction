import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import tkinter as tk
from tkinter import filedialog

def upload_file():
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(title="Select CSV File", filetypes=[("CSV files", "*.csv")])
    if file_path:
        print(f"File selected: {file_path}")
        return pd.read_csv(file_path)
    else:
        print("No file selected. Exiting.")
        exit()

data = upload_file()

print("\nFirst few rows of the uploaded data:")
print(data.head())


if 'Date' not in data.columns:
    print("Error: The uploaded file must contain a 'Date' column.")
    exit()


data['Date'] = pd.to_datetime(data['Date'])
data = data.sort_values(by='Date')


stock_columns = [col for col in data.columns if col != 'Date']
print(f"\nAvailable stock columns for prediction: {stock_columns}")

stock_choice = input(f"Choose the stock column to predict (e.g., {stock_columns[0]}): ")
if stock_choice not in stock_columns:
    print(f"Invalid choice. Please choose from {stock_columns}. Exiting.")
    exit()


data['Close'] = data[stock_choice]
data['SMA_20'] = data['Close'].rolling(window=20).mean()
data['EMA_20'] = data['Close'].ewm(span=20, adjust=False).mean()
data['Daily_Return'] = data['Close'].pct_change()
data = data.dropna()


X = data[['Close', 'SMA_20', 'EMA_20', 'Daily_Return']]
y = data['Close'].shift(-1)
X = X[:-1]
y = y[:-1]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


model = LinearRegression()
model.fit(X_train_scaled, y_train)


y_pred = model.predict(X_test_scaled)


mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print(f"\nMean Squared Error (MSE): {mse}")
print(f"Root Mean Squared Error (RMSE): {rmse}")

plt.figure(figsize=(10, 6))
plt.plot(y_test.index, y_test, label='Actual Prices', color='blue')
plt.plot(y_test.index, y_pred, label='Predicted Prices', linestyle='dashed', color='orange')
plt.legend()
plt.title('Stock Price Prediction: Actual vs Predicted')
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()


last_data = X.iloc[-1].to_frame().T  
last_data_scaled = scaler.transform(last_data)  


next_day_price = model.predict(last_data_scaled)
predicted_price = next_day_price.item()

print(f"\nPredicted Next Day Price: {predicted_price:.2f}")
