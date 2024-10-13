import numpy as np
from matplotlib import pyplot as plt
from sklearn import preprocessing


def plot_results(results,product_id,freqency):
    # Plot Actual vs Predicted Sales
    print(f"\nFinal {freqency} Predictions for Product_ID {product_id}:")
    plt.figure(figsize=(12,6))
    plt.plot(results['Date'], results['Actual Sales'], label='Actual Sales', marker='o')
    plt.plot(results['Date'], results['Predicted Sales'], label='Predicted Sales', marker='o')
    plt.title(f'Actual vs Predicted {freqency} Sales for Product_ID {product_id}')
    plt.xlabel('Date')
    plt.ylabel('Units Sold')
    plt.legend()
    plt.show()

def create_sequences(data, seq_length):
    # This function is to create x and y as time series forecasting input and output
    X = []
    y = []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)
