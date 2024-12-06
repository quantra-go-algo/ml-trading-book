import talib as ta
import numpy as np
from sklearn import metrics
import seaborn as sns

import matplotlib.pyplot as plt

def get_target_features(data):    
    # --------------------------------
    # Create features
    # Create a column 'pct_change' with the 15-minute prior percentage change
    data['pct_change'] = data['close'].pct_change()

    # Create a column 'pct_change2' with the half an hour prior percentage change
    data['pct_change2'] = data['close'].pct_change(2)

    # Create a column 'pct_change5' with the 75-minute prior percentage change
    data['pct_change5'] = data['close'].pct_change(5)
    
    # Create a column by the name RSI, and assign the RSI values to it
    data['rsi'] = ta.RSI(data['close'].values, timeperiod=int(6.5*4))

    # Create a column by the name ADX, and assign the ADX values to it
    data['adx'] = ta.ADX(data['high'].values, data['low'].values, 
                         data['open'].values, timeperiod=int(6.5*4))
    
    # Create a column by the name sma, and assign the SMA values to it
    data['sma'] = data['close'].rolling(window=int(6.5*4)).mean()

    # Create a column by the name corr, and assign the correlation values to it
    data['corr'] = data['close'].rolling(window=int(6.5*4)).corr(data['sma'])
    
    # 1-day and 2-day volatility
    data['volatility'] = data.rolling(
        int(6.5*4), min_periods=int(6.5*4))['pct_change'].std()*100

    data['volatility2'] = data.rolling(
        int(6.5*8), min_periods=int(6.5*8))['pct_change'].std()*100
    
    # -------------------------------------------------
    # Define target variable
    # Create a column 'future_returns' with the calculation of percentage change
    data['future_returns'] = data['close'].pct_change().shift(-1)

    # Create the signal column
    data['signal'] = np.where(data['future_returns'] > 0, 1, 0)
    
    data = data.dropna()

    return data['signal'], data[['pct_change', 'pct_change2',
          'pct_change5', 'rsi', 'adx', 'corr', 
          'volatility']]


def get_performance(strategy_data):
    plot_equity_curve(strategy_data)
    plot_dd(strategy_data)
    sharpe_ratio(strategy_data)


def plot_equity_curve(strategy_data):
    
    # Calculate the cumulative returns
    strategy_data['strategy_cumulative_returns'] = (
        1+strategy_data['strategy_returns']).cumprod()
    
    strategy_data['benchmark_cumulative_returns'] = (
        1+strategy_data['pct_change']).cumprod()

    # ---------------------Equity Curve---------------------
    # Plot cumulative strategy returns
    strategy_data['strategy_cumulative_returns'].plot(figsize=(8, 5), color='green')
    strategy_data['benchmark_cumulative_returns'].plot(figsize=(8, 5), color='blue')
    plt.title('Equity Curve', fontsize=14)
    plt.legend()
    plt.ylabel('Cumulative returns')
    plt.tight_layout()
    plt.show()

    
def plot_dd(strategy_data):
    
    # Calculate the running maximum
    running_max = np.maximum.accumulate(
        strategy_data['strategy_cumulative_returns'].dropna())
    # Ensure the value never drops below 1
    running_max[running_max < 1] = 1
    # Calculate the percentage drawdown
    drawdown = ((strategy_data['strategy_cumulative_returns'])/running_max - 1) * 100

    # Calculate the maximum drawdown
    max_dd = drawdown.min()
    print("The maximum drawdown is {0:.2f}%.".format(max_dd))

    # ---------------------DD plot---------------------
    fig = plt.figure(figsize=(8, 5))

    # Plot max drawdown
    plt.plot(drawdown, color='red')
    # Fill in-between the drawdown
    plt.fill_between(drawdown.index, drawdown.values, color='red')
    plt.title('Strategy Drawdown', fontsize=14)
    plt.ylabel('Drawdown(%)', fontsize=12)
    plt.xlabel('Year', fontsize=12)

    plt.tight_layout()
    plt.show()
    
def sharpe_ratio(strategy_data):
    sharpe_ratio = round(strategy_data['strategy_returns'].mean() /
                     strategy_data['strategy_returns'].std() * np.sqrt(252*6.5*4), 2)
    print("The Sharpe ratio is {0:.2f}.".format(sharpe_ratio))
    
    
def get_metrics(y_test, predicted):
    confusion_matrix_data = metrics.confusion_matrix(y_test, predicted)
    # Plot the data

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(confusion_matrix_data, fmt="d",
                cmap='Blues', cbar=False, annot=True, ax=ax)

    # Set the axes labels and the title
    ax.set_xlabel('Predicted Labels', fontsize=12)
    ax.set_ylabel('Actual Labels', fontsize=12)
    ax.set_title('Confusion Matrix', fontsize=14)
    ax.xaxis.set_ticklabels(['No Position', 'Long Position'])
    ax.yaxis.set_ticklabels(['No Position', 'Long Position'])

    # Display the plot
    plt.show()

    print(metrics.classification_report(y_test, predicted))
