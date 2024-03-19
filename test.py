'''
testing of IVIX.py for learning purposes
'''
import random
from datetime import datetime
import numpy as np
import pandas as pd
from scipy import interpolate
import iVIX
import matplotlib.pyplot as plt



def plot(data, col_name):
    plt.plot(data.index, data['1M'], label='1M')

    plt.xlabel('Date')
    plt.ylabel('Rate')
    plt.title('shibor')
    plt.legend()
    plt.grid(True)

    # Show plot
    plt.show()


def main():
    shibor_rate = pd.read_csv('shibor.csv', index_col=0, encoding='GBK', parse_dates=[0])
    options_data = pd.read_csv('options.csv', index_col=0, encoding='GBK')
    trade_days = pd.read_csv('tradeday.csv', encoding='GBK')
    # true_ivix = pd.read_csv('ivixx.csv', encoding='GBK')

    # test cubic spline
    # plot(shibor_rate, '1M')
    fig = plt.plot()

    # for i in range(1):
    date_index = random.randint(1, len(trade_days))
    test_date = pd.to_datetime(trade_days.iloc[date_index]['DateTime'])
    shibor_values = shibor_rate.loc[test_date.strftime('%Y-%m-%d')].values
    interpolation_period = np.asarray([1.0, 7.0, 14.0, 30.0, 90.0, 180.0, 270.0, 360.0]) / 360.0
    cs = interpolate.CubicSpline(interpolation_period, shibor_values)
    # ax.plot(interpolation_period, shibor_values, 'o', label='data')
    xs = np.arange(1 / 365, 1, 1 / 365)
    plt.plot(xs, cs(xs), label=test_date.strftime('%Y-%m-%d'))
    plt.plot(0.456, cs(0.456), 'o')

    plt.legend()
    plt.xlabel('time period')
    plt.ylabel('rates')
    plt.show()


if __name__ == "__main__":
    main()
