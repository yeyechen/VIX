import iVIX
import os
from datetime import datetime
import numpy as np
import pandas as pd
from scipy import interpolate
from pyecharts.charts import Line
import pyecharts.options as opts


def outputSigmaSquare(options, expiration_time, risk_free_rates, maturity):
    """
    Parameters:
        options：当前时间点所有options的数据
        expiration_time: 一个给定的option的到期时间点
        risk_free_rates: risk-free rates
        maturity: 一个给定的option的年化期限

    Return:
        sigma_square: 一个给定option到期时间点对应的volatility
    """
    # get risk-free rate corresponding to an expiration time
    risk_free_rate: float = risk_free_rates[expiration_time]

    # select options corresponding to an expiration time
    options: pd.DataFrame = options.rename(columns={'k_x': 'EXE_PRICE'})
    options_with_expiration_time: pd.DataFrame = options[pd.to_datetime(options.EXE_ENDDATE) == expiration_time]

    # select call and put options corresponding to an expiration time, set strike price as index, and then sort
    call_options: pd.DataFrame = options_with_expiration_time.rename(columns={'price_call': 'CLOSE', 'k_x': 'EXE_PRICE'}).set_index('EXE_PRICE').sort_index()
    put_options: pd.DataFrame = options_with_expiration_time.rename(columns={'price_put': 'CLOSE', 'k_x': 'EXE_PRICE'}).set_index('EXE_PRICE').sort_index()

    # calculate forward price and volatility
    forward_price: float = iVIX.calcForwardPrice(call_options, put_options, risk_free_rate, maturity)
    sigma_square: float = iVIX.calcSigmaSquare(call_options, put_options, options_with_expiration_time, forward_price, risk_free_rate, maturity)

    return sigma_square


def calcVixIndex(vix_time: datetime):
    """
        Parameters:
            vix_time：计算VIX的时间点

        Return:
            vix: VIX Index值
        """
    # select options for a particular vix_time
    options: pd.DataFrame = new_options_dataset.loc[vix_time, :]

    # get near/next-term expiration time
    (near_term_expiration_time, next_term_expiration_time) = iVIX.getNearAndNextTermOptionExpirationDates(options, vix_time)

    # calculate near/next-term 年化期限
    near_maturity_data = options.loc[options['EXE_ENDDATE'] == near_term_expiration_time.strftime('%Y/%m/%d %H:%M'), 'd2e':'timeleft']
    next_maturity_data = options.loc[options['EXE_ENDDATE'] == next_term_expiration_time.strftime('%Y/%m/%d %H:%M'), 'd2e':'timeleft']
    near_maturity_days, near_maturity_minutes = near_maturity_data.iloc[0,0], near_maturity_data.iloc[0,1]
    next_maturity_days, next_maturity_minutes = next_maturity_data.iloc[0, 0], next_maturity_data.iloc[0, 1]
    near_maturity = (near_maturity_days + near_maturity_minutes / TRADING_MINUTES_IN_A_DAY) / TRADING_DAYS_IN_A_YEAR
    next_maturity = (next_maturity_days + next_maturity_minutes / TRADING_MINUTES_IN_A_DAY) / TRADING_DAYS_IN_A_YEAR

    # calculate volatility for both near/next-term options
    risk_free_rates = {near_term_expiration_time: 0.0, next_term_expiration_time: 0.0}
    near_sigma_square: float = outputSigmaSquare(options, near_term_expiration_time, risk_free_rates, near_maturity)
    next_sigma_square: float = outputSigmaSquare(options, next_term_expiration_time, risk_free_rates, next_maturity)

    # the weight is computed based on number of trading days in a year/month, rather than number of calendar days
    near_weight = (next_maturity - CONSTANT_MATURITY_TERM / TRADING_DAYS_IN_A_YEAR) / (next_maturity - near_maturity)
    next_weight = 1 - near_weight
    vix = 100.0 * np.sqrt((near_maturity * near_weight * near_sigma_square + next_maturity * next_weight *
                           next_sigma_square) * TRADING_DAYS_IN_A_YEAR / CONSTANT_MATURITY_TERM)
    return vix


def main():
    # calculate vix index and implied volatility for every available trading time
    ivix = []
    iv = []
    trading_time = sorted(new_options_dataset.index.unique())

    for time in trading_time:

        ivix.append(calcVixIndex(time))

        # compute the average of implied volatility for each trading time
        options_for_iv: pd.DataFrame = new_options_dataset.loc[time, :]
        implied_volatility = options_for_iv['iv'].mean()
        iv.append(implied_volatility * 100)

        # render chart
        attr = trading_time
        line = Line().set_global_opts(title_opts=opts.TitleOpts(title='中国波指'))
        line.add_xaxis(attr)
        line.add_yaxis('隐含波动率', iv,
                        markpoint_opts=opts.MarkPointOpts(data=[opts.MarkPointItem(type_="max")]))
        line.add_yaxis('手动计算', ivix, markline_opts=opts.MarkLineOpts(
            data=[opts.MarkLineItem(type_='max'), opts.MarkLineItem(type_='average')]))
        line.render('new_vix.html')



if __name__ == '__main__':

    TRADING_MINUTES_IN_A_DAY = 240
    TRADING_DAYS_IN_A_YEAR = 244
    MEANINGFUL_VIX_INDEX_MATURITY_THRESHOLD = 5
    # trading days in a month
    CONSTANT_MATURITY_TERM = 20

    file_path = os.getcwd()

    new_options_file_name = 'option_total.csv'
    new_options_file_path = os.path.join(file_path, new_options_file_name)
    new_options_dataset = pd.read_csv(new_options_file_name, encoding='GBK')

    # adjust the structure of options dataset to fit in original code
    # change: setting 'vix_time' as index, selecting useful columns, renaming 'expire_date' to 'EXE_ENDDATE'
    new_options_dataset['time'] = new_options_dataset['time'].astype(str).str.split(' ').str[-1]
    new_options_dataset['vix_time'] = pd.to_datetime(new_options_dataset['date'] + ' ' + new_options_dataset['time'])
    new_options_dataset = new_options_dataset.set_index('vix_time')
    selected_columns = ['price_call', 'price_put', 'k_x', 'd2e', 'timeleft', 'r', 'expire_date', 'iv']
    new_options_dataset = new_options_dataset[selected_columns]
    new_options_dataset = new_options_dataset.rename(columns={'expire_date': 'EXE_ENDDATE'})
    trading_close_time = '15:00'
    new_options_dataset['EXE_ENDDATE'] = new_options_dataset['EXE_ENDDATE'].str.replace('-', '/') + ' ' + trading_close_time

    main()

