import os
from datetime import datetime
import numpy as np
import pandas as pd
from scipy import interpolate
from pyecharts.charts import Line
import pyecharts.options as opts


def calcInterpolatedRiskFreeInterestRates(options, vix_date):
    """
    Parameters:
        options: 计算VIX的当天的options的数据
        vix_date: 用来计算VIX的日期

    Return：
        interpolated_shibor_rates：与到期日对应的年化期限(maturity)的risk free interest rates
    """
    vix_date_in_datetime_format = datetime.strptime(vix_date, '%Y/%m/%d')
    vix_date_in_str_format = vix_date_in_datetime_format.strftime('%Y-%m-%d')

    # sort in ascending order for all unique expiration dates
    expiration_dates = np.sort(options.EXE_ENDDATE.unique())

    # collect all maturities from 'date' to all expiration dates, into a dictionary.
    # {key: expiration date, value: maturity in % in year}
    maturities = {}
    for day in expiration_dates:
        day = pd.to_datetime(day)
        # % in year
        maturities[day] = float((day - vix_date_in_datetime_format).days) / 365.0

    # 选取最近的有数据的日期，如果没有当天interest rate数据，则选取最近日的interest rate数据
    latest_shibor_date = datetime.strptime(shibor_rate_dataset.index[0], '%Y-%m-%d')
    if vix_date_in_datetime_format >= latest_shibor_date:
        vix_date_shibor_rates: np.ndarray = shibor_rate_dataset.iloc[0].values
    else:
        vix_date_shibor_rates: np.ndarray = shibor_rate_dataset.loc[vix_date_in_str_format].values

    # collect interpolated rates from given shibor interest rate dataset as a dictionary
    # {key: expiration date, value: interpolated shibor rate}
    interpolated_shibor_rates = {}

    # standard interest rate periods from the data that we have
    interest_rate_periods: np.ndarray = np.array([1.0, 7.0, 14.0, 30.0, 90.0, 180.0, 270.0, 365.0]) / 365.0
    # perform interpolation for different maturities
    cs = interpolate.CubicSpline(interest_rate_periods, vix_date_shibor_rates)
    min_period = min(interest_rate_periods)
    max_period = max(interest_rate_periods)

    for day in maturities.keys():
        # 与到期日对应的年化期限
        maturity = maturities[day]
        # 把值拉回到定义到区间内
        if maturities[day] > max_period:
            maturity = max_period * 0.99999
        elif maturities[day] < min_period:
            maturity = min_period * 1.00001

        # convert the interpolated interest rate into exact value
        interpolated_shibor_rates[day] = cs(maturity) / 100.0

    return interpolated_shibor_rates


def getNearAndNextTermOptionExpirationDates(options, vix_date):
    """
    Parameters:
        options: 计算VIX的当天的options的数据
        vix_date: 用来计算VIX的日期

    Return:
        near_term_expiration_date: 当月合约的到期日
        next_term_expiration_date: 次月合约的到期日
    """
    vix_date_in_datetime_format = datetime.strptime(vix_date, '%Y/%m/%d')
    # convert all unique expiration dates to datetime format into a python list
    expiration_dates = [datetime.strptime(i, '%Y/%m/%d %H:%M') for i in list(options.EXE_ENDDATE.unique())]

    # we want to get near term expiration date by selecting the nearest expiration date,
    # but if time to maturity from current vix_date
    # is less than a certain number of days that makes the resulting vix index
    # not as meaningful, we roll to the next expiration date until threshold condition is satisfied.
    # next term expiration date is next available expiration date to near term expiration date
    sorted_expiration_dates = sorted(expiration_dates)
    near_term_expiration_date_index = 0
    next_term_expiration_date_index = 1

    while ((next_term_expiration_date_index < len(expiration_dates)) and
           ((sorted_expiration_dates[near_term_expiration_date_index] - vix_date_in_datetime_format).days <
            MEANINGFUL_VIX_INDEX_MATURITY_THRESHOLD)):
        near_term_expiration_date_index += 1
        next_term_expiration_date_index += 1

    near_term_expiration_date = sorted_expiration_dates[near_term_expiration_date_index]
    next_term_expiration_date = sorted_expiration_dates[next_term_expiration_date_index]

    return near_term_expiration_date, next_term_expiration_date


def calcForwardPrice(call_options, put_options, risk_free_rate, maturity):
    """
    Parameters:
        call_options: 当天的call options的数据
        put_options: 当天的put options的数据
        risk_free_rate: 与年化期限对应的risk free interest rate
        maturity: 年化期限

    Return:
        forward_price: forward price / forward index level
    """
    # get at-the-money strike price, which the strike price when |call price - put price| is the smallest
    atm_strike_price: float = abs(call_options.CLOSE - put_options.CLOSE).idxmin()
    # calculate the price difference between call option and put option for at-the-money strike price
    atm_call_put_price_diff: float = (call_options.CLOSE - put_options.CLOSE)[atm_strike_price].min()
    # apply the formula for calculating forward prices
    forward_price = atm_strike_price + np.exp(maturity * risk_free_rate) * atm_call_put_price_diff
    return forward_price


def calcSigmaSquare(call_options, put_options, options, forward_price, risk_free_rate, maturity):
    """
    Parameters:
        call_options: 当天的call options的数据
        put_options: 当天的put options的数据
        options: 当天所有options的数据
        forward_price: forward price / forward index level
        risk_free_rate: 与年化期限对应的risk free interest rate
        maturity: 年化期限

    Return:
        sigma_square: 用于计算VIX index的volatility
    """
    # get K_0 (first price below the forward price) and use it to get out-of-the-money call/put options
    options = options.set_index('EXE_PRICE').sort_index()
    if options[options.index < forward_price].empty:
        K_0: float = options[options.index >= forward_price].index[0]
    else:
        K_0: float = options[options.index < forward_price].index[-1]

    otm_call_options: pd.DataFrame = call_options[call_options.index > K_0].copy()
    otm_put_options: pd.DataFrame = put_options[put_options.index < K_0].copy()

    # Interval between strike prices (Delta K_i)
    # get a series of strike prices for call/put options (the value for index = strike prices)
    # and calculate the DELTA_K value for all options with index i
    # at the upper and lower edges of any given strip of options,
    # DELTA_K is simply the difference between K_i and the adjacent strike price.
    otm_call_strikes = otm_call_options.index
    if otm_call_strikes.empty:
        pass
    elif len(otm_call_strikes) < 3:
        otm_call_options['DELTA_K'] = otm_call_strikes[-1] - otm_call_strikes[0]
    else:
        for i in range(1, len(otm_call_strikes) - 1):
            otm_call_options.loc[otm_call_strikes[i], 'DELTA_K'] = (otm_call_strikes[i + 1] - otm_call_strikes[
                i - 1]) / 2.0
        otm_call_options.loc[otm_call_strikes[0], 'DELTA_K'] = otm_call_strikes[1] - otm_call_strikes[0]
        otm_call_options.loc[otm_call_strikes[-1], 'DELTA_K'] = otm_call_strikes[-1] - otm_call_strikes[-2]

    otm_put_strikes = otm_put_options.index
    if otm_put_strikes.empty:
        pass
    elif len(otm_put_strikes) < 3:
        otm_put_options['DELTA_K'] = otm_put_strikes[-1] - otm_put_strikes[0]
    else:
        for i in range(1, len(otm_put_strikes) - 1):
            otm_put_options.loc[otm_put_strikes[i], 'DELTA_K'] = (otm_put_strikes[i + 1] - otm_put_strikes[i - 1]) / 2.0
        otm_put_options.loc[otm_put_strikes[0], 'DELTA_K'] = otm_put_strikes[1] - otm_put_strikes[0]
        otm_put_options.loc[otm_put_strikes[-1], 'DELTA_K'] = otm_put_strikes[-1] - otm_put_strikes[-2]

    # calculate (Delta_K_i / K_i^2) * price(K_i) part in the formula, we need to calculate this part for K_0 explicitly
    call_component = pd.Series()
    put_component = pd.Series()
    delta_K_0 = 0
    if otm_call_strikes.empty and otm_put_strikes.empty:
        pass
    elif (not otm_call_strikes.empty) and otm_put_strikes.empty:
        call_component = otm_call_options.CLOSE * otm_call_options.DELTA_K / otm_call_options.index / otm_call_options.index
        delta_K_0 = otm_call_strikes[0] - K_0
    elif (not otm_put_strikes.empty) and otm_call_strikes.empty:
        put_component = otm_put_options.CLOSE * otm_put_options.DELTA_K / otm_put_options.index / otm_put_options.index
        delta_K_0 = K_0 - otm_put_strikes[-1]
    else:
        call_component = otm_call_options.CLOSE * otm_call_options.DELTA_K / otm_call_options.index / otm_call_options.index
        put_component = otm_put_options.CLOSE * otm_put_options.DELTA_K / otm_put_options.index / otm_put_options.index
        delta_K_0 = (otm_call_strikes[0] - otm_put_strikes[-1]) / 2

    atm_option_price = (call_options.loc[K_0, 'CLOSE'].mean() + put_options.loc[K_0, 'CLOSE'].mean()) / 2
    k_0_component = atm_option_price * delta_K_0 / K_0 / K_0

    sigma_square = ((sum(call_component) + sum(put_component) + k_0_component) * np.exp(maturity * risk_free_rate) * 2 /
                    maturity - (forward_price / K_0 - 1) ** 2 / maturity)
    return sigma_square


def outputSigmaSquare(options, expiration_date, interpolated_shibor_rates, maturity):
    """
    Parameters:
        options：当天所有options的数据
        expiration_date: 一个给定的option到期日
        interpolated_shibor_rates: 插值计算得出的 risk-free rates
        maturity: 一个给定的option年化期限

    Return:
        sigma_square: 一个给定option到期日对应的volatility
    """
    # get risk-free interest rate corresponding to an expiration date
    risk_free_rate: float = interpolated_shibor_rates[expiration_date]

    # select options with an expiration date
    options_with_expiration_date: pd.DataFrame = options[pd.to_datetime(options.EXE_ENDDATE) == expiration_date]

    # split calls and puts from options of a particular term,
    # and set the index of the pandas dataframe to the values of the 'EXE_PRICE' column (strike),
    # then sort the resulting dataframe based on the index
    call_options: pd.DataFrame = (options_with_expiration_date[options_with_expiration_date.EXE_MODE == '认购']
                                  .set_index('EXE_PRICE').sort_index())
    put_options: pd.DataFrame = (options_with_expiration_date[options_with_expiration_date.EXE_MODE == '认沽']
                                 .set_index('EXE_PRICE').sort_index())

    forward_price: float = calcForwardPrice(call_options, put_options, risk_free_rate, maturity)
    sigma_square: float = calcSigmaSquare(call_options, put_options, options_with_expiration_date,
                                          forward_price, risk_free_rate, maturity)
    return sigma_square


def calcVixIndex(vix_date, options, interest_rate):
    """
    Parameters:
        vix_date：计算VIX的当天日期

    Return:
        vix: VIX Index值
    """

    # determine the near and next term expiration dates
    (near_term_expiration_date,
     next_term_expiration_date) = getNearAndNextTermOptionExpirationDates(options, vix_date)

    # time to maturity in % year
    vix_date_in_datetime_format = datetime.strptime(vix_date, '%Y/%m/%d')
    near_maturity = (near_term_expiration_date - vix_date_in_datetime_format).days / 365.0
    next_maturity = (next_term_expiration_date - vix_date_in_datetime_format).days / 365.0

    # calculate volatility for both near-term and next-term options
    near_sigma_square: float = outputSigmaSquare(options, near_term_expiration_date, interest_rate, near_maturity)
    next_sigma_square: float = outputSigmaSquare(options, next_term_expiration_date, interest_rate, next_maturity)

    # get the VIX index value by calculating the weighted average of above volatility, where the weight is
    # proportional to the absolute time difference between the expiration date and the number of days in the
    # future that we want to measure VIX in.
    near_weight = (next_maturity - CONSTANT_MATURITY_TERM / 365.0) / (next_maturity - near_maturity)
    next_weight = 1 - near_weight
    vix = 100.0 * np.sqrt((near_maturity * near_weight * near_sigma_square + next_maturity * next_weight *
                           next_sigma_square) * 365.0 / CONSTANT_MATURITY_TERM)
    return vix


def main():
    # calculate vix index for every available trade day
    # ivix is the name for 中国波指
    ivix = []
    for day in trade_day_dataset['DateTime']:
        # select options that is available at the particular date of vix_date
        options = pd.DataFrame = options_dataset.loc[day, :]

        # get interpolated risk-free interest rates
        interpolated_shibor_rates = calcInterpolatedRiskFreeInterestRates(options, day)

        ivix.append(calcVixIndex(day, options, interpolated_shibor_rates))

    # render chart
    attr = true_ivix_dataset['日期'].tolist()
    ivix_data = true_ivix_dataset['收盘价(元)'].tolist()
    line = Line().set_global_opts(title_opts=opts.TitleOpts(title='中国波指'))
    line.add_xaxis(attr)
    line.add_yaxis('中证指数发布', ivix_data, markpoint_opts=opts.MarkPointOpts(data=[opts.MarkPointItem(type_="max")]))
    line.add_yaxis('手动计算', ivix, markline_opts=opts.MarkLineOpts(
        data=[opts.MarkLineItem(type_='max'), opts.MarkLineItem(type_='average')]))
    line.render('vix.html')


if __name__ == '__main__':
    MEANINGFUL_VIX_INDEX_MATURITY_THRESHOLD = 5
    CONSTANT_MATURITY_TERM = 30

    file_path = os.getcwd()
    risk_free_rates_file_name = 'shibor.csv'
    options_file_name = 'options.csv'
    trade_days_file_name = 'tradeday.csv'
    true_ivix_file_name = 'ivixx.csv'

    # get necessary datasets from paths
    risk_free_rates_file_path = os.path.join(file_path, risk_free_rates_file_name)
    options_file_path = os.path.join(file_path, options_file_name)
    trade_days_file_path = os.path.join(file_path, trade_days_file_name)
    true_ivix_file_path = os.path.join(file_path, true_ivix_file_name)

    shibor_rate_dataset = pd.read_csv(risk_free_rates_file_path, index_col=0, encoding='GBK')
    options_dataset = pd.read_csv(options_file_path, index_col=0, encoding='GBK')
    trade_day_dataset = pd.read_csv(trade_days_file_path, encoding='GBK')
    true_ivix_dataset = pd.read_csv(true_ivix_file_path, encoding='GBK')
    main()