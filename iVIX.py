from datetime import datetime
import numpy as np
import pandas as pd
from scipy import interpolate

MEANINGFUL_VIX_INDEX_MATURITY_THRESHOLD = 5

shibor_rate_dataset = pd.read_csv('shibor.csv', index_col=0, encoding='GBK')
options_dataset = pd.read_csv('options.csv', index_col=0, encoding='GBK')
trade_day_dataset = pd.read_csv('tradeday.csv', encoding='GBK')
true_ivix_dataset = pd.read_csv('ivixx.csv', encoding='GBK')


def calc_interpolated_risk_free_interest_rates(options, vix_date):
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
        vix_date_shibor_rates = shibor_rate_dataset.iloc[0].values
    else:
        vix_date_shibor_rates = shibor_rate_dataset.loc[vix_date_in_str_format].values

    # collect interpolated rates from given shibor interest rate dataset as a dictionary
    # {key: expiration date, value: interpolated shibor rate}
    interpolated_shibor_rates = {}

    # standard interest rate periods from the data that we have
    interest_rate_periods = np.array([1.0, 7.0, 14.0, 30.0, 90.0, 180.0, 270.0, 365.0]) / 365.0
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


def get_near_and_next_term_option_expiration_dates(options, vix_date):
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


def calc_sigma_square(call_options, put_options, options, forward_price, risk_free_rate, maturity):
    """
    Parameters:
        call_options: 当天的call options的数据
        put_options: 当天的put options的数据
        forward_price: forward price / forward index level
        risk_free_rate: 与年化期限对应的risk free interest rate
        maturity: 年化期限

    Return:
        sigma_square: 用于计算VIX index的volatility
    """
    # get K_0 (first price below the forward price) and use it to get out-of-the-money call/put options
    options = options.set_index('EXE_PRICE').sort_index()
    if options[options.index < forward_price].empty:
        K_0 = options[options.index >= forward_price].index[0]
    else:
        K_0 = options[options.index < forward_price].index[-1]

    otm_call_options = call_options[call_options.index > K_0].copy()
    otm_put_options = put_options[put_options.index < K_0].copy()

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
            otm_call_options.loc[otm_call_strikes[i], 'DELTA_K'] = (otm_call_strikes[i + 1] - otm_call_strikes[i - 1]) / 2.0
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

    # calculate (Delta_K_i / K_i^2) * price(K_i) in the formula, we need to calculate this part for K_0 explicitly
    call_component = pd.Series()
    put_component = pd.Series()
    delta_k_0 = 0
    if not otm_call_strikes.empty:
        call_component = otm_call_options.CLOSE * otm_call_options.DELTA_K / otm_call_options.index / otm_call_options.index
        delta_k_0 = K_0 - otm_call_strikes[0]
    if not otm_put_strikes.empty:
        put_component = otm_put_options.CLOSE * otm_put_options.DELTA_K / otm_put_options.index / otm_put_options.index
        delta_k_0 = otm_put_strikes[-1] - K_0
    if (not otm_call_strikes.empty) and (not otm_put_strikes.empty):
        delta_k_0 = (otm_call_strikes[0] - otm_put_strikes[-1]) / 2

    atm_option_price = (call_options.loc[K_0, 'CLOSE'].mean() + put_options.loc[K_0, 'CLOSE'].mean()) / 2
    k_0_component = atm_option_price * delta_k_0 / K_0 / K_0
    sigma_square = ((sum(call_component) + sum(put_component) + k_0_component) * np.exp(maturity * risk_free_rate) * 2
                    / maturity - (forward_price / K_0 - 1) ** 2 / maturity)
    return sigma_square


def calc_vix_index(vix_date):
    """
    Parameters:
        vix_date：计算VIX的当天日期，'%Y/%m/%d' 字符串格式

    Return:
        vix: VIX Index值
    """

    # 在options_dataset里选取以vix_date为起始日期的所有options数据
    options = options_dataset.loc[vix_date, :]

    near_term_expiration_date, next_term_expiration_date = get_near_and_next_term_option_expiration_dates(options, vix_date)

    # get risk-free interest rates for both near-term and next-term options to expiration (R)
    interpolated_shibor_rates = calc_interpolated_risk_free_interest_rates(options, vix_date)
    near_term_risk_free_rate = interpolated_shibor_rates[near_term_expiration_date]
    next_term_risk_free_rate = interpolated_shibor_rates[next_term_expiration_date]

    # time to maturity in % year (T)
    vix_date_in_datetime_format = datetime.strptime(vix_date, '%Y/%m/%d')
    near_maturity = (near_term_expiration_date - vix_date_in_datetime_format).days / 365.0
    next_maturity = (next_term_expiration_date - vix_date_in_datetime_format).days / 365.0

    # get near term and next term options which are pandas dataframes
    near_term_options = options[pd.to_datetime(options.EXE_ENDDATE) == near_term_expiration_date]
    next_term_options = options[pd.to_datetime(options.EXE_ENDDATE) == next_term_expiration_date]

    # start calculating forward prices for near and next term options (F):

    # split calls and puts from near and next options, and set the index of the panda dataframe to the values of the
    # 'EXE_PRICE' column, and then sort the resulting dataframe based on the index
    near_call_options = near_term_options[near_term_options.EXE_MODE == '认购'].set_index('EXE_PRICE').sort_index()
    near_put_options = near_term_options[near_term_options.EXE_MODE == '认沽'].set_index('EXE_PRICE').sort_index()
    next_call_options = next_term_options[next_term_options.EXE_MODE == '认购'].set_index('EXE_PRICE').sort_index()
    next_put_options = next_term_options[next_term_options.EXE_MODE == '认沽'].set_index('EXE_PRICE').sort_index()

    # at-the-money strike price, the strike price when |call price - put price| is smallest
    near_atm_strike_price = abs(near_call_options.CLOSE - near_put_options.CLOSE).idxmin()
    next_atm_strike_price = abs(next_call_options.CLOSE - next_put_options.CLOSE).idxmin()

    # calculate the price difference between call option and put option for at-the-money strike price
    # todo: why min()?
    near_atm_call_put_price_diff = (near_call_options.CLOSE - near_put_options.CLOSE)[near_atm_strike_price].min()
    next_atm_call_put_price_diff = (next_call_options.CLOSE - next_put_options.CLOSE)[next_atm_strike_price].min()

    # apply the formula for calculating forward prices (or called forward index level)
    near_forward_price = near_atm_strike_price + np.exp(near_maturity * near_term_risk_free_rate) * near_atm_call_put_price_diff
    next_forward_price = next_atm_strike_price + np.exp(next_maturity * next_term_risk_free_rate) * next_atm_call_put_price_diff

    # 计算不同到期日期权对于VIX的贡献
    near_sigma = calc_sigma_square(near_call_options, near_put_options, near_term_options, near_forward_price, near_term_risk_free_rate, near_maturity)
    next_sigma = calc_sigma_square(next_call_options, next_put_options, next_term_options, next_forward_price, next_term_risk_free_rate, next_maturity)

    # 利用两个不同到期日的期权对VIX的贡献sig1和sig2，
    # 已经相应的期权剩余到期时间T1和T2；
    # 差值得到并返回VIX指数(%)
    w = (next_maturity - 30.0 / 365.0) / (next_maturity - near_maturity)
    vix = near_maturity * w * near_sigma + next_maturity * (1 - w) * next_sigma
    return 100 * np.sqrt(abs(vix) * 365.0 / 30.0)


ivix = []
for day in trade_day_dataset['DateTime']:
    ivix.append(calc_vix_index(day))
    # break
    # print ivix

# Render the chart
from pyecharts.charts import Line
import pyecharts.options as opts

attr = true_ivix_dataset[u'日期'].tolist()
ivix_data = true_ivix_dataset[u'收盘价(元)'].tolist()
line = Line().set_global_opts(title_opts=opts.TitleOpts(title="中国波指"))
line.add_xaxis(attr)
line.add_yaxis("中证指数发布", ivix_data, markpoint_opts=opts.MarkPointOpts(data=[opts.MarkPointItem(type_="max")]))
line.add_yaxis("手动计算", ivix, markline_opts=opts.MarkLineOpts(
    data=[opts.MarkLineItem(type_="max"), opts.MarkLineItem(type_="average")]))

line.render('vix.html')
