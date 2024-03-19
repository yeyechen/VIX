from datetime import datetime
import numpy as np
import pandas as pd
from scipy import interpolate

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

    # 频繁的remove是因为想在optionsExpDate里向后平移，逻辑是对的
    # we want to get near term expiration date by selecting the nearest expiration date,
    # but if the nearest expiration date is less than 1 day to current vix_date, we roll
    # to the next expiration date.
    # next term expiration date is next nearest expiration
    # todo: refactor
    near_term_expiration_date = min(expiration_dates)
    expiration_dates.remove(near_term_expiration_date)
    if near_term_expiration_date.day - vix_date_in_datetime_format.day < 5:  # todo: 严谨来说，是要加month约束的，但是在vix里我们只考虑<30天内的期权，这里原作者偷懒了
        near_term_expiration_date = min(expiration_dates)
        expiration_dates.remove(near_term_expiration_date)
    next_term_expiration_date = min(expiration_dates)

    return near_term_expiration_date, next_term_expiration_date


def calSigmaSquare(options, FF, R, T):
    # 计算某个到期日期权对于VIX的贡献sigma；
    # 输入为期权数据options，FF为forward index price，
    # R为无风险利率， T为期权剩余到期时间
    """
    params: options:该date为交易日的所有期权合约的基本信息和价格信息
            FF: 根据上一步计算得来的strike，然后再计算得到的forward index price， 根据它对所需要的看涨看跌合约进行划分。
                取小于FF的第一个行权价为中间行权价K0， 然后选取大于等于K0的所有看涨合约， 选取小于等于K0的所有看跌合约。
                对行权价为K0的看涨看跌合约，删除看涨合约，不过看跌合约的价格为两者的均值。
            R： 这部分期权合约到期日对应的无风险利率 shibor
            T： 还有多久到期（年化）
    return：Sigma：得到的结果是传入该到期日数据的Sigma
    """
    callAll = options[options.EXE_MODE == u"认购"].set_index(u"EXE_PRICE").sort_index()
    putAll = options[options.EXE_MODE == u"认沽"].set_index(u"EXE_PRICE").sort_index()
    callAll['deltaK'] = 0.05  # me: 数据特殊，deltaK_i都等于0.05
    putAll['deltaK'] = 0.05

    # Interval between strike prices
    index = callAll.index
    if len(index) < 3:
        callAll['deltaK'] = index[-1] - index[0]
    else:
        for i in range(1, len(index) - 1):
            callAll.loc[index[i], 'deltaK'] = (index[i + 1] - index[i - 1]) / 2.0
        callAll.loc[index[0], 'deltaK'] = index[1] - index[0]
        callAll.loc[index[-1], 'deltaK'] = index[-1] - index[-2]
    index = putAll.index
    if len(index) < 3:
        putAll['deltaK'] = index[-1] - index[0]
    else:
        for i in range(1, len(index) - 1):
            putAll.loc[index[i], 'deltaK'] = (index[i + 1] - index[i - 1]) / 2.0
        putAll.loc[index[0], 'deltaK'] = index[1] - index[0]
        putAll.loc[index[-1], 'deltaK'] = index[-1] - index[-2]

    # me: 这里把F当成K_0了
    call = callAll[callAll.index > FF]
    put = putAll[putAll.index < FF]
    FF_idx = FF
    if put.empty:
        FF_idx = call.index[0]
        callComponent = call.CLOSE * call.deltaK / call.index / call.index
        sigma = (sum(callComponent)) * np.exp(T * R) * 2 / T
        sigma = sigma - (FF / FF_idx - 1) ** 2 / T
    elif call.empty:
        FF_idx = put.index[-1]
        putComponent = put.CLOSE * put.deltaK / put.index / put.index
        sigma = (sum(putComponent)) * np.exp(T * R) * 2 / T
        sigma = sigma - (FF / FF_idx - 1) ** 2 / T
    else:
        # me: 此时FF_idx的意义就是K_0
        FF_idx = put.index[-1]
        try:
            if len(putAll.loc[FF_idx, 'CLOSE']) > 1:
                # me: 计算Q(K_0), 但是不知道values[1]和[0]的含义
                put.loc[put.index[-1], 'CLOSE'] = (putAll.loc[FF_idx, 'CLOSE'].values[1] +
                                                   callAll.loc[FF_idx, 'CLOSE'].values[0]) / 2.0

        except:
            put.loc[put.index[-1], 'CLOSE'] = (putAll.loc[FF_idx, 'CLOSE'] + callAll.loc[FF_idx, 'CLOSE']) / 2.0

        callComponent = call.CLOSE * call.deltaK / call.index / call.index
        putComponent = put.CLOSE * put.deltaK / put.index / put.index
        sigma = (sum(callComponent) + sum(putComponent)) * np.exp(T * R) * 2 / T
        sigma = sigma - (FF / FF_idx - 1) ** 2 / T
    return sigma


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

    # get risk-free interest rates for both near-term and next-term options to expiration
    interpolated_shibor_rates = calc_interpolated_risk_free_interest_rates(options, vix_date)
    near_term_risk_free_rate = interpolated_shibor_rates[near_term_expiration_date]
    next_term_risk_free_rate = interpolated_shibor_rates[next_term_expiration_date]

    # time to maturity in % year
    vix_date_in_datetime_format = datetime.strptime(vix_date, '%Y/%m/%d')
    near_maturity = (near_term_expiration_date - vix_date_in_datetime_format).days / 365.0
    next_maturity = (next_term_expiration_date - vix_date_in_datetime_format).days / 365.0

    # get near term and next term options which are pandas dataframes
    near_term_options = options[pd.to_datetime(options.EXE_ENDDATE) == near_term_expiration_date]
    next_term_options = options[pd.to_datetime(options.EXE_ENDDATE) == next_term_expiration_date]

    # start calculating forward prices for near and next term options:

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
    # todo: comment, and why min()?
    near_atm_call_put_price_diff = (near_call_options.CLOSE - near_put_options.CLOSE)[near_atm_strike_price].min()
    next_atm_call_put_price_diff = (next_call_options.CLOSE - next_put_options.CLOSE)[next_atm_strike_price].min()

    # apply the formula for calculating forward prices (or called forward index level)
    near_forward_price = near_atm_strike_price + np.exp(near_maturity * near_term_risk_free_rate) * near_atm_call_put_price_diff
    next_forward_price = next_atm_strike_price + np.exp(next_maturity * next_term_risk_free_rate) * next_atm_call_put_price_diff

    # 计算不同到期日期权对于VIX的贡献
    # todo: refactor
    near_sigma = calSigmaSquare(near_term_options, near_forward_price, near_term_risk_free_rate, near_maturity)
    next_sigma = calSigmaSquare(next_term_options, next_forward_price, next_term_risk_free_rate, next_maturity)

    # 利用两个不同到期日的期权对VIX的贡献sig1和sig2，
    # 已经相应的期权剩余到期时间T1和T2；
    # 差值得到并返回VIX指数(%)
    w = (next_maturity - 30.0 / 365.0) / (next_maturity - near_maturity)
    vix = near_maturity * w * near_sigma + next_maturity * (1 - w) * next_sigma
    return 100 * np.sqrt(abs(vix) * 365.0 / 30.0)


ivix = []
for day in trade_day_dataset['DateTime']:
    ivix.append(calc_vix_index(day))
    break
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
