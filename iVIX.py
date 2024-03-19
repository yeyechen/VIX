from datetime import datetime
import numpy as np
import pandas as pd
from scipy import interpolate

shibor_rate_dataset = pd.read_csv('shibor.csv', index_col=0, encoding='GBK')
options_dataset = pd.read_csv('options.csv', index_col=0, encoding='GBK')
trade_day_dataset = pd.read_csv('tradeday.csv', encoding='GBK')
true_ivix_dataset = pd.read_csv('ivixx.csv', encoding='GBK')


# ==============================================================================
# 开始计算ivix部分
# ==============================================================================
def calc_interpolated_risk_free_interest_rates(options, vix_date):
    """
    Parameters:
        options: 计算VIX的当天的options的数据
        vix_date: 用来计算VIX的日期

    Return：
        interpolated_shibor_rates：与到期日对应的年化期限(maturity)的risk free interest rates
    """
    vix_date_in_datetime_format = datetime.strptime(vix_date, '%Y/%m/%d')
    vix_date_in_str_format = vix_date_in_datetime_format.strftime("%Y-%m-%d")

    # me: sort in ascending order for all unique expiration dates
    expiration_dates = np.sort(options.EXE_ENDDATE.unique())

    # me: collect all maturities from 'date' to all expiration dates, into a dictionary.
    # {key: expiration date, value: maturity in % in year}
    maturities = {}
    for day in expiration_dates:
        day = pd.to_datetime(day)
        # % in year
        maturities[day] = float((day - vix_date_in_datetime_format).days) / 365.0

    # me: 选取最近的有数据的日期，如果没有当天interest rate数据，则选取最近日的interest rate数据
    latest_shibor_date = datetime.strptime(shibor_rate_dataset.index[0], "%Y-%m-%d")
    if vix_date_in_datetime_format >= latest_shibor_date:
        vix_date_shibor_rates = shibor_rate_dataset.iloc[0].values
    else:
        vix_date_shibor_rates = shibor_rate_dataset.loc[vix_date_in_str_format].values

    # collect interpolated rates from given shibor interest rate dataset as a dictionary
    # {key: expiration date, value: interpolated shibor rate}
    interpolated_shibor_rates = {}

    # me: standard interest rate periods from the data that we have
    interest_rate_periods = np.array([1.0, 7.0, 14.0, 30.0, 90.0, 180.0, 270.0, 360.0]) / 360.0
    # me: perform interpolation for different maturities
    cs = interpolate.CubicSpline(interest_rate_periods, vix_date_shibor_rates)
    min_period = min(interest_rate_periods)
    max_period = max(interest_rate_periods)

    for day in maturities.keys():
        # me: 与到期日对应的年化期限
        maturity = maturities[day]
        # 把值拉回到定义到区间内
        if maturities[day] > max_period:
            maturity = max_period * 0.99999
        elif maturities[day] < min_period:
            maturity = min_period * 1.00001

        # convert the interpolated interest rate into exact value
        interpolated_shibor_rates[day] = cs(maturity) / 100.0

    return interpolated_shibor_rates


def getHistDayOptions(options, vixDate):
    options = options.loc[vixDate, :]
    return options


def getNearNextOptExpDate(options, vixDate):
    # 找到options中的当月和次月期权到期日；
    # 用这两个期权隐含的未来波动率来插值计算未来30隐含波动率，是为市场恐慌指数VIX；
    # 如果options中的最近到期期权离到期日仅剩1天以内，则抛弃这一期权，改
    # 选择次月期权和次月期权之后第一个到期的期权来计算。
    # 返回的near和next就是用来计算VIX的两个期权的到期日
    """
    params: options: 该date为交易日的所有期权合约的基本信息和价格信息
            vixDate: VIX的计算日期
    return: near: 当月合约到期日（ps：大于1天到期）
            next：次月合约到期日
    """
    vixDate = datetime.strptime(vixDate, '%Y/%m/%d')
    optionsExpDate = list(pd.Series(options.EXE_ENDDATE.values.ravel()).unique())
    optionsExpDate = [datetime.strptime(i, '%Y/%m/%d %H:%M') for i in optionsExpDate]
    # me: 频繁的remove是因为想在optionsExpDate里向后平移，逻辑是对的
    near = min(optionsExpDate)
    optionsExpDate.remove(near)
    if near.day - vixDate.day < 1:  # me: 严谨来说，是要加month约束的，但是在vix里我们只考虑<30天内的期权，这里原作者偷懒了
        near = min(optionsExpDate)
        optionsExpDate.remove(near)
    nt = min(optionsExpDate)
    return near, nt


def getStrikeMinCallMinusPutClosePrice(options):
    # options 中包括计算某日VIX的call和put两种期权，
    # 对每个行权价，计算相应的call和put的价格差的绝对值，
    # 返回这一价格差的绝对值最小的那个行权价，
    # 并返回该行权价对应的call和put期权价格的差
    """
    params:options: 该date为交易日的所有期权合约的基本信息和价格信息
    return: strike: 看涨合约价格-看跌合约价格 的差值的绝对值最小的行权价
            priceDiff: 以及这个差值，这个是用来确定中间行权价的第一步
    """
    call = options[options.EXE_MODE == u"认购"].set_index(u"EXE_PRICE").sort_index()
    put = options[options.EXE_MODE == u"认沽"].set_index(u"EXE_PRICE").sort_index()
    callMinusPut = call.CLOSE - put.CLOSE
    strike = abs(callMinusPut).idxmin()
    priceDiff = callMinusPut[strike].min()
    return strike, priceDiff


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


def changeste(t):
    if t.month >= 10:
        str_t = t.strftime('%Y/%m/%d ') + '0:00'
    else:
        # me: 把月份中的0去掉了，2015/03/22 -> 2015/3/22
        str_t = t.strftime('%Y/%m/%d ')
        str_t = str_t[:5] + str_t[6:] + '0:00'
    return str_t


def calDayVIX(vixDate):
    # 利用CBOE的计算方法，计算历史某一日的未来30日期权波动率指数VIX
    """
    params：vixDate：计算VIX的日期  '%Y/%m/%d' 字符串格式
    return：VIX结果
    """

    # 拿取所需期权信息
    options = getHistDayOptions(options_dataset, vixDate)
    near, nexts = getNearNextOptExpDate(options, vixDate)
    shibor = calc_interpolated_risk_free_interest_rates(options, vixDate)
    R_near = shibor[datetime(near.year, near.month, near.day)]
    R_next = shibor[datetime(nexts.year, nexts.month, nexts.day)]

    str_near = changeste(near)
    str_nexts = changeste(nexts)
    optionsNearTerm = options[options.EXE_ENDDATE == str_near]
    optionsNextTerm = options[options.EXE_ENDDATE == str_nexts]
    # time to expiration
    vixDate = datetime.strptime(vixDate, '%Y/%m/%d')
    T_near = (near - vixDate).days / 365.0
    T_next = (nexts - vixDate).days / 365.0
    # the forward index prices
    nearPriceDiff = getStrikeMinCallMinusPutClosePrice(optionsNearTerm)
    nextPriceDiff = getStrikeMinCallMinusPutClosePrice(optionsNextTerm)
    near_F = nearPriceDiff[0] + np.exp(T_near * R_near) * nearPriceDiff[1]
    next_F = nextPriceDiff[0] + np.exp(T_next * R_next) * nextPriceDiff[1]
    # 计算不同到期日期权对于VIX的贡献
    near_sigma = calSigmaSquare(optionsNearTerm, near_F, R_near, T_near)
    next_sigma = calSigmaSquare(optionsNextTerm, next_F, R_next, T_next)

    # 利用两个不同到期日的期权对VIX的贡献sig1和sig2，
    # 已经相应的期权剩余到期时间T1和T2；
    # 差值得到并返回VIX指数(%)
    w = (T_next - 30.0 / 365.0) / (T_next - T_near)
    vix = T_near * w * near_sigma + T_next * (1 - w) * next_sigma
    return 100 * np.sqrt(abs(vix) * 365.0 / 30.0)


ivix = []
for day in trade_day_dataset['DateTime']:
    ivix.append(calDayVIX(day))
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
