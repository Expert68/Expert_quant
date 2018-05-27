'#######################################################################################################################'
"""
趋势跟踪与均值回复
趋势跟踪与均值回复是很多量化策略的理论基础
1、趋势跟踪：
    趋势跟踪模型里，假设之前的价格的上涨预示着之后一段时间内也会上涨，很多交易策略都是围绕着趋势跟踪模型，比如各种向上突破信号、分批
    跟随趋势建仓策略等交易策略。使用趋势跟踪一定要做好止损，保护好资金，要认识到趋势跟踪策略将导致胜率降低，即亏损的次数比盈利的次数多，
    但每次盈利要高于每次亏损
2、均值回复
    均值回复模型中，假设之前的股价上涨(下跌)只是暂时的，价格会回复都一个相对正常的水平，也就是说随后的一段时间内股价将下跌(上涨)，
    它的理论依据为价格将围绕价值上下波动。算法交易之父托马斯.彼得菲最早就是利用均值回复策略编写自动化交易程序，均值回复模型属于统计套利的一种
3、趋势心理学
    每一个瞬间的股票价格都是全体交易者对价值所达成的一种瞬间共识，它代表某个特定股票的瞬间价值的投票结果。大家可以选择买进或卖出，
    表达他的看法(投票)，或者不交易观望(弃权票)
4、趋势跟踪 VS 均值回复
    趋势跟踪和均值回复是两种截然相反的量化策略，使用模型的时候，判断当前时间序列是会形成趋势还是服从均值回复非常重要，因为需要根据它来
    判断使用哪种模型
    均值回复策略比趋势跟踪策略跟容易让普通交易者接受，这是因为：
    *一些交易者把自己设定为价值投资者，他们设想初入股市的人都喜欢在股票价格大幅下跌后选择买入，而不敢再上涨中买入，这是因为人性的本质
     导致的：交易者在股票下跌很多后买入股票，会有一种错觉，反正有很多人买的价格比我高，我不害怕，要死大家一起死，相反交易者在股票突破后
     会不敢买入股票的想法是不能再最高点买入股票，害怕买入后股票下跌
    *趋势跟踪可以理解为追涨杀跌，很多人认为这是一种投机行为，非正途，其实最主要原因是趋势跟踪的成功率普遍低于50%，他们认为不确定性太大，
     而真实市场上唯一确定的东西只有手续费，其他的都是概率，我们做量化交易的人，一定要否认确定性，量化交易不是预测未来，只是利用概率
     以及人性的弱点等，挖掘优势，提升优势

量化交易的基础还是交易，不会任何没有在市场中经历过大的亏损的人能够做好量化策略，交易者在下单时都会有一种兴奋感，这种兴奋感随着
风险越大(收益越大)程度越高，尤其是那些经历过期权期货交易的人，都会体验过这种兴奋、激情以及对未来美好的种种幻想，并且当幻想破灭
的那一刻，面对现实的无奈与懊悔，真正合格的量化交易者也应该正确认识到这种现象，即可以做到面对盈利及亏损都无过激的感觉，只是概率
上的数字，实际上这种交易减少了交易所带来的快感，但是同时也规避了人性的贪婪和恐惧，并且得以用正确的交易系统贯穿整个交易，为了在
市场中存活，就应该放弃这种乐趣
"""
#均值回复策略
from abupy import ABuSymbolPd
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
tsla_df = ABuSymbolPd.make_kl_df('usTSLA',n_folds=4)
print(tsla_df.head())
#              close    high     low  p_change    open  pre_close   volume  \
# 2016-05-24  217.91  218.74  215.18     0.782  216.60     216.22  3013843
# 2016-05-25  219.58  221.36  216.51     0.766  217.91     217.91  3132615
# 2016-05-26  225.12  225.26  219.05     2.523  220.50     219.58  4072424
# 2016-05-27  223.04  225.93  220.75    -0.924  224.99     225.12  3650272
# 2016-05-31  223.23  224.75  221.50     0.085  223.04     223.04  2789002
#                 date  date_week   atr21   atr14  key
# 2016-05-24  20160524          1  3.5600  3.5600    0
# 2016-05-25  20160525          2  4.3876  4.4064    1
# 2016-05-26  20160526          3  5.4705  5.5309    2
# 2016-05-27  20160527          4  5.7228  5.7815    3
# 2016-05-31  20160531          1  5.2185  5.2175    4
sns.set_context(rc={'figure:figsize':(14,7)})
sns.regplot(x=np.arange(0,tsla_df.shape[0]),y=tsla_df.close.values,marker='+')
# 结果如图入门--1所示
"""
以下代码将实现一个简单的均值回复策略，首先分割训练测试集，如图入门--2所示
因为去年是美股牛市，股价已经远离的正常价值，所以选取4年的数据
"""
#选取头2年([:504])作为训练数据，美股交易中一年的交易日有252天
train_kl = tsla_df[:504]
#后两年([504:])作为回测数据
test_kl = tsla_df[504:]
#分别画出两部分数据的收盘价格曲线
#下面代码在pycharm中无法运行，是因为pycharm中会将np.array([train_kl.close.values, test_kl.close.values])识别成
# 一个2行的矩阵，所以这里用join的方法得到相应的结果
# tmp_df = pd.DataFrame(np.array([train_kl.close.values, test_kl.close.values]).T,columns=['train', 'test'])
tmp_df = pd.DataFrame(train_kl.close.values,columns=['train'])
tmp_df = tmp_df.join(pd.DataFrame(test_kl.close.values,columns=['test']),how='outer')

tmp_df[['train','test']].plot(subplots=True,grid=True,figsize=(14,7))
# 结果如图入门--4所示
"""
策略总体思路如下：
    1、将两年收盘价格分开，一年收盘价格作为训练数据，另一年的收盘价格作为回归测试数据
    2、训练数据通过收盘价格的均值和标准差构造买入信号与卖出信号
    3、将训练数据计算出信号代入回归测试数据，对比策略收益结果与基准收益结果
"""
#美股交易一年的交易日有252天，1个月21天
"""
下面计算出训练数据(头一年)收盘价格的均值和标准差，通过它们构造信号阀值，当时间序列触及买入信号时买入股票
当时间序列触及卖出信号阀值时卖出股票
"""
#训练数据的收盘价格均值
close_mean = train_kl.close.mean()
#训练数据的收盘价格标准差
close_std = train_kl.close.std()
#构造卖出信号阀值
sell_signal = close_mean + close_std/3
#构造买入信号阀值
buy_signal = close_mean - close_std/3
#可视化训练数据卖出信号阀值，买入信号阀值及均值线
plt.figure(figsize=(14,7))
train_kl.close.plot()
#水平线，买入信号线，lw代表线的粗度
plt.axhline(buy_signal,color='r',lw=3)
plt.axhline(close_mean,color='black',lw=1)
plt.axhline(sell_signal,color='g',lw=3)
plt.legend(['train close','buy_signal','close_mean','sell_signal'],loc='best')
plt.show()

#将卖出信号阀值及买入信号阀值代入回归测试数据可视化
plt.figure(figsize=(14,7))
#测试集收盘价格可视化
test_kl.close.plot()
plt.axhline(buy_signal,color='r',lw=3)
plt.axhline(close_mean,color='black',lw=1)
plt.axhline(sell_signal,color='g',lw=3)
plt.legend(['test close','buy_signal','close_mean','sell_signal'],loc='best')
plt.show()

print('买入信号阀值：{} 卖出信号阀值: {}'.format(buy_signal,sell_signal))
# 买入信号阀值：221.82782329937191 卖出信号阀值: 247.9380497165012
"""
下面通过：
    1、通过buy_signal和sell_signal构建操作信号
    2、将操作信号转化为持股状态
"""
# 构建买入信号signal=1
buy_index = test_kl[test_kl['close'] <= buy_signal].index
#行赋值操作，将找到的买入时间系列的信号设置为1，代表买入操作
test_kl.loc[buy_index,'signal'] = 1
#去年是美股牛市，所以在测试数据中没有买入也没有卖出信号
# 从输出中可以看到多了一列signal，符合买入条件的行数据都被赋予1，其他的为NAN
# 构建卖出信号signal=0
#寻找测试数据中满足卖出条件的时间序列
sell_index = test_kl[test_kl['close'] >= sell_signal].index
#将找到的卖出时间系列的信号设置为0，代表卖出操作
test_kl.loc[sell_index,'signal'] = 0
"""
上面添加了新列signal代表信号将要触发的操作，这里假设都是全仓操作，即一旦第一个信号成交后，
后面虽然仍有信号发出，但由于买入全仓即没有钱再买了
同理，卖出信号由于都卖了，所以连续的信号只有第一个是由实际的操作意思
"""
"""
下一步将操作转化为持股状态得到一个新的列数据，代码如下：
"""
#假设为全仓操作，signal=keep,1代表买入，0代表空仓
test_kl['keep'] = test_kl['signal']  #增加新列并赋值的操作
#将keep列中的NAN使用向下填充的方式填充，结果使keep可以代表最终的交易持股状态
test_kl['keep'].fillna(method='ffill',inplace=True)
"""
上面使用fillna()函数向下填充NAN可以理解为一旦状态被设置为1(买入持有),那么只有遇到0(卖出空仓时)
keep状态才会改变，否则向下的所有NAN都应该与其前面的元素保持一致
"""
"""
接下来：
    1、计算基准收益
    2、计算使用均值回复策略的收益
    3、可视化收益的情况对比

计算基准收益：
    计算的目的是为了与使用策略后的收益进行对比，所以新加入数据列benchmark_profit来计算每一天的收益
    基准收益简单来说就是，从时间序列第一天开始就持有股票，直到时间序列的最后一天(即从第一天开始就加入群体直到最后一天)
"""
test_kl['benchmark_profit'] = np.log(test_kl['close']/test_kl['close'].shift(1))
#为了说明np.log()函数的意义，添加了benchmark_profit2,只为了对比得到的结果是否一致
test_kl['benchmark_profit2'] = test_kl['close']/test_kl['close'].shift(1) - 1
#可视化对比两种方式计算出的profit，可见，两者的结果时一致的
test_kl[['benchmark_profit','benchmark_profit2']].plot(subplots=True,grid=True,figsize=(14,7))

#输出结果如图入门--3所示,benchmark_profit和benchmark_profit两者的结果基本一致，但数据上有略微的差别
#test_kl['close'].shift(1)的作用是移动股价序列
#股价序列移动前：
print(test_kl['close'][:5])
# 2016-05-26    225.12
# # 2016-05-27    223.04
# # 2016-05-31    223.23
# # 2016-06-01    219.56
# # 2016-06-02    218.96
# # Name: close, dtype: float64
#使用shift(1)将股价序列向后移动一个单位后：
print(test_kl['close'].shift(1)[:5])
# 2016-05-26       NaN
# 2016-05-27    225.12
# 2016-05-31    223.04
# 2016-06-01    223.23
# 2016-06-02    219.56
# Name: close, dtype: float64
"""
shift(1)是对序列的value再index不变的情况下向后移动一个单位，所以上述代码：
test_kl['close']/test_kl['close'].shift(1)=今日收盘价格序列/昨日收盘价格序列
"""
#假设今日收盘价格220，昨日收盘价格218，使用np.log()函数计算得到结果如下：
np.log(220/218)
# 0.009132483563272474
#而使用
print(220/218 - 1.0)
# 0.00917431192660545
#两者结果不一样但是非常相近，这是由于在log(x/y) 在x和y十分相近的时候约等于 x/y  可以由拉格朗日展开式证明
"""
以下代码计算使用策略后的收益，test_kl['keep']列是一个只有元素1和0的数据列，
用它乘以test_kl['benchmark_profit']的结果，就类似于创建了一个滤波器，这个滤波器过滤输入信号为0的结果，
得到的收益结果图如图入门--4所示
"""
test_kl['trend_profit'] = test_kl['keep'] * test_kl['benchmark_profit']
test_kl['trend_profit'].plot(figsize=[14,7])
#最后将基准收益和策略收益放在一起可视化查看对比，如图入门--5所示
test_kl[['benchmark_profit','trend_profit']].cumsum().plot(grid=True,figsize=(14,7))
"""
如图入门--5所示，仔细观察就能更好的理解本章开始阶段所说的量化策略产生的信号就是选择什么时候加入群体
什么时候离开群体
"""

"""
实例2：趋势跟踪策略
《海龟交易法则》是量化经典书籍中的经典作品，其中介绍过一种趋势跟踪策略，即N日趋势突破策略：
    趋势突破定义为当天收盘价格超过N天内的最高价或最低价，超过最高价格作为买入信号，认为上升趋势成立买入股票持有，超过最低价格作为卖出信号
    下面代码实现了海龟交易法则这种趋势跟踪策略，
    设定海龟趋势突破规则的两个主要参数如下：
    N1:当天收盘价格高于N1天内最高价格作为买入信号，认为上升趋势成立买入股票；
    N2：当天收盘价格低于N2天内最低价格作为卖出信号，认为下跌趋势成立卖出股票
    N1大于N2的原因是为了打造一个非均衡胜负收益及非均衡胜负比例环境，这一点很重要，因为量化的目标结果就是非均衡(我们想要赢得的钱比输出的钱多)
"""
#当天收盘价格超过N1天内最高价格作为买入信号
N1=42
#当天收盘价格超过N2天内最低价格作为卖出信号
N2=21
#本例中计算N天内最高值将使用pd.rolling_max()函数，使用示例及详解如下：
demo_list = np.array([1,2,1,1,100,1000])
#对示例序列以3个为一组，寻找每一组中的最大值
pd.rolling_max(demo_list,window=3)
# 输出如下：array([  nan,   nan,    2.,    2.,  100., 1000.])
"""
下面继续使用特斯拉4年内的股票走势数据tsla_df,下面加入新的数据列n1_high,代表N1天内最高价格序列：
"""
#通过rolling_max()方法计算最近N1个交易日的最高价
tsla_df['n1_high'] = pd.rolling_max(tsla_df['high'],window=N1)
tsla_df.head()
"""
可以发现上面新加入列n1_high的前N1个都是nan，因为需要从第N1个开始计算最大价格值，本例将使用pd.expanding_max()函数方法得到的值
填充前N1行n1_high数据
pd.expanding_max()函数的操作即为从序列第一个数据开始依次寻找目前出现过得最大值，示例如下:
"""
demo_list = np.array([1,2,1,1,100,100])
pd.expanding_max(demo_list)
# 输出如下：array([  1.,   2.,   2.,   2., 100., 100.])
#下面利用pd.expanding_max()填充n1_high前N1行数据：
expan_max = pd.expanding_max(tsla_df['close'])
#fillna()使用序列对应的expan_max
tsla_df['n1_high'].fillna(value=expan_max,inplace=True)
tsla_df.head()
#结果如下所示：
#              close    high     low  p_change    open  pre_close   volume  \
# 2014-05-28  210.24  212.77  205.26    -0.624  210.02     211.56  5496278
# 2014-05-29  210.24  212.49  207.72     0.000  210.57     210.24  3694596
# 2014-05-30  207.77  214.80  207.02    -1.175  210.30     210.24  5586068
# 2014-06-02  204.70  209.35  201.67    -1.478  207.33     207.77  4668115
# 2014-06-03  204.94  208.00  202.59     0.117  203.49     204.70  3866182
#
#                 date  date_week   atr21   atr14  key  n1_high
# 2014-05-28  20140528          2  7.5100  7.5100    0   210.24
# 2014-05-29  20140529          3  6.0748  6.0421    1   210.24
# 2014-05-30  20140530          4  6.6981  6.7060    2   210.24
# 2014-06-02  20140602          0  7.2350  7.2763    3   210.24
# 2014-06-03  20140603          1  6.7973  6.7894    4   210.24
#下面使用类似的方式构建N2天内最低价格卖出信号n2_low:
#rolling_min()函数和rolling_max()函数类似
tsla_df['n2_low'] = pd.rolling_min(tsla_df['low'],window=N2)
expan_min = pd.expanding_min(tsla_df['close'])
tsla_df['n2_low'].fillna(value=expan_min,inplace=True)
#下面根据突破的定义来构建signal列：
#当天的收盘价格超过N天内的最高价或最低价，超过最高价格作为买入信号买入股票持有
buy_index = tsla_df[tsla_df['close']>tsla_df['n1_high'].shift(1)].index
tsla_df.loc[buy_index,'signal'] = 1
#当天收盘价格超过N天内的最高价格或最低价格，超过最低价格作为卖出信号
sell_index = tsla_df[tsla_df['close']<tsla_df['n2_low'].shift(1)].index
tsla_df.loc[sell_index,'signal'] = 0
#筛选条件 今天的收盘价格>截止到昨天的最高价格 和 今天的收盘价格 < 截止到昨天的最低价格
#下面使用饼图显示在整个交易中信号的产生情况，可以发现买入信号比卖出信号多
#如下图所示
tsla_df.signal.value_counts().plot(kind='pie',figsize=(5,5))
# 1.0    54
# 0.0    53
# Name: signal, dtype: int64
"""
下面的过程为：
    1、讲操作信号转化为持股状态，得到一个新的列数据
    2、计算基准收益
    3、计算使用趋势突破策略的收益
    4、可视化收益的情况对比图

将信号操作序列移动一个单位，代表第二天再执行操作信号，转换得到的持股状态，
这列不进行shift(1)操作也可以，代表信号产生当天执行，但是由于收盘价格是在收盘后才确定的，
计算突破使用了收盘价格，所以使用shift(1)会更接近真实情况
"""
tsla_df['keep'] = tsla_df['signal'].shift(1)
tsla_df['keep'].fillna(method='ffill',inplace=True)
tsla_df['benchmark_profit'] = np.log(tsla_df['close']/tsla_df['close'].shift(1))
tsla_df['trend_profit'] = tsla_df['keep'] * tsla_df['benchmark_profit']
tsla_df[['benchmark_profit','trend_profit']].cumsum().plot(grid=True,figsize=(14,7))
#结果如图入门--6所示
"""
成功的交易，无非就是在低点买入特定的股票，然后再高点卖出(选股，择时)
量化交易的有事就是利用统计学寻找一些特定的市场行为，这些行为会再时间序列中重复出现，
投资者应当捕捉这些行为，提高自己在赌局中的优势，想办法提高胜率
当身在赌局中时，如果没有自己的优势，那么一定处于不利地位
"""


