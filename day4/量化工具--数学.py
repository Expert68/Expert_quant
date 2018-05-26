'#######################################################################################################################'
"""
量化工具：数学
数学不能控制金融市场，而心理因素才是控制市场的关键，更确切的说，只有掌握住群众的本能才能控制市场。研究人性会让人受益匪浅。
人们经常会犯过去犯过的相同的错误。当规则被参加者习以为常后，游戏的规则会随之发生变化。但是人性没变，从这个角度讲，规则从来没有变过
                                                                                                ---股票大作手回忆录
"""
"""
数学只是一种工具，帮助人们更好、更快地解决实际问题，在工程上使用的数学已经高度的抽象为工具，使用者只需要理解大概的原理和使用规则就可以了，
在有限的时间内做成一件复杂的事情并达成目标，必须要站在巨人的肩膀上，借助工具，分析好问题的主要矛盾和次要矛盾，不要重复地造轮子，尽量用
最高效最短的时间完成任务80%以上的需求之后，再次快速取舍评估问题
"""
'#######################################################################################################################'
"""
回归与差值
统计上度量拟合程度常用以下三种方式：
1、偏差绝对值之和最小(MAE)
2、偏差平方和最小(MSE)
3、偏差平方和开平方最小(RMSE)
*MAE的特点是使用简单容易理解
*MSE的特点是对误差极值得惩罚程度大(平方放大了大的误差)
*RMSE的特点是对误差的评估更好理解(平方后误差为向量norm)
"""
'#######################################################################################################################'

"""
线性回归
"""
from abupy import ABuSymbolPd
import numpy as np
import matplotlib.pyplot as plt
tsla_close = ABuSymbolPd.make_kl_df('usTSLA').close #make_kl_df函数的默认值为2
#x序列：0,1,2...len(tsla_close)
x = np.arange(0,tsla_close.shape[0])
#收盘价格序列
y = tsla_close.values
"""
下面通过statsmodel.api.OLS()函数实现一次多项式拟合计算，即最简单的 y=kx+b
使用summary()函数可以看到Method = Least Squares,即使用了最小二乘法，示例如下：
"""
import statsmodels.api as sm
from statsmodels import regression
def regress_y(y):
    y = y
    #x序列：0,1,2，...len(y)
    x = np.arange(0,len(y))
    x = sm.add_constant(x)
    # array([[1., 0.],
    #        [1., 1.],
    #        [1., 2.],
    #        [1., 3.],
    #        [1., 4.],
    #        ......
    #使用OLS做拟合
    model = regression.linear_model.OLS(y,x).fit()
    return model
model = regress_y(y)
# <class 'statsmodels.regression.linear_model.RegressionResultsWrapper'>
b = model.params[0]
k = model.params[1]
#y_fit = kx + b
y_fit = k*x + b
#这里x = np.arange(0,tsla_close.shape[0])
plt.plot(x,y)
plt.plot(x,y_fit,'r')
# summary()函数模型拟合概述，如下表所示，绘图结果如math--1所示
#                             OLS Regression Results
# ==============================================================================
# Dep. Variable:                      y   R-squared:                       0.592
# Model:                            OLS   Adj. R-squared:                  0.591
# Method:                 Least Squares   F-statistic:                     718.8
# Date:                Tue, 22 May 2018   Prob (F-statistic):           1.79e-98
# Time:                        22:35:07   Log-Likelihood:                -2505.4
# No. Observations:                 497   AIC:                             5015.
# Df Residuals:                     495   BIC:                             5023.
# Df Model:                           1
# Covariance Type:            nonrobust
# ==============================================================================
#                  coef    std err          t      P>|t|      [0.025      0.975]
# ------------------------------------------------------------------------------
# const        204.3050      3.358     60.838      0.000     197.707     210.903
# x1             0.3143      0.012     26.811      0.000       0.291       0.337
# ==============================================================================
# Omnibus:                        4.276   Durbin-Watson:                   0.034
# Prob(Omnibus):                  0.118   Jarque-Bera (JB):                3.443
# Skew:                           0.096   Prob(JB):                        0.179
# Kurtosis:                       2.641   Cond. No.                         572.
# ==============================================================================
"""
下面分别计算MAE，MSE，RMSE的值，来度量y和y_fit的误差
"""
#MAE值：
MAE = sum(np.abs(y-y_fit))/len(y)
print('偏差绝对值之和(MAE)={}'.format(MAE))
# 偏差绝对值之和(MAE)=30.481837761958086
MSE = sum(np.square(y-y_fit))/len(y)
print('偏差绝对值之和(MAE)={}'.format(MSE))
# 偏差绝对值之和(MAE)=1445.3477631329977
RMSE = np.sqrt(sum(np.square(y-y_fit))/len(y))
print('偏差绝对值之和(RMSE)={}'.format(RMSE))
# 偏差绝对值之和(RMSE)=38.01772958940339
"""
上述度量的计算，更常用的方法是直接使用sklearn.metrics模块下的度量方法
"""
from sklearn import metrics
print('偏差绝对值之和(MAE)={}'.format(metrics.mean_absolute_error(y,y_fit)))
print('偏差绝对值之和(MAE)={}'.format(metrics.mean_squared_error(y,y_fit)))
print('偏差绝对值之和(RMSE)={}'.format(np.sqrt(metrics.mean_squared_error(y,y_fit))))
# 得到的结果如下：
# 偏差绝对值之和(MAE)=30.48183776195808
# 偏差绝对值之和(MAE)=1445.3477631329974
# 偏差绝对值之和(RMSE)=38.01772958940338
#可以看到两者得到的结果是一致的
'#######################################################################################################################'

"""
多项式回归
观察上面的误差值，由于一次线性回归所以误差很大，多项式回归拟合最简单的方式就是使用np.polynomial()函数
通过以下代码计算1~9次多项式回归，计算MSE值，可以看到随着poly的增大，MSE的值逐步降低，结果如图math--2所示
"""

import itertools
#生成9个subplots 3*3
_,axs = plt.subplots(nrows=3,ncols=3,figsize=(15,15))
# axs.shape为(3,3)
# axs=array([[<matplotlib.axes._subplots.AxesSubplot object at 0x00000120CFFE9438>,
#         <matplotlib.axes._subplots.AxesSubplot object at 0x00000120CFF15780>,
#         <matplotlib.axes._subplots.AxesSubplot object at 0x00000120CFF46DA0>],
#        [<matplotlib.axes._subplots.AxesSubplot object at 0x00000120CFF1E630>,
#         <matplotlib.axes._subplots.AxesSubplot object at 0x00000120D00626D8>,
#         <matplotlib.axes._subplots.AxesSubplot object at 0x00000120D05E7CC0>],
#        [<matplotlib.axes._subplots.AxesSubplot object at 0x00000120D0627320>,
#         <matplotlib.axes._subplots.AxesSubplot object at 0x00000120D0657E80>,
#         <matplotlib.axes._subplots.AxesSubplot object at 0x00000120D068A9B0>]],
#       dtype=object)
#将3*3转换成一个线性list
axs_list = list(itertools.chain.from_iterable(axs))
# axs_list=[<matplotlib.axes._subplots.AxesSubplot at 0x120cffe9438>,
#  <matplotlib.axes._subplots.AxesSubplot at 0x120cff15780>,
#  <matplotlib.axes._subplots.AxesSubplot at 0x120cff46da0>,
#  <matplotlib.axes._subplots.AxesSubplot at 0x120cff1e630>,
#  <matplotlib.axes._subplots.AxesSubplot at 0x120d00626d8>,
#  <matplotlib.axes._subplots.AxesSubplot at 0x120d05e7cc0>,
#  <matplotlib.axes._subplots.AxesSubplot at 0x120d0627320>,
#  <matplotlib.axes._subplots.AxesSubplot at 0x120d0657e80>,
#  <matplotlib.axes._subplots.AxesSubplot at 0x120d068a9b0>]
#1~9次多项式回归
poly = np.arange(1,10,1)
for p_cnt, ax in zip(poly,axs_list):
    #使用polynomial.chebyshev.chebfit(x,y,p_cnt)
    p = np.polynomial.Chebyshev.fit(x,y,p_cnt)
    #使用p直接对x序列代入即得到拟合结果序列
    y_fit = p(x)
    #度量mse值
    mse = metrics.mean_squared_error(y,y_fit)
    #使用拟合次数和mse误差大小设置标题
    ax.set_title('{} ploy MSE={}'.format(p_cnt,mse))
    ax.plot(x,y,'g',x,y_fit,'r.')

"""
回归拟合在量化交易中有着多种多样的用途,比如想要从1000只股票中发现走势最相似的股票，但如果直接使用原始数据，
会有很多极值噪音而干扰最终的结果。这时可以先使用多项式回归数据，使用拟合后的数据再次进行相似度计算。
"""
'#######################################################################################################################'

"""
蒙特卡罗方法与凸优化
蒙特卡罗方法(Monte Carlo Method),也称为统计模拟方法，是20世纪40年代中期由于科学技术的发展和电子计算机的发明，
而被提出的一种以概率统计理论为指导的一类非常重要的数值计算方法
该方法是指使用随机数(或更常见的伪随机数)来解决很多计算问题的方法
凸优化是指一种比较特殊的优化，是指求取最小值得目标函数作为凸函数的一类优化问题
其中，目标函数为凸函数且定义域为凸集的优化问题称为无约束凸优化问题，而目标函数和不等式约束函数均为凸函数，等式约束函数为仿射函数，
并且定义域为凸集的优化问题为约束优化问题
"""
"""
在量化交易领域寻找最优参数是常见的需求，如针对多因子的因子重要程度匹配、仓位管理的买入配比、模型的最优参数选择等，
可以说在量化交易领域中一定要掌握的数学工具就是寻找最优参数，凸优化与蒙特卡罗方法是寻找最优参数的方法中最普遍使用的两种方法
"""
"""
人一生的追求到底能带来多少幸福
我们应该怎样度过并不长的一生才最好呢？下面将使用蒙特卡罗方法与凸优化方法，计算应该怎样度过我们的一生才算最幸福的
由于是数学方法，必然需要一个数学模型，下面首先构建我们充满追求的一生的数学模型：
"""
"""
(1)Person人类，初始人能活75，一生会积累幸福、财富权利，live_one_day()函数为度过一天，参数seek为追求什么，代码如下：
"""
from abc import ABCMeta,abstractmethod
import six
#每个人平均寿命期望是75年，约75*365=27375天
K_INIT_LIVING_DAYS = 27375
class person(object):
    """
    人类
    """
    def __init__(self):
        self.living = K_INIT_LIVING_DAYS
        self.happiness = 0
        self.wealth = 0
        self.fame = 0
        self.living_day = 0

    def live_one_day(self,seek):
        """
        每天只能进行一个seek，这个seek决定了一个人今天的追求是什么，得到了什么
        seek的类型下面将编写BaseSeekDay
        :param seek:
        :return:
        """
        consume_living,happiness,wealth,fame = seek.do_seek_day()
        self.living -= consume_living
        self.wealth += wealth
        self.wealth += wealth
        self.fame += fame
        self.living_day += 1
"""
BaseSeekDay为追求xxx的一天基类，do_seek_day()函数的返回值为这一天的追求结果，代码如下：
"""
class BaseSeekDay(six.with_metaclass(ABCMeta,object)):
    def __init__(self):
        self.living_consume = 0
        self.happiness_base = 0
        self.wealth_base = 0
        self.fame_base = 0
        self.living_factor = [0]
        self.happiness_factor = [0]
        self.wealth_factor = [0]
        self.fame_factor = [0]
        self.do_seek_day_cnt = 0
        self._init_self()

    @abstractmethod
    def _init_self(self,*args,**kwargs):
        pass
    @abstractmethod
    def _gen_living_days(self,*args,**kwargs):
        pass

    def do_seek_day(self):
        """
        每一天的追求具体seek
        :return:
        """
        if self.do_seek_day_cnt >= len(self.living_factor):
            consume_living = self.living_factor[-1] * self.living_consume
        else:
            consume_living = self.living_factor[self.do_seek_day_cnt] * self.living_consume

        if self.do_seek_day_cnt >= len(self.happiness_factor):
            happiness = self.happiness_factor[-1] * self.happiness_base
        else:
            happiness = self.happiness_factor[self.do_seek_day_cnt] * self.happiness_base

        if self.do_seek_day_cnt >= len(self.wealth_factor):
            wealth = self.wealth_factor[-1] * self.wealth_base
        else:
            wealth = self.wealth_factor[self.do_seek_day_cnt] * self.wealth_base

        if self.do_seek_day_cnt >= len(self.fame_factor):
            fame = self.fame_factor[-1] * self.fame_base
        else:
            fame = self.fame_factor[self.do_seek_day_cnt] * self.wealth_base

        self.do_seek_day_cnt += 1
        return consume_living,happiness,wealth,fame

def regular_mm(group):
    return (group-group.min()) / (group.max()-group.min())


class HealthSeekDay(BaseSeekDay):
    """
    HealthSeekDay追求健康长寿的一天：
    形象：健身，旅游，娱乐，做感兴趣的事情
    抽象：追求健康长寿
    """
    def _init_self(self):
        self.living_consume = 1
        self.happiness_base = 1
        self._gen_living_days()

    def _gen_living_days(self):
        days = np.arange(1,12000)
        living_days = np.sqrt(days)
        self.living_factor = regular_mm(living_days) * 2 -1
        self.happiness_factor = regular_mm(days)[::-1]



'#######################################################################################################################'

"""
线性代数
使用ABuSymbolPd.make_kl_df同时获取多个股票的交易数据，my_stock_df是pandas三维面板数据Panel，
通过Panel轴向变换生成多只股票的收盘价格DataFrame对象my_stock_df_close
"""
from abupy import ABuSymbolPd
#获取多只股票数据组成panel
my_stock_df = ABuSymbolPd.make_kl_df(
    ['usBIDU','usGOOG','usFB','usAAPL','us.IXIC'],n_folds=2
)
#变化轴向，形成新的切面
my_stock_df = my_stock_df.swapaxes('items','minor')
my_stock_df_close = my_stock_df['close'].dropna(axis=0)

my_stock_df_close.tail()
#              us.IXIC  usAAPL  usBIDU    usFB   usGOOG
# 2018-05-16  7398.295  188.18  284.07  183.20  1081.77
# 2018-05-17  7382.470  186.99  279.68  183.76  1078.59
# 2018-05-18  7354.339  186.31  253.01  182.68  1066.36
# 2018-05-21  7394.036  187.63  240.51  184.49  1079.58
# 2018-05-22  7378.455  187.16  239.97  183.80  1069.73
#以下代码将收盘价格数据标准化后可视化显示
def regular_std(group):
    #z-score规范化也称零-均值规范化
    return (group - group.mean())   / group.std()
my_stock_df_close_std = regular_std(my_stock_df_close)
my_stock_df_close_std.plot()
# 得到的结果如图math--3所示
'#######################################################################################################################'

"""
矩阵基础知识
矩阵Matrix是一个按照长方阵列排列的复数或实数集合
线性代数中有很多基础方法只适用于方阵
"""







