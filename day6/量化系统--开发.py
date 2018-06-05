'#######################################################################################################################'
"""
量化系统一般分为回测模块、实盘模块
    回测模块：首先交易者编写实现一个交易策略，它基于一段历史的交易数据，根据交易策略进行模拟买入卖出，策略中可以涉及买入规则、
    卖出规则、选股规则、仓位控制及滑点策略等，回测的目的是验证交易策略是否可行
    实盘模块：将回测通过的策略应用于每天的实时交易数据，根据策略发出的买入信号、卖出信号、进行实际的买入卖出操作
"""
"""
回测模块中最重要的组成部分是择时、选股：
择时：即什么时候投资
选股：即投资什么股票
只有在对的时间买入对的股票才能获利
"""
"""
量化系统之择时
以下代码AbuFactorBuyBreak为之前入门示例中讲述的，还贵交易法则中的N日去世突破策略在abu量化系统中作为一个买入因子的实现代码
"""
#需要继承自AbuFactorBuyBase
from abupy import AbuFactorBuyBase
class AbuFactorBuyBreak(AbuFactorBuyBase):
    def _init_self(self, **kwargs):
        #突破参数xd，如20、30、40天突破
        self.xd = kwargs['xd']
        #忽略连续创新高，比如买入后第2天又突破新高，忽略
        self.skip_days = 0
        #在输出生成的orders_pd中显示的名字
        self.factor_name = '{}:{}'.format(self.__class__.__name__,self.xd)

    def fit_day(self, today):
        day_ind = int(today.key)
        #忽略不符合买入日(统计周期内前xd天及最后一天)
        if day_ind < self.xd - 1 or day_ind >= self.kl_pd.shape[0]-1:
            return None
        if self.skip_days > 0:
            #执行买入订单后的忽略
            self.skip_days -= 1
            return None

        #今日的收盘价格达到xd天内最高价格则符合条件
        if today.close == self.kl_pd.close[day_ind - self.xd+1:day_ind+1].max():
            #把xd赋值给忽略买入日，即xd天内又再次创新高，也不买了
            self.skip_days = self.xd
            #生成买入订单
            return self.make_buy_order(day_ind)
        return None

#下面使用字典形式初始化buy_factors,首先针对一只股票的择时操作
from abupy import AbuFactorBuyBreak
from abupy import AbuBenchmark
#buy_factors 60日向上突破，42日向上突破两个因子
buy_factors = [{'xd':20,'class':AbuFactorBuyBreak},
               {'xd':42,'class':AbuFactorBuyBreak}]
benchmark = AbuBenchmark()

#benchmark的意义为参考基准，基准默认使用回测股票对应市场的大盘指数

"""
下面构建择时操作主类AbuPickTimeWorker,通过fit()函数执行回测，%time可以看到整个回测的用时情况：
"""
from abupy import AbuPickTimeWorker
from abupy import AbuCapital
from abupy import AbuKLManager

capital = AbuCapital(1000000,benchmark)
kl_pd_manager = AbuKLManager(benchmark,capital)
#获取特斯拉择时阶段的走势数据
kl_pd = kl_pd_manager.get_pick_stock_kl_pd('usTSLA')
abu_worker = AbuPickTimeWorker(capital,kl_pd,buy_factors,buy_factors,None)
abu_worker.fit()

"""
下面使用ABuTradeProxy.trade_summary()函数将abu_worker中生成的所有orders对象
进行转换及可视化，由图开发--1所示，由于AbuPickTimeWorker没有设置sell_factors,
所以所有的交易单子都一只保留没有卖出
orders_pd:所有交易的相关数据
action_pd:所有交易的行为数据
"""
from abupy import ABuTradeProxy
order_pd,action_pd,_  = ABuTradeProxy.trade_summary(abu_worker.orders,kl_pd,draw=True,show_info=True)

"""
最后将交易行为作用于资金上进行资金的走势模拟，如图所示为策略资金走势
"""
from abupy import ABuTradeExecute
#将action_pd作用在capital上，即开始涉及资金
ABuTradeExecute.apply_action_to_capital(capital,action_pd,kl_pd_manager)
#绘制资产曲线
capital.capital_pd.capital_blance.plot()

"""
卖出因子的实现
上面所有单子都没有成交的原因是没有卖出因子，下面首先实现类似买入策略的N日趋势突破策略AbuFactorSellBreak
当股价向下突破N日最低价格时卖出股票
"""
#需要继承自AbuFactorSellBase
from abupy import AbuFactorSellBase
class AbuFactorSellBreak(AbuFactorSellBase):
    def _init_self(self, **kwargs):
        #向下突破参数xd(x_days),比如20，30,40天突破
        self.xd = kwargs['xd']
        #在输出生成的orders_pd中显示名字
        self.sell_type_extra = '{}:{}'.format(self.__class__.__name__,self.xd)

    def fit_day(self, today, orders):
        day_ind = int(today.key)
        #今天的收盘价格达到xd天内最低价格则符合条件
        if today.close == self.kl_pd.close[day_ind-self.xd+1:day_ind+1].min():
            for order in orders:
                order.fit_sell_order(day_ind,self)
#同理，使用字典组装卖出因子

from abupy import AbuFactorSellBreak
#使用120天向下突破为卖出信号
sell_factor1 = {'xd':120,'class':AbuFactorSellBreak}
#继续使用之前的buy_factors,但不在使用AbuPickTimeWorker等零散的类操作，使用AbuPickTimeWorker.do_symbols_with_factors()函数，
#封装以上零散的操作，结果如图开发--3所示
from abupy import ABuPickTimeExecute
#只使用120天向下突破为卖出因子
sell_factors = [sell_factor1]
capital = AbuCapital(1000000,benchmark)
order_pd1,action_pd1,_  = ABuPickTimeExecute.do_symbols_with_same_factors(['usTSLA'],benchmark,buy_factors,sell_factors,capital,show=True)
ABuTradeExecute.apply_action_to_capital(capital,action_pd1,kl_pd_manager)
capital.capital_pd.capital_blance.plot()

"""
从图入门--3中可以看大，大多数的交易卖出因子都生效了，但是效果很不好，量化交易系统一般都会有止盈止损卖出策略，
下面使用真实波幅ATR作为最大止盈和最大止损的常数值，代码如下：
"""
"""
ATR又称为Average true range平均真实波动范围，简称ATR指标
ATR指标主要是用来衡量市场波动的强烈度，即为了显示市场变化率的指标
ATR指标的计算方法如下：
* TR = |最高价-最低价|,|最高价-昨收|,|昨收-最低价|中的最大值
* 真实波幅(ATR) = MA(TR,N)(TR的N日简单平均移动)
* 常用参数N的设置为14日或者21日
"""
class AbuFactorAtrNStop(AbuFactorSellBase):
    def _init_self(self, **kwargs):
        if 'stop_loss_n' in kwargs:
            #设置止损的ATR倍数
            self.stop_loss_n = kwargs['stop_loss_n']

            self.sell_type_extra_loss = '{}:stop_loss={}'.format(self.__class__.__name__,self.stop_loss_n)

        if 'stop_win_n' in kwargs:
            #设置止盈的ATR倍数
            self.stop_win_n = kwargs['stop_win_n']
            self.sell_type_extra_win = '{}:stop_win={}'.format(self.__class__.__name__, self.stop_win_n)

    def fit_day(self, today, orders):
        for order in orders:
            profit = today.close - order.buy_price
            stop_base = today.atr21 + today.atr14

            if hasattr(self,'stop_win_n') and profit > 0 \
                and profit > self.stop_win_n * stop_base:
                #满足止盈条件,卖出股票
                self.sell_type_extra = self.sell_type_extra_win
                order.fit_sell_order(int(today.key),self)

            if hasattr(self, 'stop_loss_n') and profit < 0 \
                    and profit < -self.stop_loss_n * stop_base:
                # 满足止损条件,卖出股票
                self.sell_type_extra = self.sell_type_extra_win
                order.fit_sell_order(int(today.key), self)

            self.sell_type_extra = self.sell_type_extra_loss
            order.fit_sell_order(int(today.key),self)


from abupy import AbuFactorAtrNStop
#趋势耿总策略止盈要大于止损设置值，这里分别为0.3和3.0
sell_factor2 = {'stop_loss_n':0.3,'stop_win_n':3.0,'class':AbuFactorAtrNStop}

#两个卖出因子并行同时生效
sell_factors = [sell_factor1,sell_factor2]
capital = AbuCapital(1000000,benchmark)
order_pd2,action_pd2,_  = ABuPickTimeExecute.do_symbols_with_same_factors(['usTSLA'],benchmark,buy_factors,sell_factors,capital,show=True)
ABuTradeExecute.apply_action_to_capital(capital,action_pd2,kl_pd_manager)
capital.capital_pd.capital_blance.plot()

"""
注意：
    1、上面的很多拟合优化操作在实际应用中是不可取的，比如最后使用的AbuFactorAtrNstop使交易盈利，以及之后即将说明的另一些手段
    使整体交易变好的做法，不应该因为某些特定股票或者特定交易修改参数或者添加因子等方式使结果变好，这样就是过拟合择时交易系统，在
    后面将通过示例讲解如何挑选参数及因子的选择问题
    2、ATR只是仓位控制以及止盈止损控制的手段之一，并不是所有仓位控制和止盈止损的方法都会用到ATR，应用ATR是因为这个概念比较好理解，
    当股价涨幅过高的时候必然会下调，当股价跌幅过大的是偶必然会上涨，所以一定要有止盈和止损。
"""

'#################################################################################################################################################'

"""
滑点买入，卖出价格的确定及策略的实现
    abu量化系统为非高频操作，所以生成的买入单子以及卖出单子产生的信号，均在下一个交易日执行，
    这种策略框架适合个人投资者，并非和大型基金私募去拼杀设备的好坏、拼杀速度，个人投资者不要妄想可以使用统计套利进行高频交易，
    比如跨市场期权套利，它们确实是最安全的套利方式，但所付出的高额成本也不是小资金能承受的，而且这种非高频操作也会有相当多的
    好处，比如可以使用机器学习甚至深度学习技术融入策略中
"""

"""
既然是第二天执行订单，那应该使用第二天的什么价格买入或者卖出呢？
默认的策略都是使用当天的均价买入卖出，示例如下：
"""
from abupy import AbuSlippageBuyBase
import numpy as np
from abc import ABCMeta
# g_open_dow_rate = 0.07
import six

#需要继承AbuSlippageBuyBase


# class AbuSlippageBuyMean(AbuSlippageBuyBase):
#     def fit_price(self):
#         if (self.kl_pd_buy.open / self.kl_pd_buy.preClose) < (1-g_open_dow_rate):
#             #开盘下跌g_open_down_rate以上，单子失效
#              return np.inf
#         #买入价格为当天价格
#         self.buy_price = np.mean([self.kl_pd_buy['high'],self.kl_pd_buy['low']])
#         return self.buy_price

#AbuSlippageBuyBase的代码如下：
# class AbuSlippageBuyBase(six.with_metaclass(ABCMeta,object)):
#     def __init__(self,kl_pd_buy,factor_name):
#         self.buy_price = np.inf
#         self.kl_pd_buy = kl_pd_buy
#         self.factor_name = factor_name
#
#     # @abstractmethod
#     def fit_price(self):
#         pass

"""
    卖出执行策略AbuSlippageSellMean基本与AbuSlippageBuyMean相同，当然也可以实现多种复杂的当日交易策略，
设置限价单，市价单，获取当日的分时数据，再次进行策略分析执行操作，但是如果回测数量足够多的情况下，比如
全市场回测，按照大数定理，这个均值执行其实是最好的模拟，而且简单、运行速度快
    上面的代码使用了一个小策略，即设置g_open_down_rate为0.07,即当当天开盘价格直接下探7%时，放弃买单
"""
"""
下面编写一个独立的Slippage(滑点)策略，且修改g_open_down_rate的值为0.02
示例如下：
"""
from abupy import AbuSlippageBuyBase

g_open_dow_rate = 0.02

class AbuSlippageBuymean(AbuSlippageBuyBase):
    def fit_price(self):
        if (self.kl_pd_buy.open / self.kl_pd_buy.preClose) < (1-g_open_dow_rate):
            #开盘下跌g_open_down_rate以上，单子失效
             return np.inf
        #买入价格为当天价格
        self.buy_price = np.mean([self.kl_pd_buy['high'],self.kl_pd_buy['low']])
        return self.buy_price

"""
通过buy_factors的字典对象中传入slippage便可以自行设置滑点类，由于上面的交易是60日突破产生的买单，
所以我们只修改60日突破的字典对象，执行后可以看到结果如图开发--4所示，过滤了两个60日突破的买单
"""
#针对60日使用AbuSlippageBuyMean
buy_factors2 = [{'slippage':AbuSlippageBuymean,'xd':60,'class':AbuFactorBuyBreak},{'xd':42,'class':AbuFactorBuyBreak}]

capital = AbuCapital(1000000,benchmark)
order_pd3,action_pd3,_ = ABuPickTimeExecute.do_symbols_with_same_factors(['usTSLA'],benchmark,buy_factors2,sell_factors,capital,show=True)
#输出结果如图开发--5所示








