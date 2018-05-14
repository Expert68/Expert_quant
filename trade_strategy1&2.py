import six
from abc import ABCMeta,abstractmethod
from collections import namedtuple
from collections import OrderedDict
from functools import reduce
import matplotlib.pyplot as plt

class StockTradeDays(object):
    def __init__(self,price_array,start_date,date_array=None):
        # 私有价格序列
        self.__price_array = price_array
        #私有日期序列
        self.__date_array = self._init_days(start_date,date_array)
        #私有涨幅序列
        self.__change_array = self.__init_change()
        #进行ordereddict的组装
        self.stock_dict = self._init_stock_dict()
    def __init_change(self):
        """
        从price_array中生成change_array
        :return:
        """
        price_float_array = [float(price_str) for price_str in self.__price_array]
        #通过将时间平移形成两个错开的收盘价序列，通过zip()函数打包成一个新的序列
        #每个元素为相邻的两个收盘价格
        pp_array = [(price1,price2) for price1,price2 in zip(price_float_array[:-1],price_float_array[1:])]
        #round(x,3) 保留3位小数
        change_array = map(lambda pp: reduce(lambda a,b:round((b-a)/a,3),pp),pp_array)
        #list中insert()函数插入数据，将第一天的涨幅设置为0
        change_array.insert(0,0)
        return change_array
    def _init_days(self,start_date,date_array):
        """
        protect 方法，
        :param start_date: 初始日期
        :param date_array: 给定日期序列
        :return:
        """
        if date_array is None:
            #由start_date和self.price__price_array来确定日期序列
            # list = [1, 2, 3, 4, 5]
            # for item, _ in enumerate(list):
            #     print(('%s,%s') %(item,_))
            #         结果:
            #         0, 1
            #         1, 2
            #         2, 3
            #         3, 4
            #         4, 5
            # _代表的意义

            date_array = [str(start_date + ind) for ind,_ in enumerate(self)]
        else:
            #如果外面设置了date_array，就直接转换str类型组成新date_array
            date_array = [str(date) for date in date_array]

        return date_array

    def _init_stock_dict(self):
        """
        使用namedtuple，OrderedDict将结果合并
        :return:
        """
        stock_namedtuple = namedtuple('stock',('date','price','change'))
        #使用以被赋值的__date_array等进行OrderedDict的组装
        #Ordereddict 有序字典，key是由顺序的
        stock_dict = OrderedDict((date,stock_namedtuple(date,price,change)) for date,price,change in zip(self.__date_array,self.__price_array,self.__change_array))
        return stock_dict

    def filter_stock(self,want_up=True,want_calc_sum=False):
        """
        筛选结果子集
        :param want_up:
        :param want_calc_sum:
        :return:
        """
        #python中的三目表达式的写法
        filter_func = (lambda day:day.change > 0) if want_up else(lambda day: day.change < 0)
        #使用filter_func作为筛选函数
        want_days = filter(filter_func,self.stock_dict.values())
        if not want_calc_sum:
            return want_days
        #需要计算涨跌幅总和
        change_sum = 0.0
        for day in want_days:
            change_sum += day.change

        return change_sum

    def __str__(self):
        return str(self.stock_dict)

    __repr__ = __str__

    def __iter__(self):
        """
        通过代理stock_dict的迭代，yield元素
        :return:
        """
        for key in self.stock_dict:
            yield self.stock_dict[key]

    def __getitem__(self,ind):
        date_key = self.__date_array[ind]
        return self.stock_dict[date_key]
    def __len__(self):
        return len(self.stock_dict)





class tradestrategybase(six.with_metaclass(ABCMeta,object)):
    @abstractmethod
    def buy_strategy(self,*args,**kwargs):
        pass

    @abstractmethod
    def sell_strategy(self,*args,**kwargs):
        pass

class TradeStrategy1():
    """
    交易策略1：追涨策略：当股价上涨一个阀值默认为7%时，买入股票并持有s_keep_stock_threshold(20)天
    """
    s_keep_stock_threshold = 20
    def __init__(self):
        self.keep_stock_day = 0
        #7%的上涨幅度作为阀值
        self.__buy_change_threshold = 0.07
    def buy_strategy(self,trade_ind,trade_day,trade_days):
        if self.keep_stock_day == 0 and \
            trade_day.change > self.__buy_change_threshold:
            self.keep_stock_day +=1

        elif self.keep_stock_day > 0:
            self.keep_stock_day += 1

        if self.keep_stock_day >= \
            TradeStrategy1.s_keep_stock_threshold:
            self.keep_stock_day = 0

    @property
    def buy_change_threshold(self):
        return self.__buy_change_threshold

    @buy_change_threshold.setter
    def buy_change_threshold(self,buy_change_threshold):
        if not isinstance(buy_change_threshold,float):
            raise TypeError('buy_change_threshold must be float!')

        self.__buy_change_threshold = round(buy_change_threshold,2)



class TradeLoopBack(object):
    """
    交易回测系统
    """
    def __init__(self,trade_days,trade_strategy):
        """
        使用前面封装的StockTradeDays类和TradeStrategy1类
        :param trade_days:
        :param trade_strategy:
        """
        self.trade_days = trade_days
        self.trade_strategy = self.trade_strategy

        #交易盈亏结果序列
        self.profit_array = []

    def execute_trade(self):
        """
        执行交易回测
        :return:
        """
        for index,day in enumerate(self.trade_days):
            #以时间驱动，完成交易回测
            if self.trade_strategy.keep_stock_day > 0:
                self.profit_array.append(day.change)

            if hasattr(self.trade_strategy,'buy_strategy'):
                #买入策略执行
                self.trade_strategy.buy_strategy(index,day,self.trade_days)

            if hasattr(self.trade_strategy,'sell_strategy'):
                #买入策略执行
                self.trade_strategy.buy_strategy(index,day,self.trade_days)


class TradeStrategy2(tradestrategybase):
    """
    交易策略2：均值回复策略，当股价连续两个交易日下跌，
    且下跌幅度超过阀值默认 s_buy_change_threshold(-10%),
    买入股票并持有s_keep_stock_threshold(10)天
    """
    #买入后持有天数
    s_keep_stock_threshold = 10
    #下跌买入阀值
    s_buy_change_threshold = -0.10

    def __init__(self):
        self.keep_stock_day = 0

    def buy_strategy(self,trade_ind,trade_day,trade_days):
        if self.keep_stock_day == 0 and trade_ind >= 1:
            """
            当没有持有股股票的时候self.keep_stock_day == 0并且
            trade_ind >=1,不是交易开始的第一天，因为需要yesterday数据
            """
            #trade_day.change < 0  布尔判断：今天股价是否下跌
            today_down = trade_day.change < 0
            #昨天股价是否下跌
            yesterday_down = trade_days[trade_ind - 1].change < 0
            #两天总跌幅
            down_rate = trade_day.change + trade_days[trade_ind-1].change
            if today_down and yesterday_down and down_rate < TradeStrategy2.s_buy_change_threshold:
                #买入条件成立：连跌两天，跌幅超过s_buy_change_threshold，注意s_buy_change_threshold为负数
                self.keep_stock_day += 1
            elif self.keep_stock_day > 0:
                #self.keep_stock_day表示已经持有股票，持有股票天数递增
                self.keep_stock_day += 1

    def sell_strategy(self,trade_ind,trade_day,trade_days):
        if self.keep_stock_day >= TradeStrategy2.s_keep_stock_threshold:
            #当持有股票天数超过阀值s_keep_stock_threshold,卖出股票
            self.keep_stock_day = 0

    @classmethod
    def set_keep_stock_threshold(cls,keep_stock_threshold):
        cls.s_keep_stock_threshold = keep_stock_threshold

    @staticmethod
    def set_buy_change_threshold(buy_change_threshold):
        TradeStrategy2.s_buy_change_threshold = buy_change_threshold


