import six
from abc import ABCMeta,abstractmethod

class tradestrategybase(six.with_metaclass(ABCMeta,object)):
    @abstractmethod
    def buy_strategy(self,*args,**kwargs):
        pass

    @abstractmethod
    def sell_strategy(self,*args,**kwargs):
        pass

class TradeStrategy1(tradestrategybase):
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

    def sell_strategy(self,trade_ind,trade_day,trade_days):
        if self.keep_stock_day >= \
            TradeStrategy1.s_keep_stock_threshold:
            #当持有股票天数超过阀值s_keep_stock_threshold,卖出股票
            self.keep_stock_day = 0

    @property
    def buy_change_threshold(self):
        return self.__buy_change_threshold

    @buy_change_threshold.setter
    def buy_change_threshold(self,buy_change_threshold):
        if not isinstance(buy_change_threshold,float):
            raise TypeError('buy_change_threshold must be float!')

        self.__buy_change_threshold = round(buy_change_threshold,2)


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







