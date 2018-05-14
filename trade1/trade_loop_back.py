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
        self.trade_strategy = trade_strategy

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
