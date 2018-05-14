import os
import sys

path = os.path.dirname(__file__)
sys.path.append(path)

from trade1 import trade_days
from trade1 import trade_loop_back
from trade1 import trade_strategy
from trade1 import TSLA
from functools import reduce
if __name__ == '__main__':


    trade_day = trade_days.StockTradeDays(TSLA.price_array,TSLA.date_base,TSLA.date_array)
    # print(trade_day)
    trade_loop_back1 = trade_loop_back.TradeLoopBack(trade_day,trade_strategy.TradeStrategy1())
    trade_loop_back1.execute_trade()
    print('回测策略1 总盈亏为： {}%'.format(reduce(lambda a,b:a+b,trade_loop_back1.profit_array)*100))


    trade_loop_back2 = trade_loop_back.TradeLoopBack(trade_day,trade_strategy.TradeStrategy2())
    trade_loop_back2.execute_trade()
    print('回测策略2 总盈亏为： {}%'.format(reduce(lambda a, b: a + b, trade_loop_back2.profit_array) * 100))
