import os
import sys
sys.path.append(os.path.dirname(__file__))
print(sys.path)
import trade_strategy
from abupy import ABuSymbolPd
"""
标准库中的itertools提供了很多生成循环器的工具，其中很重要的用途是生成集合中所有可能方式的元素排列组合
在量化数据处理过程中经常需要使用itertools来完成数据的各种排列组合以寻找最有参数
"""
import itertools
"""
(1)permutations()函数，考虑顺序组合元素：
"""


items = [1,2,3]
for item in itertools.permutations(items):
    print(item)
    # (1, 2, 3)
    # (1, 3, 2)
    # (2, 1, 3)
    # (2, 3, 1)
    # (3, 1, 2)
    # (3, 2, 1)

"""
(2)combinations()函数，不考虑属性，不放回数据
"""
for item in itertools.combinations(items,2):
    print(item)
    # (1, 2)
    # (1, 3)
    # (2, 3)

"""
(3)combinations_with_replacement()函数，不考虑属性，有放回数据
"""
for item in itertools.combinations_with_replacement(items,2):
    print(item)
    # (1, 1)
    # (1, 2)
    # (1, 3)
    # (2, 2)
    # (2, 3)
    # (3, 3)

"""
(4)product()函数，笛卡尔积
product函数与上述方法最大的不同点是：其针对多个输入序列进行排列组合，实例如下：
"""
ab = ['a','b']
cd = ['c','d']
for item in itertools.product(ab,cd):
    print(item)
    # ('a', 'c')
    # ('a', 'd')
    # ('b', 'c')
    # ('b', 'd')

"""
使用笛卡尔积求出TradeStrategy2的最有参数，即求出下跌幅度买入阀值(s_buy_change_threshold)与买入股票后持有天数(s_keep_threshold)如何取值，
可以让策略最终盈利最大化

首先将TradeStrategy2策略基础参数与执行回测的代码抽象出一个函数calc(),该函数的输入参数有两个，分别是持股天数和下跌买入阀值：输出的返回值为3个，
分别是盈亏情况、输入的持股天数和下跌买入阀值
"""

def calc(keep_stock_threshold,buy_change_threshold):
    """
    :param keep_stock_threshold: 持股天数
    :param buy_change_threshold: 下跌买入阀值
    :return: 盈亏情况，输入的持股天数，输入的下跌买入阀值
    """

    #实例化TradeStrategy2
    trade_strategy2 = trade_strategy.TradeStrategy2()
    #通过类方法设置买入后的持股天数
    trade_strategy.TradeStrategy2.set_keep_stock_threshold(keep_stock_threshold)
    #通过静态方法设置下跌买入阀值
    trade_strategy.TradeStrategy2.set_buy_change_threshold(buy_change_threshold)
    #进行回测




"""
多进程 VS 多线程

上面加单的回测没有进行负载的计算，也没有繁多的I/O操作，所以通过for循环串行计算每组参数的结果
也没有速度上的问题，真实的回测不但具有复杂的计算，繁多的I/O操作且回测本身的复杂度也不是上面能比拟的

针对上面的问题，一般使用多任务并行的方式来解决，有如下几种方式：
1、启多个进程
2、启多个线程
3、启多个进程，每个进程启动多个线程

由于win python中存在全局解释器锁 GIL，python的线程被限制为同一个时刻一个进程中只允许一个线程执行，所以
python的多线程适用于处理I/O密集型任务和并发执行的阻塞操作，用多进程来处理计算密集型任务

这里使用concurrent.futures库来实现多线程和多进程

"""

"""
1、使用多进程(ProcessPoolExecutor)

"""








