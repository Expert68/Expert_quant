# list = [1,2,3,4,5]
# list.insert(5,6)
# for item,_ in enumerate(list):
#     print(('%s,%s') %(item,_))
# import timeit
#
# normal_list = range(10000)
# print(timeit.timeit(stmt='[i**2 for i in range(10000)]'))

# from abupy import ABuSymbolPd
# #两年的TSLA收盘数据tolist()
# price_array = ABuSymbolPd.make_kl_df('TSLA',n_folds=2).close.tolist()
# #两年的TSLA收盘日期tolist()
# data_array = ABuSymbolPd.make_kl_df('TSLA',n_folds=2).date.tolist()
# print(price_array[:5])
# print(data_array[:5])





