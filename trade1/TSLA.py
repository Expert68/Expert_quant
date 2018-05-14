from abupy import ABuSymbolPd

def get():
    #两年的TSLA收盘数据tolist()
    price_array = ABuSymbolPd.make_kl_df('TSLA',n_folds=2).close.tolist()
    #两年的TSLA收盘日期tolist()
    date_array = ABuSymbolPd.make_kl_df('TSLA',n_folds=2).date.tolist()
    # print(price_array[:5])
    # print(date_array[:5])
    # print(date_array)
    return price_array,date_array

price_array,date_array = get()

date_base = 20160516

# [208.29, 204.66, 211.17, 215.21, 220.28]
# [20160516, 20160517, 20160518, 20160519, 20160520]