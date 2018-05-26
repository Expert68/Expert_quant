"""
可视化是量化中的一大关键辅助工具，量化中往往需要通过可视化技术来更清晰地理解交易，理解数据
通过可视化技术可以更加快速地对量化系统中的问题进行分析，更进一步的指导策略的开发，以及策略中的问题发现
"""

'#######################################################################################################################'
"""
使用Matplotlib可视化数据
Matplotlib是python中最基础也是最常用的可视化工具，许多更高级的可视化库都是在Matplotlib上再次封装提供更简单易用的功能，
比如seaborn库。而且Matplotlib的使用方式和绘制思想已经成为python中绘图库的标杆，其他大多数绘图库都会特意使用与Matplotlib
类似的函数名称参数及思想，如果掌握了Matplotlib的绘制方式以及思想，那么在python中任何一个可视化库的使用都会感到简单易用
"""
'#######################################################################################################################'

"""
Matplotlib可视化基础
一般习惯以及推荐引用Matplotlib的方式如下：
import matplotlib.pyplot as plt
首先获取特斯拉电动车两年的股票数据：
"""
import matplotlib.pyplot as plt
from abupy import ABuSymbolPd
#默认封装了pandas模块所以可以不用导入pandas模块，同时pandas模块又默认封装了numpy模块，
# 所以可以不用导入pandas模块和numpy模块
tsla_df = ABuSymbolPd.make_kl_df('usTSLA',n_folds=2)
#获取头尾各5行数据
print(tsla_df.head())
#              close    high     low  p_change    open  pre_close   volume  \
# 2016-05-18  211.17  215.31  207.75     3.181  209.15     204.66  5617519
# 2016-05-19  215.21  216.79  207.30     1.913  213.62     211.17  6866321
# 2016-05-20  220.28  220.55  216.35     2.356  216.99     215.21  9007076
# 2016-05-23  216.22  222.60  215.86    -1.843  219.87     220.28  5102479
# 2016-05-24  217.91  218.74  215.18     0.782  216.60     216.22  3013843
#                 date  date_week  key    atr21    atr14
# 2016-05-18  20160518          2    0  10.6500  10.6500
# 2016-05-19  20160519          3    1  11.4252  11.4429
# 2016-05-20  20160520          4    2  10.6776  10.6548
# 2016-05-23  20160523          0    3   9.7347   9.6560
# 2016-05-24  20160524          1    4   8.6232   8.4674
print(tsla_df.tail())
#              close    high     low  p_change    open  pre_close   volume  \
# 2018-05-09  306.85  307.01  300.05     1.616  300.41     301.97  5727365
# 2018-05-10  305.02  312.99  304.11    -0.596  307.50     306.85  5651561
# 2018-05-11  301.06  308.88  299.08    -1.298  307.70     305.02  4679649
# 2018-05-15  284.18  286.96  280.50    -2.668  285.01     291.97  9519173
# 2018-05-16  286.48  288.81  281.56     0.809  283.83     284.18  5674019
#                 date  date_week  key    atr21    atr14
# 2018-05-09  20180509          2  495  14.7541  14.2905
# 2018-05-10  20180510          3  496  14.4147  13.8544
# 2018-05-11  20180511          4  497  13.9952  13.3138
# 2018-05-15  20180515          1  498  14.9520  14.8080
# 2018-05-16  20180516          2  499  14.5391  14.2216

"""
plt.plot()是最简单也是最常用的绘图方式，可以通过pandas的Series直接作为参数进行绘制，可以通过numpy对象，
甚至python的list对象它都是支持的。以下代码函数plot_demo()封装了plt.plot()方法,分别使用Series对象，numpy对象以及list绘制TSLA的收盘价格
"""
def plot_demo(axs=None,just_series=False):
    """
    绘制tsla的收盘价格曲线
    :param axs: axs为子画布
    :param just_series: 是否只绘制一条收盘曲线使用Series，后面会用到
    :return:
    """
    #如果参数传入子画布则使用子画布绘制
    drawer = plt if axs is None else axs #三目表达式，else左边成立则执行plt，否则执行axs
    #Series对象tsla_df.close，红色
    drawer.plot(tsla_df.close,c='r')
    if not just_series:
        #为了使曲线不重叠，y变量加了10个单位 tsla_df.close.values + 10
        #numpy对象tsla_df.close.index + tsla_df.close.values,绿色
        drawer.plot(tsla_df.close.index,tsla_df.close.values+10,c='g')
        #为了使曲线不重叠，y变量增加20个单位
        #list对象，numpy.tolist()将numpy转换为list对象，蓝色
        drawer.plot(tsla_df.close.index.tolist(),(tsla_df.close.values+20),c='b')
    plt.xlabel('time')
    plt.ylabel('close')
    plt.title('TSLA CLOSE')
    plt.grid(True)

plot_demo()
#得到的结果如图matplotlib--1所示

'#######################################################################################################################'
"""
Matplotlib子画布及loc的使用
上面的plot_demo分别使用色Series，numpy,list序列对象绘制了股价的走势，但得到的结果中并没有标注哪一条曲线是使用
Series，哪一条是使用numpy或list绘制
Matplotlib使用legend()函数操作标注，参数loc代表标注位置
下面的代码通过生成多个子画布，在不同的子画布上使用不同的loc值来示例标注及loc的使用，得到的结果如图matplotlib--2所示：
"""
#subplots 返回fig和axs，下面的代码得到4个子图和一个axs对象
_,axs = plt.subplots(nrows=2,ncols=2,figsize=(14,10))
#画布0，loc：0 plot_demo中传入画布，则使用传入的画布绘制
drawer = axs[0][0]
plot_demo(drawer)
drawer.legend(['Series','Numpy','List'],loc=0)
#画布1，loc：1
drawer = axs[0][1]
plot_demo(drawer)
drawer.legend(['Series','Numpy','List'],loc=1)
#画布2，loc：2
drawer = axs[1][0]
plot_demo(drawer)
drawer.legend(['Series','Numpy','List'],loc=2)
#画布3，loc：2,设置bbox_to_anchor，在画布外的相对位置绘制
drawer = axs[1][1]
plot_demo(drawer)
drawer.legend(['Series','Numpy','List'],bbox_to_anchor=(1.05,1),loc=2,borderaxespad=0)
#得到的结果如图matplotlib--2所示

"""
上述代码使用plt.subplots()函数生成多个子画布，返回的axs是numpy类型数据，
参数nrows和ncols分别代表x与y轴的shape值，所以plt.subplot(nrows=2,ncols=2,figsize=(14,10))实际会生成一个2*2的numpy对象axs
在之前plot_demo()函数的基础上使用legend()函数对绘制进行标注，这里的loc使用不同的值，将导致标注的位置发生变化，axs[1][1]使用bbox_to_anchor将标注绘制在画布的外面
如果loc无特殊要求，一般使用loc='best'简单指定，即自动选择最优的位置进行标注绘制
"""
'#######################################################################################################################'

"""
K线图的绘制
Matplotlib对绘制K线图有直接的函数封装，所以使用Matplotlib绘制K线图非常简单
绘制K线图的主要的工作就是制作一个qutotes、
qutotes里每一个数据是一根蜡烛，绘制标准K线图使用matplotlib.finance.candlestick_ochl()函数，
函数后缀的ohcl的意思是按照开盘、最高、收盘、最低的顺序组织数据加入到qutotes中
另外还有matplotlib.finance.candlestick2_ochl()函数，其用法与matplotlib.finance.candlestick_ochl()函数显示
"""
import matplotlib.finance as mpf
__colorup__ = 'red'
__colordown__ = 'green'
#为了示例清晰，只拿出前30天的交易数据绘制K线图
tsla_part_df = tsla_df[:30]
fig,ax = plt.subplots(figsize=(14,7))
qutotes = []
for i,(d,o,c,h,l) in enumerate(zip(tsla_part_df.index,tsla_part_df.open,tsla_part_df.close,tsla_part_df.high,tsla_part_df.low)):
    #K线图的日期要使用matplotlib.finance.date2num进行转换为特有的数字值
    d = mpf.date2num(d)
    #日期，开盘、收盘、最高和最低组成tuple对象val
    val = (d,o,c,h,l)
    #将val加入qutotes
    qutotes.append(val)
    #使用mpf.candlestick_ohcl()进行k线图的绘制
    mpf.candlestick_ochl(ax,qutotes,width=0.6,colorup=__colorup__,colordown=__colordown__)
    ax.autoscale_view()
    ax.xaxis_date()
# for item in zip(tsla_part_df.index, tsla_part_df.open, tsla_part_df.close, tsla_part_df.high, tsla_part_df.low):  #zip函数，将列表元素一一对应包成元组
#     print(item)
# (Timestamp('2016-05-23 00:00:00'), 219.87, 216.22, 222.6, 215.86)
# (Timestamp('2016-05-24 00:00:00'), 216.6, 217.91, 218.74, 215.18)
# (Timestamp('2016-05-25 00:00:00'), 217.91, 219.58, 221.36, 216.51)
# (Timestamp('2016-05-26 00:00:00'), 220.5, 225.12, 225.26, 219.05)
# (Timestamp('2016-05-27 00:00:00'), 224.99, 223.04, 225.93, 220.75)
# (Timestamp('2016-05-31 00:00:00'), 223.04, 223.23, 224.75, 221.5)

'#######################################################################################################################'

"""
使用Bokeh进行可视化交互
有些时候将一段数据可视化后有交互操作的需求，如K线图，当K线数量比较多的时候，可能有横向平移、放大某一时刻等交互需求，
Python实现这些交互需求的解决方案一般是通过网页形式的可视化，配合JavaScript（JS)与网页完成交互
Bokeh模块是一个专门针对Web浏览器实现呈现功能的交互式可视化Python库，使用示例如下：
"""
from bokeh.plotting import figure, output_file, show

# prepare some data
x = [1, 2, 3, 4, 5]
y = [6, 7, 2, 4, 5]

# output to static HTML file
output_file("matplotlib.html")

# create a new plot with a title and axis labels
p = figure(title="simple line example", x_axis_label='x', y_axis_label='y')

# add a line renderer with legend and line thickness
p.line(x, y, legend="Temp.", line_width=2)

# show the results
show(p)

#abupy模块封装了bokeh模块，可以通过ABuMarketDrawing.plot_candle_form_klpd()函数绘制可交互的K线图
#使用的时候设置html_bk=True,可以支持K线图拖拽、平移、放大等操作
from abupy import ABuMarketDrawing
ABuMarketDrawing.plot_candle_form_klpd(tsla_df,html_bk=True)
#得到的结果如matplotlib.html所示

'#######################################################################################################################'

"""
使用pandas可视化数据
pandas封装了matplotlib，使用matplotlib可以实现的绘制可视化都可以使用pandas用更简单的方式实现
由于pandas是为了金融量化而产生的库，包含很多金融量化接口，所以用pandas直接可视化金融数据非常简单
"""

"""
绘制股票的收益以及收益波动情况
计算收益波动一般会使用pd.rolling_std()函数和rolling_std()函数，示例如下：
"""
import pandas as pd
import numpy as np

#示例序列
demo_list = np.array([2,4,16,20])
#以3天为周期计算波动
demo_window = 3
#pd.rolling_std * np.sqrt
pd.rolling_std(demo_list,window=demo_window,center=False) * np.sqrt(demo_window)
# 输出：array([    nan,     nan, 13.1149, 14.4222])

"""
rolling_std()函数的意思是根据参数demo_window的大小、从原始序列依次取出demo_window个元素做std()操作，
波动即等于每次取出子序列做std()之后的结果乘以demo_window开方值，代码如下：
pd.Series([2,4,16]).std() * np.sqrt(demo_window)
输出如下：13.11487750
"""

'#######################################################################################################################'

"""
使用Seaborn可视化数据
Seaborn是在Matplotlib的基础上进行了更高级的API封装，从而使得作图更加容易，并且更加漂亮
一般使用以下形式导入seaborn库(使用sb会导致误解)
import seaborn as sns
应用seaborn就可以使用一行代码将直方图与概率密度图一起绘制出来，代码如下：
"""
import seaborn as sns
sns.distplot(tsla_df['p_change'],bins=80)
#结果如图seaborn--1所示

"""
针对pandas的DataFrame数据，使用列名称定轴绘制箱形图（能显示出一组数据的最大值、最小值、中位数、及上下四分位数），
tsla_df中的date_week代表星期几，使用date_week作为x轴，
p_change涨跌幅数据作为y轴，使用boxplot()函数绘制箱形图来可视化振幅和周几之间的关系，示例如下：
"""
sns.boxplot(x='date_week',y='p_change',data=tsla_df)
#结果如图seaborn--2所示
#可以看到周一的箱体最高，即TSLA周一的股价振幅最大(即p_change普遍偏大);周四箱体最矮，即TSLA周四相对振幅最小(即netChangeRatio普遍偏小）

"""
通过sns.joinplot()函数，可视化两组数的相关性和概率密度分布(点落在某一区间内的概率)，代码如下：
"""
sns.jointplot(tsla_df['high'],tsla_df['low'],color='r')
#结果如图seaboor--3所示

"""
通过pd.DataFrame.join()函数可以将多组股票的p_change涨跌幅数组连接起来组成一个新的DataFrame，代码如下：
"""
change_df = pd.DataFrame({'tsla':tsla_df.p_change}) #构造新DataFrame的方法

change_df = change_df.join(pd.DataFrame({'goog':ABuSymbolPd.make_kl_df('usGOOG',n_folds=2).p_change}))

change_df = change_df.join(pd.DataFrame({'aapl':ABuSymbolPd.make_kl_df('usAAPL',n_folds=2).p_change}))

change_df = change_df.join(pd.DataFrame({'fb':ABuSymbolPd.make_kl_df('usFB',n_folds=2).p_change}))

change_df = change_df.join(pd.DataFrame({'bidu':ABuSymbolPd.make_kl_df('usBIDU',n_folds=2).p_change}))

change_df.dropna()

print(change_df.head())
#              tsla   goog   aapl     fb   bidu
# 2016-05-23 -1.843 -0.775  1.271 -1.176 -0.165
# 2016-05-24  0.782  2.251  1.524  1.492  3.617
# 2016-05-25  0.766  0.719  1.757  0.161  0.455
# 2016-05-26  2.523 -0.159  0.793  1.340  0.543
# 2016-05-27 -0.924  1.179 -0.060 -0.075  4.131
"""
下面使用pd.DataFrame.corr()函数计算每组数据的协方差，使用热力图展示每组股票涨跌幅的相关性，示例如下：
"""
#使用corr计算数据的相关性：
corr = change_df.corr()
_,ax = plt.subplots(figsize=(8,5))
#sns.heatmap(corr,ax=ax)
sns.heatmap(corr,ax=ax)
#结果如图seaborn--4所示

"""
通过图seaborn--4发现，对角线的格子颜色最浅，因为他们代表本股票相对自身的相关性，其值都是1，
非对角线的格子代表本股票相对其他股票的相关性，股票颜色相对较深
"""








