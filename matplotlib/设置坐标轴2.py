import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(-3, 3, 50)
y1 = 2*x + 1
y2 = x**2

plt.figure()
plt.plot(x, y2)
# plot the second curve in this figure with certain parameters
plt.plot(x, y1, color='red', linewidth=1.0, linestyle='--')
# set x limits
plt.xlim((-1, 2))
plt.ylim((-2, 3))

# set new ticks
new_ticks = np.linspace(-1, 2, 5)
plt.xticks(new_ticks)
# set tick labels
plt.yticks([-2, -1.8, -1, 1.22, 3],
           ['$really\ bad$', '$bad$', '$normal$', '$good$', '$really\ good$'])
# to use '$ $' for math text and nice looking, e.g. '$\pi$'

# gca = 'get current axis'
ax = plt.gca()
#设置右、上边框
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')


# 设置坐标轴上的数字显示的位置，top:显示在顶部  bottom:显示在底部,默认是none
# 使用.xaxis.set_ticks_position设置x坐标刻度数字或名称的位置：bottom.（所有位置：top，bottom，both，default，none）
ax.xaxis.set_ticks_position('bottom')
# ACCEPTS: [ 'top' | 'bottom' | 'both' | 'default' | 'none' ]

# 把x轴原点绑定在y=0处   使用.set_position设置边框位置：y=0的位置；（位置所有属性：outward，axes，data）
ax.spines['bottom'].set_position(('data', 0))
# the 1st is in 'outward' | 'axes' | 'data'
# axes: percentage of y axis
# data: depend on y data

ax.yaxis.set_ticks_position('left')
# ACCEPTS: [ 'left' | 'right' | 'both' | 'default' | 'none' ]

ax.spines['left'].set_position(('data', 0))
plt.show()
