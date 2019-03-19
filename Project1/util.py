import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')


def get_pos(x, samples):
    """samples是从小到大排序的列表，
    返回索引i，使得samples[:i]均小于x，samples[i:]均大于x
    """
    for i, point in enumerate(samples):
        # 找到第一个不小于x的点，返回其索引
        if point >= x:
            return i
    # 若samples中的点均小于x
    if i == len(samples) - 1:
        return len(samples)


def plot_normal_results(mean, sigma, x, p_x, suffix):
    """绘制非参数估计拟合正太分布 的结果"""
    # 绘制标准概率密度函数图像
    plt.clf()
    std_x = np.arange(mean-4*sigma, mean+4*sigma, 0.001)
    std_y = normfun(std_x, mean, sigma)
    plt.subplot(121, title="Standard")
    plt.plot(std_x, std_y)

    # 拟合概率密度函数图像
    plt.subplot(122, title="Estimated "+suffix)
    plt.plot(x, p_x)

    # 保存图片
    fig_name = suffix+"-normal.jpg"
    plt.savefig("./results/"+fig_name)


def normfun(x, mean, sigma):
    """计算正太分布"""
    y = np.exp(-(x-mean)**2 / 2*sigma**2) / (sigma * np.sqrt(2*np.pi))
    return y
