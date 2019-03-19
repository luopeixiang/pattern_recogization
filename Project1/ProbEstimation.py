import numpy as np

from util import get_pos, normfun


class KNNEstimation(object):
    """使用K近邻方法对概率密度函数进行估计"""

    def __init__(self, kn):
        """参数：
            kn：int，表示每个小舱内拥有的样本个数
        """
        self.kn = kn

    def estimation(self, samples, win_num=80):
        """参数：
            samples是经过从小到大排序的采样点
            win_num：指定窗口数
           返回：
            point_x:一系列窗口的中心
            point_px:以及这些中心对应的密度
        """

        min_x, max_x = samples[0], samples[-1]
        step = (max_x - min_x) / win_num
        N = len(samples)

        point_x = []
        point_px = []
        # 在x的取值范围里面即[min_x, max_x]
        # 以每一个小点为小舱中心用式3-63进行估计
        x = min_x  # 从最小的点开始
        while x < max_x:
            # 估计窗口大小 返回窗口的宽度
            width = self._get_win_width(x, samples)
            # 模式识别式3-63
            p_x = self.kn / N / width
            point_x.append(x)
            point_px.append(p_x)
            # 下一个小点
            x += step

        assert len(point_x) == len(point_px)
        return point_x, point_px

    def _get_win_width(self, win_mid, samples):
        """获取窗口宽度
        参数：
            win_mid:窗口中点
            samples：一系列采样点
        """

        # 先找出win_mid在samples里面所处的位置
        # 从这个位置出发向两边延伸，将kn个数包括进来的窗口即为所求
        pos = get_pos(win_mid, samples)

        # 两个指针分别代表左右两边窗口的界限
        left_pointer = pos - 1
        right_pointer = pos

        # 限制窗口的边界
        LEFT_END = -1
        RIGHT_END = len(samples)

        count = 0
        while left_pointer != LEFT_END \
                and right_pointer != RIGHT_END \
                and count < self.kn:
            if samples[right_pointer] - win_mid < samples[left_pointer] - win_mid:
                # 向右扩大窗口
                right_pointer += 1
            else:
                # 向左扩大窗口
                left_pointer -= 1
            count += 1

        if count != self.kn:
            if left_pointer == LEFT_END:
                # 从右边的窗口加入self.kn - count 个数
                right_pointer += (self.kn - count)
            elif right_pointer == RIGHT_END:
                left_pointer -= (self.kn - count)

        assert right_pointer - left_pointer - 1 == self.kn
        # width = samples[right_pointer-1] - samples[left_pointer+1]
        width = 2 * max(samples[right_pointer-1] - win_mid,
                        win_mid - samples[left_pointer+1])

        return width


class ParzenEstimation(object):
    """Parzen窗法估计"""

    def __init__(self, h, win_type="Square"):
        """参数:
            h: 对应书中的超立方体的棱长，在一维情况下就是窗口大小
            win_type: Square表示方窗，Gauss表示高斯窗
        """
        assert win_type in ["Square", "Gauss"]
        self.width = h
        self.win_type = win_type

    def estimation(self, samples, win_num=80):
        min_x, max_x = samples[0], samples[-1]
        step = (max_x - min_x) / win_num

        point_x = []
        point_px = []
        x = min_x  # 从最小的点开始
        while x < max_x:
            p_x = self._knernel_func(x, samples)
            point_x.append(x)
            point_px.append(p_x)
            # 下一个小点
            x += step

        assert len(point_x) == len(point_px)
        return point_x, point_px

    def _knernel_func(self, x, samples):
        """根据式3-70计算p(x)"""
        N = len(samples)
        kx = 0.
        # 方窗函数
        if self.win_type == "Square":
            for xi in samples:
                # 式3-72
                kx += 1./self.width if abs(x - xi) <= self.width/2 else 0.
        elif self.win_type == "Gauss":
            # 式3-74
            xs = np.array([x] * N)
            samples = np.array(samples)
            sigma = self.width / np.sqrt(N)
            kx += normfun(xs, samples, sigma).sum()
        px = kx / N
        return px
