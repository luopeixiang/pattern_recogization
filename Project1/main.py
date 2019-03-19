import numpy as np

from ProbEstimation import KNNEstimation, ParzenEstimation
from util import plot_normal_results

# 生成正太分布的数据
MEAN = 5  # 平均值
SIGMA = 2  # 标准差


def knn_estimation():
    print("K近邻估计")
    # K近邻估计方法
    # 尝试不同的参数组合来估计概率密度 (K_N, N)
    param_pairs = [(50, 10000), (1000, 50000),
                   (1000, 500000), (1000, 1000000),
                   (1000, 5000000), (50000, 50000000)]
    for K_N, N in param_pairs:
        print("使用参数 KN={} N={}进行估计...".format(K_N, N))
        # 从已知概率分布获取样本
        samples = list(np.random.normal(MEAN, SIGMA, N))
        samples.sort()

        # 使用KNN方法进行非参数估计
        knn_est = KNNEstimation(K_N)
        x, p_x = knn_est.estimation(samples)
        # 绘制估计的结果
        # import pdb
        # pdb.set_trace()
        suffix = "KNN-KN{}-N-{}".format(K_N, N)
        plot_normal_results(MEAN, SIGMA, x, p_x, suffix)


def parzen_estimation():
    # Parzen窗口估计方法
    print("Parzen窗估计...")

    # 尝试不同的参数估计 (h, N, win_type)
    h_list = [0.25, 1, 4]
    N_list = [1000, 10000, 100000]
    win_type_list = ["Square", "Gauss"]
    for h in h_list:
        for N in N_list:
            for win_type in win_type_list:
                print("使用参数 h={} N={} 核函数={} 进行估计...".format(h, N, win_type))
                samples = list(np.random.normal(MEAN, SIGMA, N))
                samples.sort()

                # 使用Parzen方法进行非参数估计
                parzen_est = ParzenEstimation(h, win_type)
                x, p_x = parzen_est.estimation(samples)
                # 绘制结果
                suffix = "Parzen-h{}-N{}-{}".format(h, N, win_type)
                plot_normal_results(MEAN, SIGMA, x, p_x, suffix)


def main():
    knn_estimation()
    # parzen_estimation()


if __name__ == "__main__":
    main()
