import numpy as np
from random import random, randint
from time import sleep
from matplotlib import pyplot as plt

class NormalDistBandit:
    # 构造一个n臂的bandit，每个臂的奖励服从正态分布，均值和标准差由means和stds指定
    def __init__(self, means, stds):
        assert len(means) == len(stds), "Means and stds must be the same length."
        self.n = len(means)
        self.means = np.array(means)
        self.stds = np.array(stds)
        assert all(self.stds >= 0), "Stds must be positive."
    
    def pull(self, k):
        assert 0 <= k < self.n, f"Invalid arm {k}."
        # 拉动第k个臂，返回奖励
        return np.random.normal(loc=self.means[k], scale=self.stds[k])

def epsilon_greedy(values, epsilon):
    # 返回的是当前当前选择的臂的索引
    assert len(values) > 1, "There should be 2 or more values."
    # 标准情况下，epsilon-greedy策略选择最优臂的概率为1-epsilon + epsilon/len(values)
    # 选择非最优臂的概率为epsilon/len(values)
    # 为了使得选择最优臂的概率为1-epsilon，选择非最优臂的概率为epsilon/(len(values)-1)，我们需要对epsilon进行调整
    # epsilon' = epsilon * len(values) / (len(values) - 1)
    # 则有p(best) = 1 - epsilon' + epsilon' / len(values) = 1 - epsilon
    # p(non-best) = epsilon' / len(values) = epsilon / (len(values) - 1)
    eps = epsilon * len(values) / (len(values) - 1)
    if random() <= eps: # 随机选择一个臂
        return randint(0, len(values)-1) # [0, len(values)-1]
    return int(np.argmax(values))
   
        
if __name__ == "__main__":
    n = 5
    bandit = NormalDistBandit(means = np.array(range(-n, n+1)), stds = np.ones(11))
    
    # Task：把下面这几种epsilon的曲线画到一张图中，分析你所观察到的结果。
    epsilons = [0.01, 0.05, 0.1, 0.2]
    
    iter = 10000
    x = np.array(range(iter))
    
    plt.figure()
    for eps in epsilons:
        # 初始化每次实验的平均奖励曲线
        y = np.zeros(iter, dtype=np.float64)
        values = np.zeros(n*2+1, dtype=np.float64)
        counts = np.zeros(n*2+1, dtype=np.int64)
        for i in range(1, iter):
            # 选择一个臂
            action = epsilon_greedy(values, eps)
            # 拉动臂，获得奖励，记录拉动次数
            counts[action] += 1
            value = bandit.pull(action)
            # 增量更新特定bandit的平均值
            values[action] = (values[action] * (counts[action] - 1) + value) / counts[action]
            # 增量更新平均奖励曲线
            y[i] = (y[i-1] * (i-1) + value) / i
        plt.plot(x, y, label=f'ε = {eps}')
    plt.xlabel('Iterations')
    plt.ylabel('Average reward')
    plt.legend()
    plt.savefig('bandit.png', dpi=150)