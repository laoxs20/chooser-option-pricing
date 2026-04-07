# scripts/bsm_model/chooser_option.py
import numpy as np
import pandas as pd
from scipy.stats import norm


class ChooserOptionBSM:
    """
    BSM 两期模拟的 Chooser Option 定价器
    参数:
        S0: 当前标的资产价格
        K: 行权价
        T1: 决策时间（年）
        T2: 到期时间（年）
        r: 无风险利率（连续复利）
        sigma: 波动率
        q: 股息率（连续）
        n_sim: 模拟次数
    """

    def __init__(self, S0, K, T1, T2, r, sigma, q, n_sim=10000):
        self.S0 = S0
        self.K = K
        self.T1 = T1
        self.T2 = T2
        self.r = r
        self.sigma = sigma
        self.q = q
        self.n_sim = n_sim
        np.random.seed(42)  # 可复现

    def simulate_price(self, S_start, T, dt=1 / 252):
        """从 S_start 模拟到 T 年后的价格（几何布朗运动）"""
        n_steps = int(T / dt)
        S = np.full(self.n_sim, S_start)
        for _ in range(n_steps):
            z = np.random.standard_normal(self.n_sim)
            S = S * np.exp((self.r - self.q - 0.5 * self.sigma ** 2) * dt
                           + self.sigma * np.sqrt(dt) * z)
        return S

    def price(self):
        """返回 Chooser Option 的当前理论价格"""
        # 第一步：模拟决策日 T1 的股价
        S_T1 = self.simulate_price(self.S0, self.T1)

        # 在 T1 决定是 call 还是 put
        is_call = S_T1 > self.K

        # 第二步：根据选择，分别模拟到期日股价
        S_T2 = np.zeros(self.n_sim)
        for i in range(self.n_sim):
            if is_call[i]:
                # 看涨：从 S_T1[i] 模拟到 T2
                S_T2[i] = self.simulate_price(S_T1[i], self.T2 - self.T1)[0]  # 注意只取一个样本路径
            else:
                # 看跌
                S_T2[i] = self.simulate_price(S_T1[i], self.T2 - self.T1)[0]

        # 计算 payoff
        payoff_call = np.maximum(S_T2 - self.K, 0)
        payoff_put = np.maximum(self.K - S_T2, 0)
        payoff = np.where(is_call, payoff_call, payoff_put)

        # 折现到当前
        option_price = np.exp(-self.r * self.T2) * np.mean(payoff)
        return option_price

    def price_with_analytical_formula(self):
        """
        论文中未给出解析解，但 Chooser Option 有闭式解（Rubinstein 1991）：
        C = C(S0, K, T2) + P(S0, K, T1)  其中 C 和 P 是普通欧式期权价格
        注意：这里假设股息率为 q
        """

        # 计算 d1, d2 函数
        def d1(S, K, T, r, sigma, q):
            return (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))

        def d2(d1, sigma, T):
            return d1 - sigma * np.sqrt(T)

        # 普通欧式看涨价格
        def call_price(S, K, T, r, sigma, q):
            d1_val = d1(S, K, T, r, sigma, q)
            d2_val = d2(d1_val, sigma, T)
            return S * np.exp(-q * T) * norm.cdf(d1_val) - K * np.exp(-r * T) * norm.cdf(d2_val)

        # 普通欧式看跌价格
        def put_price(S, K, T, r, sigma, q):
            d1_val = d1(S, K, T, r, sigma, q)
            d2_val = d2(d1_val, sigma, T)
            return K * np.exp(-r * T) * norm.cdf(-d2_val) - S * np.exp(-q * T) * norm.cdf(-d1_val)

        # Chooser 解析公式（European chooser with same strike）
        C_T2 = call_price(self.S0, self.K, self.T2, self.r, self.sigma, self.q)
        P_T1 = put_price(self.S0, self.K, self.T1, self.r, self.sigma, self.q)
        return C_T2 + P_T1


# 测试参数（论文 Table 2）
if __name__ == "__main__":
    params = {
        "S0": 156.7,
        "K": 150,
        "T1": 0.5,
        "T2": 1.0,
        "r": 0.0015,
        "sigma": 0.282,
        "q": 0.0233,
        "n_sim": 50000
    }
    chooser = ChooserOptionBSM(**params)
    price_mc = chooser.price()
    price_analytical = chooser.price_with_analytical_formula()
    print(f"蒙特卡洛模拟价格: {price_mc:.4f}")
    print(f"解析公式价格: {price_analytical:.4f}")