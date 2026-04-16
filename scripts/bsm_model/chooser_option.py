
import numpy as np
import pandas as pd
from scipy.stats import norm
import yaml
from pathlib import Path

class ChooserOptionBSM:
    """
    BSM 模型下的 Chooser Option 定价
    验证：    - Exploration_of_JPMorgan_Chooser_Option_Pricing
    """
    def __init__(self, S0, K, T1, T2, r, sigma, q, n_sim=100000, seed=42):
        self.S0 = S0
        self.K = K
        self.T1 = T1
        self.T2 = T2
        self.r = r
        self.sigma = sigma
        self.q = q
        self.n_sim = n_sim
        self.seed = seed
        np.random.seed(seed)

    # ---------- 辅助函数 ----------
    def _simulate_price(self, S_start, T):
        """
        向量化一步模拟终期价格
        S_start: 标量 或 数组
        T: 时间间隔（年）
        返回: 模拟后的价格数组，长度 = self.n_sim (若S_start标量) 或 len(S_start) (若S_start数组)
        """
        if hasattr(S_start, '__len__'):
            n = len(S_start)
        else:
            n = self.n_sim  # ✅ 关键修正：使用 self.n_sim
        mu = (self.r - self.q - 0.5 * self.sigma ** 2) * T
        sigma_sqrtT = self.sigma * np.sqrt(T)
        z = np.random.standard_normal(n)
        return S_start * np.exp(mu + sigma_sqrtT * z)

    # ---------- 蒙特卡洛定价（返回平均价格）----------
    def price_mc(self):
        """蒙特卡洛模拟 Chooser Option 当前价值（向量化）"""
        # 第一步：模拟决策日股价
        S_T1 = self._simulate_price(self.S0, self.T1)   # shape (n_sim,)
        is_call = S_T1 > self.K

        # 第二步：模拟到期日股价（基于每个 S_T1 的一条路径）
        S_T2 = self._simulate_price(S_T1, self.T2 - self.T1)

        # 计算 payoff
        payoff_call = np.maximum(S_T2 - self.K, 0)
        payoff_put  = np.maximum(self.K - S_T2, 0)
        payoff = np.where(is_call, payoff_call, payoff_put)

        # 折现
        price = np.exp(-self.r * self.T2) * np.mean(payoff)
        return price

    # ---------- 解析公式定价 ----------
    def price_analytical(self):
        """Chooser Option 解析公式"""
        def d1(S, K, T, r, sigma, q):
            return (np.log(S/K) + (r - q + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
        def d2(d1, sigma, T):
            return d1 - sigma*np.sqrt(T)

        def call_price(S, K, T, r, sigma, q):
            d1_val = d1(S, K, T, r, sigma, q)
            d2_val = d2(d1_val, sigma, T)
            return S * np.exp(-q*T) * norm.cdf(d1_val) - K * np.exp(-r*T) * norm.cdf(d2_val)

        def put_price(S, K, T, r, sigma, q):
            d1_val = d1(S, K, T, r, sigma, q)
            d2_val = d2(d1_val, sigma, T)
            return K * np.exp(-r*T) * norm.cdf(-d2_val) - S * np.exp(-q*T) * norm.cdf(-d1_val)

        # Chooser = 欧式看涨(T2) + 欧式看跌(T1)
        C_T2 = call_price(self.S0, self.K, self.T2, self.r, self.sigma, self.q)
        P_T1 = put_price(self.S0, self.K, self.T1, self.r, self.sigma, self.q)
        return C_T2 + P_T1

    # ---------- 生成路径表格（论文 Table 3 对比）----------
    def generate_path_table(self, n_paths=10):
        """生成 n_paths 条模拟路径，返回 DataFrame"""
        # 临时修改模拟次数
        original_n_sim = self.n_sim
        self.n_sim = n_paths
        np.random.seed(self.seed)   # 保证可复现

        # 模拟 T1 和 T2 股价
        S_T1 = self._simulate_price(self.S0, self.T1)
        choices = np.where(S_T1 > self.K, 'CALL', 'PUT')
        S_T2 = self._simulate_price(S_T1, self.T2 - self.T1)

        # 计算 payoff
        payoff_call = np.maximum(S_T2 - self.K, 0)
        payoff_put  = np.maximum(self.K - S_T2, 0)
        payoff = np.where(S_T1 > self.K, payoff_call, payoff_put)

        # 恢复原模拟次数
        self.n_sim = original_n_sim

        df = pd.DataFrame({
            '1st ST': np.round(S_T1, 2),
            'Choice': choices,
            '2nd ST (based)': np.round(S_T2, 2),
            'Payoff': np.round(payoff, 2)
        })
        return df

    def set_sigma_from_history(self, returns_series, window=20):
        """
        根据历史收益率序列计算滚动波动率（年化），并更新 self.sigma
        returns_series: 日收益率序列（pandas Series）
        window: 滚动窗口大小（交易日）
        """
        rolling_std = returns_series.rolling(window).std() * np.sqrt(252)
        # 返回最新值，但这里我们会在外部循环中逐日使用
        return rolling_std

# ---------- 命令行测试 ----------
if __name__ == "__main__":
    # 加载参数（也可直接写死在代码中）
    params_path = Path(__file__).parent.parent.parent / "params.yaml"
    if params_path.exists():
        with open(params_path, 'r', encoding="utf-8") as f:
            params = yaml.safe_load(f)
    else:
        # 默认参数（论文 Table 2）
        params = {
            "S0": 156.7, "K": 150, "T1": 0.5, "T2": 1.0,
            "r": 0.0015, "sigma": 0.282, "q": 0.0233,
            "n_sim_mc": 100000, "random_seed": 42
        }

    chooser = ChooserOptionBSM(
        S0=params['S0'],
        K=params['K'],
        T1=params['T1'],
        T2=params['T2'],
        r=params['r'],
        sigma=params['sigma'],
        q=params['q'],
        n_sim=params.get('n_sim_mc', 100000),
        seed=params.get('random_seed', 42)
    )

    # 蒙特卡洛价格 vs 解析价格
    price_mc = chooser.price_mc()
    price_ana = chooser.price_analytical()
    print(f"蒙特卡洛价格 (n_sim={chooser.n_sim}): {price_mc:.4f}")
    print(f"解析公式价格: {price_ana:.4f}")

    # 生成 10 次模拟路径表格
    df_paths = chooser.generate_path_table(n_paths=10)
    print("\n10次模拟路径（与论文 Table 3 对比）:")
    print(df_paths.to_string(index=False))