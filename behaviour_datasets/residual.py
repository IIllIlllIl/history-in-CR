import numpy as np
from scipy import stats


observed = np.array([[52, 225, 47], [230, 930, 207], [43, 216, 129]])
print(observed)
chi2, p, dof, expected = stats.chi2_contingency(observed)

row_sums = observed.sum(axis=1)
col_sums = observed.sum(axis=0)
total = observed.sum()
row_proportions = row_sums / total
col_proportions = col_sums / total

print(f"卡方统计量: {chi2:.2f}")
print(f"P值: {p:.6f}")
print(f"自由度: {dof}")
print("期望频数矩阵:\n", expected.round(2))
standardized_residuals = (observed - expected) / np.sqrt(expected)
print(np.round(standardized_residuals, 2))

adjusted_residuals = np.zeros_like(observed, dtype=float)
for i in range(observed.shape[0]):
    for j in range(observed.shape[1]):
        E = expected[i, j]
        # 调整残差分母
        denominator = np.sqrt(E * (1 - row_proportions[i]) * (1 - col_proportions[j]))
        # 调整残差
        adjusted_residuals[i, j] = (observed[i, j] - E) / denominator
print(np.round(adjusted_residuals, 2))
