import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency

# === Настройки ===
path = "data-251019.xlsx"
exclude_ids = [2105212553, 2105364012, 2105434991, 2117312175, 2117477460]
col_factor = "Как часто интерфейсы в кодовой базе имели единственную реализацию"
col_target = "На момент начала вашей работы, описываемый далее проект был на ваш взгляд поддерживаемым"

# === Загрузка и фильтрация ===
df = pd.read_excel(path)
df = df[~df["ID"].isin(exclude_ids)].copy()
data = df[[col_factor, col_target]].dropna()

# === Таблица сопряжённости ===
contingency = pd.crosstab(data[col_factor], data[col_target])

# === Статистика ===
chi2, p, dof, expected = chi2_contingency(contingency)
n = contingency.values.sum()
r, k = contingency.shape
phi2 = chi2 / n
phi2_corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
k_corr = k - ((k-1)**2)/(n-1)
r_corr = r - ((r-1)**2)/(n-1)
denom = min((k_corr-1), (r_corr-1))
cramers_v = np.sqrt(phi2_corr / denom) if denom > 0 else np.nan

# === Вывод ===
print("=== Таблица сопряжённости ===")
print(contingency.to_string())
print("\n=== χ² и Cramer's V ===")
print(f"χ² = {chi2}")
print(f"p = {p}")
print(f"Cramer's V = {cramers_v}")
print(f"N = {n}")
print(f"Степеней свободы = {dof}")
