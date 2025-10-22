import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency

# === Настройки ===
path = "data-251019.xlsx"
col_support = "На момент начала вашей работы, описываемый далее проект был на ваш взгляд поддерживаемым"
col_time_single = "Сколько времени в среднем занимал запуск одного теста в цикле разработки в секундах"

# === Загрузка и очистка данных ===
df = pd.read_excel(path)

exclude_ids = [2105212553, 2105364012, 2105434991, 2117312175, 2117477460]
df = df[~df["ID"].isin(exclude_ids)]

df = df[[col_support, col_time_single]].dropna()
df = df[df[col_support].isin(["да", "нет"])]
df = df[df[col_time_single] != "не помню"]

# === Таблица сопряжённости ===
cont_table = pd.crosstab(df[col_time_single], df[col_support])
print("=== Таблица сопряжённости ===")
print(cont_table)

# === χ²-тест ===
chi2, p, dof, expected = chi2_contingency(cont_table)

# === Cramer's V ===
n = cont_table.sum().sum()
r, k = cont_table.shape
cramers_v = np.sqrt(chi2 / (n * (min(r, k) - 1)))

print("\n=== χ² и Cramer's V ===")
print(f"χ² = {chi2:.6f}")
print(f"p = {p:.6f}")
print(f"Cramer's V = {cramers_v:.3f}")
print(f"N = {n}")
print(f"Степеней свободы = {dof}")