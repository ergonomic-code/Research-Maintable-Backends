import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency

# === Настройки ===
path = "data-251019.xlsx"

# Исключаем анкеты с ошибками
exclude_ids = [2105212553, 2105364012, 2105434991, 2117312175, 2117477460]

# Загружаем данные
df = pd.read_excel(path)
df = df[~df["ID"].isin(exclude_ids)]

# Целевая переменная
col_support = "На момент начала вашей работы, описываемый далее проект был на ваш взгляд поддерживаемым"

# Проверяемые факторы
questions = [
    "Применялся ли шаблон Command Query Responsibility Segregation",
    "Применялось ли реактивное программирование (Project Reactor, Rx* и т.п.)"
]

def cramers_v(conf_matrix):
    """Вычисляет коэффициент Крамера V."""
    chi2 = chi2_contingency(conf_matrix)[0]
    n = conf_matrix.sum().sum()
    phi2 = chi2 / n
    r, k = conf_matrix.shape
    return np.sqrt(phi2 / min(k - 1, r - 1))

# === Анализ ===
for col in questions:
    print(f"\n=== {col} ===")
    table = pd.crosstab(df[col], df[col_support])

    print("\n=== Таблица сопряжённости ===")
    print(table)

    chi2, p, dof, _ = chi2_contingency(table)
    V = cramers_v(table)
    N = table.to_numpy().sum()

    print(f"\nχ² = {chi2:.6f}")
    print(f"p = {p:.6f}")
    print(f"Cramer's V = {V:.6f}")
    print(f"N = {N}")
    print(f"Степеней свободы = {dof}")
