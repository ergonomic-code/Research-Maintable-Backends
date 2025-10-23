import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency

# === Настройки ===
path = "data-251019.xlsx"
exclude_ids = [2105212553, 2105364012, 2105434991, 2117312175, 2117477460]
col_factor = "В какой момент писались тесты на этапе разработки"
col_target = "На момент начала вашей работы, описываемый далее проект был на ваш взгляд поддерживаемым"

# === Загрузка и фильтрация ===
df = pd.read_excel(path)
df = df[~df["ID"].isin(exclude_ids)].copy()
data = df[[col_factor, col_target]].dropna()

# === Таблица сопряжённости ===
crosstab = pd.crosstab(data[col_factor], data[col_target])

# Таблица сопряженности и χ²
chi2, p, dof, expected = chi2_contingency(crosstab)

n = crosstab.sum().sum()
k = min(crosstab.shape)
cramers_v = np.sqrt(chi2 / (n * (k - 1))) if k > 1 else np.nan

print('χ² =', chi2)
print('p =', p)
print('Cramer’s V =', cramers_v)
print('N =', n)
print('Степеней свободы =', dof)
print('Таблица сопряженности:')
print(crosstab)