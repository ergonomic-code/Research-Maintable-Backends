import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency

# Загружаем Excel
path = 'data-251019.xlsx'
df = pd.read_excel(path)

# Исключаем некорректные ответы
exclude_ids = [2105212553, 2105364012, 2105434991, 2117312175, 2117477460]
df = df[~df['ID'].isin(exclude_ids)]

# Колонки для анализа
col_sync = 'Какой процент межсервисного взаимодействия проходил в синхронном режиме'
col_support = 'На момент начала вашей работы, описываемый далее проект был на ваш взгляд поддерживаемым'

# Таблица сопряженности и χ²
crosstab = pd.crosstab(df[col_sync], df[col_support])
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