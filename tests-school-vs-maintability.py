import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency

# Загружаем Excel
path = 'data-251019.xlsx'
df = pd.read_excel(path)

# Исключаем некорректные ответы
exclude_ids = [2105212553, 2105364012, 2105434991, 2117312175, 2117477460]
df = df[~df['ID'].isin(exclude_ids)]

# Объединяем стили тестирования: детройтскую и лондонскую школы
col_style = 'Какой стиль тестирования использовался чаще всего'
col_support = "На момент начала вашей работы, описываемый далее проект был на ваш взгляд поддерживаемым"

def normalize_style(x):
    if pd.isna(x):
        return np.nan
    s = str(x).lower()
    if 'детройт' in s or 'состояние' in s or 'выходных' in s:
        return 'Детройтская'
    if 'лондон' in s or 'взаимодейств' in s:
        return 'Лондонская'
    if 'оба' in s or 'смешан' in s:
        return 'Смешанная'
    return 'Другое'

df['style_group'] = df[col_style].apply(normalize_style)

# Таблица сопряженности и χ²
crosstab = pd.crosstab(df['style_group'], df[col_support])
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