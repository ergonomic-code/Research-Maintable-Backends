import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency

# Загрузка Excel
path = 'data-251019.xlsx'
df = pd.read_excel(path)

# Исключаем анкеты с ошибками
exclude_ids = [2105212553, 2105364012, 2105434991, 2117312175, 2117477460]
df = df[~df['ID'].isin(exclude_ids)]

# Колонки для анализа
col_srp = 'Следовала ли команда The Single Responsibility Principle'
col_ocp = 'Следовала ли команда The Open/Closed Principle'
col_lsp = 'Следовала ли команда The Liskov Substitution Principle'
col_isp = 'Следовала ли команда The Interface Segregation Principle'
col_dip = 'Следовала ли команда The Dependency Inversion Principle'
col_support = 'На момент начала вашей работы, описываемый далее проект был на ваш взгляд поддерживаемым'

# Функция для расчёта χ², p и Cramer's V
def chi_summary(col):
    crosstab = pd.crosstab(df[col], df[col_support])
    chi2, p, dof, expected = chi2_contingency(crosstab)
    n = crosstab.sum().sum()
    k = min(crosstab.shape)
    cramers_v = np.sqrt(chi2 / (n * (k - 1))) if k > 1 else np.nan
    print(f'=== {col} ===')
    print('χ² =', chi2)
    print('p =', p)
    print('Cramer’s V =', cramers_v)
    print('N =', n)
    print('Степеней свободы =', dof)
    print('Таблица сопряженности:', crosstab, ' ')


chi_summary(col_isp)
chi_summary(col_lsp)
chi_summary(col_ocp)

chi_summary(col_srp)
chi_summary(col_dip)