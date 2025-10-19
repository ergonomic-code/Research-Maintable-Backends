import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency

# Загрузка Excel
path = 'data-251019.xlsx'
df = pd.read_excel(path)

# Исключаем анкеты с ошибками
exclude_ids = [2105212553, 2105364012, 2105434991, 2117312175, 2117477460]
df = df[~df['ID'].isin(exclude_ids)]

# Поиск всех колонок, относящихся к вопросу про архитектуру
cols_arch = [c for c in df.columns if c.startswith('Какая использовалась архитектура системы')]
col_support = 'На момент начала вашей работы, описываемый далее проект был на ваш взгляд поддерживаемым'

# Создаём бинарную таблицу признаков для каждого типа архитектуры
arch_df = pd.DataFrame()
for c in cols_arch:
    arch_df[c.split('/')[-1].strip()] = df[c].apply(lambda x: 1 if pd.notna(x) and str(x).strip() != '' else 0)
arch_df[col_support] = df[col_support]

# Рассчитываем χ², p и Cramer’s V для каждого типа архитектуры
results = {}
for col in arch_df.columns[:-1]:
    crosstab = pd.crosstab(arch_df[col], arch_df[col_support])
    if crosstab.shape[0] > 1 and crosstab.shape[1] > 1:
        chi2, p, dof, expected = chi2_contingency(crosstab)
        n = crosstab.sum().sum()
        k = min(crosstab.shape)
        cramers_v = np.sqrt(chi2 / (n * (k - 1))) if k > 1 else np.nan
        results[col] = dict(chi2=chi2, p_value=p, cramers_v=cramers_v, n=n, dof=dof)
    else:
        results[col] = dict(chi2=np.nan, p_value=np.nan, cramers_v=np.nan, n=crosstab.sum().sum(), dof=0)

# Вывод результатов
for k, v in results.items():
    print(f'Архитектура: {k}')
    print(f"χ² = {v['chi2']:.2f}, p = {v['p_value']:.3f}, Cramer’s V = {v['cramers_v']:.2f}, N = {v['n']}, dof = {v['dof']}")
    print('---')