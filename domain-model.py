import pandas as pd

# Загружаем таблицу
path = 'data-251019.xlsx'
df = pd.read_excel(path)

# Исключаем анкеты с ошибками
exclude_ids = [2105212553, 2105364012, 2105434991, 2117312175, 2117477460]
df = df[~df['ID'].isin(exclude_ids)]

col_support = 'На момент начала вашей работы, описываемый далее проект был на ваш взгляд поддерживаемым'
cols_model = [c for c in df.columns if c.startswith('Какой подход использовался для представления и организации модели данных в приложении')]

percentages = {}
for col in cols_model:
    subset = df[df[col].notna()]
    if len(subset) > 0:
        total = len(subset)
        supported = (subset[col_support] == 'да').sum()
        percentages[col.split('/')[-1].strip()] = round(supported / total * 100, 1)

print('Проценты поддерживаемых проектов по каждому подходу:')
for k, v in percentages.items():
    print(f'{k}: {v}%')