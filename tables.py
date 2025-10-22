import pandas as pd
import numpy as np
from scipy.stats import mannwhitneyu

# === Настройки ===
path = 'data-251019.xlsx'

# Загружаем таблицу
df = pd.read_excel(path)

# Исключаем анкеты с ошибками
exclude_ids = [2105212553, 2105364012, 2105434991, 2117312175, 2117477460]
df = df[~df["ID"].isin(exclude_ids)]

# Названия нужных колонок
col_loc = "Количество таблиц (коллекций) во всех хранилищах данных проекта"
col_support = "На момент начала вашей работы, описываемый далее проект был на ваш взгляд поддерживаемым"

# Преобразуем в числовой тип и убираем пропуски
df[col_loc] = pd.to_numeric(df[col_loc], errors="coerce")
valid = df[[col_loc, col_support]].dropna()

# Разделяем по группам
supported = valid[valid[col_support] == "да"][col_loc]
unsupported = valid[valid[col_support] == "нет"][col_loc]

# Проверяем распределения
print("Медиана поддерживаемых:", np.median(supported))
print("Медиана неподдерживаемых:", np.median(unsupported))

# Тест Манна–Уитни
stat, p = mannwhitneyu(supported, unsupported, alternative="two-sided")

print("\nMann–Whitney U =", stat)
print("p-value =", p)

if p < 0.05:
    print("→ Различие статистически значимо (p < 0.05)")
else:
    print("→ Различие статистически незначимо (p ≥ 0.05)")
