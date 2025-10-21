import pandas as pd
import statsmodels.api as sm
from sklearn.preprocessing import LabelEncoder

# === Настройки ===
path = "data-251019.xlsx"

# Загружаем таблицу
df = pd.read_excel(path)

# Исключаем анкеты с ошибками
exclude_ids = [2105212553, 2105364012, 2105434991, 2117312175, 2117477460]
df = df[~df["ID"].isin(exclude_ids)]

# Целевая переменная — поддерживаемость
df["supportive"] = df["На момент начала вашей работы, описываемый далее проект был на ваш взгляд поддерживаемым"].map({
    "да": 1,
    "нет": 0
})

# Преобразуем частоту применения шаблонов проектирования в числовую шкалу
order = ["никогда", "редко", "иногда", "часто", "всегда", "не знаю/не помню"]
df["DesignPatterns"] = df["Применялись ли шаблоны проектирования"].str.lower().map(lambda x: x if x in order else "не знаю/не помню")
mapping = {v: i for i, v in enumerate(order)}
df["DesignPatterns"] = df["DesignPatterns"].map(mapping)

# === Логистическая регрессия ===
X = sm.add_constant(df["DesignPatterns"])
y = df["supportive"]

model = sm.Logit(y, X)
result = model.fit()

print(result.summary())
