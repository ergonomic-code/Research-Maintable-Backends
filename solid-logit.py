import pandas as pd
import statsmodels.api as sm
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

# === Загружаем таблицу ===
path = "data-251019.xlsx"
df = pd.read_excel(path)

# Исключаем анкеты с ошибками
exclude_ids = [2105212553, 2105364012, 2105434991, 2117312175, 2117477460]
df = df[~df["ID"].isin(exclude_ids)]

# === Подготовка признаков ===
# Бинаризуем целевую переменную
df["supportive"] = df["На момент начала вашей работы, описываемый далее проект был на ваш взгляд поддерживаемым"].map({"да": 1, "нет": 0})

# Выбираем столбцы с SOLID-принципами
principles = {
    "SRP": "Следовала ли команда The Single Responsibility Principle",
    "OCP": "Следовала ли команда The Open/Closed Principle",
    "LSP": "Следовала ли команда The Liskov Substitution Principle",
    "ISP": "Следовала ли команда The Interface Segregation Principle",
    "DIP": "Следовала ли команда The Dependency Inversion Principle"
}

# Преобразуем ответы в порядковую шкалу
order = ["никогда", "редко", "иногда", "часто", "всегда", "не знаю/не помню"]
encoder = LabelEncoder()
encoder.classes_ = order  # фиксируем порядок

for key, col in principles.items():
    df[key] = df[col].str.lower().map(lambda x: x if x in order else "не знаю/не помню")
    df[key] = df[key].map({v: i for i, v in enumerate(order)})

# === Строим модель ===
X = df[list(principles.keys())]
X = sm.add_constant(X)
y = df["supportive"]

model = sm.Logit(y, X)
result = model.fit()

print(result.summary())

# class weights для компенсации 64%/36%
weights = compute_class_weight(class_weight='balanced', classes=np.array([0, 1]), y=y)
class_weights = {0: weights[0], 1: weights[1]}

clf = LogisticRegression(class_weight=class_weights, max_iter=1000)
clf.fit(X, y)

pd.Series(clf.coef_[0], index=X.columns)