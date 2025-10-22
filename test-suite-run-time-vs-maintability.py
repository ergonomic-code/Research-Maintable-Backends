import pandas as pd
from scipy.stats import mannwhitneyu

# === Настройки ===
path = "data-251019.xlsx"
col_support = "На момент начала вашей работы, описываемый далее проект был на ваш взгляд поддерживаемым"
col_time_full = "Сколько времени занимал запуск тестов в цикле разработки в секундах"

# === Загрузка данных ===
df = pd.read_excel(path)

# Исключаем анкеты с ошибками
exclude_ids = [2105212553, 2105364012, 2105434991, 2117312175, 2117477460]
df = df[~df["ID"].isin(exclude_ids)]

# Отбираем нужные колонки
df = df[[col_support, col_time_full]].dropna()

# === Преобразуем категории во внутренние численные ранги ===
order = {
    "0-1 секунду": 1,
    "2-10 секунд": 2,
    "10-60 секунд": 3,
    "более минуты": 4,
    "не помню": None  # выбрасываем
}

df[col_time_full] = df[col_time_full].map(order)
df = df.dropna(subset=[col_time_full])

# Бинаризуем поддерживаемость
df = df[df[col_support].isin(["да", "нет"])]
df[col_support] = df[col_support].map({"да": 1, "нет": 0})

# Разделяем группы
support_yes = df[df[col_support] == 1][col_time_full]
support_no = df[df[col_support] == 0][col_time_full]

# Тест Манна–Уитни
stat, p = mannwhitneyu(support_yes, support_no, alternative="two-sided")

# === Результаты ===
print("=== Время запуска тестов в цикле разработки (категориально) ===")
print(f"N = {len(df)}")
print(f"Медиана (поддерживаемые): {support_yes.median():.2f}")
print(f"Медиана (неподдерживаемые): {support_no.median():.2f}")
print(f"U = {stat:.3f}")
print(f"p = {p:.5f}")

# Величина эффекта (rank-biserial)
rbc = 1 - (2 * stat) / (len(support_yes) * len(support_no))
print(f"Rank-biserial correlation = {rbc:.3f}")

if p < 0.05:
    print("→ Различия статистически значимы (p < 0.05)")
else:
    print("→ Различия статистически незначимы")
