import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# === Настройки ===
path = "data-251019.xlsx"
col_loc = "Количество таблиц (коллекций) во всех хранилищах данных проекта"
col_support = "На момент начала вашей работы, описываемый далее проект был на ваш взгляд поддерживаемым"

# === Загрузка и подготовка ===
df = pd.read_excel(path)
exclude_ids = [2105212553, 2105364012, 2105434991, 2117312175, 2117477460, 2119818009]
df = df[~df["ID"].isin(exclude_ids)]

# Приводим к числовым типам и фильтруем аномалии
df = df[pd.to_numeric(df[col_loc], errors="coerce").notna()]
df[col_loc] = df[col_loc].astype(float)
df = df[df[col_loc] > 0]

# === Boxplot + swarmplot ===
plt.figure(figsize=(8, 6))
sns.boxplot(data=df, x=col_support, y=col_loc, whis=1.5, showfliers=False)
sns.swarmplot(data=df, x=col_support, y=col_loc, color="0.25", size=4)
plt.yscale("log")  # логарифмическая шкала для удобства
plt.xlabel("Поддерживаемость проекта (да / нет)")
plt.ylabel("Количество таблиц (лог шкала)")
plt.title("Количество таблиц vs Поддерживаемость")
plt.tight_layout()
plt.savefig("tables_vs_maint_boxplot.png", dpi=300)
plt.close()

# === Создаем бины LOC по квантилям ===
df["LOC_bin"] = pd.qcut(df[col_loc], q=6, duplicates="drop")

# === Доля поддерживаемых в каждом бине ===
support_rate = (
    df.groupby("LOC_bin")[col_support]
    .apply(lambda x: (x == "да").mean())
    .reset_index(name="Support share")
)

# === Средний LOC в каждом бине (для оси X) ===
loc_mean = df.groupby("LOC_bin")[col_loc].mean().reset_index(name="Mean LOC")
support_rate["Mean LOC"] = loc_mean["Mean LOC"]

# === Визуализация ===
plt.figure(figsize=(8, 6))
sns.lineplot(data=support_rate, x="Mean LOC", y="Support share", marker="o")
plt.xscale("log")
plt.xlabel("Количество таблиц (лог шкала)")
plt.ylabel("Доля поддерживаемых проектов")
plt.title("Тренд поддерживаемости в зависимости от количества таблиц")
plt.ylim(0, 1)
plt.grid(True, which="both", axis="both", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig("tables_vs_maint_trend.png", dpi=300)
plt.close()
