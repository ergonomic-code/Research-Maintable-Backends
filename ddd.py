import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
import difflib
import re

# Загрузка Excel
path = 'data-251019.xlsx'
df = pd.read_excel(path)

# Исключаем анкеты с ошибками
exclude_ids = [2105212553, 2105364012, 2105434991, 2117312175, 2117477460]
df = df[~df['ID'].isin(exclude_ids)]

# Поиск всех колонок, относящихся к вопросу про DDD
cols_ddd = [
'Ограниченные контексты (bounded contexts - Отдельные модели и термины в разных частях системы.)',
'Единый язык (ubiquitous Language - общий словарь между разработчиками и бизнесом, отражённый в коде)',
'Объекты-значения (value objects - использовались неизменяемые объекты, определяемые по значению)',
'Не применялись',
'Богатая доменная модель (rich domain model - Сущности и value-объекты содержат поведение, а не только данные)',
'Доменные события (domain events - Важные бизнес-события публиковались и обрабатывались как часть модели.)',
'Явные агрегаты (aggregates - Границы консистентности определены, один "root" управляет изменениями внутри агрегата.)',
'Участие экспертов предметной области в проектировании модели',
'Эвентшторминг (event storming - фасилитированная сессия по разработке модели предметной области, при которой участники совместно создают гибкую визуальную карту бизнес-процессов с фокусом на события, происходящие в системе.)',
'Карты контекстов (context map - явное описание взаимодействия между ограниченными контекстами)'
]

col_support = 'На момент начала вашей работы, описываемый далее проект был на ваш взгляд поддерживаемым'

# Создаём бинарную таблицу признаков для каждого признака DDD
results = {}


def _find_best_column_name(desired: str, columns: list[str]) -> str | None:
    """Try to match a desired term to an actual DataFrame column name.

    Strategies (in order):
    - exact match
    - case-insensitive exact
    - case-insensitive substring
    - token intersection (all tokens from desired are present in column)
    - fuzzy close match using difflib (cutoff 0.6)

    Returns actual column name or None if no reasonable match found.
    """
    if desired in columns:
        return desired

    desired_l = desired.lower()
    cols_l = [c.lower() for c in columns]

    # case-insensitive exact
    for orig, low in zip(columns, cols_l):
        if low == desired_l:
            return orig

    # substring
    for orig, low in zip(columns, cols_l):
        if desired_l in low:
            return orig

    # token match: all non-empty tokens of desired appear in column
    desired_tokens = [t for t in re.split(r"\W+", desired_l) if t]
    if desired_tokens:
        for orig, low in zip(columns, cols_l):
            if all(tok in low for tok in desired_tokens):
                return orig

    # fuzzy
    cand = difflib.get_close_matches(desired, columns, n=1, cutoff=0.6)
    if cand:
        return cand[0]

    return None


for c in cols_ddd:
    matched = _find_best_column_name(c, df.columns.tolist())
    if matched is None:
        print(f"Warning: DDD column not found for '{c}' — skipping")
        # keep a placeholder in results for completeness
        results[c.split('/')[-1].strip()] = dict(chi2=np.nan, p=np.nan, cramers_v=np.nan, n=0, dof=0)
        continue

    # create binary indicator from the matched column without overwriting original
    series = df[matched]
    binary = series.apply(lambda x: 1 if pd.notna(x) and str(x).strip() != '' else 0)
    crosstab = pd.crosstab(binary, df[col_support])

    if crosstab.shape[0] > 1 and crosstab.shape[1] > 1:
        chi2, p, dof, expected = chi2_contingency(crosstab)
        n = int(crosstab.sum().sum())
        k = min(crosstab.shape)
        cramers_v = np.sqrt(chi2 / (n * (k - 1))) if k > 1 else np.nan
        results[matched.split('/')[-1].strip()] = dict(
            chi2=chi2,
            p=p,
            cramers_v=cramers_v,
            n=n,
            dof=dof,
            crosstab=crosstab,
        )
    else:
        results[matched.split('/')[-1].strip()] = dict(
            chi2=np.nan,
            p=np.nan,
            cramers_v=np.nan,
            n=int(crosstab.sum().sum()),
            dof=0,
            crosstab=crosstab,
        )

# Вывод результатов
for k, v in results.items():
    print(f'DDD-признак: {k}')
    print(f"χ² = {v['chi2']:.2f}, p = {v['p']:.3f}, Cramer’s V = {v['cramers_v']:.2f}, N = {v['n']}, dof = {v['dof']}")
    crosstab = v.get('crosstab')
    if crosstab is not None:
        print('Таблица сопряжённости:')
        print(crosstab)
    print('---')


def find_columns_with_terms(df: pd.DataFrame, terms: list[str], *, case_sensitive: bool = False, regex: bool = False, min_matches: int = 1) -> dict:
    """Search dataframe columns for cells containing any of the provided terms.

    Returns a dict mapping column name -> dict(match_count=int, samples=list[str])

    - case_sensitive: perform case-sensitive matching
    - regex: interpret terms as regular expressions
    - min_matches: only include columns with at least this many matching rows
    """
    out: dict = {}
    if not terms:
        return out

    # prepare compiled terms if regex
    if regex:
        import re

        flags = 0 if case_sensitive else re.IGNORECASE
        patterns = [re.compile(t, flags=flags) for t in terms]

        def matches_any(s: str) -> bool:
            if not isinstance(s, str):
                return False
            return any(p.search(s) for p in patterns)

    else:
        lowered = [t if case_sensitive else t.lower() for t in terms]

        def matches_any(s: str) -> bool:
            if not isinstance(s, str):
                return False
            ss = s if case_sensitive else s.lower()
            return any(t in ss for t in lowered)

    # scan columns
    for col in df.columns:
        # only apply to object/string columns and numeric columns converted to str
        series = df[col].dropna().astype(str)
        mask = series.apply(matches_any)
        count = int(mask.sum())
        if count >= min_matches:
            samples = series[mask].unique().tolist()[:5]
            out[col] = dict(match_count=count, samples=samples)

    return out
