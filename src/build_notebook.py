"""Build the reproducible research notebook with nbformat."""
from pathlib import Path
import nbformat as nbf


def build():
    cells = []
    def md(text): cells.append(nbf.v4.new_markdown_cell(text))
    def code(text): cells.append(nbf.v4.new_code_cell(text))
    md('''# Экономические соседи муниципалитетов
Ноутбук проверяет исходные данные, повторяет кластеризацию и строит графики
результатов. СберИндекс, направление «Кластеризация». Данные получены 09.10.2026.

## Основные результаты
Полная история охватывает 2 016 МО за 24 месяца. 1 003 МО не меняют согласованную
группу ни разу; средняя месячная доля переходов — 9,65%. Сеть потребительского
сходства выбрана для поиска аналогов, а не для рейтинга благосостояния.
KMeans имеет больший силуэт, чем выбранный сетевой метод.
Переходы и связь с доступностью рынков не доказывают причинность.''')
    md('''## Задача и методика
Вопрос: как меняются группы муниципалитетов, похожих по структуре потребления?
Методы реализованы в `src/prepare.py`, `src/network.py`, `src/pipeline.py`.
Ноутбук заново готовит панель и выполняет расчёт из исходных файлов.

### Допущения и ограничения
- Расходы — модельные средние безналичные расходы жителей, рубли.
- Доли нормируются на сумму пяти категорий, а не на все расходы.
- Пропуски не равны нулям. Используется полная панель фиксированных territory_id.
- Признаки sqrt(долей); веса kNN отражают сходство, а не потоки покупателей.
- Настройка только на январе 2023; затем параметры фиксированы.
- PCA служит отображением, не определяет соседей. Подробные метрики: `docs/metrics.md`.
- Два года не позволяют надёжно отделить сезонность от структурных изменений.

Источники: [официальный архив](https://www.sberbank.com/common/img/uploaded/files/pdf/sberindex/hackathonlicence.zip),
[справочник территорий](https://sberindex.ru/ru/research/dataset-borders-and-changes-of-municipalities).
Исходные и производные данные — СберИндекс, CC BY-SA 4.0. Изменения в проекте:
отбор полной панели, расчёт долей, сетей, групп и показателей.''')
    md('### 1. Настройка среды и конфигурация\nПеред открытием: `pip install -r requirements.txt`. Запуск из корня проекта либо `notebooks/`.')
    code('''from pathlib import Path
import os, sys, json, contextlib, io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display, Markdown

ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents]
            if (p / "config.json").is_file() and (p / "src/prepare.py").is_file())
os.chdir(ROOT)
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
os.environ["LOKY_MAX_CPU_COUNT"] = "2"
CONFIG = json.loads(Path("config.json").read_text())
plt.rcParams.update({"figure.figsize": (10, 4.6), "figure.dpi": 120,
    "font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": .2, "axes.axisbelow": True})
COLORS = ["#3568a5", "#728445", "#ca9643", "#c16688", "#986940"]
display(pd.Series(CONFIG, name="Конфигурация").to_frame())''')
    md('## Данные\n### 2. Контроль файлов, схемы и полноты')
    code('''from src.inventory import inventory
from src.prepare import prepare, CATEGORIES
files = inventory("data/raw")
display(pd.DataFrame(files["files"])[["file", "bytes", "sha256"]])
audit = prepare("data/raw/hackathonlicence", "data/processed")
display(pd.Series({k:v for k,v in audit.items() if not isinstance(v,list)}, name="Аудит").to_frame())
frame = pd.read_parquet("data/raw/hackathonlicence/consumption.parquet")
assert not frame.duplicated(["territory_id", "date", "category"]).any()
assert not frame.isna().any().any()
assert (frame.value > 0).all()
panel = np.load("data/processed/panel.npz")
assert np.allclose(panel["shares"].sum(axis=2), 1)
assert np.all(panel["values"].sum(axis=2) <= panel["total"])
assert audit["included_ids"] == 2016 and audit["excluded_ids"] == 174
display(Markdown("**Проверки пройдены:** ключи уникальны, значения положительны; доли суммируются в 1, корзина не превышает total."))''')
    md('### 3. Где отсутствуют наблюдения\nЗнаменатель: 2 190 исходных МО × 6 категорий в каждом месяце. Это пропущенные строки, не нулевые расходы.')
    code('''expected = frame.territory_id.nunique() * frame.category.nunique()
missing = (expected - frame.groupby("date").size()).reindex(panel["months"])
assert missing.sum() == audit["missing_panel_rows"]
fig, ax = plt.subplots()
ax.bar(pd.to_datetime(missing.index), missing.values, width=24, color=COLORS[0])
ax.set(title="Пропуски в исходной панели · 2023–2024 · 2 190 МО", ylabel="Отсутствующие строки МО × категория", xlabel="Месяц")
fig.autofmt_xdate(); fig.tight_layout(); plt.show()
display(Markdown(f"Отсутствуют **{missing.sum():,}** строк (**{missing.sum() / (expected*24):.2%}**). Для динамики оставлены МО с полной историей; это может влиять на представительность."))''')
    md('### 4. Проверка вспомогательных таблиц\nДороги и железные дороги — статический срез конца 2024, не измерение мобильности. В модели используется только автомобильная близость, и только в дополнительном сравнении.')
    code('''connections = pd.read_parquet("data/raw/hackathonlicence/connection.parquet")
connection_audit = []
for kind, part in connections.groupby("type"):
    pairs = pd.DataFrame({"a":np.minimum(part.territory_id_x, part.territory_id_y),
                          "b":np.maximum(part.territory_id_x, part.territory_id_y)})
    connection_audit.append({"Тип":kind, "Строк":len(part),
        "Неположительных расстояний":int((part.distance<=0).sum()),
        "Повторов неупорядоченных пар":int(pairs.duplicated().sum())})
display(pd.DataFrame(connection_audit))
market = pd.read_parquet("data/raw/hackathonlicence/market_access.parquet")
assert not market.territory_id.duplicated().any()
display(pd.DataFrame({"Источник":["Доступность рынков"], "МО":[len(market)],
    "Минимум":[market.market_access.min()], "Максимум":[market.market_access.max()]}))
del connections
display(Markdown("148 неположительных дорожных расстояний исключаются. Дубли обратных железнодорожных пар не интерпретируются как дополнительные связи; железные дороги в модель не входят."))''')
    md('## Результаты расчёта\n### 5. Повторный расчёт\nХод расчёта записывается в `outputs/analysis/notebook-run.log`. Результаты сохраняются в `outputs/analysis/`.')
    code('''from src.pipeline import run
from threadpoolctl import threadpool_limits
with open("outputs/analysis/notebook-run.log", "w") as log, contextlib.redirect_stdout(log), threadpool_limits(limits=2):
    summary = run("data/processed", "data/raw/hackathonlicence", "config.json", "outputs/analysis")
base = Path("outputs/analysis")
comparison = pd.read_csv(base/"comparison.csv")
dynamics = pd.read_csv(base/"dynamics.csv")
smoothing = pd.read_csv(base/"smoothing-sensitivity.csv")
profiles = pd.read_csv(base/"profiles.csv")
display(pd.Series(summary["selected"], name="Выбрано на январе 2023").to_frame())''')
    md('### 6. Сравнение методов на одной сети\nПоказаны лучшие по SW конфигурации каждого метода при kNN=30 и минимальном размере группы 2%. Полные 27 вариантов — `comparison.csv`. Сетевой метод выбирается среди Spectral и Louvain; KMeans — признаковый ориентир.')
    code('''same_graph = comparison[(comparison.neighbors==30) & (comparison.min_cluster_fraction>=.02)]
best = same_graph.sort_values("SW",ascending=False).groupby("method",sort=False).head(1)
display(best[["method", "parameter", "clusters", "SW", "CH", "S_Dbw", "AVI", "AVU", "MQ"]].round(4))
fig, ax = plt.subplots(figsize=(9,4))
ax.barh(best.method, best.SW, color=COLORS[:len(best)])
ax.set(xlim=(0,.4), xlabel="SW · больше — лучше", title="Силуэт на одинаковых признаках и МО · январь 2023")
for y,value in enumerate(best.SW): ax.text(value+.006,y,f"{value:.3f}",va="center")
fig.tight_layout(); plt.show()
display(Markdown("KMeans разделяет признаки лучше по SW; Spectral выбран для исследования сети. Высокие сетевые индексы на графе из тех же признаков не являются внешней валидацией."))''')
    md('### 7. Как менялась корзина\nКаждый МО имеет одинаковый вес. Доля — от суммы пяти категорий. Это не распределение всех расходов населения России.')
    code('''basket = panel["shares"].mean(axis=1)
fig, ax = plt.subplots(figsize=(10,5))
for index, category in enumerate(CATEGORIES):
    ax.plot(pd.to_datetime(panel["months"]), 100*basket[:,index],
            color=COLORS[index], label=category, marker="o", markersize=3)
ax.set(ylabel="Средняя доля выбранной корзины, %", xlabel="Месяц", ylim=(0,70),
       title="Структура потребления · равный вес 2 016 МО · 2023–2024")
ax.legend(loc="upper center",bbox_to_anchor=(.5,-.19),ncol=3,frameon=False)
fig.autofmt_xdate(); fig.tight_layout(); plt.show()
display(pd.DataFrame({"Категория":CATEGORIES,"Январь 2023, %":100*basket[0],
                      "Декабрь 2024, %":100*basket[-1]}).round(2))
display(Markdown("Маркетплейсы занимают большую долю выбранной корзины к концу периода. Сравнение января с декабрём не очищено от сезонности; изменение долей не равно изменению доходов."))''')
    md('### 8. Переходы и сглаживание на одинаковых месяцах\nДля обоих рядов 21 переход: апрель 2023 — декабрь 2024. Сглаживание: среднее уровней за три текущих/предыдущих месяца, затем новые доли и группы.')
    code('''common = smoothing.dropna(subset=["ARI_previous_trailing3", "ARI_previous_raw"])
assert len(common)==21
fig, ax = plt.subplots()
ax.plot(pd.to_datetime(common.month),common.ARI_previous_raw,label="Исходные уровни",color=COLORS[0],marker="o")
ax.plot(pd.to_datetime(common.month),common.ARI_previous_trailing3,label="Среднее за 3 месяца",color=COLORS[2],linestyle="--",marker="s")
ax.set(ylim=(0,1), ylabel="ARI соседних месяцев", xlabel="Месяц",title="Сглаживание повышает устойчивость · 2 016 МО")
ax.legend(frameon=False); fig.autofmt_xdate(); fig.tight_layout(); plt.show()
seed_check = pd.read_csv(base/"seed-stability.csv")
yearly = pd.read_csv(base/"same-month-year.csv")
display(pd.DataFrame({"Показатель":["Средний ARI исходного ряда, общие месяцы", "Средний ARI сглаженного ряда, общие месяцы", "Минимальный ARI проверки seeds", "Средний ARI одинаковых месяцев 2023/2024"],
    "Значение":[common.ARI_previous_raw.mean(),common.ARI_previous_trailing3.mean(),seed_check.ARI_to_seed42.min(),yearly.ARI.mean()]}).round(4))
display(Markdown(f"**{summary['never_changed']:,}** МО сохранили группу во всех месяцах. Средняя доля переходов во всём периоде: **{summary['mean_changed_fraction']:.2%}**. Сглаживание уменьшает краткосрочную вариативность, но не устраняет сезонность."))''')
    md('### 9. Потребительская и транспортная близость\nТри подхода рассчитаны на одинаковых 2 004 МО крупнейшей дорожной компоненты, декабрь 2024; Spectral, K=3, kNN=30. Оценка в одном потребительском пространстве.')
    code('''proximity = pd.read_csv(base/"proximity-common-population.csv")
display(proximity[["graph", "municipalities", "clusters", "SW", "CH", "S_Dbw", "AVI", "AVU", "MQ"]].round(4))
assert proximity.municipalities.nunique()==1
display(Markdown("Дорожные группы почти не разделяют потребительские профили (SW около 0,004). Это аргумент для поиска аналогов по расходам, а не оценка транспортной эффективности."))''')
    md('### 10. Внешний показатель: доступность рынков\nПоказатель 2024 года не включён в обучение. Выборка — 2 004 МО с известной доступностью; это отдельное правило отбора от дорожного сравнения.')
    code('''associations = pd.read_csv(base/"market-associations.csv")
fig, ax = plt.subplots(figsize=(10,4.6))
ax.barh(associations.category, associations.rho, color=COLORS)
ax.axvline(0,color="#444444",linewidth=1)
ax.set(xlim=(-1,1), xlabel="Spearman ρ",title="Доступность рынков и средняя корзина 2024 · 2 004 МО")
for index,value in enumerate(associations.rho):
    ax.text(value+(.025 if value>=0 else -.025),index,f"{value:+.3f}",
            ha="left" if value>=0 else "right",va="center")
fig.tight_layout(); plt.show()
display(associations[["category", "n", "rho"]].round(4))
display(Markdown("Продовольственная доля отрицательно связана с доступностью, доля здоровья — положительно. Возможны общие причины и композиционные зависимости; причинный эффект не установлен."))''')
    md('''## Выводы
Практическая цель — выбрать потребительские аналоги муниципалитета и проверить,
как их профиль и состав меняются по месяцам. Фиксированная PCA-проекция используется
для отображения профилей; соседи определяются по полному пространству признаков.
Пример паспорта Майкопа приведён в презентации. Истории и связи всех МО сохранены
в `outputs/analysis/explorer-nodes.json`.

Группы описательны. Их нельзя прямо считать рейтингом благосостояния, инвестиционной
привлекательности или работы администрации. Не идентифицированы влияния покрытия
банка, наличных, инфляции и состава жителей.

Исполняемый код и исходные определения доступны в проекте; контрольные суммы —
`outputs/input-inventory.json`, метрики — `docs/metrics.md`, методологический отчёт —
`docs/methodological-report.md`. Лицензия данных: **CC BY-SA 4.0**, СберИндекс.''')
    notebook = nbf.v4.new_notebook(cells=cells,metadata={
        'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},
        'language_info':{'name':'python','version':'3.12'}})
    nbf.validate(notebook)
    Path('notebooks').mkdir(exist_ok=True)
    nbf.write(notebook,'notebooks/economic-neighbors.ipynb')
    print('Created notebooks/economic-neighbors.ipynb')


if __name__=='__main__': build()
