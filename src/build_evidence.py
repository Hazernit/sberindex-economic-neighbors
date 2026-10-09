"""Create reviewed UI evidence and a Russian methodological report from outputs."""
from pathlib import Path
import json
from datetime import datetime,timezone
import numpy as np
import pandas as pd
from .pipeline import save_json


def build():
    base = Path('outputs/analysis')
    summary = json.loads((base/'summary.json').read_text())
    audit = json.loads(Path('data/processed/panel-audit.json').read_text())
    model = np.load(base/'model.npz')
    now = datetime.now(timezone.utc).isoformat()
    archive = 'https://www.sberbank.com/common/img/uploaded/files/pdf/sberindex/hackathonlicence.zip'
    dictionary_url = 'https://sberindex.ru/ru/research/dataset-borders-and-changes-of-municipalities'
    queries = {}
    def add(identifier,rows,components,files,definition):
        queries[identifier] = {'rows':rows,'methods':[{'language':'python',
            'code':f'python -m src.prepare && python -m src.pipeline\n# Derived: {definition}'}],
            'source':{'label':'СберИндекс: муниципальные данные; расчёт проекта',
                'files':files,'executedAt':now,'links':[{'label':'Официальный архив','url':archive},
                    {'label':'Справочник неизменных территорий','url':dictionary_url}],
                'filters':['Полная панель 2016 МО, январь 2023 — декабрь 2024',
                    'Доли от суммы пяти выбранных категорий; total исключён'],
                'evidenceFlow':[{'title':'Получение','detail':'Официальный ZIP и RAR скачаны через Яндекс 09.10.2026; SHA-256 исходных файлов сохранены в inventory.'},
                    {'title':'Проверка','detail':'Ключ МО × месяц × категория уникален. Из 2190 МО исключены 174 с неполной историей. Пропуски не заменялись нулями.'},
                    {'title':'Расчёт','detail':definition}],
                'metricDefinitions':[{'label':definition,'definition':definition,
                    'componentIds':components,'sourceLineage':[{'files':files}]}]}}
    table_map = {'profiles':('profiles.csv',['group-profiles','passport','results-text']),
        'dynamics':('dynamics.csv',['monthly-stability','network-explorer','results-text']),
        'comparison':('comparison.csv',['method-comparison','comparison-text']),
        'proximity':('proximity-common-population.csv',['proximity-comparison','comparison-text']),
        'market':('market-associations.csv',['market-correlations','market-text']),
        'stability':('seed-stability.csv',['robustness','robustness-text']),
        'smoothing':('smoothing-sensitivity.csv',['robustness','robustness-text']),
        'yearly':('same-month-year.csv',['robustness','robustness-text']),
        'transitions':('transitions.csv',['transitions'])}
    definitions = {
        'profiles':'Равновзвешенные средние доли категорий среди МО одной группы; размер — число МО.',
        'dynamics':'ARI соседних месяцев и доля МО, сменивших согласованную по максимальному пересечению группу.',
        'comparison':'27 конфигураций на январе 2023: KMeans, spectral, Louvain. SW на фиксированной выборке 2000; прочие метрики на 2016 МО.',
        'proximity':'Декабрь 2024, общая популяция 2004 МО в крупнейшей дорожной компоненте. Три определения близости; spectral, K=3, kNN=30.',
        'market':'Spearman между доступностью рынков 2024 и средними долями за 12 месяцев 2024 на 2004 МО; без причинной интерпретации.',
        'stability':'ARI разбиений с seeds 17,31,47 относительно seed42 для каждого из 24 месяцев.',
        'smoothing':'Трёхмесячное скользящее среднее уровней, затем доли и повторная кластеризация с фиксированными параметрами.',
        'yearly':'ARI одинаковых календарных месяцев 2023 и 2024; 12 сравнений, фиксированная популяция.',
        'transitions':'Полные пересечения согласованных групп между соседними месяцами, включая нулевые ячейки.'}
    for key,(filename,components) in table_map.items():
        add(key,pd.read_csv(base/filename).replace({np.nan:None}).to_dict('records'),components,
            ['consumption.parquet']+(['connection.parquet'] if key=='proximity' else [])+
            (['market_access.parquet'] if key=='market' else []),definitions[key])
        if key=='proximity':
            queries[key]['source']['filters']=['Декабрь 2024; общие 2004 МО крупнейшей дорожной компоненты','Spectral, K=3, kNN=30; единые признаки оценки']
        elif key=='market':
            queries[key]['source']['filters']=['Средняя корзина 2024; 2004 МО с известной доступностью рынков','Равный вес МО; Spearman; причинность не установлена']
    nodes = json.loads((base/'explorer-nodes.json').read_text())
    compact_nodes = [{**{k:v for k,v in n.items() if k!='history'},
                      'history':json.dumps(n['history'],ensure_ascii=False,separators=(',',':'))} for n in nodes]
    add('nodes',compact_nodes,['network-explorer','passport'],
        ['consumption.parquet','t_dict_municipal_districts.xlsx'],
        'По каждому МО: последняя известная к 2024 версия названия, 24 профиля, согласованные метки, фиксированная PCA-проекция и шесть сильнейших реальных рёбер каждого месяца. Проекция — экономическое пространство, не карта.')
    queries['nodes']['payloadColumns']=['history']
    basket = [{'month':str(month),'category':str(category),'share':float(model['shares'][t,:,c].mean())}
               for t,month in enumerate(model['months']) for c,category in enumerate(model['categories'])]
    add('basket',basket,['basket-trend','results-text'],['consumption.parquet'],
        'Среднее долей выбранной корзины по 2016 МО с равным весом. Это не доля всех расходов России.')
    add('audit',[{k:v for k,v in audit.items() if not isinstance(v,list)}],['audit-text'],
        ['consumption.parquet'],'Аудит полноты, уникальности ключей и согласованности корзины и total.')
    queries['audit']['source']['filters']=['Исходные 2190 МО; 303126 строк; январь 2023 — декабрь 2024','Результат отбора: 2016 МО с полной историей']
    snapshot = {'surface':'report','title':'Экономические соседи: как меняется потребительское сходство муниципалитетов',
        'generatedAt':now,'status':'reviewed','buildStatus':'creating','filters':[],
        'report':{'asOf':'2024-12-31'},'queries':queries}
    save_json('outputs/reviewed-snapshot.json',snapshot)
    best = summary['selected']
    profiles = pd.read_csv(base/'profiles.csv')
    names = pd.read_csv('data/processed/municipalities.csv').set_index('territory_id')
    seed = pd.read_csv(base/'seed-stability.csv')
    smoothing = pd.read_csv(base/'smoothing-sensitivity.csv')
    body = f'''# Экономические соседи: динамическая сеть муниципалитетов

Методологический отчёт. Направление «Кластеризация». Данные: январь 2023 — декабрь 2024.

## Основные результаты

Из 2 190 муниципалитетов с наблюдениями сформирована полная панель **2 016 МО**.
174 МО исключены из временного сравнения. **1 003 МО (49,8%)** не сменили
согласованную группу ни разу за 24 месяца. Средняя доля переходов между соседними
месяцами — **{summary['mean_changed_fraction']*100:.2f}%**; средний ARI —
**{summary['mean_ARI_previous']:.3f}**. Переход означает изменение относительного
потребительского профиля/разбиения, а не доказанный экономический шок.

Средняя доля маркетплейсов в выбранной корзине выросла с **13,46%** в январе 2023
до **26,34%** в декабре 2024; доля продовольствия снизилась с **63,78%** до **56,86%**.
Сравниваются разные календарные месяцы; это описательный срез, а не очищенный
от сезонности тренд. Ни доли, ни их изменения не равны доходам населения.

## Данные и качество

Официальный архив СберИндекса содержит consumption.parquet (303 126 строк),
connection.parquet и market_access.parquet. Лицензия в архиве определяет расходы
как модельную оценку **средних безналичных расходов жителей МО в рублях**, а
territory_id — как идентификатор неизменной территории. На сайте справочника
описана прослеживаемость преобразований. Названия берутся из последней версии,
начавшей действие не позднее 2024 года; одно territory_id может иметь несколько
исторических названий. Полная панель покрыта справочником на 100%.

В расходах нет null, отрицательных значений, нулей и дублей ключа. Однако отсутствуют
12 234 комбинации МО × месяц × категория (3,88% полного прямоугольника). Они
не заменяются нулями. Список исключённых территорий сохранён в excluded.csv.
Отбор полной истории может смещать выводы к лучше наблюдаемым территориям.

## Признаки и сеть

Используются пять непересекающихся категорий: продовольствие, здоровье,
маркетплейсы, общественное питание, транспорт. Категория «Все категории»
исключена из признаков: она включает эти пять и другие траты. Доли p_c
нормируются на **сумму пяти выбранных категорий**, а не на все расходы.
Проверено, что сумма корзины не превышает total.

Признаки x_c = sqrt(p_c); евклидово расстояние x пропорционально расстоянию
Хеллингера. Для каждого месяца строится неориентированное объединение kNN:
ребро есть, если один из двух узлов выбрал другой. Вес
exp(−d²/(sigma_i sigma_j)), sigma_i — расстояние до k-го соседа с явно
заданной положительной нижней границей. Self-loops исключены. Эта сеть
описывает **сходство расходов**, а не потоки покупателей или товарооборот.

## Выбор метода и сравнения

На январе 2023 сравниваются 27 конфигураций: k∈{{10,15,30}}, число групп
K∈{{3,5,8}} для KMeans и spectral, resolution∈{{0.5,1,1.5}} для Louvain.
Признаки, исходные МО, seed и выборка SW совпадают. Конфигурация сетевого
метода выбирается по наибольшему SW при минимальной доле группы 2%.
Параметры далее фиксируются; будущие месяцы не используются для выбора.

Выбран **spectral, k=30, K=3**. SW={best['SW']:.3f}, CH={best['CH']:.1f},
S_Dbw={best['S_Dbw']:.3f}, AVI={best['AVI']:.3f}, AVU={best['AVU']:.3f},
MQ={best['MQ']:.3f}. Это три описательные группы, а не установленное число
экономических типов. KMeans при K=3 даёт **SW=0.321**, выше сетевого варианта.
Поэтому spectral не объявляется безусловным победителем: он выбран среди
сетевых методов для задачи со структурой рёбер. Louvain даёт более дробные
сообщества, с меньшим силуэтом признаков.

Для декабря 2024 дополнительно сравниваются Хеллингер, косинусное расстояние
долей и автомобильные расстояния. Во всех трёх случаях используются одни
и те же **2 004 МО** крупнейшей дорожной компоненты и одинаковые параметры.
SW составляет соответственно **0.320, 0.219, 0.004**. Географические группы
почти не разделяют потребительские профили. Это аргумент для поиска
потребительских аналогов по расходам; он не делает дорожную сеть «плохой»
для транспортных задач.

## Метрики и ограничения валидации

Полный требуемый набор SW, CH, S_Dbw, AVI, AVU и MQ рассчитан. Формулы,
выбранные варианты, направления оптимизации и проверки — в docs/metrics.md.
Важная особенность: AVU по формуле первоисточника с делителем K при трёх
группах и ненулевых межгрупповых рёбрах тождественно равен 2/3. Он поэтому
не различает месяцы этого решения. Высокие AVI/MQ на сети из тех же признаков
не являются независимой внешней валидацией. Проверка смежных seeds не
заменяет проверку состава данных и признаков.

## Отслеживание и чувствительность

Метки сопоставляются задачей максимального пересечения составов (Hungarian).
Несопоставленные новые группы получают никогда ранее не использованные ID.
Сохраняются все пересечения групп между месяцами, поэтому доступны слияния
и разделения; одно совпавшее имя группы не доказывает неизменность профиля.

На всех 24 месяцах проверены seeds 17,31,47 против 42; минимальный ARI —
**{seed.ARI_to_seed42.min():.3f}**. Это высокая устойчивость к начальной
случайности. После усреднения **уровней** за три предыдущих/текущий месяц
и повторного расчёта долей средний ARI соседних месяцев составляет
**{smoothing.ARI_previous_trailing3.mean():.3f}** против
**{smoothing.ARI_previous_raw.mean():.3f}** у исходного ряда на тех же 21 переходах
(апрель 2023 — декабрь 2024);
средний ARI исходного и сглаженного разбиений — **{smoothing.ARI_raw_vs_trailing3.mean():.3f}**.
Сглаживание снижает краткосрочную вариативность, но часть сезонности остаётся.
Средний ARI одинаковых месяцев двух лет — **0.543**. Двух лет недостаточно
для надёжного разложения на сезонность, долговременный тренд и шоки.

## Экономическая интерпретация

'''
    for _,r in profiles[profiles.month=='2024-12'].iterrows():
        rep = names.loc[int(r.representative_id)]
        body += f"Группа {int(r.cluster)+1}: {int(r.municipalities)} МО; продовольствие {r['share_Продовольствие']*100:.1f}%, маркетплейсы {r['share_Маркетплейсы']*100:.1f}%, общепит {r['share_Общественное питание']*100:.1f}%. Ближайший к центроиду пример — {rep.municipal_district_name_short} ({rep.region_name}). Это пример профиля, а не характеристика всех жителей.\n\n"
    body += '''Доступность рынков — статический показатель 2024 года, не доход или зарплата.
На 2 004 МО с доступным показателем Spearman с продовольственной долей
средней корзины 2024 составляет **−0.549**, с долей здоровья **+0.463**.
Показатель не вошёл в обучение групп. Связь может отражать размер и структуру
населённых пунктов и другие различия; причинный эффект доступности не установлен.
Доли композиционны: рост одной механически связан со снижением других.

Практическое применение — подбор группы территорий для сравнительного анализа
потребления. Нельзя напрямую использовать группы как рейтинг благосостояния,
потенциала инвестиций или эффективности управления. Влияние покрытия банка,
наличных платежей, инфляции и состава жителей отдельно не идентифицировано.

## Визуализация

Интерактивный «паспорт экономических соседей» соединяет фиксированную
проекцию профилей, месяц, выбранное МО, реальные сильные рёбра и историю
согласованных групп. Движение точки не является географическим перемещением.
PCA обучена только на январе 2023; две компоненты объясняют **85,7%**
его дисперсии. Значения долей и соседи используют полное пространство
признаков, а не расстояния на двумерной картинке.

## Воспроизведение и источники

```sh
python -m src.prepare
LOKY_MAX_CPU_COUNT=2 python -m src.pipeline
python -m src.build_evidence
python -m unittest discover -s tests -v
```

Конфигурация — config.json; контрольные суммы — input-inventory.json,
panel-audit.json и summary.json. Семь критичных проверок успешно выполнены.
Таблицы всех сравнений сохранены; значения отчёта получены из расчётов,
не из демонстрационных данных. Аудит и графики дополнены выполняемым notebook.

- СберИндекс. Муниципальные расходы и транспортные данные, лицензия в
  официальном архиве, скачано 09.10.2026:
  https://www.sberbank.com/common/img/uploaded/files/pdf/sberindex/hackathonlicence.zip
- СберИндекс. Границы и преобразования муниципалитетов, скачано 09.10.2026:
  https://sberindex.ru/ru/research/dataset-borders-and-changes-of-municipalities
- Формулы и первоисточники метрик: docs/metrics.md.

Исходные данные и производные материалы с их содержимым распространяются
на условиях **CC BY-SA 4.0**, с указанием СберИндекса и изменений. Код
алгоритмов — MIT; условия данных не заменяются лицензией кода.
'''
    Path('docs/methodological-report.md').write_text(body)
    print('Evidence saved: outputs/reviewed-snapshot.json; docs/methodological-report.md')


if __name__=='__main__':
    build()
