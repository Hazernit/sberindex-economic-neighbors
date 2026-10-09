"""Validate official inputs and construct a balanced municipality panel."""
from pathlib import Path
import argparse
import json
import hashlib
import numpy as np
import pandas as pd

CATEGORIES = ['Продовольствие','Здоровье','Маркетплейсы','Общественное питание','Транспорт']


def prepare(raw_dir, output, dictionary='data/raw/t_dict_municipal_districts.xlsx'):
    raw_dir, output = Path(raw_dir), Path(output)
    output.mkdir(parents=True,exist_ok=True)
    frame = pd.read_parquet(raw_dir/'consumption.parquet')
    keys = ['territory_id','date','category']
    if set(frame.columns) != {*keys,'value'}:
        raise ValueError(f'Unexpected schema: {frame.columns.tolist()}')
    if frame.duplicated(keys).any() or frame.isna().any().any():
        raise ValueError('Duplicate keys or nulls in official input')
    if (frame.value<0).any():
        raise ValueError('Negative expenditure')
    months = sorted(frame.date.unique())
    expected_months = pd.period_range('2023-01','2024-12',freq='M').astype(str).tolist()
    if months != expected_months:
        raise ValueError('Period changed; review metadata before analysis')
    if set(frame.category.unique()) != {*CATEGORIES,'Все категории'}:
        raise ValueError('Category definitions changed')
    pivot = frame.pivot(index=['date','territory_id'],columns='category',values='value')
    observed_ids = np.sort(frame.territory_id.unique())
    full_index = pd.MultiIndex.from_product([months,observed_ids],names=['date','territory_id'])
    full = pivot.reindex(full_index)
    valid_count = full.notna().all(axis=1).groupby('territory_id').sum()
    ids = valid_count[valid_count==len(months)].index.to_numpy()
    excluded = pd.DataFrame({'territory_id':observed_ids, 'complete_months':valid_count.reindex(observed_ids).values})
    excluded = excluded[excluded.complete_months<len(months)]
    excluded['reason'] = 'Нет полного набора шести категорий во всех 24 месяцах'
    excluded.to_csv(output/'excluded.csv',index=False)
    panel = pivot.loc[pd.IndexSlice[:,ids],:].sort_index()
    residual = panel['Все категории']-panel[CATEGORIES].sum(axis=1)
    if (residual<0).any() or (panel[CATEGORIES].sum(axis=1)<=0).any():
        raise ValueError('Basket/total inconsistency')
    values = panel[CATEGORIES].to_numpy().reshape(len(months),len(ids),len(CATEGORIES))
    shares = values/values.sum(axis=2,keepdims=True)
    np.savez_compressed(output/'panel.npz',ids=ids,months=np.array(months),categories=np.array(CATEGORIES),
        values=values,shares=shares,total=panel['Все категории'].to_numpy().reshape(len(months),len(ids)))
    # Auxiliary values only describe the 2024 cross-section.
    market = pd.read_parquet(raw_dir/'market_access.parquet')
    if market.territory_id.duplicated().any():
        raise ValueError('Duplicate market-access ids')
    market.to_csv(output/'market_access.csv',index=False)
    dictionary_frame = pd.read_excel(dictionary)
    if dictionary_frame.groupby('territory_id').region_name.nunique().max()>1:
        raise ValueError('Conflicting region names for a territory')
    name_fields = ['territory_id','municipal_district_name_short','municipal_district_name','region_name','year_from','year_to']
    # Select the latest name known by 2024; territory_id identifies fixed borders.
    latest = dictionary_frame[dictionary_frame.year_from<=2024].sort_values(
        ['territory_id','year_from','year_to'],ascending=[True,False,False])
    latest = latest.drop_duplicates('territory_id',keep='first')[name_fields]
    if not pd.Series(ids).isin(latest.territory_id).all():
        raise ValueError('Municipality dictionary does not cover the balanced panel')
    latest.to_csv(output/'municipalities.csv',index=False)
    summary = {'source_ids':int(len(observed_ids)),'included_ids':int(len(ids)),
        'excluded_ids':int(len(excluded)),'months':months,'categories':CATEGORIES,
        'source_rows':int(len(frame)),'missing_panel_rows':int(full.isna().sum().sum()),
        'measure':'Оценка средних безналичных расходов жителей, рубли',
        'denominator':'Сумма пяти выбранных категорий, не все расходы',
        'source_sha256':hashlib.sha256((raw_dir/'consumption.parquet').read_bytes()).hexdigest()}
    (output/'panel-audit.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw',default='data/raw/hackathonlicence')
    parser.add_argument('--output',default='data/processed')
    parser.add_argument('--dictionary',default='data/raw/t_dict_municipal_districts.xlsx')
    args = parser.parse_args()
    print(json.dumps(prepare(args.raw,args.output,args.dictionary),ensure_ascii=False,indent=2))
