"""Reproducible monthly attributed-network clustering on official SberIndex data."""
from pathlib import Path
import argparse
import json
import time
import hashlib
import warnings
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score
from threadpoolctl import threadpool_limits
from .network import knn_graph,fit_partition,graph_summary,align_labels
from .metrics import evaluate


def save_json(path, value):
    # JSON null is used for undefined statistics; no NaN tokens in browser data.
    def clean(obj):
        if isinstance(obj,dict): return {k:clean(v) for k,v in obj.items()}
        if isinstance(obj,list): return [clean(v) for v in obj]
        if isinstance(obj,(np.integer,)): return int(obj)
        if isinstance(obj,(np.floating,float)): return float(obj) if np.isfinite(obj) else None
        return obj
    Path(path).write_text(json.dumps(clean(value),ensure_ascii=False,indent=2)+'\n')


def road_graph(path, ids, k):
    """Static Dec.2024 road graph: reject zero distances, one unordered pair."""
    roads = pd.read_parquet(path,filters=[('type','==','highway')])
    roads = roads[(roads.distance>0)&roads.territory_id_x.isin(ids)&roads.territory_id_y.isin(ids)]
    mapping = {int(value):i for i,value in enumerate(ids)}
    n = len(ids)
    distance = np.full((n,n),np.inf)
    rows = roads.territory_id_x.map(mapping).to_numpy()
    cols = roads.territory_id_y.map(mapping).to_numpy()
    distance[rows,cols] = roads.distance
    distance[cols,rows] = roads.distance
    # Keep nodes without known road distances as isolated; do not invent links.
    indices = np.argsort(distance,axis=1)[:,:k]
    ds = np.take_along_axis(distance,indices,axis=1)
    finite = np.isfinite(ds)
    sigma = np.max(np.where(finite,ds,0),axis=1)
    sigma = np.maximum(sigma,1e-9)
    weights = np.zeros_like(ds)
    weights[finite] = np.exp(-ds[finite]**2/(sigma[:,None]*sigma[indices])[finite])
    a = sparse.csr_matrix((weights.ravel(),(np.repeat(np.arange(n),k),indices.ravel())),shape=(n,n))
    a = a.maximum(a.T); a.setdiag(0); a.eliminate_zeros()
    return a


def run(processed, raw, config_path, output):
    start = time.time()
    output = Path(output); output.mkdir(parents=True,exist_ok=True)
    cfg = json.loads(Path(config_path).read_text())
    panel = np.load(Path(processed)/'panel.npz')
    ids,months,categories = panel['ids'],panel['months'],panel['categories']
    shares,values = panel['shares'],panel['values']
    features = np.sqrt(shares)
    sample_size, seed = cfg['silhouette_sample_size'],cfg['seed']
    comparison, cache = [],{}
    for k in cfg['neighbors']:
        a = knn_graph(features[0],k)
        for method,params in [('kmeans',cfg['cluster_counts']),('spectral',cfg['cluster_counts']),
                              ('louvain',cfg['louvain_resolutions'])]:
            for parameter in params:
                labels = fit_partition(features[0],a,method,parameter,seed)
                record = dict(month=str(months[0]),graph='hellinger',method=method,
                    neighbors=k,parameter=parameter,**evaluate(features[0],a,labels,sample_size,seed),
                    **graph_summary(a))
                comparison.append(record)
                cache[(method,k,parameter)] = labels
                print(f"grid {method} k={k} p={parameter} SW={record['SW']:.4f}",flush=True)
    candidates = [r for r in comparison if r['method']!='kmeans'
                  and r['min_cluster_fraction']>=cfg['min_cluster_fraction']]
    if not candidates:
        raise ValueError('No eligible network partition; review cluster-size threshold')
    best = max(candidates,key=lambda r:r['SW'])
    save_json(output/'selected.json',best)
    pd.DataFrame(comparison).to_csv(output/'comparison.csv',index=False)
    method,k,parameter = best['method'],best['neighbors'],best['parameter']
    print(f'Selected {method} k={k} p={parameter}',flush=True)
    projector = PCA(n_components=2,svd_solver='full').fit(features[0])
    assignments,profiles,dynamics,transitions,bridge_rows = [],[],[],[],[]
    aligned_all,raw_all,adjacencies = [],[],[]
    projected = []
    previous = None; next_id = 0
    for t,month in enumerate(months):
        x = features[t]; a = knn_graph(x,k)
        labels = cache[(method,k,parameter)].copy() if t==0 else fit_partition(x,a,method,parameter,seed)
        raw_all.append(labels)
        if previous is None:
            aligned = labels.copy(); next_id = int(labels.max())+1
        else:
            aligned,next_id = align_labels(previous,labels,next_id)
            for old in np.unique(previous):
                for new in np.unique(aligned):
                    transitions.append({'from_month':str(months[t-1]),'to_month':str(month),
                         'from_cluster':int(old),'to_cluster':int(new),
                         'municipalities':int(((previous==old)&(aligned==new)).sum())})
        ari = adjusted_rand_score(previous,aligned) if previous is not None else None
        dynamics.append({'month':str(month), 'ARI_previous':ari,
            'changed_fraction':float(np.mean(previous!=aligned)) if previous is not None else None,
            **evaluate(x,a,labels,sample_size,seed),**graph_summary(a)})
        coords = projector.transform(x); projected.append(coords)
        cluster_weight = np.eye(len(np.unique(aligned)))[np.searchsorted(np.unique(aligned),aligned)]
        node_group_weights = np.asarray(a @ cluster_weight)
        degree = np.asarray(a.sum(axis=1)).ravel()
        external_fraction = 1-node_group_weights[np.arange(len(ids)),np.searchsorted(np.unique(aligned),aligned)]/degree
        for i,identifier in enumerate(ids):
            assignments.append({'month':str(month),'territory_id':int(identifier),'cluster':int(aligned[i]),
                'external_edge_fraction':float(external_fraction[i])})
        for cluster in np.unique(aligned):
            mask = aligned==cluster
            centroid = x[mask].mean(axis=0)
            group_indices = np.flatnonzero(mask)
            representative = group_indices[np.argmin(np.linalg.norm(x[mask]-centroid,axis=1))]
            record = {'month':str(month),'cluster':int(cluster),'municipalities':int(mask.sum()),
                'representative_id':int(ids[representative])}
            for c,category in enumerate(categories):
                record[f'share_{category}'] = float(shares[t,mask,c].mean())
                record[f'rubles_{category}'] = float(np.median(values[t,mask,c]))
            profiles.append(record)
        for i in np.argsort(-external_fraction)[:10]:
            bridge_rows.append({'month':str(month),'territory_id':int(ids[i]),'cluster':int(aligned[i]),
                               'external_edge_fraction':float(external_fraction[i])})
        aligned_all.append(aligned); adjacencies.append(a); previous = aligned
        print(f'month {month} clusters={len(np.unique(aligned))} ARI={ari}',flush=True)
    pd.DataFrame(assignments).to_csv(output/'assignments.csv',index=False)
    pd.DataFrame(profiles).to_csv(output/'profiles.csv',index=False)
    pd.DataFrame(dynamics).to_csv(output/'dynamics.csv',index=False)
    pd.DataFrame(transitions).to_csv(output/'transitions.csv',index=False)
    pd.DataFrame(bridge_rows).to_csv(output/'boundary-nodes.csv',index=False)
    # Separate temporal variation from optimization randomness in every month.
    stability = []
    for t in range(len(months)):
        for alternate_seed in cfg['stability_seeds']:
            labels = fit_partition(features[t],adjacencies[t],method,parameter,alternate_seed)
            stability.append({'month':str(months[t]),'seed':alternate_seed,
                'ARI_to_seed42':float(adjusted_rand_score(raw_all[t],labels)),
                **evaluate(features[t],adjacencies[t],labels,sample_size,seed)})
    pd.DataFrame(stability).to_csv(output/'seed-stability.csv',index=False)
    # Trailing 3-month mean of levels, then recomputed shares (not average of ratios).
    smoothed_labels, sensitivity = [],[]
    for t in range(2,len(months)):
        avg = values[t-2:t+1].mean(axis=0)
        sx = np.sqrt(avg/avg.sum(axis=1,keepdims=True))
        sa = knn_graph(sx,k)
        sl = fit_partition(sx,sa,method,parameter,seed)
        smooth_ari = adjusted_rand_score(smoothed_labels[-1],sl) if smoothed_labels else None
        sensitivity.append({'month':str(months[t]),'ARI_raw_vs_trailing3':float(adjusted_rand_score(raw_all[t],sl)),
                            'ARI_previous_raw':dynamics[t]['ARI_previous'] if smoothed_labels else None,
                            'ARI_previous_trailing3':smooth_ari})
        smoothed_labels.append(sl)
    pd.DataFrame(sensitivity).to_csv(output/'smoothing-sensitivity.csv',index=False)
    year_over_year = [{'month_2023':str(months[t]),'month_2024':str(months[t+12]),
        'ARI':float(adjusted_rand_score(raw_all[t],raw_all[t+12]))} for t in range(12)]
    pd.DataFrame(year_over_year).to_csv(output/'same-month-year.csv',index=False)
    # Compare alternate proximity definitions at a common 2024 cross-section.
    definitions = []
    for definition, a in [('hellinger',adjacencies[-1]),('cosine',knn_graph(shares[-1],k,'cosine'))]:
        labels = fit_partition(features[-1],a,method,parameter,seed)
        definitions.append({'graph':definition,'month':str(months[-1]),
            'ARI_to_hellinger':float(adjusted_rand_score(raw_all[-1],labels)),
            **evaluate(features[-1],a,labels,sample_size,seed),**graph_summary(a)})
    roads = road_graph(Path(raw)/'connection.parquet',ids,k)
    active = np.asarray(roads.sum(axis=1)).ravel()>0
    _, component_labels = sparse.csgraph.connected_components(roads,directed=False)
    largest_component = int(np.argmax(np.bincount(component_labels)))
    active &= component_labels==largest_component
    # All definitions rerun on the same road-covered population for fair comparison.
    subset_definitions = []
    for definition,a in [('hellinger',knn_graph(features[-1,active],k)),
                         ('cosine',knn_graph(shares[-1,active],k,'cosine')),
                         ('road',roads[active][:,active])]:
        labels = fit_partition(features[-1,active],a,method,parameter,seed)
        subset_definitions.append({'graph':definition,'month':str(months[-1]),'municipalities':int(active.sum()),
            **evaluate(features[-1,active],a,labels,sample_size,seed),**graph_summary(a)})
    pd.DataFrame(definitions).to_csv(output/'proximity-full.csv',index=False)
    pd.DataFrame(subset_definitions).to_csv(output/'proximity-common-population.csv',index=False)
    market = pd.read_csv(Path(processed)/'market_access.csv').set_index('territory_id').market_access.reindex(ids)
    yearavg = shares[12:].mean(axis=0)
    market_associations = []
    known = market.notna().to_numpy()
    for c,category in enumerate(categories):
        rho,pvalue = spearmanr(market.to_numpy()[known],yearavg[known,c])
        market_associations.append({'category':str(category),'rho':float(rho),'pvalue_uncorrected':float(pvalue),
                                    'n':int(known.sum())})
    pd.DataFrame(market_associations).to_csv(output/'market-associations.csv',index=False)
    aligned_all = np.array(aligned_all); projected = np.array(projected)
    np.savez_compressed(output/'model.npz',ids=ids,months=months,categories=categories,shares=shares,
        values=values,labels=aligned_all,projection=projected,market_access=market.to_numpy(),
        pca_explained_variance_ratio=projector.explained_variance_ratio_)
    # Compact per-MO history used by the interactive passport; source links are public.
    nodes = []
    names = pd.read_csv(Path(processed)/'municipalities.csv').set_index('territory_id')
    for i,identifier in enumerate(ids):
        history = []
        for t in range(len(months)):
            row = adjacencies[t].getrow(i)
            order = np.argsort(-row.data)[:6]
            neighbors = [[int(ids[row.indices[o]]),round(float(row.data[o]),6)] for o in order]
            history.append({'month':str(months[t]),'cluster':int(aligned_all[t,i]),
                'x':round(float(projected[t,i,0]),6),'y':round(float(projected[t,i,1]),6),
                'shares':[round(float(v),6) for v in shares[t,i]],'neighbors':neighbors})
        nodes.append({'territory_id':int(identifier),'name':names.loc[int(identifier),'municipal_district_name_short'],
            'full_name':names.loc[int(identifier),'municipal_district_name'],
            'region':names.loc[int(identifier),'region_name'],
            'market_access':float(market.iloc[i]) if known[i] else None,
            'changes':int(np.sum(aligned_all[1:,i]!=aligned_all[:-1,i])), 'history':history})
    save_json(output/'explorer-nodes.json',nodes)
    summary = {'municipalities':int(len(ids)),'period':[str(months[0]),str(months[-1])],
        'selected':best, 'mean_ARI_previous':float(pd.DataFrame(dynamics).ARI_previous.dropna().mean()),
        'mean_changed_fraction':float(pd.DataFrame(dynamics).changed_fraction.dropna().mean()),
        'never_changed':int(np.sum(np.all(aligned_all==aligned_all[0],axis=0))),
        'seed_stability':stability, 'year_over_year':year_over_year,
        'smoothing':sensitivity,'proximity':subset_definitions,'market_associations':market_associations,
        'pca_explained_variance_ratio':projector.explained_variance_ratio_.tolist(),
        'seconds':round(time.time()-start,2), 'config_sha256':hashlib.sha256(Path(config_path).read_bytes()).hexdigest()}
    save_json(output/'summary.json',summary)
    print(json.dumps({k:v for k,v in summary.items() if k not in ['selected','seed_stability','year_over_year','smoothing','proximity','market_associations']},ensure_ascii=False),flush=True)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--processed',default='data/processed')
    parser.add_argument('--raw',default='data/raw/hackathonlicence')
    parser.add_argument('--config',default='config.json')
    parser.add_argument('--output',default='outputs/analysis')
    args = parser.parse_args()
    with threadpool_limits(limits=2):
        run(args.processed,args.raw,args.config,args.output)
