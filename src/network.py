"""Economic proximity networks and temporal label alignment."""
import numpy as np
import networkx as nx
from scipy import sparse
from scipy.optimize import linear_sum_assignment
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans, SpectralClustering


def knn_graph(x, neighbors, metric='euclidean'):
    n = len(x)
    if not 0 < neighbors < n:
        raise ValueError('Invalid neighbors')
    distances, indices = NearestNeighbors(n_neighbors=neighbors+1, metric=metric).fit(x).kneighbors(x)
    # Do not rely on self being first when duplicate profiles are present.
    pairs = [(ds[ix != i][:neighbors], ix[ix != i][:neighbors])
             for i, (ds,ix) in enumerate(zip(distances, indices))]
    ds = np.stack([p[0] for p in pairs])
    ix = np.stack([p[1] for p in pairs])
    positive = ds[ds>0]
    floor = max(float(np.median(positive))*1e-6 if len(positive) else 1e-12, 1e-12)
    sigma = np.maximum(ds[:, -1], floor)
    weights = np.exp(-ds**2 / (sigma[:,None]*sigma[ix]))
    a = sparse.csr_matrix((weights.ravel(), (np.repeat(np.arange(n),neighbors), ix.ravel())),shape=(n,n))
    a = a.maximum(a.T)
    a.setdiag(0); a.eliminate_zeros()
    return a


def fit_partition(x, adjacency, method, parameter, seed):
    if method == 'kmeans':
        return KMeans(n_clusters=int(parameter), n_init=20, random_state=seed).fit_predict(x)
    if method == 'spectral':
        return SpectralClustering(n_clusters=int(parameter), affinity='precomputed',
            assign_labels='kmeans', n_init=20, random_state=seed, eigen_tol=1e-6).fit_predict(adjacency)
    if method == 'louvain':
        graph = nx.from_scipy_sparse_array(adjacency)
        communities = nx.community.louvain_communities(graph, weight='weight',
                            resolution=float(parameter), seed=seed)
        labels = np.empty(len(x), dtype=int)
        for label, members in enumerate(sorted(communities, key=lambda c:min(c))):
            labels[list(members)] = label
        return labels
    raise ValueError(f'Unknown method: {method}')


def graph_summary(adjacency):
    count, labels = sparse.csgraph.connected_components(adjacency,directed=False)
    sizes = np.bincount(labels)
    n = adjacency.shape[0]
    return {'edges':int(adjacency.nnz//2),'density':float(adjacency.nnz/(n*(n-1))),
            'components':int(count),'largest_component_fraction':float(sizes.max()/n)}


def align_labels(previous, current, next_id):
    """Maximum-overlap one-to-one matches; unmatched labels get new IDs."""
    old, oi = np.unique(previous,return_inverse=True)
    new, ni = np.unique(current,return_inverse=True)
    overlaps = np.zeros((len(old),len(new)),dtype=int)
    np.add.at(overlaps,(oi,ni),1)
    rows,cols = linear_sum_assignment(-overlaps)
    mapping = {int(new[c]):int(old[r]) for r,c in zip(rows,cols) if overlaps[r,c]>0}
    for label in new:
        if int(label) not in mapping:
            mapping[int(label)] = next_id
            next_id += 1
    return np.array([mapping[int(label)] for label in current]), next_id
