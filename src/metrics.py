"""Internal validity in feature and network spaces; see docs/metrics.md."""
import numpy as np
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score


def s_dbw(x, labels):
    """Variance-vector S_Dbw (Shahileh et al.2025, eqs.11–17).

    The ball indicator is 1 for distance <= radius (the 2025 article's
    eq. 14 reverses this indicator; we explicitly count inside the ball).
    This differs from the standard-deviation-vector implementation in S_Dbw.
    Undefined centroid densities return NaN, never an artificial zero.
    """
    groups = [x[labels == k] for k in np.unique(labels)]
    k = len(groups)
    if k < 2 or any(len(g) < 2 for g in groups):
        return float('nan')
    centroids = [g.mean(axis=0) for g in groups]
    variance_norms = np.array([np.linalg.norm(g.var(axis=0)) for g in groups])
    global_variance = np.linalg.norm(x.var(axis=0))
    if global_variance == 0:
        return float('nan')
    scatter = variance_norms.mean() / global_variance
    radius = np.sqrt(variance_norms.sum()) / k
    densities = [int((np.linalg.norm(g-c, axis=1) <= radius).sum())
                 for g, c in zip(groups, centroids)]
    between = 0.0
    for i in range(k):
        for j in range(i+1, k):
            denominator = max(densities[i], densities[j])
            if not denominator:
                return float('nan')
            midpoint = (centroids[i]+centroids[j])/2
            union = np.concatenate([groups[i], groups[j]])
            density = (np.linalg.norm(union-midpoint, axis=1) <= radius).sum()
            between += 2*density/denominator
    return float(scatter + between/(k*(k-1)))


def network_indices(adjacency, labels):
    """Weighted AVI, AVU eqs.18–21 Shahileh et al.2025; standard Q (gamma=1).

    A is symmetric and each undirected edge appears twice. AVU divides the
    ordered pair sum by K, NOT K(K-1). A disconnected pair with no cut has
    unifiability 0. Zero-volume groups invalidate the partition.
    """
    unique, inverse = np.unique(labels, return_inverse=True)
    membership = np.eye(len(unique))[inverse]
    block = np.asarray(membership.T @ (adjacency @ membership))
    volume = block.sum(axis=1)
    if (volume <= 0).any():
        return dict(AVI=float('nan'), AVU=float('nan'), MQ=float('nan'))
    internal = np.diag(block)
    cut = volume-internal
    denominator = cut[:, None]+cut[None, :]-block
    ratios = np.divide(block, denominator, out=np.zeros_like(block), where=denominator>1e-12)
    np.fill_diagonal(ratios, 0)
    total = volume.sum()
    return dict(AVI=float(np.mean(internal/volume)),
                AVU=float(ratios.sum()/len(unique)),
                MQ=float(np.sum(internal/total-(volume/total)**2)))


def evaluate(x, adjacency, labels, sample_size=2000, seed=42):
    counts = np.unique(labels, return_counts=True)[1]
    if not 1 < len(counts) < len(labels):
        raise ValueError('Need between 2 and N-1 clusters')
    return {
        'clusters': int(len(counts)), 'min_cluster_fraction': float(counts.min()/len(labels)),
        'SW': float(silhouette_score(x, labels, sample_size=min(sample_size,len(x)),random_state=seed)),
        'CH': float(calinski_harabasz_score(x, labels)),
        'DB': float(davies_bouldin_score(x, labels)), 'S_Dbw': s_dbw(x, labels),
        **network_indices(adjacency, labels),
    }
