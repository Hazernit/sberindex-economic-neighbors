"""Checks for mistakes that would invalidate the scientific interpretation."""
import unittest
import numpy as np
import networkx as nx
from scipy import sparse
from sklearn.datasets import make_blobs
from src.network import knn_graph,align_labels
from src.metrics import network_indices,s_dbw


class ScientificChecks(unittest.TestCase):
    def test_graph_is_symmetric_and_no_self_links_with_duplicates(self):
        a = knn_graph(np.array([[0.,0.],[0.,0.],[1.,0.],[1.,1.]]),2)
        self.assertEqual((a-a.T).nnz,0)
        self.assertTrue(np.all(a.diagonal()==0))
        self.assertTrue(np.isfinite(a.data).all())
        self.assertTrue((a.data>0).all())

    def test_modularity_matches_networkx_and_ratios_have_known_values(self):
        a = sparse.csr_matrix(np.array([[0.,2.,1.,0.],[2.,0.,0.,0.],[1.,0.,0.,2.],[0.,0.,2.,0.]]))
        labels = np.array([0,0,1,1])
        metrics = network_indices(a,labels)
        self.assertAlmostEqual(metrics['AVI'],.8)
        self.assertAlmostEqual(metrics['AVU'],1.)
        self.assertAlmostEqual(metrics['MQ'],nx.community.modularity(nx.from_scipy_sparse_array(a),[{0,1},{2,3}]))

    def test_disconnected_groups_have_no_merge_tendency(self):
        a = sparse.csr_matrix(np.array([[0.,1.,0.,0.],[1.,0.,0.,0.],[0.,0.,0.,1.],[0.,0.,1.,0.]]))
        metrics = network_indices(a,np.array([4,4,7,7]))
        self.assertEqual(metrics['AVI'],1.)
        self.assertEqual(metrics['AVU'],0.)

    def test_label_permutation_has_no_temporal_changes(self):
        old = np.array([4,4,7,7,9,9])
        current = np.array([2,2,0,0,1,1])
        aligned,next_id = align_labels(old,current,10)
        np.testing.assert_array_equal(aligned,old)
        self.assertEqual(next_id,10)

    def test_new_groups_get_never_used_ids(self):
        aligned,next_id = align_labels(np.array([0,0,1,1,1,1]),np.array([0,0,1,1,2,2]),12)
        self.assertIn(12,aligned)
        self.assertEqual(next_id,13)

    def test_sdbw_prefers_separated_groups_and_is_scale_invariant(self):
        x,labels = make_blobs(n_samples=200,centers=[[-3,0],[3,0]],cluster_std=1.5,random_state=5)
        mixed = np.random.default_rng(42).permutation(labels)
        self.assertLess(s_dbw(x,labels),s_dbw(x,mixed))
        self.assertAlmostEqual(s_dbw(x,labels),s_dbw(x*7+2,labels))

    def test_empty_centroid_density_is_undefined(self):
        x,labels = make_blobs(n_samples=200,centers=[[-5,0],[5,0]],cluster_std=.2,random_state=5)
        self.assertTrue(np.isnan(s_dbw(x,np.random.default_rng(42).permutation(labels))))


if __name__ == '__main__':
    unittest.main()
