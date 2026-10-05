import unittest
from unittest.mock import patch

import networkx as nx
import numpy as np

from wm_sar import amplification as amp
from wm_sar.region_extractor import ReCore, ReCoreConfig, WMSAR, WMSARConfig


def _three_node_graph():
    graph = nx.DiGraph()
    graph.add_node("seed", err=1.0, cost=1.0)
    graph.add_node("a", err=2.0, cost=1000.0)
    graph.add_node("b", err=1.0, cost=1.0)
    graph.add_edges_from([("seed", "a"), ("seed", "b")])
    return graph


class PaperCoreTests(unittest.TestCase):
    def test_paper_defaults_and_public_aliases(self):
        self.assertIs(ReCore, WMSAR)
        self.assertIs(ReCoreConfig, WMSARConfig)
        config = ReCoreConfig()
        self.assertEqual(config.weight_norm, 0.9)
        self.assertEqual(config.max_region_size, 20)
        self.assertFalse(config.merge_candidates)

    def test_local_amplification_has_no_spectral_floor(self):
        graph = _three_node_graph()
        self.assertEqual(amp.geaf_node(graph, "seed"), 0.0)
        graph.add_edge("a", "seed")
        self.assertAlmostEqual(amp.geaf_node(graph, "a", H=2), 2.0 * 0.9 ** 2)

    def test_seed_score_and_error_only_ablation(self):
        graph = _three_node_graph()
        extractor = ReCore(ReCoreConfig(n_seeds=1))
        extractor._geaf_cache = {"seed": 1.0, "a": 1.0, "b": 1.0}
        extractor._kappa_cache = {"seed": 9.0, "a": 0.0, "b": 0.0}
        self.assertEqual(extractor.seeds(graph), ["seed"])
        extractor.cfg.use_geaf = False
        self.assertEqual(extractor.seeds(graph), ["a"])

    def test_growth_uses_additive_redundancy_cost(self):
        graph = _three_node_graph()
        extractor = ReCore(ReCoreConfig(max_region_size=2, use_coupling=False))
        with patch.object(amp, "rho_B_complement", side_effect=lambda _, region, __: 11 - len(region)):
            self.assertEqual(extractor.grow(graph, "seed"), {"seed", "a"})

    def test_growth_requires_positive_spectral_relief(self):
        graph = _three_node_graph()
        extractor = ReCore(ReCoreConfig(max_region_size=2))
        with patch.object(amp, "rho_B_complement", return_value=1.0):
            self.assertEqual(extractor.grow(graph, "seed"), {"seed"})

    def test_positive_relief_can_grow_despite_negative_gain(self):
        graph = _three_node_graph()
        extractor = ReCore(ReCoreConfig(max_region_size=2, lambda3=100.0))
        with patch.object(amp, "rho_B_complement", side_effect=lambda _, region, __: 11 - len(region)):
            self.assertEqual(extractor.grow(graph, "seed"), {"seed", "a"})

    def test_growth_without_spectral_relief_uses_error_gain(self):
        graph = _three_node_graph()
        extractor = ReCore(ReCoreConfig(max_region_size=2, use_rho_relief=False))
        with patch.object(amp, "rho_B_complement", return_value=1.0):
            self.assertEqual(extractor.grow(graph, "seed"), {"seed", "a"})

    def test_coupling_ablation_reaches_growth_and_region_score(self):
        graph = _three_node_graph()
        extractor = ReCore(ReCoreConfig(max_region_size=2, use_coupling=False))
        extractor._kappa_cache = {"seed": 9.0, "a": 0.0, "b": 100.0}
        extractor._rho_full = 3.0
        with patch.object(amp, "rho_B_complement", return_value=1.0):
            self.assertAlmostEqual(extractor.score(graph, {"seed", "a"}), 2.0)
        with patch.object(amp, "rho_B_complement", side_effect=lambda _, region, __: 11 - len(region)):
            self.assertEqual(extractor.grow(graph, "seed"), {"seed", "a"})

    def test_region_score_is_error_weighted_and_cardinality_normalized(self):
        graph = _three_node_graph()
        extractor = ReCore()
        extractor._kappa_cache = {"seed": 3.0, "a": 1.0}
        extractor._rho_full = 4.0
        with patch.object(amp, "rho_B_complement", return_value=2.0):
            self.assertAlmostEqual(extractor.score(graph, {"seed", "a"}), 16.0 / 3.0)
        with patch.object(amp, "rho_B_complement", return_value=4.0):
            self.assertEqual(extractor.score(graph, {"seed", "a"}), 0.0)

    def test_pruning_preserves_connectivity(self):
        graph = nx.DiGraph([("a", "b"), ("b", "c")])
        extractor = ReCore()
        values = {
            frozenset({"a", "b", "c"}): 0.0,
            frozenset({"a", "b"}): 1.0,
            frozenset({"b", "c"}): 1.0,
            frozenset({"a", "c"}): 0.0,
        }
        with patch.object(amp, "rho_B_complement", side_effect=lambda _, region, __: values[frozenset(region)]):
            self.assertEqual(extractor.prune(graph, {"a", "b", "c"}), {"a", "b", "c"})

    def test_algorithm_does_not_merge_candidates_by_default(self):
        graph = _three_node_graph()
        extractor = ReCore(ReCoreConfig(use_pruning=False))
        with patch.object(extractor, "_merge", side_effect=AssertionError("unexpected merge")):
            candidates = extractor.candidate_regions(graph)
        for candidate in candidates:
            self.assertLessEqual(len(candidate.nodes), extractor.cfg.max_region_size)

    def test_residual_operator_excludes_removed_boundary_edges(self):
        graph = _three_node_graph()
        region = {"seed", "a"}
        np.testing.assert_allclose(
            amp.coupling_blocks_region(graph, region),
            amp.coupling_blocks_region(graph.subgraph(region), region),
        )

    def test_global_and_regional_operator_estimators_match(self):
        graph = _three_node_graph()
        np.testing.assert_allclose(
            amp._estimate_propagation_gains(graph),
            amp.coupling_blocks_region(graph, set(graph)),
        )

    def test_empty_repair_has_identical_pre_and_post_bound(self):
        bounds = amp.return_error_bound(_three_node_graph(), set())
        self.assertEqual(bounds["rho_pre"], bounds["rho_post"])
        self.assertEqual(bounds["bound_pre"], bounds["bound_post"])

    def test_planning_regret_uses_geometric_amplification(self):
        self.assertEqual(amp.phi_H_regret(0, 0.5, 2.0), 0.0)
        self.assertEqual(amp.phi_H_regret(1, 0.5, 2.0), 1.0)
        self.assertEqual(amp.phi_H_regret(4, 0.5, 2.0), 4.0)
        self.assertEqual(amp.phi_H_regret(3, 1.0, 2.0), 7.0)

    def test_reward_error_has_two_policy_contributions(self):
        bounds = amp.return_error_bound(
            _three_node_graph(), set(), H=4, epsilon=0.0, epsilon_R=0.25
        )
        self.assertEqual(bounds["bound_pre"], 2.0)
        self.assertEqual(bounds["bound_post"], 2.0)

    def test_coupled_envelope_uses_both_error_channels(self):
        operator = np.array([[0.5, 0.2], [0.3, 0.4]])
        initial = np.array([1.0, 2.0])
        forcing = np.array([0.1, 0.2])
        envelope = amp.coupled_error_envelope(initial, operator, forcing, 2)
        np.testing.assert_allclose(envelope[0], initial)
        np.testing.assert_allclose(envelope[1], operator @ initial + forcing)
        np.testing.assert_allclose(envelope[2], operator @ envelope[1] + forcing)

    def test_coupled_envelope_rejects_invalid_inputs(self):
        with self.assertRaises(ValueError):
            amp.coupled_error_envelope([1.0], np.eye(2), [0.0, 0.0], 1)
        with self.assertRaises(ValueError):
            amp.coupled_error_envelope([1.0, -1.0], np.eye(2), [0.0, 0.0], 1)


if __name__ == "__main__":
    unittest.main()
