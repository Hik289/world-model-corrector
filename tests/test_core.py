import os
import unittest
from unittest.mock import patch

import networkx as nx

from experiments.exp_cascade_gain import inject_with_gain
from wm_sar import baselines
from wm_sar.agent_calling_tree import generate_calling_trees
from wm_sar.failure_graph import build_from_agent_calling_tree
from wm_sar.llm_client import LLMClient
from wm_sar.region_extractor import Region, WMSAR, WMSARConfig
from wm_sar.repair_executor import propagate_effective_error


def _path_graph():
    graph = nx.DiGraph()
    for i, err in enumerate((0.8, 0.6, 0.4, 0.2)):
        graph.add_node(
            f"n{i}",
            err=err,
            unc=0.1,
            cost=1.0,
            time_step=i,
        )
    graph.add_edges_from((f"n{i}", f"n{i + 1}") for i in range(3))
    graph.graph["t_star"] = "n3"
    graph.graph["gt_region"] = {"n0", "n1"}
    return graph


def _failure_graph_from_tree(tree):
    return build_from_agent_calling_tree(
        tree.G,
        dict(zip(tree.node_list, tree.node_states)),
        dict(zip(tree.node_list, tree.true_error)),
        tree.G.graph["t_star"],
    )


class WMSARCoreTests(unittest.TestCase):
    def test_config_and_repair_strength_validation(self):
        with self.assertRaises(ValueError):
            WMSARConfig(max_region_size=0)
        with self.assertRaises(ValueError):
            propagate_effective_error(_path_graph(), set(), strength=1.1)

    def test_budget_selects_one_region_without_overspending(self):
        graph = _path_graph()
        extractor = WMSAR()
        extractor.candidate_regions = lambda _: [
            Region(nodes={"n0", "n1"}, score=2.0),
            Region(nodes={"n2"}, score=1.0),
        ]
        selected = extractor.repair_region(graph, budget=2.0)
        self.assertEqual(selected, {"n0", "n1"})
        self.assertLessEqual(sum(graph.nodes[n]["cost"] for n in selected), 2.0)

    def test_budget_fallback_is_an_affordable_singleton(self):
        graph = _path_graph()
        graph.nodes["n0"]["cost"] = 5.0
        graph.nodes["n2"]["cost"] = 2.0
        extractor = WMSAR()
        extractor.candidate_regions = lambda _: [
            Region(nodes={"n0", "n1"}, score=2.0),
            Region(nodes={"n2"}, score=1.0),
        ]
        self.assertEqual(extractor.repair_region(graph, budget=1.0), {"n1"})
        self.assertEqual(extractor.repair_region(graph, budget=0.0), set())

    def test_extracted_region_is_connected_and_bounded(self):
        graph = _path_graph()
        config = WMSARConfig(max_region_size=3, n_seeds=2)
        region = WMSAR(config).repair_region(graph)
        self.assertLessEqual(len(region), config.max_region_size)
        if len(region) > 1:
            self.assertTrue(nx.is_connected(graph.subgraph(region).to_undirected()))

    def test_pipeline_smoke(self):
        tree = generate_calling_trees(n=1, seed=7)[0]
        graph = _failure_graph_from_tree(tree)
        plan = baselines.wm_sar(graph)
        self.assertNotIn(graph.graph["t_star"], plan.nodes)

    def test_cascade_gain_is_repeatable(self):
        tree = generate_calling_trees(n=1, seed=11)[0]
        graph = _failure_graph_from_tree(tree)
        first = inject_with_gain(graph, 1.1)
        second = inject_with_gain(graph, 1.1)
        self.assertEqual(
            nx.get_node_attributes(first, "err"),
            nx.get_node_attributes(second, "err"),
        )

    def test_llm_client_validates_configuration_and_exposes_chat(self):
        with patch.dict(
            os.environ,
            {
                "LLM_API_KEY": "",
                "OPENAI_API_KEY": "",
                "LLM_MODEL": "gpt-4o-mini",
            },
            clear=False,
        ):
            with self.assertRaisesRegex(ValueError, "missing"):
                LLMClient(backend="openai")

        client = object.__new__(LLMClient)
        client._call = lambda *_: ("ok", 2, 1, 3.0)
        result = client.chat("system", "user")
        self.assertEqual((result.text, result.total_tokens), ("ok", 3))


if __name__ == "__main__":
    unittest.main()
