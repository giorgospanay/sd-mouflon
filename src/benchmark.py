import numpy as np
import networkx as nx
import pandas as pd
import pickle
import os
import sys
import time
import concurrent.futures
from scipy.sparse import diags


# Globals for paths
obj_path="../data/obj"
log_path="../logs/"
plot_path="../plots/"

# sFairSC: maximum GB allowed for dense matrix conversion.
SFAIRSC_DENSE_LIMIT_GB = 4.0


# External module imports for Fairness-aware Louvain
sys.path.append("ext_modules")
sys.path.append("ext_modules/fairness-aware-louvain/algorithms")
sys.path.append("ext_modules/fairness-aware-louvain/community-detection")
from diversityFairness import diversityMetric
from modularityFairness import modularityFairnessMetric
from L_diversityFairness import LDiversityFairnessMetric
from L_modularityFairness import LModularityFairnessMetric
from redModularityLouvain import redFairness_louvain_communities
from blueModularityLouvain import blueFairness_louvain_communities
from LredModularityLouvain import LRedFairness_louvain_communities
from LblueModularityLouvain import LBlueFairness_louvain_communities
from diversityFairnessLouvain import diversityFairness_louvain_communities
from LdiversityLouvain import Ldiversity_louvain_communities

# MOUFLON import
sys.path.append(os.path.dirname(__file__))  # ensures src/ is on the path
from modules.fair_louvaines import fair_louvain_communities
from modules.helpers import fairness_base, fairness_fexp

# sFairSC import
from sfairsc import s_fair_sc


# --- Debug logger -------------------------------------------------------------

DEBUG = False

def dlog(*args, **kwargs):
	"""Print only when DEBUG is True."""
	if DEBUG:
		import traceback
		exc_info = kwargs.pop("exc_info", False)
		print("[DEBUG]", *args, **kwargs)
		if exc_info:
			traceback.print_exc()


# --- Algorithm registry -------------------------------------------------------
#
# Each entry is a dict:
#   name      : display name in results
#   call      : callable(G, attb_map, matrices) -> communities
#   alpha     : alpha value for this run (None for non-MOUFLON)
#   strategy  : strategy label (None for non-MOUFLON/sFairSC)
#
# F-AL algos: fn(G, weight=..., resolution=..., node_attributes=attb_map)
# MOUFLON:    fair_louvain_communities(G, color_list, alpha, strategy)
#             strategies: "base", "hybrid"
#             alphas:     [0.25, 0.5, 0.75]
# Louvain:    nx.community.louvain_communities -- standard baseline, no fairness
# sFairSC:    s_fair_sc(W, D, F, k) -- spectral, uses matrices
#
FAL_ALGOS = [
	{
		"name":     "FAL-RedModularity",
		"call":     lambda G, attb_map, mats: redFairness_louvain_communities(
						G, weight="weight", resolution=1, node_attributes=attb_map),
		"alpha":    None,
		"strategy": None,
	},
	{
		"name":     "FAL-BlueModularity",
		"call":     lambda G, attb_map, mats: blueFairness_louvain_communities(
						G, weight="weight", resolution=1, node_attributes=attb_map),
		"alpha":    None,
		"strategy": None,
	},
	{
		"name":     "FAL-LRedModularity",
		"call":     lambda G, attb_map, mats: LRedFairness_louvain_communities(
						G, weight="weight", resolution=1, node_attributes=attb_map),
		"alpha":    None,
		"strategy": None,
	},
	{
		"name":     "FAL-LBlueModularity",
		"call":     lambda G, attb_map, mats: LBlueFairness_louvain_communities(
						G, weight="weight", resolution=1, node_attributes=attb_map),
		"alpha":    None,
		"strategy": None,
	},
	{
		"name":     "FAL-Diversity",
		"call":     lambda G, attb_map, mats: diversityFairness_louvain_communities(
						G, weight="weight", resolution=1, node_attributes=attb_map),
		"alpha":    None,
		"strategy": None,
	},
	{
		"name":     "FAL-LDiversity",
		"call":     lambda G, attb_map, mats: Ldiversity_louvain_communities(
						G, weight="weight", resolution=1, node_attributes=attb_map),
		"alpha":    None,
		"strategy": None,
	},
]

# MOUFLON: "base" and "hybrid" strategies, alpha in [0.25, 0.5, 0.75]
MOUFLON_ALPHAS     = [0.25, 0.5, 0.75]
MOUFLON_STRATEGIES = ["base", "hybrid"]
MOUFLON_COLOR_LIST = ["blue", "red"]


def _sfairsc_wrapper(G, attb_map, W_sparse, D_sparse, F, k):
	"""
	Run sFairSC and convert its flat label list to a list-of-sets partition.

	W and D are kept sparse until this point. We estimate the memory cost of
	converting them to dense before doing so, and raise MemoryError cleanly
	if it would exceed SFAIRSC_DENSE_LIMIT_GB so the benchmark can log and
	continue rather than triggering the OOM killer.
	"""
	n = W_sparse.shape[0]
	# Two n×n float64 matrices (W and D): n*n*8 bytes each
	estimated_gb = 2 * n * n * 8 / 1024**3
	if estimated_gb > SFAIRSC_DENSE_LIMIT_GB:
		raise MemoryError(
			f"sFairSC dense matrix conversion would require ~{estimated_gb:.1f} GB "
			f"for n={n} (limit={SFAIRSC_DENSE_LIMIT_GB} GB) — skipping"
		)
	dlog(f"sFairSC: converting sparse to dense (~{estimated_gb:.2f} GB), n={n}, k={k}")
	W = np.array(W_sparse.todense(), dtype=float)
	D = np.array(D_sparse.todense(), dtype=float)
	labels = s_fair_sc(W, D, F, k)
	nodes  = sorted(G.nodes())
	communities = [set() for _ in range(k)]
	for node, label in zip(nodes, labels):
		communities[label].add(node)
	return [c for c in communities if c]  # drop empty communities


def build_algo_registry(sfairsc_k=(2, 3, 4, 5)):
	"""
	Return the full list of algorithm descriptors to benchmark.

	Parameters
	----------
	sfairsc_k : iterable of int
		k values to sweep for sFairSC. Pass an empty list to skip sFairSC entirely.
	"""
	registry = list(FAL_ALGOS)

	# Louvain baseline -- standard NetworkX, no fairness awareness
	registry.append({
		"name":     "Louvain",
		"call":     lambda G, attb_map, mats: list(
						nx.algorithms.community.louvain_communities(G, weight="weight")
					),
		"alpha":    None,
		"strategy": None,
	})

	# MOUFLON: sweep strategy x alpha
	for strat in MOUFLON_STRATEGIES:
		for a in MOUFLON_ALPHAS:
			_strat, _a = strat, a
			registry.append({
				"name":     f"MOUFLON-{_strat}",
				"call":     lambda G, attb_map, mats, s=_strat, alpha=_a: fair_louvain_communities(
								G,
								color_list=MOUFLON_COLOR_LIST,
								alpha=alpha,
								strategy=s,
							),
				"alpha":    _a,
				"strategy": _strat,
			})

	# sFairSC: sweep k (number of clusters)
	for k in sfairsc_k:
		_k = k
		registry.append({
			"name":     f"sFairSC-k{_k}",
			"call":     lambda G, attb_map, mats, k=_k: _sfairsc_wrapper(
							G, attb_map, mats[0], mats[1], mats[2], k
						),
			"alpha":    None,
			"strategy": f"k={_k}",
		})

	return registry


# --- Graph helpers ------------------------------------------------------------

def attbs_from_graph(G, color_attr="color"):
	"""Convert node color attributes to 0/1 mapping expected by F-AL."""
	mapping = {}
	for u, data in G.nodes(data=True):
		c = data.get(color_attr, None)
		if c is None:
			raise KeyError(f"Node {u} missing '{color_attr}' attribute.")
		if c == "red":
			mapping[u] = 0
		elif c == "blue":
			mapping[u] = 1
		else:
			raise ValueError(
				f"Node {u} has invalid {color_attr}={c!r}. Expected 'red' or 'blue'."
			)
	return mapping


def color_dist_from_graph(G):
	"""Build the {color: count} dict required by fairness_base and fairness_fexp."""
	colors = nx.get_node_attributes(G, "color")
	dist = {}
	for c in colors.values():
		dist[c] = dist.get(c, 0) + 1
	return dist


def open_graph(network):
	"""
	Load a .nx graph and prepare all required representations.

	Returns
	-------
	G          : NetworkX graph
	attb_map   : dict {node -> 0 (red) | 1 (blue)}  -- for F-AL and metrics
	color_dist : dict {"red": n, "blue": n}          -- for fairness_base / fexp
	matrices   : (W, D, F) numpy dense matrices      -- for sFairSC
	"""
	with open(f"{obj_path}/{network}.nx", "rb") as g_open:
		G = pickle.load(g_open)

	dlog(f"Loaded graph '{network}': {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

	attb_map   = attbs_from_graph(G)
	color_dist = color_dist_from_graph(G)

	dlog(f"Node colours: { {c: n for c, n in color_dist.items()} }")

	# Matrices for sFairSC — kept sparse here; _sfairsc_wrapper converts to dense
	# only after checking the memory cost.
	W_sparse = nx.adjacency_matrix(G)
	degree_values = [G.degree[node] for node in sorted(G.nodes())]
	D_sparse = diags(degree_values)
	colors = nx.get_node_attributes(G, "color")
	F = np.array(
		[[1.0] if colors[node] == "blue" else [0.0] for node in sorted(G.nodes())]
	)

	n = G.number_of_nodes()
	dlog(f"Sparse matrices ready: W and D are {n}x{n} "
		 f"(dense would be ~{2*n*n*8/1024**3:.2f} GB)")

	return G, attb_map, color_dist, (W_sparse, D_sparse, F)


# --- Metrics ------------------------------------------------------------------

def compute_metrics(G, communities, attb_map, color_dist):
	"""
	Compute all fairness and quality metrics for a partition.

	Includes:
	  - Standard modularity (NetworkX)
	  - F-AL: unfairness, red/blue modularity, L-variants, diversity, L-diversity
	  - MOUFLON: balance (fairness_base), fexp (fairness_fexp)

	Returns a flat dict of scalar values.
	"""
	dlog(f"Computing metrics over {len(communities)} communities "
		 f"(sizes: {sorted([len(c) for c in communities], reverse=True)})")

	modularity = nx.algorithms.community.modularity(G, communities, weight="weight")

	# F-AL metrics
	diversity, _ = diversityMetric(
		G, communities, attb_map, weight="weight", resolution=1
	)
	unfairness, _, _, red_mod_list, blue_mod_list = modularityFairnessMetric(
		G, communities, attb_map, weight="weight", resolution=1
	)
	l_unfairness, _, _, l_red_list, l_blue_list = LModularityFairnessMetric(
		G, communities, attb_map, weight="weight", resolution=1
	)
	l_diversity, _ = LDiversityFairnessMetric(
		G, communities, attb_map, weight="weight", resolution=1
	)

	# MOUFLON metrics -- fairness_base and fairness_fexp use G's "color" node
	# attributes directly; color_dist is the global {color: count} distribution
	balance, _ = fairness_base(G, communities, color_dist)
	fexp,    _ = fairness_fexp(G, communities, color_dist)

	result = {
		"modularity":        modularity,
		"unfairness":        unfairness,
		"red_modularity":    sum(red_mod_list),
		"blue_modularity":   sum(blue_mod_list),
		"l_unfairness":      l_unfairness,
		"l_red_modularity":  sum(l_red_list),
		"l_blue_modularity": sum(l_blue_list),
		"diversity":         diversity,
		"l_diversity":       l_diversity,
		"balance":           balance,
		"fexp":              fexp,
	}

	dlog("Metrics: " + "  ".join(f"{k}={v:.4f}" for k, v in result.items()))
	return result


# --- Single timed run (with timeout) -----------------------------------------

def _run_once(algo_call, G, attb_map, matrices):
	"""Execute one algorithm call and return (communities, elapsed_seconds)."""
	start = time.time()
	communities = algo_call(G, attb_map, matrices)
	elapsed = time.time() - start
	return communities, elapsed


def run_with_timeout(algo_call, G, attb_map, timeout, matrices=None):
	"""
	Run algo_call inside a thread with a hard timeout.
	Returns (communities, elapsed) or raises TimeoutError.
	"""
	with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
		future = executor.submit(_run_once, algo_call, G, attb_map, matrices)
		try:
			return future.result(timeout=timeout)
		except concurrent.futures.TimeoutError:
			raise TimeoutError(f"Algorithm exceeded timeout of {timeout}s")


# --- Core benchmark loop ------------------------------------------------------

def run_benchmark(network, n_runs, timeout=1800, sfairsc_k=(2, 3, 4, 5)):
	"""
	Run all algorithms on `network` for `n_runs` repetitions each.

	Parameters
	----------
	network : str
		Name of the graph file (without extension) under obj_path.
	n_runs : int
		Number of independent runs per (algorithm, alpha, strategy) combo.
	timeout : int
		Per-run wall-clock timeout in seconds (default 30 min).
	sfairsc_k : iterable of int
		k values to sweep for sFairSC (e.g. [3, 5]).
	"""
	print(f"\nRunning benchmark for network='{network}'\n{'='*50}")

	G, attb_map, color_dist, matrices = open_graph(network)
	registry = build_algo_registry(sfairsc_k=sfairsc_k)

	dlog(f"Registry has {len(registry)} algorithm configurations")

	# raw_results[key] = list of metric dicts across runs
	# key = (name, alpha, strategy)
	raw_results: dict[tuple, list[dict]] = {}

	for algo in registry:
		key   = (algo["name"], algo["alpha"], algo["strategy"])
		label = f"{algo['name']}  alpha={algo['alpha']}  strategy={algo['strategy']}"
		print(f"\n▶ {label}")

		run_metrics = []

		# sFairSC pre-flight: sqrtm(D) inside s_fair_sc always allocates a
		# dense n×n matrix, so skip all runs if the graph is too large
		# regardless of the sparse wrapper we pass in.
		if algo["name"].startswith("sFairSC"):
			n = G.number_of_nodes()
			estimated_gb = 2 * n * n * 8 / 1024**3
			if estimated_gb > SFAIRSC_DENSE_LIMIT_GB:
				print(f"  SKIP -- graph too large for sFairSC "
					  f"(n={n}, ~{estimated_gb:.1f} GB > limit {SFAIRSC_DENSE_LIMIT_GB} GB)")
				continue

		for i in range(n_runs):
			print(f"  run {i+1}/{n_runs} ...", end=" ", flush=True)
			try:
				communities, elapsed = run_with_timeout(
					algo["call"], G, attb_map, timeout, matrices=matrices
				)
				dlog(f"Algorithm returned {len(communities)} communities in {elapsed:.3f}s")
				metrics = compute_metrics(G, communities, attb_map, color_dist)
				metrics["runtime"] = elapsed
				run_metrics.append(metrics)
				print(f"done ({elapsed:.2f}s)  "
					  f"modularity={metrics['modularity']:.4f}  "
					  f"balance={metrics['balance']:.4f}  "
					  f"fexp={metrics['fexp']:.4f}")

			except TimeoutError:
				print(f"TIMEOUT after {timeout}s -- skipping remaining runs")
				break
			except Exception as e:
				print(f"ERROR: {e} -- skipping this run")
				dlog(f"Full traceback for {label} run {i+1}:", exc_info=True)

		if run_metrics:
			raw_results[key] = run_metrics
		else:
			print(f"  !! No successful runs for {label}")

	# -- Aggregate into one row per (algo, alpha, strategy) -------------------
	rows = []
	metric_keys = [
		"modularity",
		"unfairness", "red_modularity", "blue_modularity",
		"l_unfairness", "l_red_modularity", "l_blue_modularity",
		"diversity", "l_diversity",
		"balance", "fexp",
		"runtime",
	]

	for (name, alpha, strategy), runs in raw_results.items():
		row = {
			"network":      network,
			"algorithm":    name,
			"alpha":        alpha,
			"strategy":     strategy,
			"n_successful": len(runs),
		}
		for mk in metric_keys:
			values = [r[mk] for r in runs if mk in r]
			row[f"{mk}_mean"] = float(np.mean(values))        if values else np.nan
			row[f"{mk}_std"]  = float(np.std(values, ddof=1)) if len(values) > 1 else np.nan
		rows.append(row)

	df = pd.DataFrame(rows)

	dlog(f"Final DataFrame: {df.shape[0]} rows x {df.shape[1]} columns")

	# -- Save -----------------------------------------------------------------
	os.makedirs(log_path, exist_ok=True)
	out_csv = f"{log_path}/benchmark_{network}.csv"
	df.to_csv(out_csv, index=False)
	print(f"\nResults saved to {out_csv}")

	return df


# --- Entry point --------------------------------------------------------------

def main():
	import argparse

	parser = argparse.ArgumentParser(description="Fairness-aware community detection benchmark")
	parser.add_argument("networks",    nargs="+",               help="Graph names (filenames without .nx extension)")
	parser.add_argument("--runs",      type=int, default=10,    help="Number of runs per config (default: 10)")
	parser.add_argument("--timeout",   type=int, default=1800,  help="Per-run timeout in seconds (default: 1800)")
	parser.add_argument("--sfairsc-k", type=int, nargs="+", default=[2, 3, 4, 5],
						metavar="K",
						help="k values for sFairSC (default: 2 3 4 5). Pass 0 to skip sFairSC entirely.")
	parser.add_argument("--debug",     action="store_true",
						help="Enable debug output (graph stats, community sizes, all metrics per run, full tracebacks).")
	args = parser.parse_args()

	global DEBUG
	DEBUG = args.debug
	if DEBUG:
		print("[DEBUG] Debug mode enabled")

	sfairsc_k = [k for k in args.sfairsc_k if k > 0]

	all_dfs = []
	for net in args.networks:
		df = run_benchmark(net, n_runs=args.runs, timeout=args.timeout, sfairsc_k=sfairsc_k)
		all_dfs.append(df)
		print(df.to_string(index=False))

	if len(all_dfs) > 1:
		combined = pd.concat(all_dfs, ignore_index=True)
		out_path = f"{log_path}/benchmark_combined.csv"
		combined.to_csv(out_path, index=False)
		print(f"\nCombined results saved to {out_path}")


if __name__ == "__main__":
	main()