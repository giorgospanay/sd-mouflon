import os
import pickle
import random
import statistics
import sys
import time

import networkx as nx
import pandas as pd
from cdlib.classes import NodeClustering
from networkx.algorithms.community import modularity

from modules.fair_louvaines import fair_louvain_communities
from modules.helpers import (
	diversity_fairness,
	diversityMetricPaper,
	fairness_base,
	fairness_fexp,
	modularity_fairness,
)

# ── Globals ───────────────────────────────────────────────────────────────────
obj_path  = "../data/obj"
log_path  = "../logs/"
plot_path = "../plots/"

COLOR_LIST = ["blue", "red"]


# ── Helpers ───────────────────────────────────────────────────────────────────

def add_colors(G: nx.Graph, seed: int = None) -> None:
	"""Assign random blue/red colours to every node in-place."""
	rng = random.Random(seed)
	for node in G.nodes():
		G.nodes[node]["color"] = "red" if rng.random() < 0.5 else "blue"


def lfr_gt_communities(G: nx.Graph):
	"""
	Extract ground-truth communities from the LFR 'community' node attribute.
	Returns a list of sets (one per community) and a matching NodeClustering.
	"""
	community_map: dict[frozenset, list] = {}
	for node, data in G.nodes(data=True):
		key = frozenset(data["community"])   # LFR stores a frozenset per node
		community_map.setdefault(key, []).append(node)
	gt_comms = [set(members) for members in community_map.values()]
	gt_nc = NodeClustering(gt_comms, G, method_name="lfr_planted")
	return gt_comms, gt_nc


def _nx_path(name: str) -> str:
	return os.path.join(obj_path, f"{name}.nx")


def load_or_generate(name: str, generator_fn) -> nx.Graph:
	"""
	Load a graph from disk if the .nx file already exists,
	otherwise call generator_fn(), save the result, and return it.
	generator_fn must return a plain nx.Graph with node colours already set.
	"""
	path = _nx_path(name)
	if os.path.exists(path):
		with open(path, "rb") as f:
			G = pickle.load(f)
		print(f"  Loaded '{name}' from {path}  "
			  f"(N={G.number_of_nodes()}, M={G.number_of_edges()})")
	else:
		print(f"  '{name}' not found — generating …")
		G = generator_fn()
		os.makedirs(obj_path, exist_ok=True)
		with open(path, "wb") as f:
			pickle.dump(G, f)
		print(f"  Saved '{name}' to {path}  "
			  f"(N={G.number_of_nodes()}, M={G.number_of_edges()})")
	return G


# ── Graph families ────────────────────────────────────────────────────────────

# ── Family 1 · Scalability — increasing size ──────────────────────────────────
#   Mirrors ER family: p ≈ 0.001, n = 1 000 … 200 000
#   mu=0.3 → clear community structure (good for scalability tests)

LFR_SIZE_N = [1_000, 2_000, 5_000, 10_000, 20_000, 50_000, 100_000, 200_000]

def _lfr_size_name(n: int) -> str:
	return f"LFR_size_{n}"

def _lfr_safe(seed: int, **kwargs) -> nx.Graph:
	"""
	Call nx.LFR_benchmark_graph with automatic fallback when the degree-sequence
	sampler fails (ExceededMaxIterations).

	Retry strategy (up to MAX_RETRIES attempts):
	  1. Bump the seed by 1  →  different random draw, same parameters.
	  2. After SEED_RETRIES seed-only attempts, also widen max_degree by 20 %
		 and relax min_community down by 20 % to give the sampler more room.
	"""
	MAX_RETRIES   = 20
	SEED_RETRIES  =  5   # pure-seed retries before also relaxing params

	base_max_deg      = kwargs.get("max_degree")
	base_min_comm     = kwargs.get("min_community")

	for attempt in range(MAX_RETRIES):
		try_kwargs = dict(kwargs, seed=seed + attempt)

		if attempt >= SEED_RETRIES:
			relax = 1.0 + 0.2 * (attempt - SEED_RETRIES + 1)
			try_kwargs["max_degree"]    = max(
				base_max_deg + 1, int(base_max_deg * relax)
			)
			try_kwargs["min_community"] = max(
				5, int(base_min_comm / relax)
			)

		try:
			G = nx.LFR_benchmark_graph(**try_kwargs)
			if attempt > 0:
				print(f"    ↳ LFR succeeded on attempt {attempt + 1} "
					  f"(seed={seed + attempt}, "
					  f"max_degree={try_kwargs['max_degree']}, "
					  f"min_community={try_kwargs['min_community']})")
			return nx.Graph(G)
		except nx.ExceededMaxIterations:
			print(f"    ↳ LFR attempt {attempt + 1}/{MAX_RETRIES} failed "
				  f"(seed={seed + attempt}) — retrying…")

	raise RuntimeError(
		f"LFR generation failed after {MAX_RETRIES} attempts. "
		f"Parameters may be fundamentally incompatible: {kwargs}"
	)


def _build_lfr_size(n: int, seed: int = 1) -> nx.Graph:
	# Fixed density across all sizes: avg_degree=20, mu=0.3.
	# This isolates n as the only variable so runtime/quality curves are clean.
	# min_community > average_degree ensures every node can satisfy its
	# intra-community degree budget; max_community ~ n/10 keeps >=10 communities
	# at every scale.
	AVG_K    = 20
	MAX_K    = 50      # ~2.5x avg; allows hubs without a pathological tail
	MIN_COMM = 30      # > AVG_K
	MAX_COMM = max(100, n // 10)

	G = _lfr_safe(
		seed,
		n=n,
		tau1=3,
		tau2=1.5,
		mu=0.3,
		average_degree=AVG_K,
		max_degree=MAX_K,
		min_community=MIN_COMM,
		max_community=MAX_COMM,
	)
	add_colors(G, seed=seed)
	return G


# ── Family 2 · Density sweep ──────────────────────────────────────────────────
#   Fixed n=10 000, vary average_degree over LFR's comfortable range.
#   mu=0.3 is held constant so density is the only changing variable.
#   min_community scales with k (must stay > avg_degree); max_community is
#   fixed at 2 000 so community count stays broadly comparable across k values.

LFR_DENSITY_K = [10, 20, 50, 100, 200]   # average-degree sweep

def _lfr_density_name(k: int) -> str:
	return f"LFR_density_k{k}"

def _build_lfr_density(k: int, n: int = 10_000, seed: int = 1) -> nx.Graph:
	G = _lfr_safe(
		seed,
		n=n,
		tau1=3,
		tau2=1.5,
		mu=0.3,
		average_degree=k,
		max_degree=3 * k,
		min_community=k + 5,
		max_community=2_000,
	)
	add_colors(G, seed=seed)
	return G


# ── Family 3 · Quality — planted partition ────────────────────────────────────
#   Varying mu (mixing parameter): low mu = clear structure, high mu = noisy
#   n = 1 000, two mu regimes × two colour ratios

LFR_QUALITY_MU  = [0.1, 0.2, 0.3, 0.4, 0.5]   # mixing parameter sweep
LFR_QUALITY_N   = 1_000
LFR_QUALITY_K        = 20   # average degree — matches size/density families
LFR_QUALITY_KMX      = 60   # max degree (~3x avg; satisfies min_community constraint)
LFR_QUALITY_MIN_COMM = 25   # > avg_degree
LFR_QUALITY_MAX_COMM = 100
LFR_QUALITY_MU       = [0.1, 0.2, 0.3, 0.4, 0.5]   # mixing parameter sweep
LFR_QUALITY_P_SENS   = [0.1, 0.2, 0.3, 0.4, 0.5]   # sensitive-attribute prevalence


def _lfr_quality_base_name(mu: float, seed: int = 1) -> str:
	"""Shared base graph (structure only, no colour) — reused by both scenarios."""
	mus = f"{int(mu * 10):02d}"
	return f"LFR_quality_mu{mus}_s{seed}_base"


# ── Scenario A: random colouring ("color-node" analogue) ─────────────────────
#   Each node is independently assigned red with probability p_sensitive,
#   blue otherwise.  Colour is uncorrelated with community structure.

def _lfr_quality_node_name(mu: float, p: float, seed: int = 1) -> str:
	mus = f"{int(mu * 10):02d}"
	ps  = f"{int(p  * 10):02d}"
	return f"LFR_quality-node_mu{mus}_p{ps}_s{seed}"

def _build_lfr_quality_base(mu: float, seed: int = 1) -> nx.Graph:
	"""Generate the LFR structure without colours (shared by both scenarios)."""
	return _lfr_safe(
		seed,
		n=LFR_QUALITY_N,
		tau1=3,
		tau2=1.5,
		mu=mu,
		average_degree=LFR_QUALITY_K,
		max_degree=LFR_QUALITY_KMX,
		min_community=LFR_QUALITY_MIN_COMM,
		max_community=LFR_QUALITY_MAX_COMM,
	)

def _add_colors_random(G: nx.Graph, p_sensitive: float, seed: int = 1) -> None:
	"""Colour-node scenario: each node drawn independently from p_sensitive."""
	rng = random.Random(seed)
	for node in G.nodes():
		G.nodes[node]["color"] = "red" if rng.random() < p_sensitive else "blue"


# ── Scenario B: homogeneous communities ("color-full" analogue) ───────────────
#   All nodes within a community share the same colour.
#   round(p_sensitive × n_communities) communities are assigned red,
#   the rest blue.  Assignment is deterministic given the sorted community list
#   and the seed, so graphs are reproducible.

def _lfr_quality_comm_name(mu: float, p: float, seed: int = 1) -> str:
	mus = f"{int(mu * 10):02d}"
	ps  = f"{int(p  * 10):02d}"
	return f"LFR_quality-comm_mu{mus}_p{ps}_s{seed}"

def _add_colors_homogeneous(G: nx.Graph, p_sensitive: float, seed: int = 1) -> None:
	"""
	Homogeneous scenario: every node in a community gets the same colour.
	round(p_sensitive * n_communities) communities become red, rest blue.
	Communities are sorted by min-node-id for determinism, then shuffled
	with the given seed before assigning red/blue.
	"""
	# Extract communities from LFR node attribute
	comm_map: dict[frozenset, list] = {}
	for node, data in G.nodes(data=True):
		key = frozenset(data["community"])
		comm_map.setdefault(key, []).append(node)

	communities = list(comm_map.values())
	# Deterministic shuffle so same seed → same red/blue assignment
	rng = random.Random(seed)
	rng.shuffle(communities)

	n_red = round(p_sensitive * len(communities))
	red_comms = set(id(c) for c in communities[:n_red])

	for comm in communities:
		color = "red" if id(comm) in red_comms else "blue"
		for node in comm:
			G.nodes[node]["color"] = color


# ── Core metric collection ────────────────────────────────────────────────────

def _symmetric_nf1(pred_nc, gt_nc) -> float:
	"""
	Compute the symmetric NF1 score as the average of both matching directions:
	  - forward:  each predicted community matched to best ground-truth community
	  - backward: each ground-truth community matched to best predicted community
	This ensures NF1 in [0, 1] regardless of the relative number of communities
	in each partition.
	"""
	nf1_forward  = pred_nc.nf1(gt_nc).score
	nf1_backward = gt_nc.nf1(pred_nc).score
	return (nf1_forward + nf1_backward) / 2


def _collect_metrics(
	net: nx.Graph,
	res,
	color_dist: dict,
	colors: dict,
	gt_nc=None,
) -> dict:
	"""Run all metrics on a single partition result. Returns a flat dict."""
	mod = modularity(net, res, weight="weight")

	F_bal, _ = fairness_base(net, res, color_dist)
	F_exp, _ = fairness_fexp(net, res, color_dist)

	F_modf, _ = modularity_fairness(net, res, color_dist, colors)
	F_modf_norm = 1 - abs(F_modf)

	F_div, _ = diversity_fairness(net, res, color_dist, colors)
	F_div_norm = 1 - abs(F_div)

	F_div_paper, _ = diversityMetricPaper(net, res, colors)
	F_div_paper_norm = 1 - abs(F_div_paper)

	ami = nf1 = None
	if gt_nc is not None:
		pred_nc = NodeClustering([set(r) for r in res], net, method_name="")
		ami = pred_nc.adjusted_mutual_information(gt_nc).score
		# Symmetric NF1: average of both matching directions to guarantee [0, 1]
		nf1 = _symmetric_nf1(pred_nc, gt_nc)

	return dict(
		ncomms=len(res),
		modularity=mod,
		fair_bal=F_bal,
		fair_exp=F_exp,
		fair_modf=F_modf_norm,
		fair_div=F_div_norm,
		fair_div_paper=F_div_paper_norm,
		ami=ami,
		nf1=nf1,
	)


# ── Single experiment runner ──────────────────────────────────────────────────

def run_experiment(
	net: nx.Graph,
	name: str,
	color_list=None,
	alpha=None,
	n_reps: int = 3,
	strategy=None,
	planted: bool = False,
	debug_mode: bool = False,
	extra_cols: dict = None,   # e.g. {"n": 1000} or {"p": 0.3, "mu": 0.3}
):
	"""
	Run the full alpha × strategy grid on *net* and return a DataFrame.
	Optionally save to log_path/<name>.csv when not in debug mode.
	"""
	if color_list is None:
		color_list = COLOR_LIST
	if alpha is None:
		alpha = [round(a * 0.1, 1) for a in range(11)]
	if strategy is None:
		strategy = ["base", "step2", "fexp", "hybrid", "fmody", "diversity",
					"step2fmody", "step2div"]

	colors = nx.get_node_attributes(net, "color")
	color_dist = {c: 0 for c in color_list}
	for n_id in net.nodes():
		color_dist[colors[n_id]] += 1

	gt_nc = None
	if planted:
		_, gt_nc = lfr_gt_communities(net)

	rows = []

	for strat in strategy:
		print(f"  Strategy={strat}")
		for a in alpha:
			print(f"    alpha={a}")
			reps: list[dict] = []
			times = []
			for i in range(n_reps):
				t0 = time.time()
				res = fair_louvain_communities(
					net, color_list=color_list, alpha=a, strategy=strat
				)
				times.append(time.time() - t0)
				reps.append(_collect_metrics(net, res, color_dist, colors, gt_nc))

			def mean(key):
				vals = [r[key] for r in reps if r[key] is not None]
				return statistics.fmean(vals) if vals else None

			def std(key):
				vals = [r[key] for r in reps if r[key] is not None]
				return statistics.stdev(vals) if len(vals) > 1 else 0.0

			row = dict(
				network=name,
				strategy=strat,
				alpha=a,
				time=statistics.fmean(times),
				time_std=statistics.stdev(times) if n_reps > 1 else 0.0,
			)
			if extra_cols:
				row.update(extra_cols)
			for key in ("ncomms", "modularity", "fair_bal", "fair_exp",
						"fair_modf", "fair_div", "fair_div_paper", "ami", "nf1"):
				row[key]          = mean(key)
				row[f"{key}_std"] = std(key)
			rows.append(row)

	# ── Vanilla Louvain baseline ──────────────────────────────────────────────
	if not any(x in name for x in ("color-full", "color-node")):
		print("  Strategy=louvain (baseline)")
		reps = []
		times = []
		for i in range(n_reps):
			t0 = time.time()
			res = nx.community.louvain_communities(net)
			times.append(time.time() - t0)
			reps.append(_collect_metrics(net, res, color_dist, colors, gt_nc))

		def mean(key):
			vals = [r[key] for r in reps if r[key] is not None]
			return statistics.fmean(vals) if vals else None

		def std(key):
			vals = [r[key] for r in reps if r[key] is not None]
			return statistics.stdev(vals) if len(vals) > 1 else 0.0

		row = dict(
			network=name,
			strategy="louvain",
			alpha=1.0,
			time=statistics.fmean(times),
			time_std=statistics.stdev(times) if n_reps > 1 else 0.0,
		)
		if extra_cols:
			row.update(extra_cols)
		for key in ("ncomms", "modularity", "fair_bal", "fair_exp",
					"fair_modf", "fair_div", "fair_div_paper", "ami", "nf1"):
			row[key]          = mean(key)
			row[f"{key}_std"] = std(key)
		rows.append(row)

	df = pd.DataFrame(rows)

	if not debug_mode:
		os.makedirs(log_path, exist_ok=True)
		out = os.path.join(log_path, f"{name}.csv")
		df.to_csv(out, index=False)
		print(f"  Saved → {out}")
	else:
		cols = ["strategy", "alpha", "modularity", "fair_bal", "fair_exp",
				"fair_modf", "fair_div", "fair_div_paper", "ami", "nf1", "ncomms"]
		print(df[[c for c in cols if c in df.columns]])

	return df


# ── Public experiment entry-points ────────────────────────────────────────────

def lfr_scalability_size(
	alpha=None, n_reps=3, strategy=None,
	debug_mode=False, color_list=None,
):
	"""
	Family 1 — scalability by increasing graph size.
	One CSV per graph size written to log_path/.
	"""
	all_dfs = []
	for n in LFR_SIZE_N:
		name = _lfr_size_name(n)
		print(f"\n── LFR size experiment: n={n} ({'─'*40})")
		G = load_or_generate(name, lambda n=n: _build_lfr_size(n))
		df = run_experiment(
			G, name,
			color_list=color_list,
			alpha=alpha,
			n_reps=n_reps,
			strategy=strategy,
			planted=False,
			debug_mode=debug_mode,
			extra_cols={"lfr_n": n},
		)
		all_dfs.append(df)

	combined = pd.concat(all_dfs, ignore_index=True)
	if not debug_mode:
		out = os.path.join(log_path, "LFR_scalability_size.csv")
		combined.to_csv(out, index=False)
		print(f"\nCombined size results → {out}")
	return combined


def lfr_scalability_density(
	alpha=None, n_reps=3, strategy=None,
	debug_mode=False, color_list=None,
):
	"""
	Family 2 — scalability by increasing graph density.
	One CSV per density value written to log_path/.
	"""
	all_dfs = []
	for k in LFR_DENSITY_K:
		name = _lfr_density_name(k)
		print(f"\n── LFR density experiment: avg_degree={k} ({'─'*40})")
		G = load_or_generate(name, lambda k=k: _build_lfr_density(k))
		df = run_experiment(
			G, name,
			color_list=color_list,
			alpha=alpha,
			n_reps=n_reps,
			strategy=strategy,
			planted=False,
			debug_mode=debug_mode,
			extra_cols={"lfr_avg_degree": k},
		)
		all_dfs.append(df)

	combined = pd.concat(all_dfs, ignore_index=True)
	if not debug_mode:
		out = os.path.join(log_path, "LFR_scalability_density.csv")
		combined.to_csv(out, index=False)
		print(f"\nCombined density results → {out}")
	return combined


def lfr_quality(
	alpha=None, n_reps=3, strategy=None,
	debug_mode=False, color_list=None, seed=1,
):
	"""
	Family 3 — quality experiments sweeping mu × p_sensitive.

	Two scenarios per (mu, p_sensitive) combination:
	  - quality-node : colour assigned randomly per node (uncorrelated with structure)
	  - quality-comm : colour assigned per community (homogeneous communities)

	The underlying LFR structure (same mu, same seed) is shared and cached once
	as a colour-free base graph; each coloured variant is then saved separately.

	Ground truth: LFR 'community' node attribute.
	One CSV per (scenario, mu, p_sensitive) written to log_path/.
	"""
	all_dfs = []

	for mu in LFR_QUALITY_MU:
		# ── Generate / load the shared base graph for this mu ─────────────────
		base_name = _lfr_quality_base_name(mu, seed=seed)
		base_path = _nx_path(base_name)
		if os.path.exists(base_path):
			with open(base_path, "rb") as f:
				G_base = pickle.load(f)
			print(f"\n  Base graph '{base_name}' loaded.")
		else:
			print(f"\n  Generating base graph for mu={mu} …")
			G_base = _build_lfr_quality_base(mu, seed=seed)
			os.makedirs(obj_path, exist_ok=True)
			with open(base_path, "wb") as f:
				pickle.dump(G_base, f)
			print(f"  Saved base graph → {base_path}")

		for p in LFR_QUALITY_P_SENS:

			for scenario, name_fn, color_fn in [
				("node", _lfr_quality_node_name, _add_colors_random),
				("comm", _lfr_quality_comm_name, _add_colors_homogeneous),
			]:
				name = name_fn(mu, p, seed=seed)
				print(f"\n── LFR quality-{scenario}: mu={mu}, p_sensitive={p}")

				# Build coloured variant from base, or load if cached
				nx_path = _nx_path(name)
				if os.path.exists(nx_path):
					with open(nx_path, "rb") as f:
						G = pickle.load(f)
					print(f"  Loaded '{name}'.")
				else:
					G = G_base.copy()
					color_fn(G, p_sensitive=p, seed=seed)
					with open(nx_path, "wb") as f:
						pickle.dump(G, f)
					print(f"  Saved '{name}' → {nx_path}")

				df = run_experiment(
					G, name,
					color_list=color_list,
					alpha=alpha,
					n_reps=n_reps,
					strategy=strategy,
					planted=True,
					debug_mode=debug_mode,
					extra_cols={
						"lfr_scenario": scenario,
						"lfr_mu": mu,
						"lfr_p_sensitive": p,
					},
				)
				all_dfs.append(df)

	combined = pd.concat(all_dfs, ignore_index=True)
	if not debug_mode:
		out = os.path.join(log_path, "LFR_quality.csv")
		combined.to_csv(out, index=False)
		print(f"\nCombined quality results → {out}")
	return combined


# ── CLI ───────────────────────────────────────────────────────────────────────
"""
Usage:
  python lfr_experiments.py <family> [color_list] [alpha] [n_reps] [strat_list] [debug]

  family      : size | density | quality | all
  color_list  : comma-separated, e.g. blue,red   (default: blue,red)
  alpha       : comma-separated floats            (default: 0.0,0.1,…,1.0)
  n_reps      : int                               (default: 3)
  strat_list  : comma-separated strategy names   (default: all strategies)
  debug       : literal string "debug"

Examples:
  python lfr_experiments.py size
  python lfr_experiments.py quality blue,red 0.0,0.5,1.0 5 step2,hybrid debug
  python lfr_experiments.py all blue,red 0.0,0.5,1.0 3 step2,hybrid
"""

def _parse_args(args):
	color_list = COLOR_LIST
	alpha      = [round(a * 0.1, 1) for a in range(11)]
	n_reps     = 3
	strategy   = None     # None → run_experiment uses its own default
	debug_mode = False

	if len(args) >= 2:
		color_list = args[1].split(",")
	if len(args) >= 3:
		alpha = [float(x) for x in args[2].split(",")]
	if len(args) >= 4:
		n_reps = int(args[3])
	if len(args) >= 5:
		strategy = args[4].split(",")
	if len(args) >= 6 and args[5] == "debug":
		debug_mode = True

	return color_list, alpha, n_reps, strategy, debug_mode


def main():
	args = sys.argv[1:]
	if not args:
		print("Usage: python lfr_experiments.py <family> [options…]")
		print("  family: size | density | quality | all")
		sys.exit(1)

	family = args[0].lower()
	color_list, alpha, n_reps, strategy, debug_mode = _parse_args(args)

	kwargs = dict(
		alpha=alpha, n_reps=n_reps, strategy=strategy,
		debug_mode=debug_mode, color_list=color_list,
	)

	if family == "size":
		lfr_scalability_size(**kwargs)
	elif family == "density":
		lfr_scalability_density(**kwargs)
	elif family == "quality":
		lfr_quality(**kwargs)
	elif family == "all":
		lfr_scalability_size(**kwargs)
		lfr_scalability_density(**kwargs)
		lfr_quality(**kwargs)
	else:
		print(f"Unknown family '{family}'. Choose: size | density | quality | all")
		sys.exit(1)


if __name__ == "__main__":
	main()