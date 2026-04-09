import itertools
import signal
import sys
import time

import networkx as nx
import numpy as np
import pandas as pd
from networkx.algorithms.community import modularity
from networkx.generators.community import LFR_benchmark_graph
from scipy import stats
from scipy.stats import f

from modules.fair_louvaines import fair_louvain_communities
from modules.helpers import fairness_fexp

# ---------------------------------------------------------------------------
# Globals
# ---------------------------------------------------------------------------

log_path = "../logs/"
COLOR_LIST = ["blue", "red"]

# Maximum seconds to wait for a single LFR network to generate before skipping.
GENERATION_TIMEOUT_SECS = 45

LFR_PARAMS = dict(
	n=250,
	tau1=2,
	tau2=1.5,
	mu=0.3,
	average_degree=8,
	min_community=10,
	max_iters=1000,
)


# ---------------------------------------------------------------------------
# LFR generation
# ---------------------------------------------------------------------------

class _GenerationTimeout(Exception):
	pass

def _timeout_handler(signum, frame):
	raise _GenerationTimeout()


def generate_lfr_network(network_seed: int, max_retries: int = 10) -> nx.Graph:
	"""
	Generate one LFR benchmark graph with blue/red colors assigned 50/50.
	Retries use seeds in a range far above the normal network index space
	to avoid colliding with another network's seed and breaking pairing.
	"""
	retry_base = LFR_PARAMS["n"] * 100  # e.g. 25000 — well past any network index
	G = None

	for attempt in range(max_retries):
		seed = network_seed if attempt == 0 else retry_base + network_seed * max_retries + attempt
		try:
			signal.signal(signal.SIGALRM, _timeout_handler)
			signal.alarm(GENERATION_TIMEOUT_SECS)
			try:
				G = LFR_benchmark_graph(**LFR_PARAMS, seed=seed)
			finally:
				signal.alarm(0)  # always cancel alarm
			if attempt > 0:
				print(f"  [warn] network {network_seed}: seed {network_seed} failed, used fallback seed {seed}")
			break
		except (nx.ExceededMaxIterations, _GenerationTimeout) as e:
			reason = "timed out" if isinstance(e, _GenerationTimeout) else "exceeded max iters"
			if attempt == max_retries - 1:
				raise RuntimeError(
					f"LFR generation failed for network {network_seed} "
					f"after {max_retries} retries ({reason}). Try relaxing LFR_PARAMS."
				)
			print(f"  [warn] network {network_seed} attempt {attempt+1} {reason}, retrying...")

	G.remove_edges_from(nx.selfloop_edges(G))

	rng = np.random.default_rng(network_seed)
	nodes = list(G.nodes())
	n = len(nodes)
	colors = ["blue"] * (n // 2) + ["red"] * (n - n // 2)
	rng.shuffle(colors)
	for node, color in zip(nodes, colors):
		G.nodes[node]["color"] = color

	return G


# ---------------------------------------------------------------------------
# Statistical tests
# ---------------------------------------------------------------------------

def hotelling_t2_paired(X, Y, cond_threshold=1e10):
	"""
	Paired Hotelling T2 on n x p arrays X, Y (paired by row).
	Returns (T2, F-statistic, p-value, fallback_used).

	fallback_used=True if the covariance matrix is singular or ill-conditioned;
	T2, Fstat, p-value will be nan.  Rely on univariate t-tests in that case.
	"""
	D = X - Y
	n, p = D.shape

	if n <= p:
		raise ValueError(f"Need n_obs > p. Got n={n}, p={p}")

	dbar = D.mean(axis=0)
	S = np.cov(D, rowvar=False, ddof=1)

	rank = np.linalg.matrix_rank(S)
	cond = np.linalg.cond(S) if rank == p else np.inf

	if rank < p or cond > cond_threshold:
		return np.nan, np.nan, np.nan, True

	T2 = n * dbar.T @ np.linalg.inv(S) @ dbar
	Fstat = ((n - p) / (p * (n - 1))) * T2
	pval = 1 - f.cdf(Fstat, p, n - p)

	return T2, Fstat, pval, False


# ---------------------------------------------------------------------------
# Fairness
# ---------------------------------------------------------------------------

def get_fairness_value(net, res, color_dist):
	F_overall, _ = fairness_fexp(net, res, color_dist)
	return F_overall


# ---------------------------------------------------------------------------
# Core sweep
# ---------------------------------------------------------------------------

def run_alpha_sweep(
	n_networks: int = 50,
	alpha_list: list = [0.0, 0.25, 0.5, 0.75, 1.0],
	debug_mode: bool = False,
) -> pd.DataFrame:
	"""
	For each of K=n_networks LFR networks, run hybrid fair_louvain_communities
	once per alpha.  Louvain seed = network_id for reproducibility.
	"""
	run_rows = []

	for net_idx in range(n_networks):
		print(f"\nNetwork {net_idx + 1}/{n_networks}  (seed={net_idx})")

		try:
			net = generate_lfr_network(network_seed=net_idx)
		except RuntimeError as e:
			print(f"  [skip] network {net_idx} skipped: {e}")
			continue
		print(f"  N={net.number_of_nodes()}, M={net.number_of_edges()}")

		colors = nx.get_node_attributes(net, "color")
		color_dist = {c: 0 for c in COLOR_LIST}
		for node in net.nodes():
			color_dist[colors[node]] += 1

		for a in alpha_list:
			start = time.time()
			res = fair_louvain_communities(
				net,
				color_list=COLOR_LIST,
				alpha=a,
				strategy="hybrid",
				seed=net_idx,
			)
			elapsed = time.time() - start

			Q = modularity(net, res, weight="weight")
			F = get_fairness_value(net, res, color_dist)

			run_rows.append(
				{
					"network_id": net_idx,
					"alpha": a,
					"time": elapsed,
					"Q": Q,
					"F": F,
					"ncomms": len(res),
				}
			)

	runs_df = pd.DataFrame(run_rows)

	if not debug_mode:
		runs_df.to_csv(
			f"{log_path}/lfr_hybrid_fair_exp_alpha_sweep_runs.csv",
			index=False,
		)

	return runs_df


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def summarize_runs(runs_df: pd.DataFrame) -> pd.DataFrame:
	rows = []
	for a in sorted(runs_df["alpha"].unique()):
		sub = runs_df[runs_df["alpha"] == a]
		rows.append(
			{
				"alpha": a,
				"Q_mean": sub["Q"].mean(),
				"Q_std": sub["Q"].std(ddof=1),
				"F_mean": sub["F"].mean(),
				"F_std": sub["F"].std(ddof=1),
				"time_mean": sub["time"].mean(),
				"ncomms_mean": sub["ncomms"].mean(),
			}
		)
	return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Pairwise alpha tests
# ---------------------------------------------------------------------------

def compare_alpha_pairs_paired(runs_df: pd.DataFrame) -> pd.DataFrame:
	"""
	All pairwise alpha comparisons, paired by network_id.
	Tests whether the (Q, F) distribution shifts significantly between alphas.
	"""
	alpha_vals = sorted(runs_df["alpha"].unique())
	pair_rows = []

	for a1, a2 in itertools.combinations(alpha_vals, 2):
		g1 = runs_df[runs_df["alpha"] == a1].sort_values("network_id").reset_index(drop=True)
		g2 = runs_df[runs_df["alpha"] == a2].sort_values("network_id").reset_index(drop=True)

		if len(g1) != len(g2):
			raise ValueError(f"Unequal counts for alpha {a1} vs {a2}")
		if not np.array_equal(g1["network_id"].to_numpy(), g2["network_id"].to_numpy()):
			raise ValueError(f"network_ids do not align for alpha {a1} vs {a2}")

		xQ, yQ = g1["Q"].to_numpy(), g2["Q"].to_numpy()
		xF, yF = g1["F"].to_numpy(), g2["F"].to_numpy()

		tQ, pQ = stats.ttest_rel(xQ, yQ)
		tF, pF = stats.ttest_rel(xF, yF)

		X = g1[["Q", "F"]].to_numpy()
		Y = g2[["Q", "F"]].to_numpy()
		T2, Fstat, pQF, t2_fallback = hotelling_t2_paired(X, Y)

		pair_rows.append(
			{
				"alpha_1": a1,
				"alpha_2": a2,
				"Q_mean_1": xQ.mean(),
				"Q_mean_2": yQ.mean(),
				"F_mean_1": xF.mean(),
				"F_mean_2": yF.mean(),
				"Q_diff_mean": (xQ - yQ).mean(),
				"Q_diff_std": (xQ - yQ).std(ddof=1),
				"F_diff_mean": (xF - yF).mean(),
				"F_diff_std": (xF - yF).std(ddof=1),
				"t_Q": tQ,
				"p_Q": pQ,
				"t_F": tF,
				"p_F": pF,
				"T2_QF": T2,
				"Fstat_QF": Fstat,
				"p_QF": pQF,
				"T2_fallback": t2_fallback,
			}
		)

	return pd.DataFrame(pair_rows)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
	args = sys.argv[1:]

	n_networks = int(args[0]) if len(args) >= 1 else 50
	debug_mode = len(args) >= 2 and args[1].lower() == "debug"

	alpha_list = [0.0, 0.25, 0.5, 0.75, 1.0]

	print(f"Generating and sweeping {n_networks} LFR networks (hybrid only).")
	print(f"Alpha list : {alpha_list}")
	print(f"LFR params : {LFR_PARAMS}")

	runs_df = run_alpha_sweep(
		n_networks=n_networks,
		alpha_list=alpha_list,
		debug_mode=debug_mode,
	)

	summary_df = summarize_runs(runs_df)
	pairs_df = compare_alpha_pairs_paired(runs_df)

	print("\n=== Per-alpha summary ===")
	print(summary_df.to_string(index=False))

	print("\n=== Pairwise alpha tests ===")
	print(
		pairs_df[
			["alpha_1", "alpha_2", "Q_diff_mean", "F_diff_mean", "p_Q", "p_F", "p_QF"]
		].to_string(index=False)
	)

	if not debug_mode:
		base = f"{log_path}/lfr_hybrid_fair_exp"
		summary_df.to_csv(f"{base}_alpha_sweep_summary.csv", index=False)
		pairs_df.to_csv(f"{base}_alpha_sweep_paired_tests.csv", index=False)
		print(f"\nResults saved to {log_path}")


if __name__ == "__main__":
	main()