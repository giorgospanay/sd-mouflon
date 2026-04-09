import itertools
import pickle
import sys
import time

import networkx as nx
import numpy as np
import pandas as pd
from networkx.algorithms.community import modularity
from scipy import stats
from scipy.stats import f

from modules.fair_louvaines import fair_louvain_communities
from modules.helpers import fairness_fexp

# ---------------------------------------------------------------------------
# Globals
# ---------------------------------------------------------------------------

obj_path = "../data/obj"
log_path = "../logs/"
COLOR_LIST = ["blue", "red"]


# ---------------------------------------------------------------------------
# Statistical tests
# ---------------------------------------------------------------------------

def hotelling_t2_paired(X, Y, cond_threshold=1e10):
	"""
	Paired Hotelling T² on n x p arrays X, Y (paired by row).
	Returns (T2, F-statistic, p-value, fallback_used).

	fallback_used=True if the covariance matrix is singular or ill-conditioned
	(common on small/near-deterministic networks where Q and F barely vary
	across reps).  T², Fstat, p-value will be nan; rely on p_Q and p_F instead.
	"""
	D = X - Y
	n, p = D.shape

	if n <= p:
		raise ValueError(f"Need n_reps > p. Got n={n}, p={p}")

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
	network: str,
	alpha_list: list = [0.0, 0.25, 0.5, 0.75, 1.0],
	n_reps: int = 100,
	debug_mode: bool = False,
) -> pd.DataFrame:
	"""
	Run hybrid fair_louvain_communities n_reps times per alpha on a fixed
	saved network.  Same seed list across all alphas for valid pairing.
	"""
	if "color-node" not in network:
		raise ValueError(
			f"Expected a network name containing 'color-node'. Got: {network}"
		)

	with open(f"{obj_path}/{network}.nx", "rb") as g_open:
		net = pickle.load(g_open)

	print(f"Network '{network}' loaded.")
	print(f"  N={net.number_of_nodes()}, M={net.number_of_edges()}")

	colors = nx.get_node_attributes(net, "color")
	color_dist = {c: 0 for c in COLOR_LIST}
	for n_id in net.nodes():
		color_dist[colors[n_id]] += 1

	print("Color distribution:")
	for color, cnt in color_dist.items():
		print(f"  {color}: {cnt}")

	# Same seeds for every alpha — required for valid pairing
	seed_list = list(range(n_reps))
	run_rows = []

	for a in alpha_list:
		print(f"\n--- alpha={a} ---")

		for rep, seed in enumerate(seed_list):
			if n_reps <= 10 or rep % 10 == 0:
				print(f"  rep {rep + 1}/{n_reps}")

			start = time.time()
			res = fair_louvain_communities(
				net,
				color_list=COLOR_LIST,
				alpha=a,
				strategy="hybrid",
				seed=seed,
			)
			elapsed = time.time() - start

			Q = modularity(net, res, weight="weight")
			F = get_fairness_value(net, res, color_dist)

			run_rows.append(
				{
					"network": network,
					"alpha": a,
					"rep": rep,
					"seed": seed,
					"time": elapsed,
					"Q": Q,
					"F": F,
					"ncomms": len(res),
				}
			)

	runs_df = pd.DataFrame(run_rows)

	if not debug_mode:
		runs_df.to_csv(
			f"{log_path}/{network}_hybrid_fair_exp_alpha_sweep_runs.csv",
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
				"time_std": sub["time"].std(ddof=1),
				"ncomms_mean": sub["ncomms"].mean(),
				"ncomms_std": sub["ncomms"].std(ddof=1),
			}
		)
	return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Pairwise alpha tests
# ---------------------------------------------------------------------------

def compare_alpha_pairs_paired(runs_df: pd.DataFrame) -> pd.DataFrame:
	"""
	All pairwise alpha comparisons, paired by seed.
	Tests whether the (Q, F) distribution shifts significantly between alphas.
	"""
	alpha_vals = sorted(runs_df["alpha"].unique())
	pair_rows = []

	for a1, a2 in itertools.combinations(alpha_vals, 2):
		g1 = runs_df[runs_df["alpha"] == a1].sort_values(["seed", "rep"]).reset_index(drop=True)
		g2 = runs_df[runs_df["alpha"] == a2].sort_values(["seed", "rep"]).reset_index(drop=True)

		if len(g1) != len(g2):
			raise ValueError(f"Unequal run counts for alpha {a1} vs {a2}")
		if not np.array_equal(g1["seed"].to_numpy(), g2["seed"].to_numpy()):
			raise ValueError(f"Seeds do not align for alpha {a1} vs {a2}")

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

	if len(args) < 1:
		print(
			"Usage:\n"
			"  python3 test_stat_significance.py network [n_reps] [debug]\n\n"
			"Example:\n"
			"  python3 test_stat_significance.py my-color-node 100 debug"
		)
		sys.exit(1)

	network = args[0]
	n_reps = int(args[1]) if len(args) >= 2 else 100
	debug_mode = len(args) >= 3 and args[2].lower() == "debug"

	alpha_list = [0.0, 0.25, 0.5, 0.75, 1.0]

	runs_df = run_alpha_sweep(
		network=network,
		alpha_list=alpha_list,
		n_reps=n_reps,
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
		base = f"{log_path}/{network}_hybrid_fair_exp"
		summary_df.to_csv(f"{base}_alpha_sweep_summary.csv", index=False)
		pairs_df.to_csv(f"{base}_alpha_sweep_paired_tests.csv", index=False)
		print(f"\nResults saved to {log_path}")


if __name__ == "__main__":
	main()