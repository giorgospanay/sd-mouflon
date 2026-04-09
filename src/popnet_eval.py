import pickle
import random
import statistics
import sys
import time

import networkx as nx
import pandas as pd
from networkx.algorithms.community import modularity
from sklearn.metrics import adjusted_mutual_info_score

from modules.fair_louvaines import fair_louvain_communities
from modules.helpers import fairness_base, fairness_fexp


# ── Paths (mirror project conventions) ───────────────────────────────────────
obj_path = "../data/obj"
log_path = "../logs/"


# ── Helpers ───────────────────────────────────────────────────────────────────

def partition_to_labels(partition, nodes):
	"""Map a list-of-sets partition to a per-node label list (ordered by nodes)."""
	labels = {}
	for cid, comm in enumerate(partition):
		for u in comm:
			labels[u] = cid
	return [labels[u] for u in nodes]


def community_size_stats(partition):
	"""Return (mean, stdev, min, max) of community sizes."""
	sizes = [len(c) for c in partition]
	return (
		statistics.fmean(sizes),
		statistics.stdev(sizes) if len(sizes) > 1 else 0.0,
		min(sizes),
		max(sizes),
	)


def score_partition(net, partition, color_dist):
	"""Return (modularity, fair_bal, fair_exp, fbal_dist) for a single partition."""
	mod              = modularity(net, partition, weight="weight")
	F_bal, fbal_dist = fairness_base(net, partition, color_dist)
	F_exp, _         = fairness_fexp(net, partition, color_dist)
	return mod, F_bal, F_exp, fbal_dist


def community_dist_rows(partition, fbal_dist, strategy, alpha, criterion, colors, color_list):
	"""
	Build one row per community for the distribution output.

	fbal_dist is the per-community balance dict returned by fairness_base,
	keyed by community index.  Each row contains:
	  strategy, alpha, criterion, comm_id, size, fair_bal_comm,
	  and a count column per color (n_blue, n_red, …).
	"""
	rows = []
	for cid, comm in enumerate(partition):
		color_counts = {c: 0 for c in color_list}
		for node in comm:
			color_counts[colors[node]] += 1
		row = {
			"strategy"     : strategy,
			"alpha"        : alpha,
			"criterion"    : criterion,
			"comm_id"      : cid,
			"size"         : len(comm),
			"fair_bal_comm": fbal_dist[cid] if cid < len(fbal_dist) else None,
		}
		row.update({f"n_{c}": color_counts[c] for c in color_list})
		rows.append(row)
	return rows


def top_k_comm_rows(partition, fbal_dist, strategy, alpha, criterion,
					colors, color_list, top_k):
	"""
	Return rows for the top_k largest communities in a partition.
	Each row has the same schema as community_dist_rows plus a 'rank' column
	(1 = largest).  Communities smaller than top_k are silently omitted if
	the partition itself has fewer than top_k communities.
	"""
	indexed = sorted(enumerate(partition), key=lambda x: len(x[1]), reverse=True)
	rows = []
	for rank, (cid, comm) in enumerate(indexed[:top_k], start=1):
		color_counts = {c: 0 for c in color_list}
		for node in comm:
			color_counts[colors[node]] += 1
		row = {
			"strategy"     : strategy,
			"alpha"        : alpha,
			"criterion"    : criterion,
			"rank"         : rank,
			"comm_id"      : cid,
			"size"         : len(comm),
			"fair_bal_comm": fbal_dist[cid] if cid < len(fbal_dist) else None,
		}
		row.update({f"n_{c}": color_counts[c] for c in color_list})
		rows.append(row)
	return rows


# ── Small pure helpers ────────────────────────────────────────────────────────

def _best_indices(mod_idx, fexp_idx):
	"""
	Yield (criterion_label, index) pairs, deduplicating when both criteria
	select the same rep (only one block emitted, labelled 'best_mod').
	"""
	yield ("best_mod", mod_idx)
	if fexp_idx != mod_idx:
		yield ("best_fexp", fexp_idx)


def _best_run_row(strategy, alpha, criterion, rep_idx, seed,
				  mod, F_bal, F_exp, ncomms):
	return {
		"strategy" : strategy,
		"alpha"    : alpha,
		"criterion": criterion,
		"rep"      : rep_idx,
		"seed"     : seed,
		"modularity": mod,
		"fair_bal" : F_bal,
		"fair_exp" : F_exp,
		"ncomms"   : ncomms,
	}


def _summary_row(strategy, alpha,
				 mod_runs, fbal_runs, fexp_runs, ami_runs,
				 ncomms_runs,
				 sz_mean_r, sz_std_r, sz_min_r, sz_max_r,
				 time_runs):
	return {
		"strategy"       : strategy,
		"alpha"          : alpha,
		"modularity"     : statistics.fmean(mod_runs),
		"mod_std"        : statistics.stdev(mod_runs),
		"fair_bal"       : statistics.fmean(fbal_runs),
		"fair_bal_std"   : statistics.stdev(fbal_runs),
		"fair_exp"       : statistics.fmean(fexp_runs),
		"fair_exp_std"   : statistics.stdev(fexp_runs),
		"ami_louvain"    : statistics.fmean(ami_runs),
		"ami_louvain_std": statistics.stdev(ami_runs),
		"ncomms"         : statistics.fmean(ncomms_runs),
		"ncomms_std"     : statistics.stdev(ncomms_runs),
		"size_mean"      : statistics.fmean(sz_mean_r),
		"size_std"       : statistics.fmean(sz_std_r),
		"size_min"       : statistics.fmean(sz_min_r),
		"size_max"       : statistics.fmean(sz_max_r),
		"time"           : statistics.fmean(time_runs),
		"time_std"       : statistics.stdev(time_runs),
	}


# ── Main experiment ───────────────────────────────────────────────────────────

def experiment(
	network,
	color_list=["blue", "red"],
	alpha=[0.0, 0.25, 0.5, 0.75, 1.0],
	n_reps=5,
	top_k=5,
	debug_mode=False,
):

	# ── Load network ──────────────────────────────────────────────────────────
	with open(f"{obj_path}/{network}.nx", "rb") as f:
		net_full = pickle.load(f)

	print(f"Network '{network}' loaded.  N={net_full.number_of_nodes()}, M={net_full.number_of_edges()}")

	# ── Extract largest connected component ───────────────────────────────────
	lcc_nodes = max(nx.connected_components(net_full), key=len)
	net       = net_full.subgraph(lcc_nodes).copy()
	n_dropped = net_full.number_of_nodes() - net.number_of_nodes()
	print(f"LCC: N={net.number_of_nodes()}, M={net.number_of_edges()} "
		  f"({n_dropped} nodes dropped from smaller components)")

	nodes = list(net.nodes())

	# ── Colour distribution ───────────────────────────────────────────────────
	colors     = nx.get_node_attributes(net, "color")
	color_dist = {c: 0 for c in color_list}
	for n_id in nodes:
		color_dist[colors[n_id]] += 1
	print("Color distribution:", color_dist)

	# ── Shared seeds: drawn once, used identically by all conditions ──────────
	rng   = random.Random(42)
	seeds = [rng.randint(0, 2**31) for _ in range(n_reps)]
	print(f"Seeds (n={n_reps}): {seeds}")

	# ── Result accumulators ───────────────────────────────────────────────────
	summary_rows = []   # one dict per (strategy, alpha) — averaged stats
	best_rows    = []   # one dict per best run per (strategy, alpha, criterion)
	dist_rows    = []   # one dict per community in each best run
	topk_rows    = []   # one dict per top-k community in each best run

	# ═════════════════════════════════════════════════════════════════════════
	#  Louvain baseline — run once (seeds are alpha-independent)
	# ═════════════════════════════════════════════════════════════════════════
	print(f"\n{'='*50}\nSTRATEGY = louvain (baseline)\n{'='*50}")

	louv_partitions = []
	louv_mod_runs, louv_fbal_runs, louv_fexp_runs = [], [], []
	louv_ncomms_runs = []
	louv_sz_mean_r, louv_sz_std_r, louv_sz_min_r, louv_sz_max_r = [], [], [], []
	louv_time_runs  = []
	louv_rep_scores = []   # (mod, fbal, fexp, fbal_dist, partition)

	for i, seed in enumerate(seeds):
		print(f"  rep {i+1}/{n_reps}", end="\r")
		t0       = time.time()
		res_louv = nx.community.louvain_communities(net, seed=seed)
		t1       = time.time()

		mod, F_bal, F_exp, fbal_dist = score_partition(net, res_louv, color_dist)
		louv_mod_runs.append(mod)
		louv_fbal_runs.append(F_bal)
		louv_fexp_runs.append(F_exp)
		louv_ncomms_runs.append(len(res_louv))
		sm, ss, smin, smax = community_size_stats(res_louv)
		louv_sz_mean_r.append(sm); louv_sz_std_r.append(ss)
		louv_sz_min_r.append(smin); louv_sz_max_r.append(smax)
		louv_time_runs.append(t1 - t0)

		louv_partitions.append(res_louv)
		louv_rep_scores.append((mod, F_bal, F_exp, fbal_dist, res_louv))

	# Best Louvain runs
	best_louv_mod_idx  = max(range(n_reps), key=lambda i: louv_rep_scores[i][0])
	best_louv_fexp_idx = max(range(n_reps), key=lambda i: louv_rep_scores[i][2])

	for criterion, idx in _best_indices(best_louv_mod_idx, best_louv_fexp_idx):
		mod, F_bal, F_exp, fbal_dist, part = louv_rep_scores[idx]
		best_rows.append(_best_run_row("louvain", 1.0, criterion, idx, seeds[idx],
									   mod, F_bal, F_exp, len(part)))
		dist_rows += community_dist_rows(part, fbal_dist, "louvain", 1.0,
										 criterion, colors, color_list)
		topk_rows += top_k_comm_rows(part, fbal_dist, "louvain", 1.0,
									 criterion, colors, color_list, top_k)

	# AMI of Louvain with itself = 1.0 by definition
	summary_rows.append(_summary_row(
		"louvain", 1.0,
		louv_mod_runs, louv_fbal_runs, louv_fexp_runs,
		[1.0] * n_reps,
		louv_ncomms_runs,
		louv_sz_mean_r, louv_sz_std_r, louv_sz_min_r, louv_sz_max_r,
		louv_time_runs,
	))

	# ═════════════════════════════════════════════════════════════════════════
	#  Hybrid: sweep alpha, pair each rep with the Louvain run of the same seed
	# ═════════════════════════════════════════════════════════════════════════
	for a in alpha:
		print(f"\n{'='*50}\nalpha = {a}\n{'='*50}")

		hyb_mod_runs, hyb_fbal_runs, hyb_fexp_runs, hyb_ami_runs = [], [], [], []
		hyb_ncomms_runs = []
		hyb_sz_mean_r, hyb_sz_std_r, hyb_sz_min_r, hyb_sz_max_r = [], [], [], []
		hyb_time_runs  = []
		hyb_rep_scores = []

		for i, seed in enumerate(seeds):
			print(f"  rep {i+1}/{n_reps}", end="\r")
			t0      = time.time()
			res_hyb = fair_louvain_communities(
				net, color_list=color_list, alpha=a, strategy="hybrid", seed=seed
			)
			t1      = time.time()

			mod, F_bal, F_exp, fbal_dist = score_partition(net, res_hyb, color_dist)
			hyb_mod_runs.append(mod)
			hyb_fbal_runs.append(F_bal)
			hyb_fexp_runs.append(F_exp)
			hyb_ncomms_runs.append(len(res_hyb))
			sm, ss, smin, smax = community_size_stats(res_hyb)
			hyb_sz_mean_r.append(sm); hyb_sz_std_r.append(ss)
			hyb_sz_min_r.append(smin); hyb_sz_max_r.append(smax)
			hyb_time_runs.append(t1 - t0)

			# AMI paired with the Louvain run of the same seed
			labels_louv = partition_to_labels(louv_partitions[i], nodes)
			labels_hyb  = partition_to_labels(res_hyb, nodes)
			hyb_ami_runs.append(adjusted_mutual_info_score(labels_louv, labels_hyb))

			hyb_rep_scores.append((mod, F_bal, F_exp, fbal_dist, res_hyb))

		# Best hybrid runs for this alpha
		best_hyb_mod_idx  = max(range(n_reps), key=lambda i: hyb_rep_scores[i][0])
		best_hyb_fexp_idx = max(range(n_reps), key=lambda i: hyb_rep_scores[i][2])

		for criterion, idx in _best_indices(best_hyb_mod_idx, best_hyb_fexp_idx):
			mod, F_bal, F_exp, fbal_dist, part = hyb_rep_scores[idx]
			best_rows.append(_best_run_row("hybrid", a, criterion, idx, seeds[idx],
										   mod, F_bal, F_exp, len(part)))
			dist_rows += community_dist_rows(part, fbal_dist, "hybrid", a,
											 criterion, colors, color_list)
			topk_rows += top_k_comm_rows(part, fbal_dist, "hybrid", a,
										 criterion, colors, color_list, top_k)

		summary_rows.append(_summary_row(
			"hybrid", a,
			hyb_mod_runs, hyb_fbal_runs, hyb_fexp_runs, hyb_ami_runs,
			hyb_ncomms_runs,
			hyb_sz_mean_r, hyb_sz_std_r, hyb_sz_min_r, hyb_sz_max_r,
			hyb_time_runs,
		))

	# ── Assemble and output ───────────────────────────────────────────────────
	df_summary = pd.DataFrame(summary_rows)
	df_best    = pd.DataFrame(best_rows)
	df_dist    = pd.DataFrame(dist_rows)
	df_topk    = pd.DataFrame(topk_rows)

	display_cols = [
		"strategy", "alpha",
		"modularity", "fair_bal", "fair_exp",
		"ami_louvain", "ncomms", "size_mean", "size_max",
	]

	if debug_mode:
		print("\n── Summary ──")
		print(df_summary[display_cols].to_string(index=False))
		print("\n── Best runs ──")
		print(df_best.to_string(index=False))
		print(f"\n── Top-{top_k} communities (best runs) ──")
		print(df_topk.to_string(index=False))
		print("\n── Community distribution (first 20 rows) ──")
		print(df_dist.head(20).to_string(index=False))
	else:
		df_summary.to_csv(f"{log_path}/{network}_city_ab.csv",       index=False)
		df_best.to_csv(   f"{log_path}/{network}_best_runs.csv",      index=False)
		df_dist.to_csv(   f"{log_path}/{network}_community_dist.csv", index=False)
		df_topk.to_csv(   f"{log_path}/{network}_top{top_k}_comms.csv", index=False)
		print(f"\nSaved:")
		print(f"  {log_path}/{network}_city_ab.csv")
		print(f"  {log_path}/{network}_best_runs.csv")
		print(f"  {log_path}/{network}_community_dist.csv")
		print(f"  {log_path}/{network}_top{top_k}_comms.csv")
		print(df_summary[display_cols].to_string(index=False))


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
	args = sys.argv[1:]

	if len(args) < 1:
		print("Usage: python fair_cd_city.py <network> [color_list] [alpha] [n_reps] [top_k] [debug]")
		return

	network    = args[0]
	color_list = args[1].split(",")                      if len(args) > 1 else ["blue", "red"]
	alpha      = [float(a) for a in args[2].split(",")]  if len(args) > 2 else [0.0, 0.25, 0.5, 0.75, 1.0]
	n_reps     = int(args[3])                            if len(args) > 3 else 5
	top_k      = int(args[4])                            if len(args) > 4 and args[4] != "debug" else 5
	debug_mode = args[-1] == "debug"

	experiment(
		network,
		color_list=color_list,
		alpha=alpha,
		n_reps=n_reps,
		top_k=top_k,
		debug_mode=debug_mode,
	)


if __name__ == "__main__":
	main()