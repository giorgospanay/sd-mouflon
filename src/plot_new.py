import glob
import gc
import re
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.colors as mcolors
import seaborn as sns
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.ticker import LogFormatterSciNotation, LogLocator

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
obj_path      = "../data/obj"
log_path      = "../logs"
path_1e03x10  = "../logs/1e-03x10"
path_10000x10 = "../logs/10000x10"
plot_path     = "../plots/"

# ---------------------------------------------------------------------------
# Global figure aesthetics (single-column, 300 dpi)
# ---------------------------------------------------------------------------
plt.rcParams.update({
	"font.size":        9,
	"axes.titlesize":   9,
	"axes.labelsize":   9,
	"xtick.labelsize":  8,
	"ytick.labelsize":  8,
	"legend.fontsize":  7,
	"lines.linewidth":  1.5,
	"lines.markersize": 5,
	"errorbar.capsize": 2,
	"figure.dpi":       300,
})

COL_W = 3.5   # single-column width in inches
COL_H = 3.0   # panel height in inches

# ---------------------------------------------------------------------------
# Style catalogue
# ---------------------------------------------------------------------------
STYLE = {
	"modularity":      {"color": "tab:blue",   "linestyle": "-",  "marker": "x",  "label": "Modularity"},
	"balance_fm":      {"color": "tab:red",    "linestyle": ":",  "marker": "o",  "label": "Fair-mod (balance)"},
	"prop_fm":         {"color": "tab:green",  "linestyle": ":",  "marker": "^",  "label": "Fair-mod (prop_balance)"},
	"balance_mouflon": {"color": "tab:red",    "linestyle": "--", "marker": "o",  "label": "MOUFLON (balance)"},
	"prop_mouflon":    {"color": "tab:green",  "linestyle": "--", "marker": "^",  "label": "MOUFLON (prop_balance)"},
	"louvain":         {"color": "tab:orange", "linestyle": ":",  "marker": "D",  "label": "Louvain"},
	"ncomms":          {"color": "tab:purple", "linestyle": "-",  "marker": "X",  "label": "Number of communities"},
	"ami":             {"color": "tab:olive",  "linestyle": "-",  "marker": "s",  "label": "AMI"},
	"nf1":             {"color": "tab:cyan",   "linestyle": "-",  "marker": "P",  "label": "NF1"},
}

# Fairness column per MOUFLON strategy
strategy_config = {
	"step2":  {"fairness": "fair_bal", "style": STYLE["balance_mouflon"]},
	"hybrid": {"fairness": "fair_exp", "style": STYLE["prop_mouflon"]},
}

# Alpha shade mapping: 0 → lightest, 1 → darkest
ALPHA_SHADES = {0.0: 0.30, 0.5: 0.65, 1.0: 1.0}


def _alpha_style(base_style: dict, alpha_val: float) -> dict:
	"""Return copy of base_style with colour blended toward white by alpha_val."""
	shade    = ALPHA_SHADES.get(float(alpha_val), 1.0)
	base_rgb = mcolors.to_rgb(base_style["color"])
	blended  = tuple(1 - shade * (1 - c) for c in base_rgb)
	s        = base_style.copy()
	s["color"] = blended
	s["label"] = f"{base_style['label']} (a={alpha_val})"
	return s


def _mu_color(base_color: str, mu_val: float, mu_vals_sorted: list):
	"""Darker colour for higher mu."""
	idx      = mu_vals_sorted.index(mu_val)
	frac     = (idx + 1) / len(mu_vals_sorted)
	base_rgb = mcolors.to_rgb(base_color)
	return tuple(1 - frac * (1 - c) for c in base_rgb)


# ---------------------------------------------------------------------------
# Helpers: ncomms y-limits
# ---------------------------------------------------------------------------
def get_ncomms_limits(dfs, strategies=None):
	ymin, ymax = float("inf"), float("-inf")
	for df in dfs:
		sub = df if strategies is None else df[df["strategy"].isin(strategies)]
		if "ncomms" in sub.columns:
			std_col = sub["ncomms_std"] if "ncomms_std" in sub.columns else pd.Series(0, index=sub.index)
			ymin = min(ymin, (sub["ncomms"] - std_col).min())
			ymax = max(ymax, (sub["ncomms"] + std_col).max())
	if ymin == float("inf"):
		return 0.9, 1.1
	mean_val  = (ymax + ymin) / 2
	variation = (ymax - ymin) / max(mean_val, 1)
	if variation < 0.05:
		pad = max(mean_val * 0.1, 1)
		return max(mean_val - pad, 1e-6), mean_val + pad
	pad = max((ymax - ymin) * 0.05, 0.5)
	return max(ymin - pad, 1e-6), ymax + pad


INSET_LIMITS = {
	"facebook":  (1, 2700),
	"deezer":    (1, 20000),
	"twitch":    (1, 110200),
	"pokec-a":   (1, 891000),
	"pokec-g":   (1, 1070500),
}


def _with_padding(lo, hi, frac=0.05):
	return max(lo, 1), hi * (1 + frac)


# ---------------------------------------------------------------------------
# Shared helpers for 2-row quality figures
# ---------------------------------------------------------------------------
def _has_meaningful(df, col):
	"""True if column exists, is not all-NaN, and is not all-zero."""
	return (col in df.columns
			and not df[col].isna().all()
			and (df[col].fillna(0) != 0).any())



def _padded_01_lim(frac=0.05):
	"""Return y-limits for a [0,1] axis with symmetric padding."""
	return -frac, 1 + frac

def _draw_metrics_row(ax, df_strat, draw_error, x_col,
					  inset_lo, inset_hi, use_log=False,
					  ami_col="ami", ami_std_col="ami_std",
					  nf1_col="nf1", nf1_std_col="nf1_std"):
	"""
	Fill a second-row subpanel with ncomms (left y-axis, log-optional)
	and AMI + NF1 (right y-axis, 0-1) if they have meaningful values.
	ncomms uses its own left scale; AMI/NF1 share the right scale.
	"""
	has_ami = _has_meaningful(df_strat, ami_col)
	has_nf1 = _has_meaningful(df_strat, nf1_col)

	# ncomms on primary axis
	ax.plot(df_strat[x_col], df_strat["ncomms"], **STYLE["ncomms"])
	if draw_error and "ncomms_std" in df_strat.columns:
		ax.errorbar(df_strat[x_col], df_strat["ncomms"],
					yerr=df_strat["ncomms_std"], fmt="none",
					ecolor=STYLE["ncomms"]["color"], capsize=2)
	# Data-driven [1, max_ncomms + padding]
	nc_max = (df_strat["ncomms"] + df_strat.get(
		"ncomms_std", pd.Series(0, index=df_strat.index))).max()
	nc_pad = max(nc_max * 0.08, 0.5)
	ax.margins(x=0.05)
	ax.set_ylim(-nc_pad, nc_max + nc_pad)
	ax.autoscale(enable=False, axis="y")
	if use_log:
		ax.set_yscale("log")
	ax.tick_params(axis="y")
	ax.set_ylabel("Number of communities")

	# AMI + NF1 on twin right axis
	if has_ami or has_nf1:
		ax2 = ax.twinx()
		ax2.set_ylim(*_padded_01_lim())
		# ticks every 0.1 on right axis, aligned with left-axis ticks every 1
		ax2.set_yticks([round(v * 0.1, 1) for v in range(0, 11)])
		ax2.set_ylabel("AMI / NF1")
		ax2.tick_params(axis="y")
		# gridlines come from the left axis only; suppress them on the twin
		ax2.set_axisbelow(True)
		ax2.grid(False)
		if has_ami:
			ax2.plot(df_strat[x_col], df_strat[ami_col], **STYLE["ami"])
			if draw_error and ami_std_col in df_strat.columns:
				ax2.errorbar(df_strat[x_col], df_strat[ami_col],
							 yerr=df_strat[ami_std_col], fmt="none",
							 ecolor=STYLE["ami"]["color"], capsize=2)
		if has_nf1:
			ax2.plot(df_strat[x_col], df_strat[nf1_col], **STYLE["nf1"])
			if draw_error and nf1_std_col in df_strat.columns:
				ax2.errorbar(df_strat[x_col], df_strat[nf1_col],
							 yerr=df_strat[nf1_std_col], fmt="none",
							 ecolor=STYLE["nf1"]["color"], capsize=2)


# ===========================================================================
# SCALABILITY — combined ER + LFR  (Figures 3 & 4 merged into one 2x2)
# ===========================================================================

def create_time_df(filename, nodes, density_prob, path):
	df = pd.read_csv(f"{path}/{filename}.csv", header=0)
	df["nodes"]   = int(nodes)
	df["density"] = density_prob
	return df


def _draw_scalability_panel(ax, df, x_col, xlabel, panel_label,
							row_label=None, hide_xlabel=False, xscale="log"):
	"""
	Draw a single scalability panel: one line per (strategy, alpha),
	plus Louvain if present. Shared helper used by the combined figure.
	"""
	alpha_vals = sorted(df["alpha"].unique())
	mouflon_strategies = [s for s in ["step2", "hybrid"]
						  if s in df["strategy"].unique()]

	for strategy in mouflon_strategies:
		base_style = (STYLE["balance_mouflon"] if strategy == "step2"
					  else STYLE["prop_mouflon"])
		for a in alpha_vals:
			df_s = (df[(df["strategy"] == strategy) & (df["alpha"] == a)]
					.sort_values(x_col))
			if df_s.empty:
				continue
			s = _alpha_style(base_style, a)
			ax.plot(df_s[x_col], df_s["time"],
					color=s["color"], linestyle=base_style["linestyle"],
					marker=base_style["marker"])
			if "time_std" in df_s.columns:
				ax.errorbar(df_s[x_col], df_s["time"], yerr=df_s["time_std"],
							fmt="none", ecolor=s["color"], capsize=2)

	if "louvain" in df["strategy"].unique():
		df_l = (df[df["strategy"] == "louvain"]
				.groupby(x_col, as_index=False)["time"].mean()
				.sort_values(x_col))
		ax.plot(df_l[x_col], df_l["time"], **STYLE["louvain"])
		if "time_std" in df.columns:
			df_ls = (df[df["strategy"] == "louvain"]
					 .groupby(x_col, as_index=False)["time_std"].mean()
					 .sort_values(x_col))
			ax.errorbar(df_l[x_col], df_l["time"], yerr=df_ls["time_std"],
						fmt="none", ecolor=STYLE["louvain"]["color"], capsize=2)

	if not hide_xlabel:
		ax.set_xlabel(xlabel)
	ax.set_xscale(xscale)
	ax.set_yscale("log")
	ax.set_xticks(sorted(df[x_col].unique()))
	if xscale == "log":
		ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
	ax.tick_params(axis="x", rotation=45)
	if hide_xlabel:
		ax.set_xticklabels([])
	ax.margins(x=0.05)
	ax.text(0.02, 0.97, panel_label, transform=ax.transAxes,
			fontsize=10, fontweight="bold", va="top", ha="left")
	if row_label:
		ax.set_ylabel(row_label)


def plot_scalability_combined(df_er_size=None, df_er_density=None,
							  filename="Figure3"):
	"""
	2x2 scalability figure combining ER (top row) and LFR (bottom row).

	Top row   (ER):  left = size (nodes),       right = density (edge prob)
	Bottom row (LFR): left = size (nodes, LFR), right = avg degree

	Y-axis (execution time) is shared within each row.
	A single legend is shown at the top.

	df_er_size / df_er_density are passed in from main(); LFR data is loaded
	directly from the standard CSV files.
	"""
	# Load LFR data
	df_lfr_size    = pd.read_csv(f"{log_path}/LFR_scalability_size.csv")
	df_lfr_density = pd.read_csv(f"{log_path}/LFR_scalability_density.csv")

	sns.set_style("whitegrid")
	fig, axs = plt.subplots(2, 2,
							figsize=(2 * COL_W, 2 * COL_H * 0.85),
							sharey="row")

	panel_labels = [["A", "B"], ["C", "D"]]

	# ── Top row: ER ───────────────────────────────────────────────────────────
	if df_er_size is not None:
		_draw_scalability_panel(
			axs[0, 0], df_er_size, "nodes",
			"Number of nodes", panel_labels[0][0],
			row_label="ER\nTime (s)")
	if df_er_density is not None:
		_draw_scalability_panel(
			axs[0, 1], df_er_density, "density",
			"Edge probability", panel_labels[0][1], xscale="linear")

	# ── Bottom row: LFR ───────────────────────────────────────────────────────
	_draw_scalability_panel(
		axs[1, 0], df_lfr_size, "lfr_n",
		"Number of nodes", panel_labels[1][0],
		row_label="LFR\nTime (s)")
	_draw_scalability_panel(
		axs[1, 1], df_lfr_density, "lfr_avg_degree",
		"Average degree", panel_labels[1][1])

	# ── Legend — built from ER size data (or LFR if ER not available) ─────────
	ref_df     = df_er_size if df_er_size is not None else df_lfr_size
	alpha_vals = sorted(ref_df["alpha"].unique())
	handles    = []
	for strategy in [s for s in ["step2", "hybrid"]
					 if s in ref_df["strategy"].unique()]:
		base_style = (STYLE["balance_mouflon"] if strategy == "step2"
					  else STYLE["prop_mouflon"])
		for a in alpha_vals:
			s = _alpha_style(base_style, a)
			handles.append(mlines.Line2D([], [],
				color=s["color"], linestyle=base_style["linestyle"],
				marker=base_style["marker"], label=s["label"]))
	if "louvain" in ref_df["strategy"].unique():
		handles.append(mlines.Line2D([], [], **STYLE["louvain"]))

	fig.legend(handles=handles, loc="upper center",
			   ncol=min(len(handles), 4), bbox_to_anchor=(0.5, 1.04))
	fig.tight_layout(rect=[0, 0, 1, 0.88])
	fig.savefig(f"{plot_path}{filename}.png", dpi=300, bbox_inches="tight")
	plt.close(fig)


# Keep the old individual functions as thin wrappers in case they are
# called elsewhere, but they now just delegate to the combined figure.
def plot_scalability_er(df_size=None, df_density=None, filename="Figure3_ER"):
	plot_scalability_combined(df_er_size=df_size, df_er_density=df_density,
							  filename=filename)

def plot_scalability_lfr(filename="Figure3b_LFR"):
	plot_scalability_combined(filename=filename)

# ===========================================================================
# QUALITY — ER synthetic  (Figures 4-7)
# 2-row layout: row 1 = modularity + fairness, row 2 = ncomms + AMI + NF1
# ===========================================================================

def extract_p_sensitive(filename):
	match = re.search(r'_c(\d+)\.csv$', filename)
	return int(match.group(1)) / 10.0 if match else None


def _quality_2row(fig, axs_top, axs_bot, dfs, x_col,
				  top_metrics_fn, inset_lo, inset_hi,
				  draw_error, use_log=False,
				  ami_col="ami", ami_std_col="ami_std",
				  nf1_col="nf1", nf1_std_col="nf1_std",
				  panel_labels=None,
				  nc_ylim=None, ami_ylim=None,
				  xlabel=None):
	"""
	Generic 2-row filler.
	axs_top / axs_bot : 1-D arrays of axes, same length as dfs.
	top_metrics_fn(ax, df) : draws the main panel (modularity + fairness).
	nc_ylim  : optional (lo, hi) to override ncomms left-axis limits on all bot panels.
	ami_ylim : optional (lo, hi) to override AMI/NF1 right-axis limits on all bot panels.
	"""
	if panel_labels is None:
		import string
		panel_labels = list(string.ascii_uppercase)

	twin_axes = []   # collect right axes so we can share limits after drawing

	for i, (df, ax_top, ax_bot) in enumerate(zip(dfs, axs_top, axs_bot)):
		top_metrics_fn(ax_top, df)
		x_ticks = sorted(df[x_col].unique())
		ax_top.set_xticks(x_ticks)
		ax_top.margins(x=0.05)
		ax_top.set_ylim(*_padded_01_lim())
		ax_top.autoscale(enable=False, axis="y")
		ax_top.set_xticklabels([])          # x labels only on bottom row
		ax_top.text(0.02, 0.97, panel_labels[i],
					transform=ax_top.transAxes,
					fontsize=10, fontweight="bold", va="top", ha="left")

		_draw_metrics_row(ax_bot, df, draw_error, x_col,
						  inset_lo, inset_hi, use_log=use_log,
						  ami_col=ami_col, ami_std_col=ami_std_col,
						  nf1_col=nf1_col, nf1_std_col=nf1_std_col)
		ax_bot.set_xticks(x_ticks)
		ax_bot.set_xticklabels([f"{v:g}" for v in x_ticks])
		ax_bot.tick_params(axis="x", labelbottom=True)
		# Explicitly un-hide tick labels that sharex="col" may suppress
		for lbl in ax_bot.get_xticklabels():
			lbl.set_visible(True)
		if xlabel is not None:
			ax_bot.set_xlabel(xlabel)
		ax_bot.margins(x=0.05)

		# Override ncomms left-axis limits if requested
		if nc_ylim is not None:
			ax_bot.set_ylim(*nc_ylim)

		# Collect the twin right axis created by _draw_metrics_row (if any)
		twins = [c for c in ax_bot.get_shared_x_axes().get_siblings(ax_bot)
				 if c is not ax_bot]
		# twinx axes are not in shared_x siblings — find via figure children
		for child_ax in fig.axes:
			if (child_ax is not ax_bot
					and hasattr(child_ax, "_twinned_axes")
					and ax_bot in child_ax._twinned_axes.get_siblings(child_ax)):
				twin_axes.append(child_ax)

	# Apply shared AMI/NF1 right-axis limits
	target_lim = ami_ylim if ami_ylim is not None else _padded_01_lim()
	for tax in fig.axes:
		# A twinx axis shares the x-axis with one of the bot panels
		if tax in axs_bot.tolist():
			continue
		if any(tax.get_shared_x_axes().joined(tax, ab) for ab in axs_bot):
			tax.set_ylim(*target_lim)


def plot_mouflon_alpha(net_node, net_full, draw_error=True, filename="Figure4"):
	"""Left = node-coloured (hybrid), right = comm-coloured (hybrid)."""
	df1 = pd.read_csv(f"{log_path}/{net_node}.csv")
	df2 = pd.read_csv(f"{log_path}/{net_full}.csv")
	df1 = df1[df1["strategy"] == "hybrid"]
	df2 = df2[df2["strategy"] == "hybrid"]

	sns.set_style("whitegrid")
	fig, axs = plt.subplots(2, 2,
							figsize=(2 * COL_W, 2 * COL_H),
							sharex="col", sharey="row")
	axs_top = axs[0]
	axs_bot = axs[1]

	inset_lo, inset_hi = get_ncomms_limits([df1, df2])

	def _top(ax, df):
		ax.plot(df["alpha"], df["modularity"], **STYLE["modularity"])
		if draw_error:
			ax.errorbar(df["alpha"], df["modularity"], yerr=df["modularity_std"],
						fmt="none", ecolor=STYLE["modularity"]["color"], capsize=2)
		ax.plot(df["alpha"], df["fair_exp"], **STYLE["prop_mouflon"])
		if draw_error:
			ax.errorbar(df["alpha"], df["fair_exp"], yerr=df["fair_exp_std"],
						fmt="none", ecolor=STYLE["prop_mouflon"]["color"], capsize=2)
		ax.set_ylabel("Score")

	_quality_2row(fig, axs_top, axs_bot, [df1, df2],
				  "alpha", _top, inset_lo, inset_hi, draw_error,
				  nc_ylim=None,
				  ami_ylim=_padded_01_lim(),
				  xlabel="alpha",
				  panel_labels=["A", "B"])
	handles = [
		mlines.Line2D([], [], **STYLE["modularity"]),
		mlines.Line2D([], [], **STYLE["prop_mouflon"]),
		mlines.Line2D([], [], **STYLE["ncomms"]),
		mlines.Line2D([], [], **STYLE["ami"]),
		mlines.Line2D([], [], **STYLE["nf1"]),
	]
	fig.legend(handles=handles, loc="upper center",
			   ncol=5, bbox_to_anchor=(0.5, 1.02))
	fig.tight_layout(rect=[0, 0, 1, 0.95])
	fig.savefig(f"{plot_path}{filename}.png", dpi=300, bbox_inches="tight")
	plt.close(fig)

def load_and_prepare(files):
	rows = []
	for f in files:
		df  = pd.read_csv(f)
		p   = extract_p_sensitive(f)
		df  = df[(df["strategy"] == "hybrid") & (df["alpha"] == 0.5)].copy()
		df["p_sensitive"] = p
		rows.append(df)
	return pd.concat(rows, ignore_index=True).sort_values("p_sensitive")


def plot_mouflon_psensitive(file_list1, file_list2, draw_error=True, filename="Figure5"):
	"""Left = node-coloured (hybrid), right = comm-coloured (hybrid)."""
	df1 = load_and_prepare(file_list1)
	df2 = load_and_prepare(file_list2)

	sns.set_style("whitegrid")
	fig, axs = plt.subplots(2, 2,
							figsize=(2 * COL_W, 2 * COL_H),
							sharex="col", sharey="row")
	axs_top = axs[0]
	axs_bot = axs[1]

	inset_lo, inset_hi = get_ncomms_limits([df1, df2])

	metric_styles = {
		"modularity": STYLE["modularity"],
		"fair_exp":   {**STYLE["prop_mouflon"], "label": "prop_balance"},
		"fair_bal":   {**STYLE["balance_mouflon"], "label": "balance", "alpha": 0.5},
	}

	def _top(ax, df):
		for metric, s in metric_styles.items():
			sc = s.copy()
			ax.plot(df["p_sensitive"], df[metric], **sc)
			if draw_error:
				ax.errorbar(df["p_sensitive"], df[metric],
							yerr=df[f"{metric}_std"], fmt="none",
							ecolor=sc.get("color", "black"), capsize=2,
							alpha=sc.get("alpha", 1.0))
		ax.set_ylabel("Score")

	_quality_2row(fig, axs_top, axs_bot, [df1, df2],
				  "p_sensitive", _top, inset_lo, inset_hi, draw_error,
				  xlabel="p_sensitive",
				  panel_labels=["A", "B"])

	## added
	for ax in axs_bot:
		ax.set_ylim(-0.5, 10.5)

	handles = [
		mlines.Line2D([], [], **STYLE["modularity"]),
		mlines.Line2D([], [], color=STYLE["prop_mouflon"]["color"],
					  linestyle=STYLE["prop_mouflon"]["linestyle"],
					  marker=STYLE["prop_mouflon"]["marker"], label="prop_balance"),
		mlines.Line2D([], [], color=STYLE["balance_mouflon"]["color"],
					  linestyle=STYLE["balance_mouflon"]["linestyle"],
					  marker=STYLE["balance_mouflon"]["marker"],
					  label="balance", alpha=0.5),
		mlines.Line2D([], [], **STYLE["ncomms"]),
		mlines.Line2D([], [], **STYLE["ami"]),
		mlines.Line2D([], [], **STYLE["nf1"]),
	]
	fig.legend(handles=handles, loc="upper center",
			   ncol=3, bbox_to_anchor=(0.5, 1.02))
	fig.tight_layout(rect=[0, 0, 1, 0.95])
	fig.savefig(f"{plot_path}{filename}.png", dpi=300, bbox_inches="tight")
	plt.close(fig)


def plot_strategies_alpha(net_node, draw_error=True, filename="Figure6"):
	df = pd.read_csv(f"{log_path}/{net_node}.csv")
	sns.set_style("whitegrid")

	fig, axs = plt.subplots(2, 2,
							figsize=(2 * COL_W, 2 * COL_H),
							sharex="col")
	axs_top = axs[0]
	axs_bot = axs[1]

	inset_lo, inset_hi = get_ncomms_limits([df], strategies=["step2", "hybrid"])

	strategies = list(strategy_config.keys())

	def _top(ax, df_strat):
		strategy = df_strat["strategy"].iloc[0]
		cfg   = strategy_config[strategy]
		style = cfg["style"]
		fair_col = cfg["fairness"]
		ax.plot(df_strat["alpha"], df_strat["modularity"], **STYLE["modularity"])
		if draw_error:
			ax.errorbar(df_strat["alpha"], df_strat["modularity"],
						yerr=df_strat["modularity_std"], fmt="none",
						ecolor=STYLE["modularity"]["color"], capsize=2)
		ax.plot(df_strat["alpha"], df_strat[fair_col], **style)
		if draw_error:
			ax.errorbar(df_strat["alpha"], df_strat[fair_col],
						yerr=df_strat[f"{fair_col}_std"], fmt="none",
						ecolor=style["color"], capsize=2)
		ax.set_title(strategy, fontsize=9)
		ax.set_ylabel("Score")

	dfs = [df[df["strategy"] == s] for s in strategies]
	_quality_2row(fig, axs_top, axs_bot, dfs,
				  "alpha", _top, inset_lo, inset_hi, draw_error,
				  xlabel="alpha",
				  nc_ylim=(-0.8, 10.8))  # hardcoded 0-10 with padding
	handles = [
		mlines.Line2D([], [], **STYLE["modularity"]),
		mlines.Line2D([], [], **STYLE["balance_mouflon"]),
		mlines.Line2D([], [], **STYLE["prop_mouflon"]),
		mlines.Line2D([], [], **STYLE["ncomms"]),
		mlines.Line2D([], [], **STYLE["ami"]),
		mlines.Line2D([], [], **STYLE["nf1"]),
	]
	fig.legend(handles=handles, loc="upper center",
			   ncol=3, bbox_to_anchor=(0.5, 1.02))
	fig.tight_layout(rect=[0, 0, 1, 0.95])
	fig.savefig(f"{plot_path}{filename}.png", dpi=300, bbox_inches="tight")
	plt.close(fig)


def load_psensitive_dfs(files):
	rows = []
	for f in files:
		df = pd.read_csv(f)
		p  = extract_p_sensitive(f)
		df = df[df["alpha"] == 0.5].copy()
		df["p_sensitive"] = p
		rows.append(df)
	return pd.concat(rows, ignore_index=True).sort_values("p_sensitive")


def plot_strategies_psensitive(files, draw_error=True, filename="Figure7"):
	df = load_psensitive_dfs(files)
	sns.set_style("whitegrid")

	fig, axs = plt.subplots(2, 2,
							figsize=(2 * COL_W, 2 * COL_H),
							sharex="col")
	axs_top = axs[0]
	axs_bot = axs[1]

	inset_lo, inset_hi = get_ncomms_limits([df], strategies=["step2", "hybrid"])
	strategies = list(strategy_config.keys())

	def _top(ax, df_strat):
		strategy = df_strat["strategy"].iloc[0]
		cfg   = strategy_config[strategy]
		style = cfg["style"]
		fair_col = cfg["fairness"]
		ax.plot(df_strat["p_sensitive"], df_strat["modularity"], **STYLE["modularity"])
		if draw_error:
			ax.errorbar(df_strat["p_sensitive"], df_strat["modularity"],
						yerr=df_strat["modularity_std"], fmt="none",
						ecolor=STYLE["modularity"]["color"], capsize=2)
		ax.plot(df_strat["p_sensitive"], df_strat[fair_col], **style)
		if draw_error:
			ax.errorbar(df_strat["p_sensitive"], df_strat[fair_col],
						yerr=df_strat[f"{fair_col}_std"], fmt="none",
						ecolor=style["color"], capsize=2)
		ax.set_title(strategy, fontsize=9)
		ax.set_ylabel("Score")

	dfs = [df[df["strategy"] == s] for s in strategies]
	_quality_2row(fig, axs_top, axs_bot, dfs,
				  "p_sensitive", _top, inset_lo, inset_hi, draw_error,
				  xlabel="p_sensitive")
	handles = [
		mlines.Line2D([], [], **STYLE["balance_mouflon"]),
		mlines.Line2D([], [], **STYLE["prop_mouflon"]),
		mlines.Line2D([], [], **STYLE["ncomms"]),
		mlines.Line2D([], [], **STYLE["ami"]),
		mlines.Line2D([], [], **STYLE["nf1"]),
	]
	fig.legend(handles=handles, loc="upper center",
			   ncol=3, bbox_to_anchor=(0.5, 1.02))
	fig.tight_layout(rect=[0, 0, 1, 0.95])
	fig.savefig(f"{plot_path}{filename}.png", dpi=300, bbox_inches="tight")
	plt.close(fig)

# ===========================================================================
# STRATEGIES — step2 balance only, alpha + p_sensitive combined (Figures 8+9)
# ===========================================================================

def plot_step2_combined(net_node, node_files, draw_error=True, filename="Figure8"):
	"""
	2x2 figure showing step2 (balance) only.
	Left column  : alpha sweep    (x = alpha,       fixed p_sensitive network c05)
	Right column : p_sensitive sweep (x = p_sensitive, fixed alpha = 0.5)
	Row 1: modularity + balance fairness.
	Row 2: number of communities + AMI/NF1.
	No prop_balance lines.
	"""
	# Alpha sweep data
	df_alpha = pd.read_csv(f"{log_path}/{net_node}.csv")
	df_alpha = df_alpha[df_alpha["strategy"] == "step2"]

	# p_sensitive sweep data
	rows = []
	for f in node_files:
		df = pd.read_csv(f)
		p  = extract_p_sensitive(f)
		df = df[(df["strategy"] == "step2") & (df["alpha"] == 0.5)].copy()
		df["p_sensitive"] = p
		rows.append(df)
	df_psens = pd.concat(rows, ignore_index=True).sort_values("p_sensitive")

	sns.set_style("whitegrid")
	fig, axs = plt.subplots(2, 2,
							figsize=(2 * COL_W, 2 * COL_H),
							sharey="row")
	# Top-left: alpha sweep score
	# Top-right: p_sensitive sweep score
	# Bot-left: alpha sweep ncomms+AMI
	# Bot-right: p_sensitive sweep ncomms+AMI

	def _draw_top(ax, df, x_col):
		ax.plot(df[x_col], df["modularity"], **STYLE["modularity"])
		if draw_error and "modularity_std" in df.columns:
			ax.errorbar(df[x_col], df["modularity"], yerr=df["modularity_std"],
						fmt="none", ecolor=STYLE["modularity"]["color"], capsize=2)
		ax.plot(df[x_col], df["fair_bal"], **STYLE["balance_mouflon"])
		if draw_error and "fair_bal_std" in df.columns:
			ax.errorbar(df[x_col], df["fair_bal"], yerr=df["fair_bal_std"],
						fmt="none", ecolor=STYLE["balance_mouflon"]["color"], capsize=2)
		ax.margins(x=0.05)
		ax.set_ylim(*_padded_01_lim())
		ax.autoscale(enable=False, axis="y")
		ax.set_ylabel("Score")

	def _draw_bot(ax, df, x_col, xlabel):
		nc_max = (df["ncomms"] + df.get("ncomms_std",
				  pd.Series(0, index=df.index))).max()
		nc_pad = max(nc_max * 0.08, 0.5)
		ax.plot(df[x_col], df["ncomms"], **STYLE["ncomms"])
		if draw_error and "ncomms_std" in df.columns:
			ax.errorbar(df[x_col], df["ncomms"], yerr=df["ncomms_std"],
						fmt="none", ecolor=STYLE["ncomms"]["color"], capsize=2)
		ax.set_xticks(sorted(df[x_col].unique()))
		ax.set_xticklabels([f"{v:g}" for v in sorted(df[x_col].unique())])
		ax.tick_params(axis="x", labelbottom=True)
		for lbl in ax.get_xticklabels():
			lbl.set_visible(True)
		ax.set_xlabel(xlabel)
		ax.margins(x=0.05)
		ax.set_ylim(-nc_pad, nc_max + nc_pad)
		ax.autoscale(enable=False, axis="y")
		ax.tick_params(axis="y")
		ax.set_ylabel("Number of communities")

		has_ami = _has_meaningful(df, "ami")
		has_nf1 = _has_meaningful(df, "nf1")
		if has_ami or has_nf1:
			ax2 = ax.twinx()
			ax2.set_ylim(*_padded_01_lim())
			ax2.set_yticks([round(v * 0.1, 1) for v in range(0, 11)])
			ax2.set_ylabel("AMI / NF1")
			ax2.tick_params(axis="y")
			ax2.grid(False)
			if has_ami:
				ax2.plot(df[x_col], df["ami"], **STYLE["ami"])
				if draw_error and "ami_std" in df.columns:
					ax2.errorbar(df[x_col], df["ami"], yerr=df["ami_std"],
								 fmt="none", ecolor=STYLE["ami"]["color"], capsize=2)
			if has_nf1:
				ax2.plot(df[x_col], df["nf1"], **STYLE["nf1"])
				if draw_error and "nf1_std" in df.columns:
					ax2.errorbar(df[x_col], df["nf1"], yerr=df["nf1_std"],
								 fmt="none", ecolor=STYLE["nf1"]["color"], capsize=2)

	# Fill panels
	_draw_top(axs[0, 0], df_alpha, "alpha")
	_draw_top(axs[0, 1], df_psens, "p_sensitive")
	_draw_bot(axs[1, 0], df_alpha, "alpha", "alpha")
	_draw_bot(axs[1, 1], df_psens, "p_sensitive", "p_sensitive")

	# Hide x tick labels on top row
	for ax in axs[0]:
		ax.set_xticks(sorted(df_alpha["alpha"].unique()
							 if ax is axs[0, 0] else df_psens["p_sensitive"].unique()))
		ax.set_xticklabels([])

	# Panel labels
	for i, (ax, label) in enumerate(zip(axs[0], ["A", "B"])):
		ax.text(0.02, 0.97, label, transform=ax.transAxes,
				fontsize=10, fontweight="bold", va="top", ha="left")

	has_ami_any = _has_meaningful(df_alpha, "ami") or _has_meaningful(df_psens, "ami")
	has_nf1_any = _has_meaningful(df_alpha, "nf1") or _has_meaningful(df_psens, "nf1")
	handles = [
		mlines.Line2D([], [], **STYLE["modularity"]),
		mlines.Line2D([], [], **STYLE["balance_mouflon"]),
		mlines.Line2D([], [], **STYLE["ncomms"]),
	]
	if has_ami_any:
		handles.append(mlines.Line2D([], [], **STYLE["ami"]))
	if has_nf1_any:
		handles.append(mlines.Line2D([], [], **STYLE["nf1"]))

	fig.legend(handles=handles, loc="upper center",
			   ncol=len(handles), bbox_to_anchor=(0.5, 1.02))
	fig.tight_layout(rect=[0, 0, 1, 0.95])
	fig.savefig(f"{plot_path}{filename}.png", dpi=300, bbox_inches="tight")
	plt.close(fig)


# ===========================================================================
# LFR QUALITY — step2 balance only, alpha + p_sensitive combined (Figure 10)
# ===========================================================================

def plot_lfr_step2_combined(p_sensitive_fixed=0.5, alpha_fixed=0.5,
							 draw_error=True, filename="Figure10"):
	"""
	2x2 figure: step2 (balance) on LFR node-coloured scenario only.
	Left column  : alpha sweep    (x = alpha,          fixed p_sensitive)
	Right column : p_sensitive sweep (x = lfr_p_sensitive, fixed alpha)
	Row 1: modularity + balance fairness, mu as hues.
	Row 2: ncomms (left) + AMI (right), mu as hues.
	"""
	df_raw = pd.read_csv(f"{log_path}/LFR_quality.csv")
	df_raw = df_raw[(df_raw["lfr_scenario"] == "node") &
					(df_raw["lfr_mu"].isin(LFR_MU_PLOT)) &
					(df_raw["strategy"] == "step2")]

	df_alpha = df_raw[df_raw["lfr_p_sensitive"] == p_sensitive_fixed]
	df_psens = df_raw[df_raw["alpha"] == alpha_fixed]

	mu_vals  = sorted(LFR_MU_PLOT)
	has_ami  = _has_meaningful(df_raw, "ami")

	sns.set_style("whitegrid")
	fig, axs = plt.subplots(2, 2,
							figsize=(2 * COL_W, 2 * COL_H),
							sharey="row")

	def _fill_col(ax_top, ax_bot, df, x_col, xlabel):
		ax_bot_r = None
		if has_ami:
			ax_bot_r = ax_bot.twinx()
			ax_bot_r.set_ylim(*_padded_01_lim())
			ax_bot_r.set_ylabel("AMI")
			ax_bot_r.tick_params(axis="y")
			ax_bot_r.grid(False)

		for mu in mu_vals:
			df_s = df[df["lfr_mu"] == mu].sort_values(x_col)
			if df_s.empty:
				continue
			mc   = _mu_color(STYLE["modularity"]["color"],    mu, mu_vals)
			fc   = _mu_color(STYLE["balance_mouflon"]["color"], mu, mu_vals)
			nc_c = _mu_color(STYLE["ncomms"]["color"],        mu, mu_vals)

			ax_top.plot(df_s[x_col], df_s["modularity"],
						color=mc, linestyle=STYLE["modularity"]["linestyle"],
						marker=STYLE["modularity"]["marker"])
			if draw_error and "modularity_std" in df_s.columns:
				ax_top.errorbar(df_s[x_col], df_s["modularity"],
								yerr=df_s["modularity_std"], fmt="none",
								ecolor=mc, capsize=2)

			ax_top.plot(df_s[x_col], df_s["fair_bal"],
						color=fc, linestyle=STYLE["balance_mouflon"]["linestyle"],
						marker=STYLE["balance_mouflon"]["marker"])
			if draw_error and "fair_bal_std" in df_s.columns:
				ax_top.errorbar(df_s[x_col], df_s["fair_bal"],
								yerr=df_s["fair_bal_std"], fmt="none",
								ecolor=fc, capsize=2)

			ax_bot.plot(df_s[x_col], df_s["ncomms"],
						color=nc_c, linestyle=STYLE["ncomms"]["linestyle"],
						marker=STYLE["ncomms"]["marker"])
			if draw_error and "ncomms_std" in df_s.columns:
				ax_bot.errorbar(df_s[x_col], df_s["ncomms"],
								yerr=df_s["ncomms_std"], fmt="none",
								ecolor=nc_c, capsize=2)

			if ax_bot_r is not None and _has_meaningful(df_s, "ami"):
				ac = _mu_color(STYLE["ami"]["color"], mu, mu_vals)
				ax_bot_r.plot(df_s[x_col], df_s["ami"],
							  color=ac, linestyle=STYLE["ami"]["linestyle"],
							  marker=STYLE["ami"]["marker"])
				if draw_error and "ami_std" in df_s.columns:
					ax_bot_r.errorbar(df_s[x_col], df_s["ami"],
									  yerr=df_s["ami_std"], fmt="none",
									  ecolor=ac, capsize=2)

		x_ticks = sorted(df[x_col].unique())
		ax_top.set_xticks(x_ticks)
		ax_top.set_xticklabels([])
		ax_top.margins(x=0.05)
		ax_top.set_ylim(*_padded_01_lim())
		ax_top.autoscale(enable=False, axis="y")

		nc_max = (df["ncomms"] + df.get("ncomms_std",
				  pd.Series(0, index=df.index))).max()
		nc_pad = max(nc_max * 0.08, 0.5)
		ax_bot.set_xticks(x_ticks)
		ax_bot.set_xticklabels([f"{v:g}" for v in x_ticks])
		ax_bot.tick_params(axis="x", labelbottom=True)
		for lbl in ax_bot.get_xticklabels():
			lbl.set_visible(True)
		ax_bot.set_xlabel(xlabel)
		ax_bot.margins(x=0.05)
		ax_bot.set_ylim(-nc_pad, nc_max + nc_pad)
		ax_bot.autoscale(enable=False, axis="y")
		ax_bot.tick_params(axis="y")

	_fill_col(axs[0, 0], axs[1, 0], df_alpha, "alpha",           "alpha")
	_fill_col(axs[0, 1], axs[1, 1], df_psens, "lfr_p_sensitive", "p_sensitive")
	# Hardcode ncomms bottom row: 0-25 with padding
	for ax in [axs[1, 0], axs[1, 1]]:
		ax.set_ylim(-2.0, 27.0)
		ax.autoscale(enable=False, axis="y")

	axs[0, 0].set_ylabel("Score")
	axs[1, 0].set_ylabel("Number of communities")
	for i, label in enumerate(["A", "B"]):
		axs[0, i].text(0.02, 0.97, label, transform=axs[0, i].transAxes,
					   fontsize=10, fontweight="bold", va="top", ha="left")

	mu_handles = [
		mlines.Line2D([], [], color=_mu_color("tab:red", mu, mu_vals),
					  linestyle="-", marker="o", label=f"mu={mu}")
		for mu in mu_vals
	]
	metric_handles = [
		mlines.Line2D([], [], **STYLE["modularity"]),
		mlines.Line2D([], [], **STYLE["balance_mouflon"]),
		mlines.Line2D([], [], **STYLE["ncomms"]),
	]
	if has_ami:
		metric_handles.append(mlines.Line2D([], [], **STYLE["ami"]))

	fig.legend(handles=mu_handles + metric_handles, loc="upper center",
			   ncol=min(len(mu_handles) + len(metric_handles), 5),
			   bbox_to_anchor=(0.5, 1.02))
	fig.tight_layout(rect=[0, 0, 1, 0.95])
	fig.savefig(f"{plot_path}{filename}.png", dpi=300, bbox_inches="tight")
	plt.close(fig)


# ===========================================================================
# LFR QUALITY — node vs comm side-by-side (Figures 5 and 10)
# ===========================================================================

def plot_lfr_quality_node_comm(strategy, fairness_col, fairness_style,
								p_sensitive=0.5, draw_error=True,
								x_col="alpha", alpha_fixed=0.5,
								filename="Figure_LFR_node_comm"):
	"""
	2x2 figure: left = node-coloured scenario, right = comm-coloured scenario.
	Single strategy shown (step2 or hybrid). No panel titles.
	x_col="alpha"       : alpha sweep at fixed p_sensitive
	x_col="lfr_p_sensitive" : p_sensitive sweep at fixed alpha=alpha_fixed
	mu shown as hues.
	"""
	if x_col == "alpha":
		df_node = _lfr_load("node", p_sensitive)
		df_comm = _lfr_load("comm", p_sensitive)
	else:
		# p_sensitive sweep: load all p_sensitive values at fixed alpha
		df_raw = pd.read_csv(f"{log_path}/LFR_quality.csv")
		df_raw = df_raw[(df_raw["lfr_mu"].isin(LFR_MU_PLOT)) &
						(df_raw["alpha"] == alpha_fixed)]
		df_node = df_raw[df_raw["lfr_scenario"] == "node"]
		df_comm = df_raw[df_raw["lfr_scenario"] == "comm"]

	# Filter to the requested strategy + louvain reference
	mu_vals = sorted(df_node["lfr_mu"].unique())

	mouflon = [strategy]
	nc_lo_n, nc_hi_n = get_ncomms_limits([df_node[df_node["strategy"] == strategy]])
	nc_lo_c, nc_hi_c = get_ncomms_limits([df_comm[df_comm["strategy"] == strategy]])
	nc_lo = min(nc_lo_n, nc_lo_c)
	nc_hi = max(nc_hi_n, nc_hi_c)

	has_ami = _has_meaningful(df_node, "ami") or _has_meaningful(df_comm, "ami")

	sns.set_style("whitegrid")
	fig, axs = plt.subplots(2, 2,
							figsize=(2 * COL_W, 2 * COL_H * 0.8),
							sharex="col", sharey="row")

	for col_idx, (df_all, panel_label) in enumerate(
			[(df_node, "A"), (df_comm, "B")]):

		ax_top = axs[0, col_idx]
		ax_bot = axs[1, col_idx]

		ax_bot_r = None
		if has_ami:
			ax_bot_r = ax_bot.twinx()
			ax_bot_r.set_ylim(*_padded_01_lim())
			ax_bot_r.set_ylabel("AMI")
			ax_bot_r.tick_params(axis="y")
			ax_bot_r.grid(False)

		for mu in mu_vals:
			df_s = (df_all[(df_all["strategy"] == strategy) &
						   (df_all["lfr_mu"] == mu)]
					.sort_values(x_col))
			if df_s.empty:
				continue

			mc  = _mu_color(STYLE["modularity"]["color"], mu, mu_vals)
			fc  = _mu_color(fairness_style["color"],     mu, mu_vals)
			nc_c = _mu_color(STYLE["ncomms"]["color"],   mu, mu_vals)

			ax_top.plot(df_s[x_col], df_s["modularity"],
						color=mc, linestyle=STYLE["modularity"]["linestyle"],
						marker=STYLE["modularity"]["marker"])
			if draw_error and "modularity_std" in df_s.columns:
				ax_top.errorbar(df_s[x_col], df_s["modularity"],
								yerr=df_s["modularity_std"], fmt="none",
								ecolor=mc, capsize=2)

			ax_top.plot(df_s[x_col], df_s[fairness_col],
						color=fc, linestyle=fairness_style["linestyle"],
						marker=fairness_style["marker"])
			if draw_error and f"{fairness_col}_std" in df_s.columns:
				ax_top.errorbar(df_s[x_col], df_s[fairness_col],
								yerr=df_s[f"{fairness_col}_std"], fmt="none",
								ecolor=fc, capsize=2)

			ax_bot.plot(df_s[x_col], df_s["ncomms"],
						color=nc_c, linestyle=STYLE["ncomms"]["linestyle"],
						marker=STYLE["ncomms"]["marker"])
			if draw_error and "ncomms_std" in df_s.columns:
				ax_bot.errorbar(df_s[x_col], df_s["ncomms"],
								yerr=df_s["ncomms_std"], fmt="none",
								ecolor=nc_c, capsize=2)

			if ax_bot_r is not None and _has_meaningful(df_s, "ami"):
				ac = _mu_color(STYLE["ami"]["color"], mu, mu_vals)
				ax_bot_r.plot(df_s[x_col], df_s["ami"],
							  color=ac, linestyle=STYLE["ami"]["linestyle"],
							  marker=STYLE["ami"]["marker"])
				if draw_error and "ami_std" in df_s.columns:
					ax_bot_r.errorbar(df_s[x_col], df_s["ami"],
									  yerr=df_s["ami_std"], fmt="none",
									  ecolor=ac, capsize=2)

		x_ticks = sorted(df_all[df_all["strategy"] == strategy][x_col].unique())
		ax_top.set_xticks(x_ticks)
		ax_top.set_xticklabels([])
		ax_top.margins(x=0.05)
		ax_top.set_ylim(*_padded_01_lim())
		ax_top.autoscale(enable=False, axis="y")
		ax_top.text(0.02, 0.97, panel_label, transform=ax_top.transAxes,
					fontsize=10, fontweight="bold", va="top", ha="left")
		if col_idx == 0:
			ax_top.set_ylabel("Score")

		ax_bot.set_xticks(x_ticks)
		ax_bot.set_xticklabels([f"{v:g}" for v in x_ticks])
		ax_bot.tick_params(axis="x", labelbottom=True)
		for lbl in ax_bot.get_xticklabels():
			lbl.set_visible(True)
		ax_bot.set_xlabel(x_col.replace('lfr_', ''))
		nc_bot_pad = max((nc_hi - nc_lo) * 0.08, 0.5)
		ax_bot.margins(x=0.05)
		ax_bot.set_ylim(-nc_bot_pad, nc_hi + nc_bot_pad)
		ax_bot.autoscale(enable=False, axis="y")
		ax_bot.tick_params(axis="y")
		if col_idx == 0:
			ax_bot.set_ylabel("Number of communities")

	mu_handles = [
		mlines.Line2D([], [], color=_mu_color("tab:red", mu, mu_vals),
					  linestyle="-", marker="o", label=f"mu={mu}")
		for mu in mu_vals
	]
	metric_handles = [
		mlines.Line2D([], [], **STYLE["modularity"]),
		mlines.Line2D([], [], color=fairness_style["color"],
					  linestyle=fairness_style["linestyle"],
					  marker=fairness_style["marker"],
					  label=fairness_style["label"]),
		mlines.Line2D([], [], **STYLE["ncomms"]),
	]
	if has_ami:
		metric_handles.append(mlines.Line2D([], [], **STYLE["ami"]))

	fig.legend(handles=mu_handles + metric_handles, loc="upper center",
			   ncol=min(len(mu_handles) + len(metric_handles), 5),
			   bbox_to_anchor=(0.5, 1.02))
	fig.tight_layout(rect=[0, 0, 1, 0.95])
	fig.savefig(f"{plot_path}{filename}.png", dpi=300, bbox_inches="tight")
	plt.close(fig)


# ===========================================================================
# LFR QUALITY  — shared helpers + 3 figures (Q2 alpha, Q3 p_sensitive, Q5 node vs comm)
# ===========================================================================

# Only plot these three mu values to keep figures readable
LFR_MU_PLOT = [0.1, 0.3, 0.5]


def _lfr_load(scenario, p_sensitive):
	"""Load LFR_quality.csv sliced to scenario and p_sensitive, mu filtered."""
	df = pd.read_csv(f"{log_path}/LFR_quality.csv")
	df = df[(df["lfr_scenario"] == scenario) &
			(df["lfr_p_sensitive"] == p_sensitive) &
			(df["lfr_mu"].isin(LFR_MU_PLOT))]
	return df


def _draw_lfr_panel_pair(axs_top, axs_bot, df_all, x_col,
						  mu_vals, draw_error, nc_lo, nc_hi, xlabel=None):
	"""
	Fill one row of (top, bot) axes pairs — one pair per strategy column.

	Top panel : modularity (blue hues) + fairness (red/green hues).
	Bot panel : ncomms (left, purple hues) + AMI only (right, olive hues).
				No NF1, no Louvain reference line.
	"""
	strategies = ["step2", "hybrid"]
	has_ami    = _has_meaningful(df_all, "ami")
	has_nf1    = False   # not shown in LFR mu-sweep panels

	for col_idx, strategy in enumerate(strategies):
		ax_top = axs_top[col_idx]
		ax_bot = axs_bot[col_idx]

		cfg      = strategy_config[strategy]
		fair_col = cfg["fairness"]
		base_fc  = cfg["style"]["color"]
		base_ls  = cfg["style"]["linestyle"]
		base_mk  = cfg["style"]["marker"]
		mod_base = STYLE["modularity"]["color"]

		# Right twin axis for AMI/NF1 — created once per bot panel
		ax_bot_r = None
		if has_ami or has_nf1:
			ax_bot_r = ax_bot.twinx()
			ax_bot_r.set_ylim(*_padded_01_lim())
			ax_bot_r.set_ylabel("AMI / NF1")
			ax_bot_r.tick_params(axis="y")
			ax_bot_r.grid(False)

		for mu in mu_vals:
			df_s = (df_all[(df_all["strategy"] == strategy) &
						   (df_all["lfr_mu"] == mu)]
					.sort_values(x_col))
			if df_s.empty:
				continue

			mc   = _mu_color(mod_base,                  mu, mu_vals)
			fc   = _mu_color(base_fc,                   mu, mu_vals)
			nc_c = _mu_color(STYLE["ncomms"]["color"],  mu, mu_vals)

			# ── top: modularity ──────────────────────────────────────────────
			ax_top.plot(df_s[x_col], df_s["modularity"],
						color=mc, linestyle=STYLE["modularity"]["linestyle"],
						marker=STYLE["modularity"]["marker"])
			if draw_error and "modularity_std" in df_s.columns:
				ax_top.errorbar(df_s[x_col], df_s["modularity"],
								yerr=df_s["modularity_std"], fmt="none",
								ecolor=mc, capsize=2)

			# ── top: fairness ────────────────────────────────────────────────
			ax_top.plot(df_s[x_col], df_s[fair_col],
						color=fc, linestyle=base_ls, marker=base_mk)
			if draw_error and f"{fair_col}_std" in df_s.columns:
				ax_top.errorbar(df_s[x_col], df_s[fair_col],
								yerr=df_s[f"{fair_col}_std"], fmt="none",
								ecolor=fc, capsize=2)

			# ── bot: ncomms ──────────────────────────────────────────────────
			ax_bot.plot(df_s[x_col], df_s["ncomms"],
						color=nc_c, linestyle=STYLE["ncomms"]["linestyle"],
						marker=STYLE["ncomms"]["marker"])
			if draw_error and "ncomms_std" in df_s.columns:
				ax_bot.errorbar(df_s[x_col], df_s["ncomms"],
								yerr=df_s["ncomms_std"], fmt="none",
								ecolor=nc_c, capsize=2)

			# ── bot: AMI / NF1 on right axis ─────────────────────────────────
			if ax_bot_r is not None:
				if has_ami and _has_meaningful(df_s, "ami"):
					ac = _mu_color(STYLE["ami"]["color"], mu, mu_vals)
					ax_bot_r.plot(df_s[x_col], df_s["ami"],
								  color=ac, linestyle=STYLE["ami"]["linestyle"],
								  marker=STYLE["ami"]["marker"])
					if draw_error and "ami_std" in df_s.columns:
						ax_bot_r.errorbar(df_s[x_col], df_s["ami"],
										  yerr=df_s["ami_std"], fmt="none",
										  ecolor=ac, capsize=2)
				if has_nf1 and _has_meaningful(df_s, "nf1"):
					nc = _mu_color(STYLE["nf1"]["color"], mu, mu_vals)
					ax_bot_r.plot(df_s[x_col], df_s["nf1"],
								  color=nc, linestyle=STYLE["nf1"]["linestyle"],
								  marker=STYLE["nf1"]["marker"])
					if draw_error and "nf1_std" in df_s.columns:
						ax_bot_r.errorbar(df_s[x_col], df_s["nf1"],
										  yerr=df_s["nf1_std"], fmt="none",
										  ecolor=nc, capsize=2)

		# Axes formatting
		x_ticks = sorted(df_all[df_all["strategy"] == strategy][x_col].unique())
		ax_top.set_xticks(x_ticks)
		ax_top.set_xticklabels([])
		ax_top.margins(x=0.05)
		ax_top.set_ylim(*_padded_01_lim())
		ax_top.autoscale(enable=False, axis="y")
		ax_top.set_title(cfg["style"]["label"].replace("MOUFLON ", ""), fontsize=9)

		ax_bot.set_xticks(x_ticks)
		ax_bot.set_xticklabels([f"{v:g}" for v in x_ticks])
		ax_bot.tick_params(axis="x", labelbottom=True)
		if xlabel is not None:
			ax_bot.set_xlabel(xlabel)
		nc_bot_pad = max((nc_hi - nc_lo) * 0.08, 0.5)
		ax_bot.margins(x=0.05)
		ax_bot.set_ylim(-nc_bot_pad, nc_hi + nc_bot_pad)
		ax_bot.autoscale(enable=False, axis="y")
		ax_bot.tick_params(axis="y")


def _lfr_legend(mu_vals, has_ami, has_nf1=False):
	"""Shared legend for all LFR quality figures. NF1 and Louvain not shown."""
	mu_h = [
		mlines.Line2D([], [], color=_mu_color("tab:red", mu, mu_vals),
					  linestyle="-", marker="o", label=f"μ={mu}")
		for mu in mu_vals
	]
	metric_h = [
		mlines.Line2D([], [], **STYLE["modularity"]),
		mlines.Line2D([], [], **STYLE["balance_mouflon"]),
		mlines.Line2D([], [], **STYLE["prop_mouflon"]),
		mlines.Line2D([], [], **STYLE["ncomms"]),
	]
	if has_ami:
		metric_h.append(mlines.Line2D([], [], **STYLE["ami"]))
	return mu_h + metric_h


# ── Q2: alpha sweep — fixed scenario and p_sensitive, mu as hues ─────────────

def plot_lfr_quality(scenario="node", p_sensitive=0.5,
					 draw_error=True, filename="Figure_LFR_quality"):
	"""
	Q2 — effect of alpha on LFR quality.
	2 cols (step2 | hybrid) × 2 rows (score | ncomms+AMI+NF1).
	Three mu values shown as hues (lightest=0.1, darkest=0.5).
	"""
	df_all = _lfr_load(scenario, p_sensitive)
	mu_vals = sorted(df_all["lfr_mu"].unique())
	mouflon = [s for s in ["step2", "hybrid"] if s in df_all["strategy"].unique()]
	nc_lo, nc_hi = get_ncomms_limits([df_all[df_all["strategy"].isin(mouflon)]])
	has_ami = _has_meaningful(df_all, "ami")
	has_nf1 = _has_meaningful(df_all, "nf1")

	sns.set_style("whitegrid")
	fig, axs = plt.subplots(2, 2,
							figsize=(2 * COL_W, 2 * COL_H * 0.8),
							sharex="col")

	_draw_lfr_panel_pair(axs[0], axs[1], df_all, "alpha",
						 mu_vals, draw_error, nc_lo, nc_hi)

	axs[1, 0].set_ylabel("Number of communities")
	axs[0, 0].set_ylabel("Score")
	for i, ax in enumerate(axs[0]):
		ax.text(0.02, 0.97, ["A", "B"][i], transform=ax.transAxes,
				fontsize=10, fontweight="bold", va="top", ha="left")
	fig.supxlabel("alpha")

	fig.legend(handles=_lfr_legend(mu_vals, has_ami, has_nf1),
			   loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.02))
	fig.tight_layout(rect=[0, 0, 1, 0.95])
	fig.savefig(f"{plot_path}{filename}.png", dpi=300, bbox_inches="tight")
	plt.close(fig)


# ── Q3: p_sensitive sweep — fixed alpha=0.5, mu as hues ──────────────────────

def plot_lfr_quality_psensitive(scenario="node",
								 alpha_fixed=0.5,
								 draw_error=True,
								 filename="Figure_LFR_quality_psens"):
	"""
	Q3 — effect of p_sensitive (sensitive group size) on LFR quality.
	2 cols (step2 | hybrid) × 2 rows (score | ncomms+AMI+NF1).
	x-axis = p_sensitive; mu shown as hues; alpha fixed at alpha_fixed.
	"""
	df_raw = pd.read_csv(f"{log_path}/LFR_quality.csv")
	df_all = df_raw[(df_raw["lfr_scenario"] == scenario) &
					(df_raw["lfr_mu"].isin(LFR_MU_PLOT)) &
					(df_raw["alpha"] == alpha_fixed)]
	mu_vals  = sorted(df_all["lfr_mu"].unique())
	mouflon  = [s for s in ["step2", "hybrid"] if s in df_all["strategy"].unique()]
	nc_lo, nc_hi = get_ncomms_limits([df_all[df_all["strategy"].isin(mouflon)]])
	has_ami = _has_meaningful(df_all, "ami")
	has_nf1 = _has_meaningful(df_all, "nf1")

	sns.set_style("whitegrid")
	fig, axs = plt.subplots(2, 2,
							figsize=(2 * COL_W, 2 * COL_H * 0.8),
							sharex="col")

	_draw_lfr_panel_pair(axs[0], axs[1], df_all, "lfr_p_sensitive",
						 mu_vals, draw_error, nc_lo, nc_hi)

	axs[1, 0].set_ylabel("Number of communities")
	axs[0, 0].set_ylabel("Score")
	for i, ax in enumerate(axs[0]):
		ax.text(0.02, 0.97, ["A", "B"][i], transform=ax.transAxes,
				fontsize=10, fontweight="bold", va="top", ha="left")
	fig.supxlabel("p_sensitive")

	fig.legend(handles=_lfr_legend(mu_vals, has_ami, has_nf1),
			   loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.02))
	fig.tight_layout(rect=[0, 0, 1, 0.95])
	fig.savefig(f"{plot_path}{filename}.png", dpi=300, bbox_inches="tight")
	plt.close(fig)


# ── Q5: node vs comm — side-by-side scenarios, mu as hues ────────────────────

def plot_lfr_quality_scenarios(p_sensitive=0.5,
								draw_error=True,
								filename="Figure_LFR_quality_scenarios"):
	"""
	Q5 — effect of community homogeneity (node-coloured vs comm-coloured).
	4 cols (node:step2 | node:hybrid | comm:step2 | comm:hybrid)
	× 2 rows (score | ncomms+AMI+NF1).
	x-axis = alpha; mu shown as hues.
	The two scenarios share y-axes so differences are immediately visible.
	"""
	df_node = _lfr_load("node", p_sensitive)
	df_comm = _lfr_load("comm", p_sensitive)
	mu_vals = sorted(df_node["lfr_mu"].unique())

	mouflon = ["step2", "hybrid"]
	nc_lo, nc_hi = get_ncomms_limits(
		[df_node[df_node["strategy"].isin(mouflon)],
		 df_comm[df_comm["strategy"].isin(mouflon)]])
	has_ami = _has_meaningful(df_node, "ami") or _has_meaningful(df_comm, "ami")
	has_nf1 = _has_meaningful(df_node, "nf1") or _has_meaningful(df_comm, "nf1")

	sns.set_style("whitegrid")
	# 2 rows × 4 cols; share y within each row
	fig, axs = plt.subplots(2, 4,
							figsize=(4 * COL_W, 2 * COL_H * 0.8),
							sharey="row")
	import string

	for s_idx, (scenario_label, df_s) in enumerate(
			[("node-coloured", df_node), ("comm-coloured", df_comm)]):
		col_offset = s_idx * 2   # columns 0-1 for node, 2-3 for comm
		top_pair = [axs[0, col_offset],     axs[0, col_offset + 1]]
		bot_pair = [axs[1, col_offset],     axs[1, col_offset + 1]]

		_draw_lfr_panel_pair(top_pair, bot_pair, df_s, "alpha",
							 mu_vals, draw_error, nc_lo, nc_hi)

		# Scenario label spanning both columns
		mid_ax = top_pair[0]
		mid_ax.annotate(
			scenario_label,
			xy=(0.5, 1.12), xycoords="axes fraction",
			ha="center", va="bottom", fontsize=9, fontweight="bold",
			annotation_clip=False,
		)
		# Panel letters
		for i, ax in enumerate(top_pair):
			ax.text(0.02, 0.97,
					string.ascii_uppercase[col_offset + i],
					transform=ax.transAxes,
					fontsize=10, fontweight="bold", va="top", ha="left")

	axs[0, 0].set_ylabel("Score")
	axs[1, 0].set_ylabel("Number of communities")
	fig.supxlabel("alpha")

	fig.legend(handles=_lfr_legend(mu_vals, has_ami, has_nf1),
			   loc="upper center", ncol=5, bbox_to_anchor=(0.5, 1.02))
	fig.tight_layout(rect=[0, 0, 1, 0.95])
	fig.savefig(f"{plot_path}{filename}.png", dpi=300, bbox_inches="tight")
	plt.close(fig)


# ===========================================================================
# CITY NETWORKS — 2-row layout
# ===========================================================================

def plot_city_networks(city_files: dict, draw_error=True, filename="Figure_city"):
	"""
	Quality figure for city networks.
	Row 1: modularity + fairness (hybrid only) + Louvain reference line.
	Row 2: ncomms (left axis) + ami_louvain (right axis, 0-1).
	"""
	import string

	sns.set_style("whitegrid")
	n = len(city_files)

	fig, axs = plt.subplots(2, n,
							figsize=(COL_W * n, 2 * COL_H),
							sharex="col")
	axs = np.array(axs).reshape(2, n)

	for col, (city_name, csv_path) in enumerate(city_files.items()):
		df       = pd.read_csv(csv_path)
		df_h     = df[df["strategy"] == "hybrid"].sort_values("alpha")
		df_lou   = df[df["strategy"] == "louvain"]

		nc_lo, nc_hi = get_ncomms_limits([df_h])

		ax_top = axs[0, col]
		ax_bot = axs[1, col]

		# ── Row 1 ─────────────────────────────────────────────────────────
		ax_top.plot(df_h["alpha"], df_h["modularity"], **STYLE["modularity"])
		if draw_error:
			ax_top.errorbar(df_h["alpha"], df_h["modularity"],
							yerr=df_h["mod_std"], fmt="none",
							ecolor=STYLE["modularity"]["color"], capsize=2)
		ax_top.plot(df_h["alpha"], df_h["fair_exp"], **STYLE["prop_mouflon"])
		if draw_error:
			ax_top.errorbar(df_h["alpha"], df_h["fair_exp"],
							yerr=df_h["fair_exp_std"], fmt="none",
							ecolor=STYLE["prop_mouflon"]["color"], capsize=2)
		if not df_lou.empty:
			ax_top.axhline(df_lou["modularity"].iloc[0],
						   color=STYLE["louvain"]["color"],
						   linestyle=":",
						   linewidth=1.2, alpha=0.8)
		ax_top.set_xticks(sorted(df_h["alpha"].unique()))
		ax_top.set_xticklabels([])
		ax_top.margins(x=0.05)
		ax_top.set_ylim(*_padded_01_lim())
		ax_top.autoscale(enable=False, axis="y")
		ax_top.set_title(city_name, fontsize=9)
		ax_top.set_ylabel("Score")
		ax_top.text(0.02, 0.97, string.ascii_uppercase[col],
					transform=ax_top.transAxes, fontsize=10,
					fontweight="bold", va="top", ha="left")

		# ── Row 2 ─────────────────────────────────────────────────────────
		ax_bot.plot(df_h["alpha"], df_h["ncomms"], **STYLE["ncomms"])
		if draw_error and "ncomms_std" in df_h.columns:
			ax_bot.errorbar(df_h["alpha"], df_h["ncomms"],
							yerr=df_h["ncomms_std"], fmt="none",
							ecolor=STYLE["ncomms"]["color"], capsize=2)
		nc_bot_pad = max((nc_hi - nc_lo) * 0.08, 0.5)
		ax_bot.tick_params(axis="y")
		ax_bot.set_ylabel("Number of communities")
		ax_bot.margins(x=0.05)
		ax_bot.set_ylim(-nc_bot_pad, nc_hi + nc_bot_pad)
		ax_bot.autoscale(enable=False, axis="y")

		if ("ami_louvain" in df_h.columns and
				not df_h["ami_louvain"].isna().all()):
			ax_r = ax_bot.twinx()
			ax_r.set_ylim(*_padded_01_lim())
			ax_r.set_ylabel("AMI vs Louvain")
			ax_r.grid(False)
			ax_r.plot(df_h["alpha"], df_h["ami_louvain"], **STYLE["ami"])
			if draw_error and "ami_louvain_std" in df_h.columns:
				ax_r.errorbar(df_h["alpha"], df_h["ami_louvain"],
							  yerr=df_h["ami_louvain_std"], fmt="none",
							  ecolor=STYLE["ami"]["color"], capsize=2)

		ax_bot.set_xticks(sorted(df_h["alpha"].unique()))
		ax_bot.set_xticklabels([f"{v:g}" for v in sorted(df_h["alpha"].unique())])
		ax_bot.tick_params(axis="x", labelbottom=True)
		for lbl in ax_bot.get_xticklabels():
			lbl.set_visible(True)

	fig.supxlabel("alpha")

	handles = [
		mlines.Line2D([], [], **STYLE["modularity"]),
		mlines.Line2D([], [], **STYLE["prop_mouflon"]),
		mlines.Line2D([], [], **STYLE["louvain"]),
		mlines.Line2D([], [], **STYLE["ncomms"]),
		mlines.Line2D([], [], **{**STYLE["ami"], "label": "AMI vs Louvain"}),
	]
	fig.legend(handles=handles, loc="upper center",
			   ncol=5, bbox_to_anchor=(0.5, 1.02))
	fig.tight_layout(rect=[0, 0, 1, 0.95])
	fig.savefig(f"{plot_path}{filename}.png", dpi=300, bbox_inches="tight")
	plt.close(fig)


# ===========================================================================
# REAL SOCIAL NETWORKS — Figure 9
# 3 rows per network: step2, hybrid (quality), then ncomms row
# ===========================================================================

def plot_multiple_alpha_real(networks, draw_error=True, filename="Figure9"):
	"""
	Multi-network quality figure.
	Row 1 (step2)  : modularity + balance fairness + Louvain ref
	Row 2 (hybrid) : modularity + prop_balance fairness + Louvain ref
	No AMI/NF1 — not available for real social networks.
	Number of communities shown as inset (lower-right) in each panel,
	y-axis always [1, max_ncomms] in log scale, shared across all panels.
	"""
	import string

	sns.set_style("whitegrid")
	n = len(networks)
	fig, axs = plt.subplots(2, n,
							figsize=(COL_W * n, 2 * COL_H * 0.85),
							sharex="col", sharey="row")
	axs = np.array(axs).reshape(2, n)

	strategies = ["step2", "hybrid"]

	# Compute a single shared inset y-range [1, global_max_ncomms] across all
	# networks and strategies, so all insets are directly comparable.
	global_nc_hi = 1.0
	for network in networks:
		df_tmp = pd.read_csv(f"{log_path}/{network}.csv")
		for strategy in strategies:
			df_s = df_tmp[df_tmp["strategy"] == strategy]
			if "ncomms" in df_s.columns and not df_s["ncomms"].isna().all():
				std = df_s["ncomms_std"] if "ncomms_std" in df_s.columns else 0
				global_nc_hi = max(global_nc_hi, (df_s["ncomms"] + std).max())
	# Add 5% headroom above the global max
	global_nc_hi = global_nc_hi * 1.05

	for col, network in enumerate(networks):
		df     = pd.read_csv(f"{log_path}/{network}.csv")
		df_lou = df[df["strategy"] == "louvain"]

		for row, strategy in enumerate(strategies):
			ax       = axs[row, col]
			cfg      = strategy_config[strategy]
			df_strat = df[df["strategy"] == strategy].sort_values("alpha")
			fair_col = cfg["fairness"]
			style    = cfg["style"]

			# Main panel: modularity + fairness + Louvain ref
			ax.plot(df_strat["alpha"], df_strat["modularity"], **STYLE["modularity"])
			if draw_error and "modularity_std" in df_strat.columns:
				ax.errorbar(df_strat["alpha"], df_strat["modularity"],
							yerr=df_strat["modularity_std"], fmt="none",
							ecolor=STYLE["modularity"]["color"], capsize=2)
			ax.plot(df_strat["alpha"], df_strat[fair_col], **style)
			if draw_error and f"{fair_col}_std" in df_strat.columns:
				ax.errorbar(df_strat["alpha"], df_strat[fair_col],
							yerr=df_strat[f"{fair_col}_std"], fmt="none",
							ecolor=style["color"], capsize=2)
			if not df_lou.empty:
				ax.axhline(df_lou["modularity"].iloc[0],
						   color=STYLE["louvain"]["color"],
						   linestyle=":",
						   linewidth=1.0, alpha=0.7)

			# (i)  alpha ticks on all panels (bottom row gets labels, top row hidden)
			ax.set_xticks(sorted(df_strat["alpha"].unique()))
			if row == 0:
				ax.set_xticklabels([])
			else:
				ax.set_xticklabels([f"{v:g}" for v in sorted(df_strat["alpha"].unique())])
				ax.tick_params(axis="x", labelbottom=True)
				for lbl in ax.get_xticklabels():
					lbl.set_visible(True)
			ax.margins(x=0.05)
			ax.set_ylim(*_padded_01_lim())
			ax.autoscale(enable=False, axis="y")

			# (ii) y-axis title is simply "Score" on leftmost column only
			if col == 0:
				ax.set_ylabel("Score", fontsize=8)

			# (iii/iv) Inset bottom-right, log scale [1, global_nc_hi]
			inset = inset_axes(ax, width="35%", height="30%", loc="lower right")
			inset.plot(df_strat["alpha"], df_strat["ncomms"], **STYLE["ncomms"])
			if draw_error and "ncomms_std" in df_strat.columns:
				inset.errorbar(df_strat["alpha"], df_strat["ncomms"],
							   yerr=df_strat["ncomms_std"], fmt="none",
							   ecolor=STYLE["ncomms"]["color"], capsize=1)
			inset.set_ylim(1, global_nc_hi)
			inset.set_yscale("log")
			inset.tick_params(axis="y", labelsize=5)
			inset.tick_params(axis="x", labelsize=0, length=0)
			inset.yaxis.set_major_locator(LogLocator(numticks=3))
			inset.yaxis.set_major_formatter(LogFormatterSciNotation(labelOnlyBase=True))
			inset.margins(x=0.05, y=0.1)

		# Column header + panel label
		axs[0, col].set_title(network, fontsize=8)
		axs[0, col].text(-0.12, 1.08, string.ascii_uppercase[col],
						 transform=axs[0, col].transAxes,
						 fontsize=11, fontweight="bold",
						 va="top", ha="center")

	fig.supxlabel("alpha")

	handles = [
		mlines.Line2D([], [], **STYLE["modularity"]),
		mlines.Line2D([], [], **STYLE["balance_mouflon"]),
		mlines.Line2D([], [], **STYLE["prop_mouflon"]),
		mlines.Line2D([], [], **STYLE["louvain"]),
		mlines.Line2D([], [], **STYLE["ncomms"]),
	]
	fig.legend(handles=handles, loc="upper center",
			   ncol=5, bbox_to_anchor=(0.5, 1.02))
	fig.tight_layout(rect=[0, 0, 1, 0.95])
	fig.savefig(f"{plot_path}{filename}.png", dpi=300, bbox_inches="tight")
	plt.close(fig)

# ===========================================================================
# Stats helper
# ===========================================================================

def get_stats(network, log_file, only_log=True):
	if not only_log:
		with open(f"{obj_path}/{network}.nx", "rb") as g_open:
			net = pickle.load(g_open)
		print(f"{network}: N={net.number_of_nodes()}, M={net.number_of_edges()}")
		del net
		gc.collect()
	df       = pd.read_csv(f"{log_path}/{log_file}.csv")
	avg_time = df.groupby("strategy").agg({"time": "mean", "time_std": "mean"})
	print(avg_time)
	no_comms = df[df["alpha"].isin([0.0, 0.5, 1.0])][
		["strategy", "alpha", "ncomms", "ncomms_std"]]
	print(no_comms)


# ===========================================================================
# Main
# ===========================================================================

def main():
	realSN_list = ["facebook", "deezer", "twitch", "pokec-a", "pokec-g"]

	# City network files — add more cities as they become available
	city_files = {
		"Sandviken": f"{log_path}/Sandviken_filtered_edgelist_flat_all_2017_city_ab.csv",
		"Filipstad": f"{log_path}/Filipstad_filtered_edgelist_flat_all_2017_city_ab.csv",
	}

	# --- Stats
	for network in realSN_list:
		get_stats(network, network, only_log=True)


	# --- Figure 1: toy example of G ---

	# --- Figure 2: prop_fairness toy values ---

	# --- Figures 3+4: combined ER+LFR scalability (2x2)
	main_df = pd.DataFrame()
	for nodes in [1000, 2000, 5000, 10000, 20000, 50000, 100000, 200000]:
		df = create_time_df(f"ER_{nodes}_r0001_K2_c05", nodes, 0.001, log_path)
		main_df = pd.concat([main_df, df], ignore_index=True)

	main_df1 = pd.DataFrame()
	for dp in range(1, 6):
		df1 = create_time_df(f"ER_10000_r0{dp}_K2_c05",
							 10000, float(dp / 10), log_path)
		main_df1 = pd.concat([main_df1, df1], ignore_index=True)

	plot_scalability_combined(
		df_er_size=main_df,
		df_er_density=main_df1,
		filename="Figure3"
	)


	# --- p_sensitive file lists (for Figures 4-7)
	node_files = [
		f"{log_path}/color-node_1000_r01_K2_c{p}.csv"
		for p in ["01", "02", "03", "04", "05"]
	]
	full_files = [
		f"{log_path}/color-full_1000_r01_K2_c{p}.csv"
		for p in ["01", "02", "03", "04", "05"]
	]

	# --- Figure 4
	plot_mouflon_alpha("color-node_1000_r01_K2_c05",
					   "color-full_1000_r01_K2_c05",
					   filename="Figure4")

	# --- Figure 5: LFR quality - prop_balance, node vs comm
	plot_lfr_quality_node_comm(
		strategy="hybrid", fairness_col="fair_exp",
		fairness_style=STYLE["prop_mouflon"],
		p_sensitive=0.5, filename="Figure5")


	# --- Figure 6: p_sensitive sweep
	plot_mouflon_psensitive(node_files, full_files, filename="Figure6")

	# --- Figure 7: LFR p_sensitive sweep, prop_balance, node vs comm
	plot_lfr_quality_node_comm(
		strategy="hybrid", fairness_col="fair_exp",
		fairness_style=STYLE["prop_mouflon"],
		x_col="lfr_p_sensitive", alpha_fixed=0.5,
		filename="Figure7")


	# --- Figure 8 combined: step2 (balance), alpha sweep / p_sensitive sweep
	plot_step2_combined("color-node_1000_r01_K2_c05", node_files, filename="Figure8")


	# --- Figure 9: LFR step2 (balance), alpha sweep / p_sensitive sweep (node only)
	plot_lfr_step2_combined(p_sensitive_fixed=0.5, alpha_fixed=0.5,
							filename="Figure9")



	# --- Figure 10: A/B testing, city networks
	plot_city_networks(city_files, draw_error=True, filename="Figure10")

	# --- Figure 11: drawing prop_fairness example ---

	# --- Figure 12: real social networks
	plot_multiple_alpha_real(realSN_list, draw_error=True, filename="Figure12")


if __name__ == "__main__":
	main()