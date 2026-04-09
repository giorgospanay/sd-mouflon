import json
import pickle
import statistics
import sys
import time

import networkx as nx
import pandas as pd
from networkx.algorithms.community import modularity

from modules.fair_louvaines import fair_louvain_communities
from modules.helpers import (diversity_fairness, diversityMetricPaper,
							 fairness_base, fairness_fexp, modularity_fairness)


# Import CD eval metrics
from cdlib.classes import NodeClustering
from networkx.algorithms.cuts import conductance
from cdlib import evaluation


# Globals for paths
obj_path="../data/obj"
log_path="../logs/"
plot_path="../plots/"


# Add helpers
def planted_labels_cliques(n):
	# clique id = node_id // 100
	return [i // 100 for i in range(n)]


# @TODO: check again before rerunning quality labels
def partition_to_labels(partition, n):
	labels = [-1] * n
	for cid, comm in enumerate(partition):
		for u in comm:
			labels[u] = cid
	return labels


def symmetric_nf1(pred_nc, gt_nc):
	nf1_forward  = pred_nc.nf1(gt_nc).score
	nf1_backward = gt_nc.nf1(pred_nc).score
	return (nf1_forward + nf1_backward) / 2


## -------------------------------------
##          Experiment logging
## -------------------------------------

def experiment(network, color_list=["blue","red"], alpha=[0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0], n_reps=3, strategy=["base","step2","fexp","hybrid","fmody","diversity","step2fmody","step2div"], debug_mode=False, planted=False):
	net=None
	# Load file
	with open(f"{obj_path}/{network}.nx","rb") as g_open:
		net=pickle.load(g_open)

	print(f"Network object {network} loaded.")
	print(f"{network}: N={net.number_of_nodes()}, M={net.number_of_edges()}")

	gt_labels=None
	gt_nc=None

	if planted:
		n=net.number_of_nodes()
		# Set planted cliques labels
		gt_labels=planted_labels_cliques(n)
		gt_comms=[set(range(k*100,(k+1)*100)) for k in range(10)]
		gt_nc=NodeClustering(gt_comms, net, method_name="planted")


	colors=nx.get_node_attributes(net, "color")
	# print(f"Color of first 10 nodes =  {sorted(list(colors.keys()))[:10]}")

	# Calculate network color ratios here to pass
	color_dist={}
	for c in color_list:
		color_dist[c]=0
	for n_ind in net.nodes():
		color_dist[colors[n_ind]]+=1

	print("Color distribution calculated.")
	for color in color_dist:
		print(f"{color}:{color_dist[color]}")


	
	# Run algorithm
	# alpha=[0.0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,
	# 	0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1.0]

	# alpha=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
	
	#alpha=[1.0]
	#alpha=[0.5,0.6,0.7]
	# alpha=[0.5]
 
	# print(f"Alpha values: {alpha}")
	
	
	# Lists to keep stats for dataframe
	strat_l=[]
	alpha_l=[]
	time_l=[]
	time_s=[]
	ncomms_l=[]
	ncomms_s=[]
	mod_l=[]
	mod_s=[]
	fairbal_l=[]
	fairbal_s=[]
	fairexp_l=[]
	fairexp_s=[]
	fairmodf_l=[]
	fairmodf_s=[]
	diversity_l=[]
	diversity_s=[]
	div_paper_l=[]
	div_paper_s=[]
	# also add for eval metrics
	ami_l=[]
	ami_s=[]
	nf1_l=[]
	nf1_s=[]
	cond_l=[]
	cond_s=[]
 
	# results=[] 	# For printing into JSON file

	# For each strategy:
	for strat in strategy:
		print(f"------------------\nSTRATEGY={strat}")

		# Run fair-balance on various values of alpha
		for a in alpha:
			print(f"-----------------\nAlpha={a}\n-----------------")

			modularity_list=[]
			fairnessbal_list=[]
			fairnessexp_list=[]
			fairnessmodf_list=[]
			diversity_list=[]
			div_paper_list=[]
			ncomms_list=[]
			time_list=[]
			# also metrics
			ami_list=[]
			nf1_list=[]

			# and n reps
			for i in range(n_reps):
				if n_reps<=10:
					print(f"Run #{(i+1)}")
				else:
					if i%1000==0:
						print(f"Run #{(i+1)}/{n_reps}")

				# Start benchmarking
				start_time=time.time()

				# For now only keep MOUFLON. Change script for external benchmark
				res = fair_louvain_communities(net, color_list=color_list, alpha=a, strategy=strat)
				
				end_time=time.time()


				# For printing communities formed into JSON file
				# results.append({
				# 	"strategy":strat,
				# 	"alpha":a,
				# 	"n_reps":i,
				# 	"communities":[list(r) for r in res],
				# })

				# # For debug: print also communities found
				# if debug_mode:
				# print("Communities obtained:")
				# print(res)

				# Track number of communities
				ncomms_list.append(len(res))

				# Track execution time
				time_list.append(end_time-start_time)

				# Track modularity score
				mod_overall = modularity(net, res, weight="weight")
				modularity_list.append(mod_overall)

				# Track both fairness scores
				F_bal_overall, F_bal_dist = fairness_base(net, res, color_dist)
				F_exp_overall, F_exp_dist = fairness_fexp(net, res, color_dist)
				fairnessbal_list.append(F_bal_overall)
				fairnessexp_list.append(F_exp_overall)
	
				# Track modularity fairness score
				F_modf_overall, F_modf_dist = modularity_fairness(net, res, color_dist, colors)
				# Trying to normalize modularity fairness score so that it is comparable to other strats
				F_modf_overall_norm = 1 - abs(F_modf_overall)
				fairnessmodf_list.append(F_modf_overall_norm)
	
				# Track diversity fairness score
				F_div_overall, F_div_dist = diversity_fairness(net, res, color_dist, colors)
				F_div_overall_norm = 1 - abs(F_div_overall)
				diversity_list.append(F_div_overall_norm)
	
				# Track diversity metric from paper
				F_div_paper_overall, F_div_paper_dist = diversityMetricPaper(net, res, colors)
				F_div_paper_overall_norm = 1 - abs(F_div_paper_overall)
				div_paper_list.append(F_div_paper_overall_norm)


				# Track partition agreement metrics
				if planted:
					# Create NodeClustering for predicted labels
					pred_nc = NodeClustering([set(r) for r in res],net,method_name="")

					# Calculate AMI
					ami_list.append(pred_nc.adjusted_mutual_information(gt_nc).score)
					# Calculate symmetric NF1 (average of both matching directions)
					nf1_list.append(symmetric_nf1(pred_nc, gt_nc))


			# At end of reps: append to lists
			strat_l.append(strat)
			alpha_l.append(a)
			time_l.append(statistics.fmean(time_list))
			time_s.append(statistics.stdev(time_list))
			mod_l.append(statistics.fmean(modularity_list))
			mod_s.append(statistics.stdev(modularity_list))
			fairbal_l.append(statistics.fmean(fairnessbal_list))
			fairbal_s.append(statistics.stdev(fairnessbal_list))
			fairexp_l.append(statistics.fmean(fairnessexp_list))
			fairexp_s.append(statistics.stdev(fairnessexp_list))
			fairmodf_l.append(statistics.fmean(fairnessmodf_list))
			fairmodf_s.append(statistics.stdev(fairnessmodf_list))
			diversity_l.append(statistics.fmean(diversity_list))
			diversity_s.append(statistics.stdev(diversity_list))
			div_paper_l.append(statistics.fmean(div_paper_list))
			div_paper_s.append(statistics.stdev(div_paper_list))
			ncomms_l.append(statistics.fmean(ncomms_list))
			ncomms_s.append(statistics.stdev(ncomms_list))
			# calc AMI + NF1 only if planted GT enabled
			if planted:
				ami_l.append(statistics.fmean(ami_list))
				ami_s.append(statistics.stdev(ami_list))
				nf1_l.append(statistics.fmean(nf1_list))
				nf1_s.append(statistics.stdev(nf1_list))
			else:
				ami_l.append(None) 
				ami_s.append(None)
				nf1_l.append(None)
				nf1_s.append(None)

	# Here, also run n runs of Louvain and compare statistics.
	# Lists unused -- reinitialize to copy code
	modularity_list=[]
	fairnessbal_list=[]
	fairnessexp_list=[]
	fairnessmodf_list=[]
	diversity_list=[]
	div_paper_list=[]
	ncomms_list=[]
	time_list=[]
	# also metrics
	ami_list=[]
	nf1_list=[]


	# Ignore louvain run for quality tests (e.g. when color-* network.)
	if not "color" in network:
		for i in range(n_reps):
			if n_reps<=10:
				print(f"Louvain run #{(i+1)}")
			else:
				if i%1000==0:
					print(f"Run #{(i+1)}/{n_reps}")

			# Start benchmarking
			start_time=time.time()

			# Run Louvain comms
			res=nx.community.louvain_communities(net)
			
			# End benchmarking
			end_time=time.time()

			# Track number of communities
			ncomms_list.append(len(res))
			# Track execution time
			time_list.append(end_time-start_time)

			# Track modularity score
			mod_overall = modularity(net, res, weight="weight")
			modularity_list.append(mod_overall)

			# Track both fairness scores
			F_bal_overall, F_bal_dist = fairness_base(net, res, color_dist)
			F_exp_overall, F_exp_dist = fairness_fexp(net, res, color_dist)
			fairnessbal_list.append(F_bal_overall)
			fairnessexp_list.append(F_exp_overall)

			# Track modularity fairness score
			F_modf_overall, F_modf_dist = modularity_fairness(net, res, color_dist, colors)
			# Trying to normalize modularity fairness score so that it is comparable to other strats
			F_modf_overall_norm = 1 - abs(F_modf_overall)
			fairnessmodf_list.append(F_modf_overall_norm)

			# Track diversity fairness score
			F_div_overall, F_div_dist = diversity_fairness(net, res, color_dist, colors)
			F_div_overall_norm = 1 - abs(F_div_overall)
			diversity_list.append(F_div_overall_norm)

			# Track diversity metric from paper
			F_div_paper_overall, F_div_paper_dist = diversityMetricPaper(net, res, colors)
			F_div_paper_overall_norm = 1 - abs(F_div_paper_overall)
			div_paper_list.append(F_div_paper_overall_norm)

			if planted:
				# Create NodeClustering for predicted labels
				pred_nc = NodeClustering([set(r) for r in res],net,method_name="")

				# Calculate AMI
				ami_list.append(pred_nc.adjusted_mutual_information(gt_nc).score)
				# Calculate symmetric NF1
				nf1_list.append(symmetric_nf1(pred_nc, gt_nc))

		# Add stats to df
		strat_l.append("louvain")
		alpha_l.append(1.0)
		time_l.append(statistics.fmean(time_list))
		time_s.append(statistics.stdev(time_list))
		mod_l.append(statistics.fmean(modularity_list))
		mod_s.append(statistics.stdev(modularity_list))
		fairbal_l.append(statistics.fmean(fairnessbal_list))
		fairbal_s.append(statistics.stdev(fairnessbal_list))
		fairexp_l.append(statistics.fmean(fairnessexp_list))
		fairexp_s.append(statistics.stdev(fairnessexp_list))
		fairmodf_l.append(statistics.fmean(fairnessmodf_list))
		fairmodf_s.append(statistics.stdev(fairnessmodf_list))
		diversity_l.append(statistics.fmean(diversity_list))
		diversity_s.append(statistics.stdev(diversity_list))
		div_paper_l.append(statistics.fmean(div_paper_list))
		div_paper_s.append(statistics.stdev(div_paper_list))
		ncomms_l.append(statistics.fmean(ncomms_list))
		ncomms_s.append(statistics.stdev(ncomms_list))

		# calc AMI + NF1 only if planted GT enabled
		if planted:
			ami_l.append(statistics.fmean(ami_list))
			ami_s.append(statistics.stdev(ami_list))
			nf1_l.append(statistics.fmean(nf1_list))
			nf1_s.append(statistics.stdev(nf1_list))
		else:
			ami_l.append(None) 
			ami_s.append(None)
			nf1_l.append(None)
			nf1_s.append(None)


	# Results dataframe
	df=pd.DataFrame.from_dict(
		{
			"strategy":strat_l,
			"alpha":alpha_l,
			"time":time_l,
			"time_std":time_s,
			"modularity":mod_l,
			"modularity_std":mod_s,
			"fair_bal":fairbal_l,
			"fair_bal_std":fairbal_s,
			"fair_exp":fairexp_l,
			"fair_exp_std":fairexp_s,
			"fair_modf": fairmodf_l,
			"fair_modf_std": fairmodf_s,
			"fair_div":diversity_l,
			"fair_div_std":diversity_s, 
			"fair_div_paper":div_paper_l,
			"fair_div_paper_std":div_paper_s,
			"ncomms":ncomms_l,
			"ncomms_std":ncomms_s,
			"ami": ami_l,
			"ami_std": ami_s,
			"nf1": nf1_l,
			"nf1_std": nf1_s
		}, orient="columns"
	)

	# If not in debug mode: save to logfile
	if not debug_mode:
		# At end of run: create big dataframe for run stats and save to file
		df.to_csv(f"{log_path}/{network}.csv",index=False)
		# For printing communities formed into JSON file
		# with open(f"{log_path}/{network}_communities.json","w") as f:
		# 	json.dump(results, f, indent=4)
	# Otherwise simply print results to screen
	else:
		print(df[["strategy","alpha","modularity","fair_bal","fair_exp", "fair_modf", "fair_div", "fair_div_paper","ami","nf1","ncomms"]])




	
# (Kobbie) Main(oo) --how is he doing these days?
"""
Arguments: python fair_cd_main.py network [color_list] [alpha] [n_reps] [strat_list] [debug]
	- network: 	the network object to be run. Can also activate the planted flag,
				depending on which network is ran (planted partition exp:s)
	- color_list: (optional) the colors coded in the network object. Useful especially
				  for networks with multiple colours, otherwise defaults to red,blue
				  Should be input without quotes or spaces, separated by commas.
	- alpha:	  (optional) the alpha values to be used. Defaults to 0.0,0.1,...,1.0. Similar formatting 
				  to color_list: no spaces, comma-separated values.
	- n_reps:	  (optional) the number of iterations for each run of alpha. Default 3
	- strat_list: (optional) the different optimization strategies to be tried for 
				  each run. Defaults to all (base,step2,fexp,hybrid). Similar formatting 
				  to color_list: no spaces, comma-separated values.
	- debug:	  (optional) runs in debug mode (i.e. no log file, but results printed
				  on screen) if the last argument is "debug"

Example call: python3 fair_cd_main.py facebook red,blue 100 base,step2 debug
	Will run the facebook network for colors R&B for 100 iterations, using the base 
	and step2 strategies, in debug mode.

"""
def main():
	args=sys.argv[1:]

	if len(args)==1:
		experiment(args[0])
	elif len(args)>1 and len(args)<7:
		# Determine planted flag from network name given
		planted=False
		net=args[0]
		if "color-full" in net or "color-node" in net:
			planted=True

		if len(args)==2:
			clist=args[1].split(",")
			experiment(args[0], clist, planted=planted)
		elif len(args)==3:
			clist=args[1].split(",")
			alist=args[2].split(",")
			alist=[float(a) for a in alist]
			experiment(args[0], clist, alpha=alist, planted=planted)
		elif len(args)==4:
			clist=args[1].split(",")
			alist=args[2].split(",")
			alist=[float(a) for a in alist]
			experiment(args[0], clist, alpha=alist, n_reps=int(args[3]), planted=planted)
		elif len(args)==5:
			clist=args[1].split(",")
			alist=args[2].split(",")
			alist=[float(a) for a in alist]
			stratlist=args[4].split(",")
			experiment(args[0], clist, alpha=alist, n_reps=int(args[3]), strategy=stratlist, planted=planted)
		elif len(args)==6:
			clist=args[1].split(",")
			alist=args[2].split(",")
			alist=[float(a) for a in alist]
			stratlist=args[4].split(",")
			if args[5]=="debug":
				experiment(args[0], clist, alpha=alist, n_reps=int(args[3]), strategy=stratlist, debug_mode=True, planted=planted)
			else:
				experiment(args[0], clist, n_reps=int(args[3]), strategy=stratlist, planted=planted)
	else:
		print("Wrong number of arguments: expected 1-5.")
		print("Usage: python [script.py] network [color_list] [alpha] [n_reps] [strat_list] [debug]")

if __name__ == '__main__':
	main()