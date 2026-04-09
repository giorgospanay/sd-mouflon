import networkx as nx
import pickle

raw_path="../data/raw"
obj_path="../data/obj"

# Processes SNAP files for Pokec social network. Generates NX networks for age/gender
def process_pokec():
	# Edgelist: soc-pokec-relationships.txt, attributes: soc-pokec-profiles.txt
	# Data: user_id col #0, gender col #3, age col #7 
	# (Age:: median is 23 --after removing 0's only--)
	prof_lines=[]
	rel_lines=[]

	with open(f"{raw_path}/pokec/soc-pokec-profiles.txt","r") as f_prof:
		prof_lines=[line.rstrip().split("\t") for line in f_prof]

	with open(f"{raw_path}/pokec/soc-pokec-relationships.txt","r") as f_rels:
		rel_lines=[line.rstrip().split("\t") for line in f_rels]

	# Create pokec-g:: split into male (1) -- blue, female (0) -- red
	net_g=nx.Graph()
	# Create pokec-a:: split age into median, blue <= 23 / red > 23. (Ignore 0)
	# 0:: no age set, disregard.
	net_a=nx.Graph()
	# Add all nodes to pokec-a, pokec-g
	for u in prof_lines:
		# pokec-g
		if u[3]!="null":
			# Add node: 
			if int(u[3])==1:
				net_g.add_node(int(u[0]),color="blue")
			else:
				net_g.add_node(int(u[0]),color="red")
		# pokec-a
		if u[7]!="null" and int(u[7])!=0:
			# Add node: 
			if int(u[7])<=23:
				net_a.add_node(int(u[0]),color="blue")
			else:
				net_a.add_node(int(u[0]),color="red")

	# Add all edges to pokec-a, pokec-g
	for ln in rel_lines:
		# pokec-g
		if int(ln[0]) in net_g and int(ln[1]) in net_g:
			# Add edge
			net_g.add_edge(int(ln[0]),int(ln[1]),weight=1.0)
		# pokec-a
		if int(ln[0]) in net_a and int(ln[1]) in net_a:
			# Add edge
			net_a.add_edge(int(ln[0]),int(ln[1]),weight=1.0)

	# Save objects
	with open(f"{obj_path}/pokec-g.nx","wb") as pg_out:
		pickle.dump(net_g,pg_out)
	with open(f"{obj_path}/pokec-a.nx","wb") as pa_out:
		pickle.dump(net_a,pa_out)


# Process deezer
def process_deezer():
	# Edgelist: deezer_europe_edges.csv, color: deezer_europe_target.csv
	edges_lines=[]
	color_lines=[]

	with open(f"{raw_path}/deezer_europe/deezer_europe_edges.csv","r") as f_edges:
		edges_lines=[line.rstrip().split(",") for line in f_edges]

	with open(f"{raw_path}/deezer_europe/deezer_europe_target.csv","r") as f_color:
		color_lines=[line.rstrip().split(",") for line in f_color]


	# Create deezer network
	net_deezer=nx.Graph()
	# Add nodes and colors:
	for i,ln in enumerate(color_lines):
		# Ignore header line: id,target
		if i==0: continue
		if int(ln[1])==0:
			net_deezer.add_node(int(ln[0]),color="red")
		else:
			net_deezer.add_node(int(ln[0]),color="blue")

	# Add edges:
	for i,ln in enumerate(edges_lines):
		# Ignore header line: node_1,node_2
		if i==0: continue
		net_deezer.add_edge(int(ln[0]),int(ln[1]),weight=1.0)


	# Save object
	with open(f"{obj_path}/deezer.nx","wb") as deez_out:
		pickle.dump(net_deezer,deez_out)


# Process twitch
def process_twitch():
	# Edgelist: large_twitch_edges.csv, color: large_twitch_features.csv
	edges_lines=[]
	color_lines=[]

	with open(f"{raw_path}/twitch_gamers/large_twitch_edges.csv","r") as f_edges:
		edges_lines=[line.rstrip().split(",") for line in f_edges]

	with open(f"{raw_path}/twitch_gamers/large_twitch_features.csv","r") as f_color:
		color_lines=[line.rstrip().split(",") for line in f_color]


	# Create twitch network
	net_twitch=nx.Graph()
	# Add nodes and colors:
	for i,ln in enumerate(color_lines):
		# Ignore header line: 
		## views,mature,life_time,created_at,updated_at,numeric_id,dead_account,language,affiliate
		if i==0: continue
		if int(ln[1])==0:
			# Mature=0 (low maturity streams) --red
			net_twitch.add_node(int(ln[5]),color="red")
		else:
			net_twitch.add_node(int(ln[5]),color="blue")

	# Add edges:
	for i,ln in enumerate(edges_lines):
		# Ignore header line: node_1,node_2
		if i==0: continue
		net_twitch.add_edge(int(ln[0]),int(ln[1]),weight=1.0)


	# Save object
	with open(f"{obj_path}/twitch.nx","wb") as tw_out:
		pickle.dump(net_twitch,tw_out)



def process_facebook():
	# Edgelist: facebook_combined.txt
	# Features: all files *.feat (names in .names)

	edges_lines=[]

	with open(f"{raw_path}/facebook/facebook_combined.txt","r") as f_edges:
		edges_lines=[line.rstrip().split(" ") for line in f_edges]


	# Create facebook network
	net_fb=nx.Graph()
	## Open feature files manually. If first gender attb==1, make red
	# 0.feat-- gender 77,78
	with open(f"{raw_path}/facebook/0.egofeat","r") as f_color:
		color_lines=[line.rstrip().split(" ") for line in f_color]
		for ln in color_lines:
			if int(ln[78])==1:
				net_fb.add_node(0,color="red")
			else:
				net_fb.add_node(0,color="blue")
	with open(f"{raw_path}/facebook/0.feat","r") as f_color:
		color_lines=[line.rstrip().split(" ") for line in f_color]
		for ln in color_lines:
			if int(ln[78])==1:
				net_fb.add_node(int(ln[0]),color="red")
			else:
				net_fb.add_node(int(ln[0]),color="blue")
	# 107.feat-- gender 264,265
	with open(f"{raw_path}/facebook/107.egofeat","r") as f_color:
		color_lines=[line.rstrip().split(" ") for line in f_color]
		for ln in color_lines:
			if int(ln[265])==1:
				net_fb.add_node(107,color="red")
			else:
				net_fb.add_node(107,color="blue")
	with open(f"{raw_path}/facebook/107.feat","r") as f_color:
		color_lines=[line.rstrip().split(" ") for line in f_color]
		for ln in color_lines:
			if int(ln[265])==1:
				net_fb.add_node(int(ln[0]),color="red")
			else:
				net_fb.add_node(int(ln[0]),color="blue")
	# 348 -- gender 86,87
	with open(f"{raw_path}/facebook/348.egofeat","r") as f_color:
		color_lines=[line.rstrip().split(" ") for line in f_color]
		for ln in color_lines:
			if int(ln[87])==1:
				net_fb.add_node(348,color="red")
			else:
				net_fb.add_node(348,color="blue")
	with open(f"{raw_path}/facebook/348.feat","r") as f_color:
		color_lines=[line.rstrip().split(" ") for line in f_color]
		for ln in color_lines:
			if int(ln[87])==1:
				net_fb.add_node(int(ln[0]),color="red")
			else:
				net_fb.add_node(int(ln[0]),color="blue")
	# 414 -- gender 63,64
	with open(f"{raw_path}/facebook/414.egofeat","r") as f_color:
		color_lines=[line.rstrip().split(" ") for line in f_color]
		for ln in color_lines:
			if int(ln[64])==1:
				net_fb.add_node(414,color="red")
			else:
				net_fb.add_node(414,color="blue")
	with open(f"{raw_path}/facebook/414.feat","r") as f_color:
		color_lines=[line.rstrip().split(" ") for line in f_color]
		for ln in color_lines:
			if int(ln[64])==1:
				net_fb.add_node(int(ln[0]),color="red")
			else:
				net_fb.add_node(int(ln[0]),color="blue")
	# 686 -- gender 41,42
	with open(f"{raw_path}/facebook/686.egofeat","r") as f_color:
		color_lines=[line.rstrip().split(" ") for line in f_color]
		for ln in color_lines:
			if int(ln[42])==1:
				net_fb.add_node(686,color="red")
			else:
				net_fb.add_node(686,color="blue")
	with open(f"{raw_path}/facebook/686.feat","r") as f_color:
		color_lines=[line.rstrip().split(" ") for line in f_color]
		for ln in color_lines:
			if int(ln[42])==1:
				net_fb.add_node(int(ln[0]),color="red")
			else:
				net_fb.add_node(int(ln[0]),color="blue")
	# 698 -- gender 26,27
	with open(f"{raw_path}/facebook/698.egofeat","r") as f_color:
		color_lines=[line.rstrip().split(" ") for line in f_color]
		for ln in color_lines:
			if int(ln[27])==1:
				net_fb.add_node(698,color="red")
			else:
				net_fb.add_node(698,color="blue")
	with open(f"{raw_path}/facebook/698.feat","r") as f_color:
		color_lines=[line.rstrip().split(" ") for line in f_color]
		for ln in color_lines:
			if int(ln[27])==1:
				net_fb.add_node(int(ln[0]),color="red")
			else:
				net_fb.add_node(int(ln[0]),color="blue")
	# 1684 -- gender 147,148
	with open(f"{raw_path}/facebook/1684.egofeat","r") as f_color:
		color_lines=[line.rstrip().split(" ") for line in f_color]
		for ln in color_lines:
			if int(ln[148])==1:
				net_fb.add_node(1684,color="red")
			else:
				net_fb.add_node(1684,color="blue")
	with open(f"{raw_path}/facebook/1684.feat","r") as f_color:
		color_lines=[line.rstrip().split(" ") for line in f_color]
		for ln in color_lines:
			if int(ln[148])==1:
				net_fb.add_node(int(ln[0]),color="red")
			else:
				net_fb.add_node(int(ln[0]),color="blue")
	# 1912 -- gender 259,260
	with open(f"{raw_path}/facebook/1912.egofeat","r") as f_color:
		color_lines=[line.rstrip().split(" ") for line in f_color]
		for ln in color_lines:
			if int(ln[260])==1:
				net_fb.add_node(1912,color="red")
			else:
				net_fb.add_node(1912,color="blue")
	with open(f"{raw_path}/facebook/1912.feat","r") as f_color:
		color_lines=[line.rstrip().split(" ") for line in f_color]
		for ln in color_lines:
			if int(ln[260])==1:
				net_fb.add_node(int(ln[0]),color="red")
			else:
				net_fb.add_node(int(ln[0]),color="blue")
	# 3437 -- gender 117,118
	with open(f"{raw_path}/facebook/3437.egofeat","r") as f_color:
		color_lines=[line.rstrip().split(" ") for line in f_color]
		for ln in color_lines:
			if int(ln[118])==1:
				net_fb.add_node(3437,color="red")
			else:
				net_fb.add_node(3437,color="blue")
	with open(f"{raw_path}/facebook/3437.feat","r") as f_color:
		color_lines=[line.rstrip().split(" ") for line in f_color]
		for ln in color_lines:
			if int(ln[118])==1:
				net_fb.add_node(int(ln[0]),color="red")
			else:
				net_fb.add_node(int(ln[0]),color="blue")
	# 3980 -- gender 19,20
	with open(f"{raw_path}/facebook/3980.egofeat","r") as f_color:
		color_lines=[line.rstrip().split(" ") for line in f_color]
		for ln in color_lines:
			if int(ln[20])==1:
				net_fb.add_node(3980,color="red")
			else:
				net_fb.add_node(3980,color="blue")
	with open(f"{raw_path}/facebook/3980.feat","r") as f_color:
		color_lines=[line.rstrip().split(" ") for line in f_color]
		for ln in color_lines:
			if int(ln[20])==1:
				net_fb.add_node(int(ln[0]),color="red")
			else:
				net_fb.add_node(int(ln[0]),color="blue")
	
	# Add edges
	for ln in edges_lines:
		net_fb.add_edge(int(ln[0]),int(ln[1]),weight=1.0)

	# Save object
	with open(f"{obj_path}/facebook.nx","wb") as fb_out:
		pickle.dump(net_fb,fb_out)


def process_proximity():
	# Edgelist: edges.csv, color: nodes.csv
	edges_lines=[]
	color_lines=[]

	with open(f"{raw_path}/proximity/edges.csv","r") as f_edges:
		edges_lines=[line.rstrip().split(",") for line in f_edges]

	with open(f"{raw_path}/proximity/nodes.csv","r") as f_color:
		color_lines=[line.rstrip().split(",") for line in f_color]


	# Create proximity networks (class - 4 colors)
	net_prox_c=nx.Graph()
	# Add nodes and colors:
	for i,ln in enumerate(color_lines):
		# Ignore header line: 
		## index, id, class, gender, _pos
		if i==0: continue

		n_ind=int(ln[0])
		n_class=ln[2]

		if "2BIO" in n_class:
			# Red: class = 2BIO
			net_prox_c.add_node(n_ind,color="red")
		elif "PC" in n_class:
			# Blue: class = PC
			net_prox_c.add_node(n_ind,color="blue")
		elif "PSI" in n_class:
			# Green: class = PSI
			net_prox_c.add_node(n_ind,color="green")
		elif "MP" in n_class:
			# Orange: class = MP
			net_prox_c.add_node(n_ind,color="orange")

	# Add edges:
	for i,ln in enumerate(edges_lines):
		# Ignore header line: 
		## source, target, time
		if i==0: continue

		n1_ind=int(ln[0])
		n2_ind=int(ln[1])

		# Add edges ignoring time. Ignore duplicate edges
		if not net_prox_c.has_edge(n1_ind,n2_ind):
			net_prox_c.add_edge(n1_ind,n2_ind,weight=1.0)


	# Save object
	with open(f"{obj_path}/prox_c.nx","wb") as tw_out:
		pickle.dump(net_prox_c,tw_out)



def _find_median_age_pokec():
	import pandas

	pokec_profiles=pandas.read_csv(f"{raw_path}/pokec/soc-pokec-profiles.txt",sep="\t",header=None)
	print(pokec_profiles)
	print(pokec_profiles.columns)

	pokec_profiles=pokec_profiles.iloc[:,[0,7]]
	pokec_profiles.dropna(inplace=True)
	pokec_profiles=pokec_profiles.loc[(pokec_profiles.iloc[:,1]!=0)]

	print(pokec_profiles)
	print(pokec_profiles.iloc[:,1].median())
	print(pokec_profiles.iloc[:,1].min())
	print(pokec_profiles.iloc[:,1].max())





if __name__ == '__main__':
	### Processing raw data to NetworkX objects with node color attributes

	# process_facebook()
	# process_deezer()
	# process_twitch()
	# process_pokec()
	# process_proximity()




	### Testing network imports
	# fb=None
	# with open(f"{obj_path}/facebook.nx","rb") as g_open:
	# 	fb=pickle.load(g_open)
	# print(f"Facebook N={fb.number_of_nodes()}, M={fb.number_of_edges()}")

	# deezer=None
	# with open(f"{obj_path}/deezer.nx","rb") as g_open:
	# 	deezer=pickle.load(g_open)
	# print(f"Deezer N={deezer.number_of_nodes()}, M={deezer.number_of_edges()}")

	# twitch=None
	# with open(f"{obj_path}/twitch.nx","rb") as g_open:
	# 	twitch=pickle.load(g_open)
	# print(f"Twitch N={twitch.number_of_nodes()}, M={twitch.number_of_edges()}")

	# pokec_g=None
	# pokec_a=None
	# with open(f"{obj_path}/pokec-g.nx","rb") as g_open:
	# 	pokec_g=pickle.load(g_open)
	# with open(f"{obj_path}/pokec-a.nx","rb") as a_open:
	# 	pokec_a=pickle.load(a_open)
	# print(f"Pokec-g N={pokec_g.number_of_nodes()}, M={pokec_g.number_of_edges()}")
	# print(f"Pokec-a N={pokec_a.number_of_nodes()}, M={pokec_a.number_of_edges()}")

	# prox_c=None
	# with open(f"{obj_path}/prox_c.nx","rb") as g_open:
	# 	prox_c=pickle.load(g_open)
	# print(f"Proximity-c N={prox_c.number_of_nodes()}, M={prox_c.number_of_edges()}")

