import pandas as pd
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt 
import networkx as nx
import sys

trace = "data/solutions/ntw_722_040-040-020_C/obj_distance-occupation-ntw_utilization/Replicas040/Genetics/NSGA2_1_200-200_SV0-CV2-MV1_MM0.2-MC0.1-MB0.1_tracker.csv"
trace = "data/solutions/ntw_722_040-040-020_C/obj_distance-occupation-ntw_utilization/Replicas040/Genetics/NSGA2_1_020-010_SV0-CV2-MV1_MM0.2-MC0.1-MB0.1_tracker.csv"
trace = "/Users/isaac/Projects/pymoo_tracker/examples/study_tracker/data/solutions/ntw_722_040-040-020_C/obj_distance-occupation-ntw_utilization/Replicas040/Genetics/NSGA2_1_200-200_SV0-CV2-MV1_MM0.2-MC0.1-MB0.1_tracker.csv"
trace = "/Users/isaac/Projects/pymoo_tracker/examples/study_tracker/data/solutions/ntw_722_040-040-020_C/obj_distance-occupation-ntw_utilization/Replicas040/Genetics/NSGA2_1_010-010_SV0-CV2-MV1_MM0.2-MC0.1-MB0.1_tracker.csv"

population , generations  = trace.split("_")[-4].split("-")
generations = int(generations)
population = int(population)
print(generations, population)


df = pd.read_csv(trace)
print(df.head(2))
print(df.tail(2))

def clean_parents(item):
    item = list(eval(item))
    return pd.Series([item[0], item[1]])

df[["parent1","parent2"]]=df["parent"].apply(clean_parents)

# Reformat values of "isF"
df.isF.fillna(False,inplace=True)
df.loc[df.isF=="true","isF"]=True
df.loc[df.isF=="false","isF"]=False


df["rank"].fillna(0,inplace=True)


mask = df.groupby(by=["idx"],as_index=False)["iter"].agg([min,max])
mask["lifeIter"] = mask["max"]-mask["min"]
dfMutates = df[["idx","mutate"]].drop_duplicates()
dfMutates.fillna(False,inplace=True)
df2 = pd.merge(mask,dfMutates,on=["idx"],how="inner")

dft = pd.merge(df,df2,on=["idx"],how="inner")
dft["name"]=dft["idx"]+"_"+dft["iter"].astype(str)


# Visualization A
### Current configuration: format GML to Cytoscape 

edgesp1 = zip(dft["parent1"],dft["idx"])
edgesp2 = zip(dft["parent2"],dft["idx"])

G1 = nx.Graph()
G1.add_edges_from(edgesp1)
G1.add_edges_from(edgesp2)
G1.remove_node("0")
degree = dict(G1.degree)
pf_att = dict(zip(dft.idx,dft["isF"].astype(bool)))
created = dict(zip(dft.idx,dft["created_gen"]))
ranks_pr = nx.pagerank(G1)
mutate_att = dict(zip(dft.idx,dft["mutate_y"].astype(bool)))
nx.set_node_attributes(G1,values=pf_att,name="isF")
nx.set_node_attributes(G1,values=ranks_pr,name="rank")
nx.set_node_attributes(G1,values=degree,name="degree")
nx.set_node_attributes(G1,values=mutate_att,name="mutate")
nx.set_node_attributes(G1,values=created,name="created")
nx.write_gml(G1, "testA_g%i_p%i_v0.gml"%(generations,population))
# Visualization B
### Current configuration: format GML to Cytoscape 

G = nx.Graph()
# G.add_nodes_from(dft[dft.iter==1]["idx"].values)
G.add_nodes_from(dft.name.values)
print("total number of nodes:",len(G.nodes))
lastGeneration = dict(zip(dft[dft["iter"]==generations-1].name,[True]*population))
iteration_att = dict(zip(dft.name,dft["iter"].astype(int)))
rank_att = dict(zip(dft.name,dft["rank"].astype(int)))
lifeiter_att = dict(zip(dft.name,dft["lifeIter"].astype(int)))
pf_att = dict(zip(dft.name,dft["isF"].astype(bool)))
mutate_att = dict(zip(dft.name,dft["mutate_y"].astype(int)))


nx.set_node_attributes(G,values=iteration_att,name="gen")
nx.set_node_attributes(G,values=rank_att,name="rank")
nx.set_node_attributes(G,values=lifeiter_att,name="lifeIter")
nx.set_node_attributes(G,values=pf_att,name="isF")
nx.set_node_attributes(G,values=lastGeneration,name="lastGen")
nx.set_node_attributes(G,values=mutate_att,name="mutate")



posX, posY = dict(),dict()
ghost = dict()
lastpost = defaultdict(int)

totalNodes = 0
for iter in range(generations+1):
    dftmp = dft[dft["iter"]==iter].sort_values(["lifeIter"],ascending=True)
    totalNodes +=len(dftmp)
    for element in range(len(dftmp)):

        individual = dftmp.iloc[element]
        name = individual["name"]
        id = individual["idx"]
        generated = int(individual["created_gen"])
        
        # # Edge generation
        if generated==iter and iter>1: #tiene ancestors
            p1 = individual["parent1"]
            p2 = individual["parent2"]
            namep1 = p1+"_"+str(iter-1)
            namep2 = p2+"_"+str(iter-1)
            assert namep1 in G.nodes,"Error P1 no existe"
            assert namep2 in G.nodes,"Error P2 no existe"
            G.add_edge(namep1, name)
            G.add_edge(namep2, name)

        if iter > generated: #individuo ya creado
            # Crear nuevo individuo GHOST en la nueva posición
            # Vertical control position
            ghost[name]= True
            posX[name] = posX["%s_%i"%(id,iter-1)]
        else:
            # individuo nuevo
            ghost[name]=False
            # Horizontal control Position 
            if iter == 0 :
                posX[name] = lastpost[iter]
                # lastpost[gen] = element+1
            else: #3 o more generations
                if lastpost[iter]==0:
                    lastpost[iter]= lastpost[iter-1]
                posX[name] = lastpost[iter]
                # lastpost[gen] = element+1
            lastpost[iter] = lastpost[iter]+60
        posY[name] = -float(iter)*50

print("total nodes:",totalNodes)
    
position = {}
for k in posX:
    position[k] = {"x":posX[k],"y":posY[k]}

nx.set_node_attributes(G,values=posX,name="posx")
nx.set_node_attributes(G,values=posY,name="posy")
nx.set_node_attributes(G,values=position,name="graphics")
nx.set_node_attributes(G,values=ghost,name="ghost")


# nx.write_gexf(G, )
nx.write_gml(G, "test_g%i_p%i_v0.gml"%(generations,population))
# cy = nx.readwrite.json_graph.cytoscape_data(G)
# import json
# with open( "test_g%i_p%i_v0.cyjs"%(generations,population), 'w') as f:
#     json.dump(cy, f)