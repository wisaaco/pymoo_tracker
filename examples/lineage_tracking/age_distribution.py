import pandas as pd
import os

def clean_parents(item):
            item = list(eval(item))
            return pd.Series([item[0], item[1]])


problem_names = ["zdt%i"%i for i in range(1,6)]
algorithm_names = ["SMSEMOA","NSGA2","UNSGA3","CTAEA"]

rel_path = os.path.dirname(__file__) #TODO improve this part
for ia, algorithm in enumerate(algorithm_names):
    print(algorithm)
    for ip, problem in enumerate(problem_names):
        print("\t",problem)

        file_tracker = open(rel_path+"/data/test_tracker_%s_%s.csv"%(algorithm_names[ia],problem_names[ip]), "wb+")

        df = pd.read_csv(file_tracker)
        print(df.head(2))
        print(df.tail(2))

        
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