import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.termination import get_termination
from pymoo.optimize import minimize

from parameters import configs
from placement_NSGA import MyMutation,MySampling,PlacementProblem,BinaryCrossover
import pickle
from datetime import datetime
import pandas as pd
import os
import inspect
from pymoo.config import Config
from pymoo.core.callback import Callback
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

Config.warnings['not_compiled'] = False

# class MyCallback(Callback):

#     def __init__(self,n_gen) -> None:
#         super().__init__()
#         self.data["best"] = []
#         self.gen = 0 
#         self.max_gen = n_gen
#         self.updates = 0 
#         self.initializes = 0 
#         # self.fileCSV = csv.writer(open("test.csv", "wb+"))
#         self.path = os.path.dirname(inspect.getfile(self.__class__))

#         self.fileCSV = open(self.path+"/test_200g_10p_Pall.csv", "wb+")
#         self.coreDF = pd.DataFrame(columns=["iter","idx","n_gen","n_iter","rank","crowding","parent","mutate","mutate_rate","pf"])
#         self.coreDF.to_csv(self.fileCSV, index=False, encoding='utf-8')

#     def print_pop(self,algorithm):
#         pop = algorithm.pop
#         Fs = algorithm.pop.get("F")

#         # print(algorithm.n_iter)
#         # print(Fs)
#         # print(Fs.shape)
#         data = []

#         # if self.max_gen == algorithm.n_gen:
#         #      print("ultimo callback")
#         #      front = NonDominatedSorting().do(Fs)
#         #      print(len(front[-1]))
#         # # Atención la longitud del frente NO COINCIDE CON el resultado final RES.F

#         for ind in pop:
#             data.append(ind.data)
#         df = pd.json_normalize(data)
#         # print(df.head())
#         print(df.shape)
#         df["iter"] = algorithm.n_iter
#         if "mutate_rate" not in df.columns:
#              df[["mutate_rate","mutate"]] = None
#         if "rank" not in df.columns:
#              df[["rank","crowding"]] = None
#         df.to_csv(self.fileCSV,index=False,header=False,columns=["iter","idx","n_gen","n_iter","rank","crowding","parent","mutate","mutate_rate","pf"])
    
#     def print_pf(self,front,pop):
#         data = []
#         X = pop.get("X")
#         print(X.shape)
#         # print(X)
#         # print(X[0])
#         print(X[0].reshape(500,81))
#         print(np.sum(X[0].reshape(500,81),axis=1))

#         for ix,individual in enumerate(pop):
#             isPF = False
#             if ix<=len(front[-1]):
#                  isPF = True #improved with a vector of T-F
#             individual.set("pf",isPF)
#             data.append(individual.data)
#         df = pd.json_normalize(data)
#         df["iter"] = self.max_gen+1
#         if "mutate_rate" not in df.columns:
#              df[["mutate_rate","mutate"]] = None
#         df.to_csv(self.fileCSV,index=False,header=False,columns=["iter","idx","n_gen","n_iter","rank","crowding","parent","mutate","mutate_rate","pf"])



#     def notify(self, algorithm):
#         # print(algorithm.mating().n_iter) # control iter.
#         # print("NOTIFY:",self.gen)
#         # print("\t Mutations: ",algorithm.mating.mutation.pop_mut)
#         self.print_pop(algorithm)
#         self.gen +=1

def main():
    np.random.seed(configs.np_seed_train)
    
    ## DEBUG
    configs.name ="500v9"
    configs.n_devices = 499
    configs.n_jobs = 9
    configs.n_tasks = 81
    configs.n_gen = 10
    ###

    path_dt = 'examples/fpp/dt_TEST_%s_%i_%i.npz'%(configs.name,configs.n_jobs,configs.n_devices)
    dataset = np.load(path_dt)
    dataset = [dataset[key] for key in dataset]
    data = []
    for sample in range(len(dataset[0])):
        data.append((dataset[0][sample],
                     dataset[1][sample],
                     dataset[2][sample],
                     ))
    



    algorithm = NSGA2(
        pop_size=10,
        sampling=MySampling(),
        crossover=BinaryCrossover(), 
        mutation=MyMutation(prob=0.25), 
        eliminate_duplicates=True,
        save_history = False
    )

    termination = get_termination("n_gen", configs.n_gen)

    rel_path = os.path.dirname(__file__)
    
    
    for i, sample  in enumerate(data):
        if i == 1: break
        
        print("Running episode: %i"%(i+1))
        times, adj, feat = sample
        problem = PlacementProblem(n_var=(configs.n_devices+1)*configs.n_tasks,
                                   n_objectives=2, #TODO fix 2 funciones objetivos
                                   time=times,
                                   adj=adj,
                                   featHW=feat,
                                   n_devices=configs.n_devices,
                                   n_tasks=configs.n_tasks) 

        sttime = datetime.now().replace(microsecond=0)
        
        res = minimize(problem,
                    algorithm,
                    termination,
                    seed=1,
                    save_history = True,
                    save_tracker = True,
                    verbose=True)
        

        file_tracker = open(rel_path+"/test_tracker_%i.csv"%i, "wb+")
        coreDF = pd.DataFrame(columns=algorithm.record_tracker_columns)
        coreDF.to_csv(file_tracker, index=False, encoding='utf-8')
        for igen in range(len(res.history_pop)):
            # curr_sol = res.history[igen].opt.get('F')
            f_idx_objectives = list()
            for o in res.history[igen].opt:
                 f_idx_objectives.append(o.data["idx"])
                
            data = []
            for ind in res.history_pop[igen]:
                indi = ind.data 
                if indi["idx"] in f_idx_objectives:
                      indi["isF"] = True
                else:
                     indi["isF"] = False
                data.append(indi)

            df = pd.json_normalize(data)
            df["gen"] = igen #move line
            if "mutate_rate" not in df.columns: #TODO This logic should be controlleld in the long
                df[["mutate_rate","mutate"]] = None
            if "rank" not in df.columns:
                df[["rank","crowding"]] = None
            if "isPF" not in df.columns:
                 df[["isF"]] = False
            df.to_csv(file_tracker,index=False,header=False,columns=algorithm.record_tracker_columns)


        ettime = datetime.now().replace(microsecond=0)

        # print(res.F)
        front = NonDominatedSorting().do(res.F)

        # Imprimir las soluciones dominantes de la última generación
        print("Soluciones Dominantes de la Última Generación:")
        print(len(front[-1]))

        # for i in front[-1]:
        #     print(i)
        #     pop = res.pop
        #     if (res.X[i] == pop.get("X")[i]).all():
        #         print("HERE")
        
        # res.algorithm.callback.print_pf(front,res.pop)

        convergence = [res.history[i].result().f for i in range(len(res.history))]
        exec_time = [res.history[i].result().exec_time for i in range(len(res.history))]
        ct = zip(convergence,exec_time)
        # print(convergence)
        with open('logs/log_ga_pf_P10_Pall_convergence'+ str(configs.name) + "_" + str(configs.n_jobs) + '_' + str(configs.n_devices)+'_%i.pkl'%i, 'wb') as f:
                    pickle.dump(ct, f)


        log_pf = []
        for pf in res.F:
            log_pf.append([i,pf[0],pf[1],(ettime-sttime)])

        with open('logs/log_ga_pf_'+ str(configs.name) + "_" + str(configs.n_jobs) + '_' + str(configs.n_devices)+'_%i.pkl'%i, 'wb') as f:
                pickle.dump(log_pf, f)
                
        print('\tEpisode {}\t Len PF: {}\t'.format(i + 1, len(res.F)))
        print("\t\t time: ",(ettime-sttime))
        
        

if __name__ == '__main__':
    print("NSGAII-strategy")
    start_time = datetime.now().replace(microsecond=0)
    print("Started training: ", start_time)
    print("="*30)
    main()
    end_time = datetime.now().replace(microsecond=0)
    print("Finish training: ", end_time)
    print("Total time: ",(end_time-start_time))
    print("Done.")
