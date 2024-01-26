import matplotlib.pyplot as plt
import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2,binary_tournament
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.problems import get_problem
from pymoo.core.callback import Callback
from pymoo.optimize import minimize
from pymoo.core.population import Population
from pymoo.util.normalization import denormalize
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.crossover.pntx import PointCrossover, SinglePointCrossover, TwoPointCrossover

import csv
import pandas as pd
import os, inspect

def random_by_bounds(n_var, xl, xu, n_samples=1):
    val = np.random.random((n_samples, n_var))
    return denormalize(val, xl, xu)


def random(problem, n_samples=1):
    return random_by_bounds(problem.n_var, problem.xl, problem.xu, n_samples=n_samples)


class MyCallback(Callback):

    def __init__(self) -> None:
        super().__init__()
        self.data["best"] = []
        self.gen = 0 
        self.updates = 0 
        self.initializes = 0 
        # self.fileCSV = csv.writer(open("test.csv", "wb+"))
        self.path = os.path.dirname(inspect.getfile(self.__class__))

        self.fileCSV = open(self.path+"/test.csv", "wb+")
        
        self.coreDF = pd.DataFrame(columns=["iter","idx","n_gen","n_iter","rank","crowding","parent","mutate","mutate_rate"])
        self.coreDF.to_csv(self.fileCSV, index=False, encoding='utf-8')

    def print_pop(self,algorithm):
        pops = algorithm.pop
        Fs = algorithm.pop.get("F")
        # print(Fs)
        print(Fs.shape)
        data = []
        for pop in pops:
            data.append(pop.data)
        df = pd.json_normalize(data)
        # print(df.head())
        df["iter"] = algorithm.n_iter
        df.to_csv(self.fileCSV,index=False,header=False,columns=["iter","idx","n_gen","n_iter","rank","crowding","parent","mutate","mutate_rate"])


    # def _print(self,who):
    #      print(who)
    #      print("\tUpdate",self.updates)
    #      print("\tinitializes",self.initializes)
    #      print("\tNotify",self.gen)

    def notify(self, algorithm):
        # print(algorithm.mating().n_iter) # control iter.
        # print("NOTIFY:",self.gen)
        # print("\t Mutations: ",algorithm.mating.mutation.pop_mut)
        self.print_pop(algorithm)
        self.gen +=1

    # def update(self,algorithm):
    #     self.updates +=1
    #     self._print("UPDATE")

    # def initialize(self,algorithm):
    #     self.initializes +=1
    #     self._print("INIT")

gens = 200
problem = get_problem("sphere") #n_vars 5 o 10? ?
pop_size = 100

X = random(problem, n_samples=pop_size)

population = Population.new("X", X)
# TODO MOVED to sampling.py
# for ix,individual in enumerate(population):
#     individual.data = {"idx":"1_%i"%ix,"parent":["0","0"],"mutate":np.NaN,"mutate_rate":np.NaN}
#     print(individual.data)


mutation = PM(eta=20)
# rankAndCrowd = RankAndCrowding()


algorithm = NSGA2(
    pop_size= pop_size,
    mutation= mutation,
    sampling = population,
    # survival= rankAndCrowd,
    n_offsprings = 4,
    eliminate_duplicates=False,
    crossover=SBX(eta=15, prob=0.9) # Funciona la identificación de parents
    # crossover=SinglePointCrossover(prob=0.9) # Funciona la identificación de parents
)

res = minimize(problem,
               algorithm,
               ('n_gen', gens),
               seed=1,
               callback=MyCallback(),
               verbose=True)

