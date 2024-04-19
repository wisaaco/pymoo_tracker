from copy import deepcopy

import numpy as np

from pymoo.core.operator import Operator
from pymoo.core.variable import Real, get


class Mutation(Operator):

    def __init__(self, prob=1.0, prob_var=None, **kwargs) -> None:
        super().__init__(**kwargs)
        self.prob = Real(prob, bounds=(0.7, 1.0), strict=(0.0, 1.0))
        self.prob_var = Real(prob_var, bounds=(0.0, 0.25), strict=(0.0, 1.0)) if prob_var is not None else None

    def do(self, problem, pop, inplace=True, **kwargs):

        # if not inplace copy the population first
        if not inplace:
            pop = deepcopy(pop)

        n_mut = len(pop)

        # get the variables to be mutated
        X = pop.get("X")

        # retrieve the mutation variables
        # Discussion: Does mutation tracking depend on the specific mutation?
        Xp,list_mutations,mut = self._do(problem, X, **kwargs)

        if list_mutations is  None:
            list_mutations = np.zeros(shape=sum(mut))
        
        if kwargs["algorithm"].save_tracker:
            pop[mut].set("X", Xp[mut])
            pop[mut].set("mutate", True)
            pop[mut].set("mutate_rate", list_mutations)
        
        return pop

    def _do(self, problem, X, **kwargs):
        return X

    def get_prob_var(self, problem, **kwargs):
        prob_var = self.prob_var if self.prob_var is not None else min(0.5, 1 / problem.n_var)
        return get(prob_var, **kwargs)
