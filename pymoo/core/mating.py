import math

from pymoo.core.infill import InfillCriterion


class Mating(InfillCriterion):

    def __init__(self,
                 selection,
                 crossover,
                 mutation,
                 **kwargs):

        super().__init__(**kwargs)
        self.selection = selection
        self.crossover = crossover
        self.mutation = mutation

    def _do(self, problem, pop, n_offsprings, parents=None, **kwargs):

        # how many parents need to be select for the mating - depending on number of offsprings remaining
        n_matings = math.ceil(n_offsprings / self.crossover.n_offsprings)

        # if the parents for the mating are not provided directly - usually selection will be used
        if parents is None:

            # select the parents for the mating - just an index array
            parents = self.selection(problem, pop, n_matings, n_parents=self.crossover.n_parents, **kwargs)

        # do the crossover using the parents index and the population - additional data provided if necessary
        off = self.crossover(problem, parents, **kwargs)

        #TODO remove
        # print(self.__class__)
        # print(len(off))
        # print("OFFSPRINGS")
        # for ind in off:
        #     print(ind.data)
        #     print(ind.get("X"))


        # do the mutation on the offsprings created through crossover
        off = self.mutation(problem, off, **kwargs)


        # #TODO remove
        # print(self.__class__)
        # print(len(off))
        # print("post MUTATIONS")
        # for ind in off:
        #     print(ind.data)


        return off



