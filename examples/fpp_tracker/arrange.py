import numpy as np
import random
from file_utils import parse_file

# TODO: Arreglar para n dimensiones
def get_pareto_front_from_array(array):
    size  = array.shape[0]
    n_obj = array.shape[1]
    array_assigned = [True] * size

    for i in range(size):
        if not array_assigned[i]:
            continue

        for j in range(size):
            if not array_assigned[j] or i == j:
                continue

            all_o = True
            for obj in range(n_obj):
                if array[i,obj] > array[j,obj]:
                    all_o = False
                    break
            # all_o = i[o1] < j[o1] AND
            #         i[o2] < j[o2] AND
            #              ...      AND
            #         i[on] < j[on]

            if all_o: array_assigned[j] = False

    return array[array_assigned]

# OLD ALGORITHM
#def get_pareto_front_from_array(array, n_obj=2):
#    idx_sorted = np.lexsort((array[:,1], array[:,0]))
#    # Investigar si para una misma X ordena también la Y de menor a mayor
#    array_sorted = array[idx_sorted]
#
#    min_y = 1000.
#    pareto = []
#    for p in array_sorted:
#        if p[1] <  min_y:
#            pareto.append((p[0],p[1]))
#            min_y = p[1]
#
#    return pareto

def get_pareto_front_from_files(configs):
    n_obj = configs.n_objectives
    solutions = [[] for _ in range(n_obj)]
    for f in configs.input:
        generation, o = parse_file(f, n_obj)

        if o is not None and o[0] is not None:
            # Generation should preferably be an empty list
            if generation:
                # Otherwise, only get the values from the last generation
                last_idx = generation.index(generation[-1])
            else:
                last_idx = 0

            for i in range(n_obj):
                solutions[i] += o[i][last_idx:]

    return get_pareto_front_from_array(np.array(solutions).T)




if __name__ == '__main__':
    from parameters import configs
    random.seed(configs.seed)

    array = np.array([[3,3,5], [3,3,4],[2,3,4],[4,3,4],[3,2,4],[3,4,4], [3,1,3],[2,2,3],[3,2,3],[4,2,3],[1,3,3],[2,3,3],[3,3,3],[4,3,3],[5,3,3],[2,4,3],[3,4,3],[4,4,3],[3,5,3], [3,2,2],[2,3,2],[3,3,2],[4,3,2],[3,4,2], [3,3,1]])

    print(get_pareto_front_from_array(array))


