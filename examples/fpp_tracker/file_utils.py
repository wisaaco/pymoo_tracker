import numpy as np

def parse_file(f, n_obj=2):
    """
    Three possible formats for the columns:
        <ts_date> <ts_time> <generation> <o1> ... <on>
        <generation> <o1> ... <on>
        <o1> ... <on>
    """
    first = f.readline().split()
    columns = len(first)
    lst = first + f.read().split()
    f.close()

    o = [None] * n_obj
    if columns == n_obj:
        generation = []
        for i in range(n_obj):
            o[i] = [float(j) for j in lst[i::columns]]
    elif columns > n_obj:
        # Ignore timestamps
        first_obj = columns - n_obj
        generation = [int(i) for i in lst[first_obj-1::columns]]
        for i in range(n_obj):
            o[i] = [float(j) for j in lst[first_obj+i::columns]]
    else:
        generation = []

    return generation, o

def get_solution_array(f, n_obj=2):
    solutions = []
    generation, o = parse_file(f, n_obj)

    if len(generation) > 0:
        last_gen = generation[-1]

        # To avoid ValueError when calling method index
        generation.append(last_gen+1)
        
        o_slize = [[] for _ in range(n_obj)]
        for g in range(1, last_gen+1):
            try:
                i, j = generation.index(g), generation.index(g+1)
                for k in range(n_obj):
                    o_slize[k] = o[k][i:j]
            except ValueError:
                pass
            solutions.append(np.array(o_slize).T)
    else:
        solutions.append(np.array(o).T)

    return solutions

def solutions_to_string(solutions):
    s = ''
    for o in solutions:
        for o_n in o:
            s += "{} ".format(o_n)
        s += '\n'
    return s

