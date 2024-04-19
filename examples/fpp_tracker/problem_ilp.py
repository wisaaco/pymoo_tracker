import pulp

import numpy as np
import random

class ProblemILP():

    def __init__(self, network, n_replicas=1, l=0.5, verbose=False, o1_max=None, o2_min=None):
        self.network = network
        self.n_replicas = n_replicas

        # Define the constants
        self.N_TASKS = network.getNTasks()
        self.N_USERS = network.getNUsers()
        self.N_NODES = network.getNNodes()

        self.TASK_MEM_ARRAY = network.getTaskMemoryArray()
        self.NODE_MEM_ARRAY = network.getNodeMemoryArray()

        self.undm = network.getUserNodeDistanceMatrix()
        self.tuam = network.getTaskUserAssignmentMatrix()

        self.l = l # lambda for converting bimode to singlemode

        self.verbose = verbose

        # Values needed for normalization TODO: Recalcular y probar otros valores de normalización
        self.f1_min = network.getTasksMinAverageDistanceToUser_v2(undm=self.undm, tuam=self.tuam)
        self.f1_max = network.getTasksMaxAverageDistanceToUser_v2(undm=self.undm, tuam=self.tuam)
        self.f2_min = network.getMinimumNNodesNeeded()
        self.f2_max = self.N_NODES

        if self.f2_min == self.f2_max:
            self.f2_max += 1 
            # This way, we avoid dividing by 0 and enforce 0 as normalized O2
            # value, so it does not interfere with O1 in any way

        # Defining the problem
        self.prob = pulp.LpProblem('TaskNodeAssignmentMatrixBimodeToSinglemodeILP', pulp.LpMinimize)

        # Defining the variables
        self.tnam = pulp.LpVariable.dicts(
                "TaskNodeAssignmentMatrix",
                (range(self.N_TASKS), range(self.N_NODES)),
                cat="Binary")

        self.n_sel = pulp.LpVariable.dicts(
                "SelectedNodeVector",
                range(self.N_NODES),
                cat="Binary")

        if self.n_replicas > 1:
            # Used to implement the minimum operation
            # https://www.fico.com/fico-xpress-optimization/docs/dms2019-04/mipform/dhtml/chap2s1_sec_ssecminval.html
            self.tudm = pulp.LpVariable.dicts(
                    "TaskUserDistanceMatrix",
                    (range(self.N_TASKS), range(self.N_USERS)),
                    cat="Continuous")

            self.d = pulp.LpVariable.dicts(
                    "TaskUserNodeDistanceMinimumSelector",
                    (range(self.N_TASKS), range(self.N_USERS), range(self.N_NODES)),
                    cat="Binary") 

        # Objective function
        self._setObjectiveFunction()

        # Constraints
        self.prob += self._getObjectiveExpression(1) >= self.f2_min

        for t in range(self.N_TASKS):
            self.prob += pulp.lpSum(self.tnam[t]) >= 1
            self.prob += pulp.lpSum(self.tnam[t]) <= self.n_replicas

        if self.n_replicas > 1:
            # Implementation of the minimum operation in ILP for n_replicas > 1
            # https://www.fico.com/fico-xpress-optimization/docs/dms2019-04/mipform/dhtml/chap2s1_sec_ssecminval.html
            max_diff = self.undm.max() - self.undm.min()
            for t in range(self.N_TASKS):
                for u in range(self.N_USERS):
                    self.prob += pulp.lpSum(self.d[t][u]) == 1
                    for n in range(self.N_NODES):
                        self.prob += self.d[t][u][n] <= self.tnam[t][n]
                        self.prob += pulp.lpSum([
                                self.d[t][u][k] * self.undm[u][k] for k in range(self.N_NODES)
                            ]) <= self.undm[u][n] + max_diff * (1 - self.tnam[t][n])

        for t in range(self.N_TASKS):
            for n in range(self.N_NODES):
                self.prob += self.tnam[t][n] <= self.n_sel[n]
                # Ensure that a service is assigned to a selected node

        for n in range(self.N_NODES):
            self.prob += pulp.lpSum(
                    [self.tnam[t][n] for t in range(self.N_TASKS)]
                ) >= self.n_sel[n]
            # Ensure that a node is selected only when there is at least one
            # service assigned

            self.prob += pulp.lpSum(
                    [self.TASK_MEM_ARRAY[t] * self.tnam[t][n]
                        for t in range(self.N_TASKS)]
                ) <= self.NODE_MEM_ARRAY[n]
            # Ensure that node's memory limit is not surpassed


    def _setObjectiveFunction(self):
        #self.prob += self._getObjectiveExpression(0), "Services' average distance to user"
        #self.prob += self._getObjectiveExpression(1), "Number of nodes with at least one task"
        self.prob += \
                self.l       * self._getObjectiveExpressionNormalized(0) + \
                (1 - self.l) * self._getObjectiveExpressionNormalized(1),  \
                "Bi-mode to single-mode execution using lambda"

    def _normalize(self, f1, f2):
        f1_norm = (f1 - self.f1_min) / (self.f1_max - self.f1_min)
        f2_norm = (f2 - self.f2_min) / (self.f2_max - self.f2_min)
        return f1_norm, f2_norm

    def _getTaskAssignedId(self):
        """
        Only works for single-node service assignment.

        Given a tnam:
            x11 x12 x13
            x21 x22 x23
            x31 x32 x33

        Our node selection vector can be retrieved this way:
            0*x11 + 1*x12 + 2*x13
            0*x21 + 1*x22 + 2*x23
            0*x31 + 1*x32 + 2*x33

        Since xij is either 1 or 0 and tasks can only be assigned to a single
        node, we can treat these variables as selection variables, multiplying
        by unit increasing coefficients representing node ids.
        """

        exp_list = []
        for t in range(self.N_TASKS):
            exp_list.append(
                    pulp.lpSum([
                            n * self.tnam[t][n] 
                            for n in range(self.N_NODES)
                        ])
                )
        return exp_list

    def _getTaskUserDistanceMatrix(self):
        tudm = np.empty((self.N_TASKS, self.N_USERS), dtype=object)
        for t in range(self.N_TASKS):
            for u in range(self.N_USERS):
                tudm[t][u] = pulp.LpAffineExpression()
                for n in range(self.N_NODES):
                    if self.n_replicas == 1:
                        tudm[t][u] += self.tnam[t][n] * self.undm[u][n]
                        # Assuming that the service is assigned to a single node
                    else:
                        tudm[t][u] += self.d[t][u][n] * self.undm[u][n]
                        # Else get minimum value contained in array d

        return tudm
    
    def _getTasksAverageDistanceToUser(self):
        tudm = self._getTaskUserDistanceMatrix()
        tud_avg = pulp.LpAffineExpression()

        tua_sum = np.sum(self.tuam, axis=1)
        for t in range(self.N_TASKS):
            tud_avg += pulp.lpSum([
                    self.tuam[t][u] * tudm[t][u] / tua_sum[t]
                    for u in range(self.N_USERS)
                ])

        tud_avg /= self.N_TASKS

        return tud_avg
    
    def _getObjectiveExpression(self, obj_n):
        if   obj_n == 0:
            return self._getTasksAverageDistanceToUser()
        elif obj_n == 1:
            return pulp.lpSum(self.n_sel)

    def _getObjectiveExpressionNormalized(self, obj_n):
        f = self._getObjectiveExpression(obj_n)
        if   obj_n == 0:
            return (f - self.f1_min) / (self.f1_max - self.f1_min)
        elif obj_n == 1:
            return (f - self.f2_min) / (self.f2_max - self.f2_min)

    def solve(self):
        self.prob.solve(pulp.PULP_CBC_CMD(msg=1 if self.verbose else 0))
        self.prob += \
                self.l       * self._getObjectiveExpressionNormalized(0) + \
                (1 - self.l) * self._getObjectiveExpressionNormalized(1) <= \
                pulp.value(self.prob.objective) - 0.000001
        return pulp.LpStatus[self.prob.status]
    
    def getSolutionString(self):
        """Get solution for printing, only after call to method solve"""
        s = "Task/Node Assignment Matrix & Task memory\n"
        for t in range(self.N_TASKS):
            for n in range(self.N_NODES):
                s += '{: <2}'.format("{:.0f}".format(pulp.value(self.tnam[t][n])))
            s += '{: >8}'.format('{:.2f}'.format(self.TASK_MEM_ARRAY[t]))
            s += '\n'
        s += '\n'

        for n in range(self.N_NODES):
            s += '{: <2}'.format("{:.0f}".format(pulp.value(self.n_sel[n])))
        s += '\n'
        s += '\n'

        for n in range(self.N_NODES):
            s += '{: <8}'.format('{:.2f}'.format(self.NODE_MEM_ARRAY[n]))
        s += '\n'

        for n in range(self.N_NODES):
            s += '{: <8}'.format('{:.2f}'.format(pulp.value(pulp.lpSum(
                    [self.TASK_MEM_ARRAY[t] * self.tnam[t][n]
                        for t in range(self.N_TASKS)]
                ))))
        s += '\n'

        return s

    def getSingleModeObjective(self):
        return pulp.value(self.prob.objective)

    def getObjective(self, obj_n):
        return pulp.value(self._getObjectiveExpression(obj_n))

    def getObjectiveNormalized(self, obj_n):
        return pulp.value(self._getObjectiveExpressionNormalized(obj_n))

    def changeLambda(self, l):
        self.l = l
        # TODO: change objective according to this new lambda

if __name__ == '__main__':
    from parameters import configs
    from network import Network
    import pickle

    random.seed(configs.seed)
    ntw = pickle.load(configs.input)

    problem = ProblemILP(ntw, l=configs.lmb)

    problem.solve()

