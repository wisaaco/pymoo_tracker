import networkx as nx
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from hashlib import md5
from tree import TreeNode

import random

from ntw_functions import barabasi_albert_weighted_graph, get_pareto_distribution, truncate_array
from ntw_classes import Node, User, Task, Link

from default import OBJ_LIST, RANDOM_KEYS, NODE_CPU_PW_MODEL_LIST

# CLASSES
# ==============================================================================
class Network:
    """
    This class will manage the network with all the tasks, users, nodes and
    the graph resulting of the connection between these nodes. It will also
    give valuable information needed for the optimizations.
    """

    def generateRandomObjectDictionary(self):
        # Using random libary
        self.rnd = {
                key: random.Random("{}:{}".format(self.seed, key))
                for key in RANDOM_KEYS
            }

        # Using NumPy's random
        self.np_rnd = {
                key: np.random.default_rng(
                        int.from_bytes(
                            "{}:{}".format(self.seed, key).encode(),
                            'little'
                        )
                        # Only accepts positive integer
                    )
                for key in RANDOM_KEYS
            }

    def __init__(self, conf):
        self.seed = conf.seed
        random.seed(conf.seed)
        np.random.seed(conf.seed)

        # Random object dictionary for each set of operations
        self.generateRandomObjectDictionary()
        
        # Generate the graph
        self.graph = barabasi_albert_weighted_graph(
                seed=conf.seed,
                n=conf.n_nodes,
                m=conf.edges,
                maxw=conf.max_weight,
                minw=conf.min_weight,
                rnd=self.rnd['graph_weights'])

        # Betweenness centrality
        btw_cnt = nx.betweenness_centrality(self.graph, seed=conf.seed, weight='weight')
        # https://networkx.org/documentation/networkx-1.10/reference/generated/networkx.algorithms.centrality.betweenness_centrality.html
        bc_sorted = sorted(btw_cnt.items(), reverse=True, key=lambda e: e[1])

        # Memory for each node
        self.memory = truncate_array(
            get_pareto_distribution(
                conf.node_memory_pareto_shape, 
                conf.n_nodes,
                conf.node_memory_choice[0],
                rng_obj=self.np_rnd['node_memory']),
            step_array=np.array(
                conf.node_memory_choice))

        # CPU for each node
        self.cpu = truncate_array(
            get_pareto_distribution(
                conf.node_n_cpus_pareto_shape, 
                conf.n_nodes,
                conf.node_n_cpus_choice[0],
                rng_obj=self.np_rnd['node_cpu']),
            step_array=np.array(
                conf.node_n_cpus_choice))

        # Specs ratio array
        self.specs_array = np.array((self.memory, self.cpu), dtype=np.uint32).T
        self.specs_ratio = \
                0.5 * (self.memory - min(conf.node_memory_choice)) \
                    / (max(conf.node_memory_choice) - min(conf.node_memory_choice)) \
                    + \
                0.5 * (self.cpu - min(conf.node_n_cpus_choice)) \
                    / (max(conf.node_n_cpus_choice) - min(conf.node_n_cpus_choice))

        self.specs_array_sorted = self.specs_array[np.lexsort((self.specs_ratio,))][::-1]

        self.memory = np.sort(self.memory)[::-1]
        self.cpu    = np.sort(self.cpu   )[::-1]

        self.nodes = [
            Node(
                max_tasks = conf.node_max_tasks_choice[-1],
                min_power = self.np_rnd['node_power'].choice(conf.node_min_pw_choice),
                cpu_power_ratio = self.np_rnd['node_power'].choice(
                    conf.node_cpu_pw_r_choice),
                cpu_power_model = self.rnd['node_power'].randrange(
                    len(NODE_CPU_PW_MODEL_LIST)),
                mem_power_ratio = self.np_rnd['node_power'].choice(
                    conf.node_mem_pw_r_choice)
            ) for _ in range(conf.n_nodes)]

        # Assign memory to each node depending on its centrality giving more
        # memory to the nodes that have more betweenness centrality
        #np.random.shuffle(self.memory)
        for i, _ in bc_sorted:
            self.nodes[i].memory, self.nodes[i].cpus = self.specs_array_sorted[i]

        # Generate the management data for the services while also randomly
        # assigning these services to a single user. Servers for these tasks
        # will be assigned case by case depending on the optimization tasks.
        self.tasks = [
            Task(
                memory = round(
                    get_pareto_distribution(
                        conf.task_memory_pareto_shape,
                        1,
                        conf.task_min_memory,
                        rng_obj=self.np_rnd['task_memory']
                    ).clip(max=conf.task_max_memory)[0], 2),
                user_id = 0, # TODO: remove
                cpu_usage = round(
                    get_pareto_distribution(
                        conf.task_cpu_usage_pareto_shape,
                        1,
                        conf.task_min_cpu_usage,
                        rng_obj=self.np_rnd['task_cpu_usage']
                    ).clip(max=conf.task_max_cpu_usage)[0], 3)
            ) for _ in range(conf.n_tasks)]

        # Generate the management data for the users
        self.USER_REQUEST_SIZE = conf.user_request_size
        self.users = [User(
                pps = self.rnd['user_data'].uniform(
                    conf.user_min_pps, conf.user_max_pps)
            ) for _ in range(conf.n_users)]
        self._addUsers(conf.min_weight, conf.max_weight)

        # Link between devices data for the network
        self._generateLinkData(conf.edge_min_bandwidth, conf.edge_max_bandwidth)

        # Generate the user access to each service. A user can access many
        # services and a service can be accessed by many users, so the resulting
        # datastructure is a set of pairs of tasks/users.
        if conf.communities:
            """    
            Given N_NODES = 32 and GROUP_SIZE = 4:

                           Tree:       Partitions:

                            32           
                          /    \            1 +
                       16        16
                      /  \      /  \        2 +
                     8    8    8    8
                    / \  / \  / \  / \      4 +
                    4 4  4 4  4 4  4 4

                n := "Depth of tree" = log2(N_NODES/GROUP_SIZE)

                  -> 2^0 + 2^1 + ... + 2^(n-1) = (2^n)-1

                partitions = (2^log2(N_NODES/GROUP_SIZE))-1 = (N_NODES/GROUP_SIZE)-1

            """
            partitions = int(np.floor(conf.n_nodes/conf.group_size)) - 1
            self.tree = self._getCommunityTree(partitions)

            tu_prob_matrix = self._getTaskUserRequestProbabilityMatrix(
                    popularity=conf.popularity, spreadness=conf.spreadness)

            tu_matrix = np.zeros((conf.n_tasks, conf.n_users), dtype=np.uint8)
            for tid in range(conf.n_tasks):
                for uid in range(conf.n_users):
                    if self.rnd['tu_assignment'].random() < tu_prob_matrix[tid,uid]:
                        tu_matrix[tid,uid] = 1

        else:
            tu_matrix = self._generateTaskUserAssignmentMatrix_v2(p=conf.probability)

        self.task_user = np.transpose(np.nonzero(tu_matrix))

    def _addUsers(self, minw=0., maxw=1., roundw=1):
        """
        Add users as nodes of a graph. These nodes act in a different way than
        the server nodes. The users' ids start after last server's id.
        """
        n_nodes = len(self.nodes)
        n_users = len(self.users)

        # Assign a probability to each node depending on its degree of
        # centrality, giving more probability to nodes with less centrality
        bc = self.getBetweennessCentrality()
        maxv = max(bc.values())
        bc_prob = [(k, (maxv - v)/(maxv*n_nodes - maxv)) for (k, v) in bc.items()]

        for uid in range(n_users):
            # Get node's id given a random float
            acc_prob = 0
            p = self.rnd['graph_nodes'].random()
            for i in range(n_nodes):
                acc_prob += bc_prob[i][1]
                if p < acc_prob:
                    break

            nid = bc_prob[i][0]

            self.graph.add_node(n_nodes + uid)
            self.graph.add_edge(n_nodes + uid, nid,
                    weight=round(self.rnd['graph_weights'].uniform(minw, maxw), roundw))
            self.users[uid].node_id = nid

    def _generateTaskUserAssignmentMatrix_v1(self):
        """
        Old method: this randomly generates the matrix with 0.5 probability
        for each matrix cell to be 0 or 1 and retries until every service is
        requested by at least one user and every user requests at least one
        service. Less efficient than the second version.
        """
        tu_matrix = self.np_rnd['tu_assignment'].choice(
                [0,1], (len(self.tasks), len(self.users)))
        while not np.all(np.any(tu_matrix, 0)) or not np.all(np.any(tu_matrix, 1)):
            tu_matrix = self.np_rnd['tu_assigment'].choice(
                    [0,1], (len(self.tasks), len(self.users)))
            # Ensure that each user requests at least one task and each tasks is
            # requested by at least one user.
        return tu_matrix

    def _generateTaskUserAssignmentMatrix_v2(self, p=0.5):
        """
        DESCRIPTION OF THE PROCESS:
            1. Generate the Eye matrix (n×m):

                1 0 0 0 0
                0 1 0 0 0
                0 0 1 0 0
                0 0 0 1 0
                0 0 0 0 1
                0 0 0 0 0
                0 0 0 0 0
                0 0 0 0 0

            2. If rows > cols, then place a random one each row (after the
                square submatrix), else if cols > rows, do same on each column:

                1 0 0 0 0
                0 1 0 0 0
                0 0 1 0 0    Identity submatrix
                0 0 0 1 0
                0 0 0 0 1
                --------------------------------------------
                0 0 0 0 1
                1 0 0 0 0    Extra rows (or cols if m > n)
                0 1 0 0 0 

                - Notice: number of ones is equal to n

            3. Shuffle rows (shuffle cols if m > n)
                
                0 1 0 0 0
                0 0 0 0 1
                0 1 0 0 0
                1 0 0 0 0
                0 0 0 1 0
                0 0 0 0 1
                1 0 0 0 0
                0 0 1 0 0

            4. Add extra ones. Requires to calculate a new probability based
                on the amount of ones that are already placed.

                - Let p   := probability that an user requests a service
                - Let p_c := new probability given that n services are already
                             assigned (or m if m > n)
                - Let a   := remaining assignments needed

                If we treat p as the proportion of ones, we have:

                    n*m*p = n + a
                    n*m*p - n = a
                    n(m*p - 1) = a

                Then, we iterate over the zeros of the matrix and assign ones
                given the following probability:

                    p_c = a/(n*m - n) = n(m*p - 1) / n(m-1) = (m*p-1)/(m-1)

                If m > n, we change m for n

        """
        # Eye matrix
        n, m = len(self.tasks), len(self.users)
        tu_matrix = self._generateTaskUserMinimalAssignmentMatrix() # Steps 1-3

        if n >= m:
            if m == 1:
                return tu_matrix
            p_c = (m*p-1)/(m-1)

        else:
            if n == 1:
                return tu_matrix
            p_c = (n*p-1)/(n-1)

        # Set remaining ones randomly according to new probability
        if p_c > 0.:
            for i in range(n):
                for j in range(m):
                    if tu_matrix[i,j] == 0 and self.rnd['tu_assignment'].random() < p_c:
                        tu_matrix[i,j] = 1

        return tu_matrix
    
    def _generateTaskUserMinimalAssignmentMatrix(self):
        """ Steps 1-3 done """
        # Eye matrix
        n, m = len(self.tasks), len(self.users)
        tu_matrix = np.eye(n, m, dtype=np.uint8)

        if n >= m:
            # Set ones randomly remaining rows
            for i in range(m, n):
                tu_matrix[i, self.rnd['tu_assignment'].randrange(m)] = 1

            # Shuffle rows
            self.np_rnd['tu_assignment'].shuffle(tu_matrix)

        else:
            # Set ones randomly remaining columns
            for i in range(n, m):
                tu_matrix[self.rnd['tu_assignment'].randrange(n), i] = 1

            # Shuffle columns
            self.np_rnd['tu_assignment'].shuffle(tu_matrix.T)

        return tu_matrix

    def _getCommunityTree(self, partitions):
        tree = TreeNode()
        root_data = frozenset(range(len(self.nodes)))
        tree.data = root_data

        prev_set = {root_data}
        communities = self.getCommunities()
        for i in range(partitions):
            next_set = set([frozenset(e) for e in next(communities)])

            chosen = prev_set.difference(next_set).pop()
            splitted = next_set.difference(prev_set)
            left = splitted.pop()
            right = splitted.pop()

            leaf = tree.findLeafByTuple(chosen)
            leaf.left  = TreeNode()
            leaf.right = TreeNode()
            leaf.left.data  = left
            leaf.right.data = right
            leaf.left.depth  = leaf.depth + 1
            leaf.right.depth = leaf.depth + 1
            leaf.left.parent  = leaf
            leaf.right.parent = leaf
            
            prev_set = next_set

        return tree

    def _getTaskUserRequestProbabilityMatrix(self, popularity, spreadness):
        """Based on communities"""

        # nid mapping to TreeNode object
        n_nodes = [
                self.tree.findLeafByElement(nid)
                for nid in range(len(self.nodes))
            ]

        u_nids = [user.node_id for user in self.users]

        # Minimum matrix to find "mandatory" task/user assignment
        tu_matrix_min = self._generateTaskUserMinimalAssignmentMatrix()

        #                  min(
        # task -> uid_1 ->   depth1
        #      -> ...   ->   ...
        #      -> uid_n ->   depth2
        #                  )
        t_nids = [
                min([u_nids[uid] for uid in np.flatnonzero(row)],
                        key=lambda nid: n_nodes[nid].depth)
                for row in tu_matrix_min
            ]

        # TODO: Hacerlo configurable. Distribuir no de manera uniforme,
        # porque hay más servicios globales que regionales
        t_depths = []
        for nid in t_nids:
            layers = n_nodes[nid].depth + 1
            rnd = self.rnd['tu_assignment'].uniform(0,(2**layers)-1)
            k = 0
            while rnd > (2**(k+1))-1:
                k += 1
            t_depths.append(k)

        tu_prob_matrix = np.zeros((len(self.tasks), len(self.users)))
        for tid in range(len(self.tasks)):
            for uid in range(len(self.users)):
                if tu_matrix_min[tid,uid] == 1:
                    tu_prob_matrix[tid,uid] = 1.
                else:
                    t_nid = t_nids[tid]
                    u_nid = u_nids[uid]
                    t_depth = t_depths[tid]

                    t_node = self.tree.findNodeByDepthAndElement(t_depth, t_nid)
                    u_node = n_nodes[u_nid]

                    # Get common node and depth
                    common_node = self.tree.findCommonAncestorByNode(t_node, u_node)
                    c_depth = common_node.depth

                    tu_prob_matrix[tid,uid] = popularity * spreadness ** (t_depth - c_depth)
        
        return tu_prob_matrix

    def _generateLinkData(self, minbw, maxbw):
        self.links = {edge:
                Link(bandwidth = round(self.rnd['graph_weights'].uniform(
                    minbw, maxbw), 2)
            ) for edge in self.graph.edges}

    def displayGraph(self, seed=1):
        """
        Display the resulting graph of the network with the server nodes,
        users and the weights of the connections. Green nodes represent the
        server nodes, while red nodes represent the users. User's ids start
        after last server node id.
        """
        plt_gnp = plt.subplot(1,1,1)

        pos = nx.spring_layout(self.graph, seed=seed)
        color = ['lime' if node < len(self.nodes) else 'red' for node in self.graph]
        nlabels = {
                i:'{}{}'.format(
                    'N' if i < len(self.nodes) else 'U',
                    i if i < len(self.nodes) else i - len(self.nodes))
                for i in range(len(self.nodes) + len(self.users))
            }
        nx.draw_networkx(self.graph, pos, labels=nlabels, font_size=8, font_weight='bold', node_color=color)

        elabels = nx.get_edge_attributes(self.graph, 'weight')
        nx.draw_networkx_edge_labels(self.graph, pos, edge_labels=elabels)

        # Communities
        """
        communities = ntw.getCommunities()
        partitions = 2
        gradient = np.linspace(0, 1, partitions + 1)
        cm_list = set(range(len(self.nodes)))
        for i in range(partitions):
            cm_list = next(communities)
        cmap = mpl.colormaps['viridis']
        for i in range(partitions+1):
            lst = list(cm_list[i])
            nx.draw_networkx_nodes(self.graph, pos, nodelist=lst, node_color=cmap(gradient[i]))
        """

        plt.show()

    # GETTERS
    # ==========================================================================

    # Values
    def getNodeOccupiedMemory(self, node_id):
        """Get node's available memory considering the amount of tasks being
        executed on it"""
        total_task_mem = sum([t.memory for t in self.tasks if t.node_id == node_id])
        return total_task_mem

    def getNodeAvailableMemory(self, node_id):
        return self.nodes[node_id].memory - getNodeOccupiedMemory(node_id)

    def getTotalNodeMemory(self):
        return np.sum([n.memory for n in self.nodes])

    def getTotalTaskMemory(self):
        return np.sum([t.memory for t in self.tasks])

    def getNUsers(self):
        return len(self.users)

    def getNNodes(self):
        return len(self.nodes)

    def getNTasks(self):
        return len(self.tasks)

    def getMinimumNNodesNeeded(self):
        """Calculates the hipotetical minimum number of nodes needed for
        holding all the services. This method uses the sorted node memory array
        and keeps adding node memory until the sum of the memory of all
        services can be stored. Notice this is not necessarily true, as
        services' required memory cannot be splited between nodes. Just useful
        for calculating the normalized value of f2."""

        total_task_mem = np.sum([t.memory for t in self.tasks])
        total_node_mem = 0.
        nodes_needed = 0
        while total_node_mem <= total_task_mem and nodes_needed < len(self.nodes):
            total_node_mem += self.memory[nodes_needed]
            nodes_needed += 1

        return nodes_needed

    def getTasksAverageDistanceToUser(self, matrix, undm=None, tuam=None, maximize=False, **garbage):
        """
        Returns a float with the average distance of all services to
        their respective users.

        Acronyms:
        - tnam: task/node assignment matrix
        - undm: user/node distance matrix
        - tuam: task/user assignment matrix
        - tudm: task/user distance matrix

        Matrices tuam and undm can be passed by parameter for efficiency
        purposes.
        """
        if tuam is None: tuam = self.getTaskUserAssignmentMatrix()
        tudm = self.getTaskUserDistanceMatrix(
                matrix, undm=undm, tuam=tuam, includeAll=False, maximize=maximize)

        tua_sum = np.sum(tudm, axis=1) 
        tud_sum = np.sum(tuam, axis=1)
        tu_filter = (tua_sum != 0) # Filter so that there's no division by zero

        avg_row = tua_sum[tu_filter] / tud_sum[tu_filter]
        return np.average(avg_row)

    def getTasksMinAverageDistanceToUser(self, undm=None, tuam=None, **garbage):
        """
        Returns a float with the max possible average distance of all services
        to their respective users. This is achieved by passing tnam as a full
        matrix of ones, so it will traverse all nodes for each service getting
        the minimum distance.

        This method is useful for getting the normalized value of f1.
        """
        tnam = np.ones((len(self.tasks), len(self.nodes)), dtype=np.uint8)
        return self.getTasksAverageDistanceToUser(
                tnam, undm=undm, tuam=tuam)

    def getTasksMaxAverageDistanceToUser(self, undm=None, tuam=None, **garbage):
        """
        Returns a float with the max possible average distance of all services
        to their respective users. This is achieved by passing tnam as a full
        matrix of ones, so it will traverse all nodes for each service getting
        the maximum distance.

        This method is useful for getting the normalized value of f1.
        """
        tnam = np.ones((len(self.tasks), len(self.nodes)), dtype=np.uint8)
        return self.getTasksAverageDistanceToUser(
                tnam, undm=undm, tuam=tuam, maximize=True)

    def getTasksMinAverageDistanceToUser_v2(self, undm=None, tuam=None, **garbage):
        return np.average(np.min(self.getTaskNodeDistanceMatrix(undm, tuam), axis=1))

    def getTasksMaxAverageDistanceToUser_v2(self, undm=None, tuam=None, **garbage):
        return np.average(np.max(self.getTaskNodeDistanceMatrix(undm, tuam), axis=1))

    def getTasksAverageHopsToUser(self, matrix, unhm=None, tuam=None, maximize=False, **garbage):
        """
        Returns a float with the average hops of all services to
        their respective users.

        Acronyms:
        - tnam: task/node assignment matrix
        - unhm: user/node hops matrix
        - tuam: task/user assignment matrix
        - tuhm: task/user hops matrix

        Matrices tuam and unhm can be passed by parameter for efficiency
        purposes.
        """
        if tuam is None: tuam = self.getTaskUserAssignmentMatrix()
        tuhm = self.getTaskUserHopsMatrix(
                matrix, unhm=unhm, tuam=tuam, includeAll=False, maximize=maximize)

        tua_sum = np.sum(tuhm, axis=1) 
        tuh_sum = np.sum(tuam, axis=1)
        tu_filter = (tua_sum != 0) # Filter so that there's no division by zero

        avg_row = tua_sum[tu_filter] / tuh_sum[tu_filter]
        return np.average(avg_row)

    def getTasksMinAverageHopsToUser(self, unhm=None, tuam=None, **garbage):
        tnam = np.ones((len(self.tasks), len(self.nodes)), dtype=np.uint8)
        return self.getTasksAverageHopsToUser(
                tnam, unhm=unhm, tuam=tuam)

    def getTasksMaxAverageHopsToUser(self, unhm=None, tuam=None, **garbage):
        tnam = np.ones((len(self.tasks), len(self.nodes)), dtype=np.uint8)
        return self.getTasksAverageHopsToUser(
                tnam, unhm=unhm, tuam=tuam, maximize=True)

    def getTasksMinAverageHopsToUser_v2(self, unhm=None, tuam=None, **garbage):
        return np.average(np.min(self.getTaskNodeHopsMatrix(unhm, tuam), axis=1))

    def getTasksMaxAverageHopsToUser_v2(self, unhm=None, tuam=None, **garbage):
        return np.average(np.max(self.getTaskNodeHopsMatrix(unhm, tuam), axis=1))

    def getNodeOccupationRatio(self, tnam, tma=None, nma=None, **garbage):
        tnmm = self.getTaskNodeMemoryMatrix(tnam)
        nma  = self.getNodeMemoryArray()
        return np.average(np.sum(tnmm, axis=0) / nma)

    def getNodeOccupationVariance(self, tnam, tma=None, nma=None, **garbage):
        tnmm = self.getTaskNodeMemoryMatrix(tnam)
        nma  = self.getNodeMemoryArray()
        xa   = np.sum(tnmm, axis=0) / nma

        mean = np.average(xa)
        sumacc = 0.
        for xi in xa:
            sumacc += (xi - mean)**2

        return sumacc / len(self.nodes)

    def getBetweennessCentrality(self):
        return nx.betweenness_centrality(
                self.graph.subgraph(range(len(self.nodes))),
                seed=self.seed)

    def getPowerConsumptionArray(self, tnam, **kwargs):
        cpu_array = np.sum(self.getTaskNodeCPUUsageMatrix(tnam), axis=0)
        mem_array = np.sum(self.getTaskNodeMemoryMatrix(tnam), axis=0)

        pwc_array = np.zeros(len(self.nodes), np.float64)
        for n in range(tnam.shape[1]):
            pw_min = self.nodes[n].min_power

            # If no task assigned, can turn off node
            if mem_array[n] > 0:
                pwc_array[n] += pw_min
            else:
                continue

            # Calculate memory power consumption
            pw_mem_r = self.nodes[n].mem_power_ratio
            node_memory = self.nodes[n].memory
            total_task_memory = mem_array[n]

            pwc_array[n] += pw_min * pw_mem_r * (total_task_memory / node_memory)

            # Calculate CPU power consumption
            pw_cpu_r = self.nodes[n].cpu_power_ratio
            xp, fp = NODE_CPU_PW_MODEL_LIST[self.nodes[n].cpu_power_model]
            n_cpus = self.nodes[n].cpus
            total_cpu_consumption = cpu_array[n]
            pwc_array[n] += np.interp(
                    total_cpu_consumption,
                    n_cpus * xp,
                    fp * pw_min * pw_cpu_r)

        return pwc_array

    def getTotalPowerConsumption(self, tnam, **kwargs):
        return np.sum(self.getPowerConsumptionArray(tnam, **kwargs))
    
    def getNetworkUtilization(self, tnam, **kwargs):
        return np.average(tuple(
                self.getNetworkUtilizationDictionary(tnam).values()
            ))

    # Dataclasses
    def getUser(self, user_id):
        return self.users[user_id]

    def getNode(self, node_id):
        return self.nodes[node_id]

    def getTask(self, task_id):
        return self.tasks[task_id]

    # Lists
    def getUserList(self):
        return self.users

    def getTaskList(self):
        return self.tasks

    def getNodeList(self):
        return self.nodes

    def getUserTasks(self, user_id):
        """Get the list of tasks requested by an user."""
        return [t for t in self.tasks if t.user_id == user_id]
            
    def getNodeExecutingTasks(self, node_id):
        """Get the list of tasks assigned to a server node."""
        return [t for t in self.tasks if t.node_id == node_id]

    # Dictionaries
    def getEdgeWeightDictionary(self):
        return nx.get_edge_attributes(self.graph, 'weight')
    
    def getNetworkUtilizationDictionary(self, tnam):
        N_NODES = len(self.nodes)

        paths = dict(nx.all_pairs_shortest_path(self.graph))
        undm = self.getUserNodeDistanceMatrix()

        tuam = self.getTaskUserAssignmentMatrix()
        tuam_nz = np.nonzero(tuam)
        edge_acc_dict = {edge: 0.0 for edge in self.graph.edges}
        for tid, uid in np.transpose(tuam_nz):
            nodes = np.flatnonzero(tnam[tid])
            nid, _ = min([(nid, undm[uid,nid]) for nid in nodes], key=lambda i: i[1])
            path = paths[N_NODES+uid][nid]
            for i in range(len(path)-1):
                orig, dest = sorted((path[i], path[i+1]))
                edge_acc_dict[orig,dest] += self.users[uid].pps * self.USER_REQUEST_SIZE

        for k in edge_acc_dict.keys():
            edge_acc_dict[k] /= self.links[k].bandwidth

        return edge_acc_dict

    def getMaxNetworkUtilization(self):
        return np.sum([l.bandwidth for l in self.links.values()])

    # NumPy 1D arrays
    def getTaskUserAssignmentArray(self):
        """ Given that a task can only be assigned to an user, we can simplify
        the operations that retrieve this information """
        return np.array([t.user_id for t in self.tasks])

    def getTaskNodeAssignmentArray(self):
        """Given that a task can only be assigned to a node, we can simplify
        the operations that retrieve this information"""
        return np.array([t.node_id for t in self.tasks])

    def getTaskMemoryArray(self):
        return np.array([t.memory for t in self.tasks])


    def getNodeMemoryArray(self):
        return np.array([n.memory for n in self.nodes])


    def getNodeOccupiedMemoryArray(self, m=None):
        tn_nonzeros = np.nonzero(m)
        mem_array = np.zeros(len(self.nodes))
        for i in range(len(tn_nonzeros[0])):
            tid = tn_nonzeros[0][i]
            nid = tn_nonzeros[1][i]
            mem_array[nid] += self.tasks[tid].memory

        return mem_array

    def getNodeAvailableMemoryArray(self, m=None):
        if m is not None:
            capacity = self.getNodeMemoryArray()
            occupied = self.getNodeOccupiedMemoryArray(m)
            return capacity - occupied
        else:
            # Not implemented
            return np.array([])
    
    def getTaskDistanceArray(self, m):
        """Returns distances of tasks to their respective users given a
        task/node assignment matrix"""
        tu_assign_m = self.getTaskUserAssignmentMatrix()
        un_dist_m   = self.getUserNodeDistanceMatrix()

        # Find ones of task/node matrix
        tn_nonzeros = np.nonzero(m)

        # Find ones of task/user matrix
        tu_nonzeros = np.nonzero(tu_assign_m)

        tasks = np.zeros(len(self.tasks), np.float64)
        for i in range(len(tn_nonzeros[0])):
            tid = tn_nonzeros[0][i]
            x = np.where(tu_nonzeros[0] == tid)
            for uid in x:
                # this for is needed when a task is assigned to more than one user
                tasks[tid] += un_dist_m[tu_nonzeros[1][uid], tn_nonzeros[1][i]]

        return tasks

    def getTaskCPUUsageArray(self):
        return [t.cpu_usage for t in self.tasks]

    def getNodeCPUArray(self):
        return [n.cpus for n in self.nodes]
    
    def getTaskHopsArray(self, m):
        """Returns hops of tasks to their respective users given a
        task/node assignment matrix"""
        tu_assign_m = self.getTaskUserAssignmentMatrix()
        un_hops_m   = self.getUserNodeHopsMatrix()

        # Find ones of task/node matrix
        tn_nonzeros = np.nonzero(m)

        # Find ones of task/user matrix
        tu_nonzeros = np.nonzero(tu_assign_m)

        tasks = np.zeros(len(self.tasks), np.float64)
        for i in range(len(tn_nonzeros[0])):
            tid = tn_nonzeros[0][i]
            x = np.where(tu_nonzeros[0] == tid)
            for uid in x:
                # this for is needed when a task is assigned to more than one user
                tasks[tid] += un_hops_m[tu_nonzeros[1][uid], tn_nonzeros[1][i]]

        return tasks

    def getMinPowerConsumptionArray(self):
        return np.array([n.min_power for n in self.nodes])

    def getMaxPowerConsumptionArray(self):
        return np.array([
            n.min_power * (1 + n.cpu_power_ratio + n.mem_power_ratio)
            for n in self.nodes])

    # NumPy matrices
    def getTaskUserAssignmentMatrix(self):
        tu_assign_m = np.zeros((len(self.tasks), len(self.users)), np.uint8)
        for t,u in self.task_user:
            tu_assign_m[t,u] += 1
        return tu_assign_m

        """
        Old solution:

        Given that a task can only be assigned to an user, we can simplify
        the operations that retrieve this information.

        assignment = np.zeros((len(self.tasks), len(self.users)), dtype=np.int16)
        for t in self.tasks:
            assignment[t.id, t.user_id] += 1
        return assignment
        """

    def getUserNodeDistanceMatrix(self):
        """Get the distance matrix from the users (rows) to the server nodes
        (columns) using Dijkstra's algorithm."""
        distances = np.empty((len(self.users), len(self.nodes)))
        for uid in range(len(self.users)):
            dct = nx.single_source_dijkstra_path_length(self.graph, uid + len(self.nodes))
            for nid in range(len(self.nodes)):
                distances[uid, nid] = dct[nid]
        return distances
    
    def getUserNodeHopsMatrix(self):
        """Get the number of hops from the users (rows) to the server nodes
        (columns) using Dijkstra's algorithm."""
        hops = np.empty((len(self.users), len(self.nodes)), dtype=np.uint16)
        for uid in range(len(self.users)):
            i = 0
            dct_iter = nx.bfs_layers(self.graph, uid + len(self.nodes))
            for dct in dct_iter:
                for nid in dct:
                    if nid < len(self.nodes):
                        hops[uid, nid] = i
                i += 1
        return hops
    
    def getTaskUserDistanceMatrix(self, tnam, undm=None, tuam=None, includeAll=True, maximize=False, **garbage):
        """Get the distance matrix from the tasks (rows) to the users
        (columns) given a task/node assignment matrix

        Acronyms:
        - tnam: task/node assignment matrix
        - undm: user/node distance matrix
        - tuam: task/user assignment matrix
        - tudm: task/user distance matrix

        Parameter 'includeAll' means include unassigned task/user distances.
        If set to false, each unassigned element will contain a zero. Useful
        for calculating averages by adding elements by columns or rows.

        Parameter 'maximize', for multinode assignment, will take the node with
        the requested service that is further away from the user instead of the
        closer node. Useful for normalization.

        Matrices undm and tuam can be passed by parameter for efficiency
        purposes.
        """

        if undm is None: undm = self.getUserNodeDistanceMatrix()

        if not includeAll and tuam is None:
            # In order to know which to inclue or exclude (m[row,col] == 1)
            tuam = self.getTaskUserAssignmentMatrix()

        tudm = np.zeros((len(self.tasks), len(self.users)), np.float64)

        # Get tasks' and nodes' indexes of nonzero values in matrix
        tn_nonzeros = np.nonzero(tnam)

        for uid in range(len(self.users)):
            for i in range(len(tn_nonzeros[0])):
                tid = tn_nonzeros[0][i]
                nid = tn_nonzeros[1][i]
                if includeAll or tuam[tid, uid] == 1:
                    if not maximize:
                        # Will take the minimum value
                        if tudm[tid, uid] == 0:
                            tudm[tid, uid] = undm[uid, nid]
                        elif undm[uid, nid] < tudm[tid, uid]: 
                            tudm[tid, uid] = undm[uid, nid]
                    elif undm[uid, nid] > tudm[tid, uid]: 
                        # Will take the maximum value
                        tudm[tid, uid] = undm[uid, nid]


        return tudm

    def getTaskUserHopsMatrix(self, tnam, unhm=None, tuam=None, includeAll=True, maximize=False, **garbage):
        """Get the hops matrix from the tasks (rows) to the users
        (columns) given a task/node assignment matrix

        Acronyms:
        - tnam: task/node assignment matrix
        - unhm: user/node hops matrix
        - tuam: task/user assignment matrix
        - tuhm: task/user hops matrix

        Parameter 'includeAll' means include unassigned task/user hops.
        If set to false, each unassigned element will contain a zero. Useful
        for calculating averages by adding elements by columns or rows.

        Parameter 'maximize', for multinode assignment, will take the node with
        the requested service that is further away from the user instead of the
        closer node. Useful for normalization.

        Matrices unhm and tuam can be passed by parameter for efficiency
        purposes.
        """

        if unhm is None: unhm = self.getUserNodeHopsMatrix()

        if not includeAll and tuam is None:
            # In order to know which to inclue or exclude (m[row,col] == 1)
            tuam = self.getTaskUserAssignmentMatrix()

        tuhm = np.zeros((len(self.tasks), len(self.users)), np.float64)

        # Get tasks' and nodes' indexes of nonzero values in matrix
        tn_nonzeros = np.nonzero(tnam)

        for uid in range(len(self.users)):
            for i in range(len(tn_nonzeros[0])):
                tid = tn_nonzeros[0][i]
                nid = tn_nonzeros[1][i]
                if includeAll or tuam[tid, uid] == 1:
                    if not maximize:
                        # Will take the minimum value
                        if tuhm[tid, uid] == 0:
                            tuhm[tid, uid] = unhm[uid, nid]
                        elif unhm[uid, nid] < tuhm[tid, uid]: 
                            tuhm[tid, uid] = unhm[uid, nid]
                    elif unhm[uid, nid] > tuhm[tid, uid]: 
                        # Will take the maximum value
                        tuhm[tid, uid] = unhm[uid, nid]


        return tuhm

    def getTaskNodeAssignmentMatrix(self, array=None):
        """Get the matrix of the amount of instances of each task (rows) on each
        server node (columns) given an array of integers"""
        assignment = np.zeros((len(self.tasks), len(self.nodes)), dtype=np.int16)
        if array is not None:
            for tid in range(len(array)):
                if 0 <= array[tid]:
                    assignment[tid, array[tid]] += 1
        else:
            # Retrieve from tasks datastructure
            for t in self.tasks:
                assignment[t.id, t.node_id] += 1

        return assignment
    
    def getTaskNodeDistanceMatrix(self, undm=None, tuam=None, **garbage):
        """Get the matrix of the average distance that a service can have to
        the users that requests it depending on the node that it is assigned"""
        if undm is None: undm = self.getUserNodeDistanceMatrix()
        if tuam is None: tuam = self.getTaskUserAssignmentMatrix()

        tua_sum = np.sum(tuam, axis=1)
        tndm = np.zeros((len(self.tasks), len(self.nodes)))

        for t in range(len(self.tasks)):
            for n in range(len(self.nodes)):
                tnd_sum = 0.0
                for u in range(len(self.users)):
                    tnd_sum += tuam[t][u] * undm[u][n]
                tndm[t][n] = tnd_sum / tua_sum[t]

        return tndm

    def getTaskNodeHopsMatrix(self, unhm=None, tuam=None, **garbage):
        """Get the matrix of the average hops that a service can have to
        the users that requests it depending on the node that it is assigned"""
        if unhm is None: unhm = self.getUserNodeHopsMatrix()
        if tuam is None: tuam = self.getTaskUserAssignmentMatrix()

        tua_sum = np.sum(tuam, axis=1)
        tnhm = np.zeros((len(self.tasks), len(self.nodes)))

        for t in range(len(self.tasks)):
            for n in range(len(self.nodes)):
                tnh_sum = 0
                for u in range(len(self.users)):
                    tnh_sum += tuam[t][u] * unhm[u][n]
                tnhm[t][n] = tnh_sum / tua_sum[t]

        return tnhm

    def getTaskNodeMemoryMatrix(self, m=None):
        """Returns a task/node memory matrix given a task/node assignment matrix"""
        mm = np.zeros((len(self.tasks), len(self.nodes)), dtype=np.float64)
        if m is not None:
            t_memory_v = self.getTaskMemoryArray()

            # Find node of each task
            tn_nonzeros = np.nonzero(m)

            for i in range(len(tn_nonzeros[0])):
                tid = tn_nonzeros[0][i]
                nid = tn_nonzeros[1][i]
                mm[tid, nid] += t_memory_v[tid]

        else:
            # Retrieve from tasks datastructure
            # DEPRECATED
            for t in self.tasks:
                mm[t.id, t.node_id] = t.memory
        
        return mm

    def getTaskNodeCPUUsageMatrix(self, tnam):
        """
        Returns a task/node CPU usage matrix given a task/node assignment matrix
        """
        pcm = np.zeros((len(self.tasks), len(self.nodes)), dtype=np.float64)
        if tnam is not None:
            t_cpu_v = self.getTaskCPUUsageArray()

            # Find node of each task
            tn_nonzeros = np.nonzero(tnam)

            for i in range(len(tn_nonzeros[0])):
                tid = tn_nonzeros[0][i]
                nid = tn_nonzeros[1][i]
                pcm[tid, nid] += t_cpu_v[tid]

        return pcm

    # MANAGEMENT
    # ==========================================================================

    def assignTask(self, task_id, node_id, fraction=1.):
        """Assign a task to a server node."""
        self.tasks[task_id].node_id = node_id
    
    def removeTask(self, task_id, node_id=-1):
        """Remove task from server node"""
        self.tasks[task_id].node_id = -1

    # FILES
    # ==========================================================================
    def export_gexf(self, path):
        nx.write_gexf(self.graph, path)

    # ANALYSIS
    # ==========================================================================
    def checkMemoryRequirements(self):
        return np.sum(self.memory) >= np.sum([t.memory for t in self.tasks])

    def getCommunities(self):
        # TODO: Probar otros (proximidad geográfica, latencia, etc.)
        return nx.community.girvan_newman(
                self.graph.subgraph([i for i in range(len(self.nodes))]))

    # OBJECTIVE HANDLING
    # ==========================================================================
    def getObjectiveBounds(self, obj, **kwargs):
        if   obj == OBJ_LIST[0]:
            f_min = self.getTasksMinAverageDistanceToUser(**kwargs)
            f_max = self.getTasksMaxAverageDistanceToUser(**kwargs)
        elif obj == OBJ_LIST[1]:
            f_min = self.getMinimumNNodesNeeded()
            f_max = len(self.nodes)
        elif obj == OBJ_LIST[2]:
            f_min = self.getTasksMinAverageHopsToUser(**kwargs)
            f_max = self.getTasksMaxAverageHopsToUser(**kwargs)
        elif obj == OBJ_LIST[3]:
            f_min = 0.
            f_max = 1.
        elif obj == OBJ_LIST[4]:
            f_min = 0.
            f_max = 0.25
        elif obj == OBJ_LIST[5]:
            f_min = 0.
            f_max = np.sum(self.getMaxPowerConsumptionArray())
        elif obj == OBJ_LIST[6]:
            f_min = 0.
            f_max = self.getMaxNetworkUtilization()
        #elif obj == OBJ_LIST[7]:
        #    pass
        #elif obj == OBJ_LIST[8]:
        #    pass
        #elif obj == OBJ_LIST[9]:
        #    pass

        return f_min, f_max

    def evaluateObjective(self, obj, tnam, **kwargs):
        if   obj == OBJ_LIST[0]:
            return self.getTasksAverageDistanceToUser(tnam, **kwargs)
        elif obj == OBJ_LIST[1]:
            return np.count_nonzero(np.any(tnam, axis=0))
        elif obj == OBJ_LIST[2]:
            return self.getTasksAverageHopsToUser(tnam, **kwargs)
        elif obj == OBJ_LIST[3]:
            return self.getNodeOccupationRatio(tnam, **kwargs)
        elif obj == OBJ_LIST[4]:
            return self.getNodeOccupationVariance(tnam, **kwargs)
        elif obj == OBJ_LIST[5]:
            return self.getTotalPowerConsumption(tnam, **kwargs)
        elif obj == OBJ_LIST[6]:
            return self.getNetworkUtilization(tnam, **kwargs)
        #elif obj == OBJ_LIST[7]:
        #    pass
        #elif obj == OBJ_LIST[8]:
        #    pass
        #elif obj == OBJ_LIST[9]:
        #    pass

if __name__ == '__main__':
    # Use 'generate' subparser for testing

    from parameters import configs
    random.seed(configs.seed)

    ntw = Network(configs)

    tnam = np.array([
            [
                1 if random.random() < 0.5 else 0
                for n in range(configs.n_nodes)
            ] for t in range(configs.n_tasks)
        ])

    # POWER CONSUMPTION ARRAY
    print(tnam)
    print(ntw.getTaskNodeCPUUsageMatrix(tnam))
    print(ntw.getTaskNodeMemoryMatrix(tnam))
    print(ntw.getPowerConsumptionArray(tnam))

    # NODES
    ntw.displayGraph()



