class TreeNode:

    def __init__(self):
        self.data = None
        self.left = None
        self.right = None

        # Deductible, but convenient to store
        self.parent = None
        self.depth  = 0

    def findLeafByTuple(self, data):
        # Trivial case: current node is leaf
        if not self.left and not self.right:
            # Check if contains requested data
            if self.data == data:
                return self
            else:
                return None

        # Search recursively on left branch
        if self.left:
            leaf = self.left.findLeafByTuple(data)

        # Search recursively on right branch
        if self.right and leaf is None:
            leaf = self.right.findLeafByTuple(data)

        return leaf

    def findLeafByElement(self, elem):
        # Trivial case: current node is leaf
        if not self.left and not self.right:
            # Check if contains requested data
            if elem in self.data:
                return self
            else:
                return None

        # Search recursively on left branch
        if self.left:
            leaf = self.left.findLeafByElement(elem)

        # Search recursively on right branch
        if self.right and leaf is None:
            leaf = self.right.findLeafByElement(elem)

        return leaf

    def findCommonAncestorByNode(self, node1, node2):
        # Adjust until both depths match
        l1_ancestor = node1
        l2_ancestor = node2
        while l1_ancestor.depth > l2_ancestor.depth:
            l1_ancestor = l1_ancestor.parent

        while l2_ancestor.depth > l1_ancestor.depth:
            l2_ancestor = l2_ancestor.parent

        # Keep going up on tree until both have same ancestor
        while l1_ancestor != l2_ancestor:
            l1_ancestor = l1_ancestor.parent
            l2_ancestor = l2_ancestor.parent

        # Return any of them, as they reference the same node
        return l1_ancestor
    
    def findCommonAncestorByElement(self, elem1, elem2):
        leaf1 = self.findLeafByElement(elem1)
        leaf2 = self.findLeafByElement(elem2)

        return self.findCommonAncestorByNode(leaf1, leaf2)
    
    def findNodeByDepthAndElement(self, depth, element):
        node = self.findLeafByElement(element)
        while node.depth != depth:
            node = node.parent
        return node


### ### #########################################
import random

def getTaskUserRequestProbability(tid, uid):
    """Based on communities"""

    # Get user node
    u_nid = users_node[uid]
    u_node = root.findLeafByElement(u_nid)

    # Get task node
    t_nid = users_node[tasks_user[tid]]
    t_depth = tasks_depth[tid]
    t_node = root.findNodeByDepthAndElement(t_depth, t_nid)

    # Get common node and depth
    common_node = root.findCommonAncestorByNode(t_node, u_node)
    c_depth = common_node.depth

    popularity = tasks_popularity[tid]
    spreadness = tasks_spreadness[tid]

    return popularity * spreadness ** (t_depth - c_depth)

def split(input_set):
    input_list = [p for p in input_set if len(p) >= 2]
    if not input_list:
        return None

    partition_tuple = random.choice(input_list)
    partition_list = list(partition_tuple)

    random.shuffle(partition_list)
    part_idx = random.randrange(1, len(partition_list))
    left_lst, right_lst = partition_list[:part_idx], partition_list[part_idx:]
    left_lst.sort()
    right_lst.sort()

    output_set = input_set.copy()
    output_set.remove(partition_tuple)
    return output_set.union({tuple(left_lst), tuple(right_lst)})

if __name__ == '__main__':
    random.seed(722)
    
    N_NODES = 10
    N_TASKS = 7
    N_USERS = 5
    SPLITS = 5
    
    my_set = {tuple([i for i in range(N_NODES)])}
    print('     Start:', my_set)
    print()

    root = TreeNode()
    root.data = my_set.copy().pop()

    prev_set = my_set
    for i in range(SPLITS):
        next_set = split(prev_set)

        chosen = prev_set.difference(next_set).pop()
        print('  Chosen:', chosen)
        splitted = next_set.difference(prev_set)
        print('Splitted:', splitted)
        left = splitted.pop()
        right = splitted.pop()

        leaf = root.findLeafByTuple(chosen)
        leaf.left  = TreeNode()
        leaf.right = TreeNode()
        leaf.left.data  = left
        leaf.right.data = right
        leaf.left.depth  = leaf.depth + 1
        leaf.right.depth = leaf.depth + 1
        leaf.left.parent  = leaf
        leaf.right.parent = leaf
        
        prev_set = next_set

    print()

    #######################
    ### TASKS AND USERS ###
    #######################
    random.seed(1)

    import numpy as np

    tuam = np.zeros((N_TASKS, N_USERS), dtype=np.uint8)

    users_node = [random.randrange(N_NODES) for _ in range(N_USERS)]
    tasks_user = [random.randrange(N_USERS) for _ in range(N_TASKS)]

    for i in range(N_TASKS):
        tuam[i, tasks_user[i]] = 1

    print([users_node[t] for t in tasks_user])
    print(users_node)

    tasks_depth = [
        random.randrange(
            root.findLeafByElement(
                users_node[tasks_user[t]]
            ).depth
        ) for t in range(N_TASKS)
    ]

    print(tasks_depth)
    print()

    tasks_popularity = [random.uniform(0.4,0.8) for _ in range(N_TASKS)]
    tasks_spreadness = [random.uniform(0.2,0.6) for _ in range(N_TASKS)]

    getTaskUserRequestProbability(tid=2, uid=4)
    for tid in range(N_TASKS):
        for uid in range(N_USERS):
            p = getTaskUserRequestProbability(tid,uid)
            print(p, end=' ')
            if random.random() < p:
                tuam[tid,uid] = 1
        print()

    print(tuam)












