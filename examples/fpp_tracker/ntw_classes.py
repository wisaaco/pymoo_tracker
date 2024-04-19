from typing import List, ClassVar
from dataclasses import dataclass, field

@dataclass
class Node:
    """Class for storing information related to each node"""
    # Class variables
    n_nodes: ClassVar[int] = 0

    # Attributes
    id: int = field(init=False)
    memory: float = field(init=False)
    max_tasks: int
    cpus: int = field(init=False)
    min_power: float
    cpu_power_model: int
    cpu_power_ratio: float
    mem_power_ratio: float
    #tasks: List[int] = field(init=False, default_factory=list)

    def __post_init__(self):
        """Assign id automatically based on the amount of nodes generated"""
        self.id = Node.n_nodes
        Node.n_nodes += 1

@dataclass
class User:
    # Class variables
    n_users: ClassVar[int] = 0

    # Attributes
    id: int = field(init=False)
    node_id: int = field(init=False, default=-1)
    pps: float

    def __post_init__(self):
        """Assign id automatically based on the amount of users generated"""
        self.id = User.n_users
        User.n_users += 1

@dataclass
class Task:
    """Class for storing information related to each task"""
    # Class variables
    n_tasks: ClassVar[int] = 0

    # Attributes
    id: int = field(init=False)
    memory: float
    user_id: int
    node_id: int = field(init=False, default=-1)
    cpu_usage: float
    #nodes: List[int] = field(init=False, default_factory=list)

    def __post_init__(self):
        """Assign id automatically based on the amount of tasks generated"""
        self.id = Task.n_tasks
        Task.n_tasks += 1

@dataclass
class Link:
    """Class for storing information related to each link between devices"""
    # Class variables
    n_links: ClassVar[int] = 0

    # Attributes
    id: int = field(init=False)
    latency: float = field(init=False)
    bandwidth: float
    length: float = field(init=False)

    def __post_init__(self):
        """Assign id automatically based on the amount of tasks generated"""
        self.id = Link.n_links
        Link.n_links += 1
