from pymoo.model.problem import Problem
import numpy as np
import matplotlib.pyplot as plt


class FlowshopScheduling(Problem):
    """
    Flowshop scheduling problem. This problem uses permutation encoding.
    Args:
        processing_times: numpy array, where processing_time[i][j] is the processing time for job j on machine i.
    """
    def __init__(self, processing_times, **kwargs):
        n_machines, n_jobs = processing_times.shape
        self.data = processing_times

        super(FlowshopScheduling, self).__init__(
            n_var=n_jobs,
            n_obj=1,
            xl=0,
            xu=n_machines,
            type_var=np.int,
            elementwise_evaluation=True,
            **kwargs
        )

    def _evaluate(self, x, out, *args, **kwargs):
        out["F"] = self.makespan(x)

    def makespan(self, x):
        machine_times = self.get_machine_times(x)
        # The makespan is the difference between the starting time of the first job and the latest finish time of any
        # job. Minimizing the makespan amounts to minimizing the total time it takes to process all jobs from start
        # to finish.
        makespan = machine_times[-1][-1] + self.data[-1][x[-1]]  # finish time of the last job
        return makespan

    def get_machine_times(self, x):
        n_machines, n_jobs = self.data.shape

        # A 2d array to store the starting time for each job on each machine
        # machine_times[i][j] --> starting time for the j-th job on machine i
        machine_times = [[] for _ in range(n_machines)]

        # Assign the initial job to the machines
        machine_times[0].append(0)
        for i in range(1, n_machines):
            # Start the next job when the previous one is finished
            machine_times[i].append(
                machine_times[i - 1][0] + self.data[i - 1][x[0]]
            )

        # Assign the remaining jobs
        for j in range(1, n_jobs):
            # For the first machine, we can put a job when the previous one is finished
            machine_times[0].append(
                machine_times[0][j - 1] + self.data[0][x[j - 1]]
            )

            # For the remaining machines, the starting time of the current job j is the max of the following two times:
            # 1. The finish time of the previous job on the current machine
            # 2. The finish time of the current job on the previous machine
            for i in range(1, n_machines):
                machine_times[i].append(
                    max(
                        machine_times[i][j - 1] + self.data[i][x[j - 1]],  # 1
                        machine_times[i - 1][j] + self.data[i - 1][x[j]]  # 2
                    )
                )
        return machine_times
