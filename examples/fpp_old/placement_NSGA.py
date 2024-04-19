import numpy as np
from pymoo.core.problem import ElementwiseProblem
from pymoo.core.crossover import Crossover
from pymoo.core.mutation import Mutation
from pymoo.core.sampling import Sampling
from utils import getCNTimes,getCNCosts
import math

class PlacementProblem(ElementwiseProblem):
    def __init__(self,n_var,n_objectives,time,adj,featHW,n_devices,n_tasks):
        super().__init__(
            n_var = n_var,
            n_obj=n_objectives,
            n_ieq_constr=2
            )
        self.number_devices = n_devices+1 
        self.n_tasks = n_tasks 

        ## One Infraestructure, and one App
        self.executions = time
        self.adj = adj
        self.featHW = featHW

    def _evaluate(self, x, out, *args, **kwargs):
        sample = x.reshape(self.number_devices,self.n_tasks)
        f1 = np.sum(getCNTimes(sample,self.executions,self.featHW,self.adj))
        f2 = np.sum(getCNCosts(sample,self.featHW))

        g1 = np.sum(np.abs(np.sum(sample,axis=0) - np.ones(shape=(self.n_tasks),dtype=np.uint8)))
        g2t = np.sum(sample,axis=1) - [5]*self.number_devices
        g2 = 1 if np.any(g2t>=0) else 0
       
        # if g2==1:
        #     print("HJERE")
        out["F"] = [f1,f2]
        out["G"] = [g1,g2]
    
    def myevaluate(self, x):
        sample = x.reshape(self.number_devices,self.n_tasks)
        f1 = np.sum(getCNTimes(sample,self.executions,self.featHW,self.adj))
        f2 = np.sum(getCNCosts(sample,self.featHW))
        g1 = np.sum(np.abs(np.sum(sample,axis=0) - np.ones(shape=(self.n_tasks),dtype=np.uint8)))
        return [f1,f2,g1]

    def _calc_pareto_front(self, *args, **kwargs):
        F = args[""]
        return super()._calc_pareto_front(*args, **kwargs)
        



        
class MySampling(Sampling):

    
    def _do(self, problem, n_samples, **kwargs):
        samples = []
        for _ in range(n_samples):
            allocations = np.zeros((problem.number_devices,problem.n_tasks),dtype=np.uint8)

            ## Random allocations
            for ix in range(problem.n_tasks):
                rnd_device = np.random.randint(problem.number_devices)
                allocations[rnd_device,ix]=1

            samples.append(allocations.reshape(1,-1).squeeze()) ### pymoo only works with 1DArrays :/
        return samples
        

class MyMutation(Mutation):
    def __init__(self,prob):
        super().__init__()
        self.prob = prob

    def _do(self, problem, X, **kwargs):
        #TODO fix mutation
        ## DEBUG
        n_gen =kwargs["algorithm"].n_gen
        counter = 0  
        list_mutations = []
        mut = np.zeros(X.shape[0],dtype=bool)
        #
        XX = X.copy()
        if len(X.shape)==3:
            XX = XX.reshape(X.shape[0]*X.shape[1],X.shape[2])
        
        for i in range(XX.shape[0]):
            if self.prob >= np.random.random(): # M1
            # if n_gen>=75 and self.prob >= np.random.random(): # M1
                # None

                sample = XX[i].reshape(problem.number_devices,problem.n_tasks)
                rnd_task = np.random.randint(problem.n_tasks)
                currentDevice = np.where(sample[:,rnd_task])[0][0]
                alternativeDevices = list(range(problem.number_devices))
                alternativeDevices.remove(currentDevice)
                newDevice = np.random.choice(alternativeDevices,1).squeeze()
                sample[currentDevice,rnd_task]=0
                sample[newDevice,rnd_task]=1
                XX[i] = sample.reshape(1,-1).squeeze()
                list_mutations.append("M1")
                counter +=1
                mut[i] = True

                
                # print("\t\t Mutating sample: %i"%i)
            # elif n_gen<75 and self.prob >= np.random.random(): #mutaciones exclusivas #M2
            elif self.prob >= np.random.random(): #mutaciones exclusivas #M2
                # None
                # En esta mutación todos los servicios seq. irán a un mismo dispositivo
                list_mutations.append("M2")
                sample = XX[i].reshape(problem.number_devices,problem.n_tasks)
                n_jobs = int(math.sqrt(problem.n_tasks))
                row_start = np.random.randint(n_jobs-1)*n_jobs #TODO check limit of random -1?
                rows,rowe = row_start,row_start+n_jobs
                
                for itask in range(rows,rowe):
                    currentDevice = np.where(sample[:,itask])[0][0]
                    sample[currentDevice,itask]=0
                
                alternativeDevices = list(range(problem.number_devices))
                newDevice = np.random.choice(alternativeDevices,1).squeeze()
                sample[newDevice,rows:rowe]=np.ones(n_jobs)
                XX[i] = sample.reshape(1,-1).squeeze()
                #remover las otras asignaciones
                mut[i] = True
                counter +=1
            # elif n_gen>=75 and self.prob*2 >= np.random.random(): #M3
            elif self.prob*2 >= np.random.random(): #M3
                # None
                list_mutations.append("M3")
                sample = XX[i].reshape(problem.number_devices,problem.n_tasks)
                alternativeDevices = list(range(problem.number_devices))
                newDevice = np.random.choice(alternativeDevices,1).squeeze()
                anotherDevice = np.random.choice(alternativeDevices,1).squeeze()
                while anotherDevice == newDevice:
                    anotherDevice = np.random.choice(alternativeDevices,1).squeeze() #Warning
                dataswap = sample[newDevice,:]     

                sample[newDevice,:]= sample[anotherDevice,:]     
                sample[anotherDevice,:]= dataswap
                XX[i] = sample.reshape(1,-1).squeeze()
                mut[i] = True
                counter +=1

            else:
                None
                
            
        XX = XX.reshape(X.shape)
        # print("\tDebug: Counter Mutations: %i de %i"%(counter,XX.shape[0]))
        # print("\tDebug: Returned: %i"%XX.shape[0])      
        # print("\tDebug: mutation type",list_mutations)
         
        return XX, list_mutations, mut

class BinaryCrossover(Crossover):
    def __init__(self):
        super().__init__(n_parents=2, n_offsprings=2)
        # self.prob = prob_mutation #TODO remove 

    def _do(self, problem, X, **kwargs):
        n_parents, n_matings, n_var = X.shape
        _X = np.zeros((self.n_offsprings, n_matings, n_var),dtype=np.uint8)

        for k in range(n_matings):
            xx, yy = X[0, k], X[1, k]
            xx = xx.reshape(problem.number_devices,problem.n_tasks)
            yy = yy.reshape(problem.number_devices,problem.n_tasks)
            rnd_col = np.random.randint(problem.n_tasks)
            off1 = np.concatenate([xx[:,:rnd_col], yy[:,rnd_col:]],axis=1)  
            off2 = np.concatenate([yy[:,:rnd_col], xx[:,rnd_col:]],axis=1) 

            _X[0, k] = off1.reshape(1,-1).squeeze()
            _X[1, k] = off2.reshape(1,-1).squeeze()

        # _X = MyMutation(prob=self.prob)._do(problem,_X) #TODO REMOVE mutate new generation
        return _X

if __name__ == "__main__":
    print("None")
    # problem = PlacementProblem(n_var=90,n_objectives=2)
    # sampling = MySampling()
    # samples = sampling._do(problem,2)
    # print(samples)
    # print(samples[0])
    # out = {}
    # results = problem._evaluate(samples[0],out)
    # print(out)
    
    # mute = MyMutation()
    # sample_muted = mute._do(problem,samples[0])
    # print(sample_muted)
    
    # cross = BinaryCrossover() ##NO
    # _X = cross._do(problem,samples)
    # print(_X)
