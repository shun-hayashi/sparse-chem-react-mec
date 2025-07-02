import os
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
import datetime
import ast
import cma
import copy
from typing import Callable
from typing import List
import json

from src.loss import loss_log

# This class stores information about chemical species, kinetic models (l, r, k), and experimental data (d).
class ChemKinetics:
    def __init__(
        self,
        func: Callable,
        d: np.ndarray,
        d_test: np.ndarray = None,
        l: np.ndarray = None,
        r: np.ndarray = None,
        k: np.ndarray = None,
        k_max: float = 1e04,
        k_cut: float = 1e-2,
        conservation: List[float] = None,
        chemformula: List[str] = None,
    ):
        self.func = func
        self.d = d.copy()
        self.d_test = d_test.copy() if d_test is not None else None
        self.l = l.copy() if l is not None else None
        self.r = r.copy() if r is not None else None
        self.k = k.copy() if k is not None else None
        self.k_max = k_max
        self.k_cut = k_cut
        self.conservation = conservation.copy() if conservation is not None else None
        self.chemformula = chemformula.copy() if chemformula is not None else None
        self.lam = 0.
        self.f = None
        self.sim = None
        self.loss = None
        self.res_train = None
        self.res_test = None

    def optimize(self, lam: float = 0., k: np.ndarray = None, target: float = 0.) -> None:
        self.loss, self.k = optimize_CMA(self.func, self.l, self.r, self.d, self.k_max, lam, k=k, target=target)
        self.f = loss_wrap(self.func, self.d, self.l, self.r, self.k, 0.)
    
    def cut_lrk(self, k_cut: float = 0.) -> None:
        index = np.where(self.k <= k_cut)
        self.l = np.delete(self.l, index, axis=0)
        self.r = np.delete(self.r, index, axis=0)
        self.k = np.delete(self.k, index, axis=0)

    def save(self, path: str, status: str) -> None:
        self._save_makedirs(path)
        self._save_info(path)
        self._save_result(path, status)
        
    def _save_makedirs(self, path: str) -> None:
        if not os.path.exists(path):
            os.makedirs(path)
    
    def _save_result(self, path: str, status: str) -> None:
        columns = ["filename", "status", "lam", "num_eq", "loss", "MRSE_train", "MRSE_test"]
        
        path_res = path + "/" + "result.csv"
        if os.path.isfile(path_res) == False:
            df = pd.DataFrame(columns=columns)
            df.to_csv(path_res, mode="x")
        
        df = pd.read_csv(path_res)
        if df.empty == True:
            fname = "0000"
        else:
            fname = format(max(df["filename"])+1, "04")
        
        num_eq = self.l.shape[0]
        z0 = self.d[:,1:,0] # Mn7_0, Mn3_0, Mn2_0 #print(z0.shape) # (20, 3)
        z0_full = np.hstack([z0, np.zeros((self.d.shape[0], self.l.shape[1]-z0.shape[1]))])
        
        loss_lam = loss_wrap(self.func, self.d, self.l, self.r, self.k, self.lam)
        loss_train = loss_wrap(self.func, self.d, self.l, self.r, self.k, 0.)
        if self.d_test is not None:
            loss_test = loss_wrap(self.func, self.d_test, self.l, self.r, self.k, 0.)
        else:
            loss_test = np.nan
        savedata = np.array([[fname, status, self.lam, num_eq, loss_lam, loss_train, loss_test]])
    
        df_add = pd.DataFrame(savedata, columns=columns)
        df_add.to_csv(path_res, mode="a", header=False)
        
        path_lrk = path + "/" + fname + ".csv"
        lrk = np.hstack([self.l, self.r, self.k.reshape(-1, 1)])
        np.savetxt(path_lrk, lrk,  delimiter=",", fmt='%.5e')
        
        print("save result: %s" % path_lrk)
            
    def _load_lrk(self, path: str, fname: str) -> None:
        lrk = np.loadtxt(path + "/" + fname + ".csv", delimiter=",", dtype=float)
        l = lrk[:,:int((lrk.shape[1]-1)/2)]
        r = lrk[:,int((lrk.shape[1]-1)/2):lrk.shape[1]-1]
        k = lrk[:,-1]
        self.l = np.array(l)
        self.r = np.array(r)
        self.k = np.array(k)

    def _save_info(self, path: str) -> None:        
        info = {"k_max": self.k_max,
                "k_cut": self.k_cut,
                "conservation": self.conservation,
                "chemformula": self.chemformula,
               }
        fname = path + "/" + "info.json"
        with open(fname, "w") as file:
            json.dump(info, file)
            
    def _load_info(self, path: str) -> None:
        fname = path + "/" + "info.json"
        with open(fname, "r") as file:
            info = json.load(file)
        self.k_max = info["k_max"]
        self.k_cut = info["k_cut"]
        self.conservation = info["conservation"]
        self.chemformula = info["chemformula"]

    def _load_result(self, path: str, fname: str) -> None:
        df = pd.read_csv(path + "/" + "result.csv")
        df1 = df.loc[df["filename"] == int(fname)]
        self.lam = df1["lam"].iloc[0]
        self.loss = df1["loss"].iloc[0]
        self.res_train = df1["MRSE_train"].iloc[0]
        self.res_test = df1["MRSE_test"].iloc[0]
        
    def load(self, path: str, fname: str) -> None:
        self._load_info(path)
        self._load_lrk(path, fname)
        self._load_result(path, fname)
        self.f = loss_wrap(self.func, self.d, self.l, self.r, self.k, 0.)
    
    def simulate(self) -> None:
        t = self.d[:,0,:]
        z0 = self.d[:,1:,0]
        z0_full = np.hstack([z0, np.zeros((z0.shape[0], self.l.shape[1]-z0.shape[1]))])
        self.sim = np.empty((z0.shape[0], 1+self.l.shape[1], t.shape[1]))
        args = [self.l, self.r, self.k]
    
        for i in range(z0.shape[0]):
            t_span = np.array([np.min(t[i,:]), np.max(t[i,:])])
            sol = solve_ivp(fun=chemreact, t_span=t_span, y0 = z0_full[i,:], args=args, dense_output=True)
            self.sim[i,0,:] = t[i,:]
            self.sim[i,1:,:] = sol.sol(t[i,:])

# This class manages the calculation results and determines whether to accept or reject the latest outcome.
class StatusManager:
    def __init__(
        self,
        lam: float = 1E-6,
        lam_step_init: float = 1E-6,
        lam_step_amp: float = 2.,
        num_retry: int = 2,
    ):
        self.lam = lam
        self.lam_step = lam_step_init
        self.lam_step_amp = lam_step_amp
        self.status = "init"
        self.lam_list = []
        self.status_list = []
        self.dim_list = []
        self.retrycount = 0
        self.num_retry = num_retry
        self.dim = None
        self.f = np.inf
        self.should_refine = False

    def update(self, f, dim) -> None: # set_result
        if self.dim is None:
            self.dim = dim
        if self.lam == 0:
            self.dim = dim
            self.status = "init"
            print("status = init")
        elif self.dim < 7 and (dim < 6 or f > 0.02): # termination condition
            self.status = "finish"
            print("status = finish")
        elif f < self.f * 1.5: # tolerance for changes in MRSE
            self.status = "accept"
            print("status = accept")
        elif self.retrycount > self.num_retry:
            self.status = "force_accept"
            print("status = force_accept")
        elif dim < self.dim * 0.7: # tolerance for changes in dimension
            self.status = "reject"
            print("status = reject")
        else:
            self.status = "accept"
            print("status = accept")
        print("dim: %i -> %i, loss: %.5f => %.5f" % (self.dim, dim, self.f, f))
        
        self.should_refine = False
        if self.lam != 0 and (self.status != "reject"):
            if dim not in self.dim_list:
                self.should_refine = True
                print("refine; dim = %i" % dim)
                print(self.dim_list)
                
        if self.status != "reject":
            self.f = f
            self.dim = dim
            self.dim_list = np.append(self.dim_list, dim)
            print("histroy of refined dim:")
            self.lam_list = np.append(self.lam_list, self.lam)
            self.lam = self.lam + self.lam_step
            self.lam_step = self.lam_step * self.lam_step_amp
            self.retrycount = 0
            
        elif self.status == "reject":
            self.lam = 0.5 * (self.lam + self.lam_list[-1]) 
            self.retrycount += 1
            print("retry count = %i" % self.retrycount)
        print("next lam = %.5f, lam step = %.5f" % (self.lam, self.lam_step))

# This function minimizes the loss function by CMA-ES.
def optimize_CMA(funcptr_cr, l, r, d, k_max, lam, k=None, target=0.):
    ##########################################################
    ## This function was created by modifying the code found in the CMA API documentation.
    ## https://cma-es.github.io/apidocs-pycma/cma.evolution_strategy.CMAEvolutionStrategy.html#inject
    ##########################################################
    tolfun = 1e-6
    tol_iter = 1e-5
    num_restart_min = 3
    num_restart_max = 9
    popsize_amp = 1.2
    should_inject = False
    
    x_max = x_from_k(k_max)
    bestever = cma.optimization_tools.BestSolution()
    popsize = int(4+3*np.log(l.shape[0]))
    best_f =np.inf
    z0 = d[:,1:,0]
    z0_full = np.hstack([z0, np.zeros((d.shape[0], l.shape[1]-z0.shape[1]))])

    if k is not None:
        x_init = x_from_k(k)
        x_inj = [x_init.tolist()]
        should_inject = True
        
    for i in range(num_restart_max+1):
        es = cma.CMAEvolutionStrategy(np.random.rand(l.shape[0]) * x_max,  # 4-D
                                       x_max/4,  # initial std sigma0
                                       {'bounds': [0, x_max],
                                        'tolfun': tolfun,
                                        'popsize': popsize,
                                        'verb_append': bestever.evalsall})
        
        if i > 0 and should_inject:
            x_init = bestever.x
            x_inj = [x_init.tolist()]
        while not es.stop():
            if should_inject:
                es.inject(x_inj)
            
            X = es.ask()    # get list of new solutions
            fit = [loss_log(x, funcptr_cr, l, r, d, z0_full, lam) for x in X]  # evaluate each solution
            es.tell(X, fit) # besides for termination only the ranking in fit is used
            es.disp()  # uses option verb_disp with default 100
    
        print('termination:', es.stop())
        cma.s.pprint(es.best.__dict__)
        
        bestever.update(es.best)
        
        if bestever.f < target:
            should_inject = True
            
        if i > (num_restart_min-1) and np.abs(best_f-es.best.f) < tol_iter:
            break 
        best_f = min(best_f, es.best.f)
        popsize = popsize * popsize_amp
    return bestever.f, k_from_x(bestever.x)

# This function generates simplified models by gradually increasing lambda from a given initial model. 
def find_sparse_model(ck, path_save, dmode="split"):
    ck.save(path_save, "init")
    dim0 = ck.k.size
    ck.cut_lrk(ck.k_cut)
    if ck.k.size < dim0:
        ck.optimize(lam=0., k=ck.k)
        ck.save(path_save, "refine")
    
    sm = StatusManager(lam=1E-6, lam_step_init=1E-06)
    sm.dim_list = np.append(sm.dim_list, ck.k.size)
    for _ in range(100):
        _ck = copy.deepcopy(ck)
        _ck.lam = sm.lam
        print("lam = %.5f" % _ck.lam)
        _ck.optimize(lam=_ck.lam, k=_ck.k)
        _ck.cut_lrk(_ck.k_cut)
        sm.update(_ck.f, _ck.k.size)
        _ck.save(path_save, sm.status)
        
        if sm.should_refine and _ck.k.size > 2:
            _ck.optimize(lam=0., k=_ck.k)
            _ck.save(path_save, "refine")
            
        if sm.status == "finish":
            break
        elif sm.status != "reject":
            ck = _ck

def loss_wrap(funcptr_cr, d, l, r, k, lam):
    z0 = d[:,1:,0]
    z0_full = np.hstack([z0, np.zeros((z0.shape[0], l.shape[1]-z0.shape[1]))])
    x = np.log10(k+1)
    loss = loss_log(x, funcptr_cr, l, r, d, z0_full, lam)
    return loss

def chemreact(t, z, l, r, k):
    dz = np.sum((k * np.prod(np.power(z, l), axis = 1)).reshape(-1, 1)*(r - l), axis = 0)
    return dz
    
def x_from_k(k):
    return np.log10(k+1)

def k_from_x(x):
    return 10**x-1