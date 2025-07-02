import numpy as np
import numba as nb
from numba import types
from numbalsoda import lsoda, address_as_void_pointer

# rate equations with x (=log(k+1)) as the argument
def cr_log(t, z_, dz_, l, r, x):
    z = nb.carray(z_, l.shape[1])
    dz = nb.carray(dz_, l.shape[1])
    # dz = np.sum((k * np.prod(np.power(z, l), axis = 1)).reshape(-1, 1)*(r - l), axis = 0)
    temp = np.zeros(l.shape[1])
    for j in range(x.shape[0]):
        temp += (10**x[j]-1) * np.prod(np.power(z, l[j,:]))*(r[j,:]-l[j,:])
    for i in range(dz.shape[0]):
        dz[i] = temp[i]

args_dtype_cr = types.Record.make_c_struct([
                    ('address_l', types.int64),
                    ('len_l_0', types.int64),
                    ('len_l_1', types.int64),
                    ('address_r', types.int64),
                    ('len_r_0', types.int64),
                    ('len_r_1', types.int64),
                    ('address_x', types.int64),
                    ('len_x', types.int64)])

# this function will create the numba function to pass to lsoda.
def create_jit_cr(cr, args_dtype):
    ##########################################################
    ## This function was created by modifying the code found in the numbalsoda documentation.
    ## https://github.com/Nicholaswogan/numbalsoda/blob/main/passing_data_to_rhs_function.ipynb
    ##########################################################
    jitted_cr = nb.njit(cr)
    @nb.cfunc(types.void(types.double,
             types.CPointer(types.double),
             types.CPointer(types.double),
             types.CPointer(args_dtype)))
    def wrapped(t, u, du, user_data_p):
        # unpack p and arr from user_data_p
        user_data = nb.carray(user_data_p, 1)
        len_l_0 = user_data[0].len_l_0
        len_l_1 = user_data[0].len_l_1
        l = nb.carray(address_as_void_pointer(user_data[0].address_l),(len_l_0, len_l_1), dtype=np.float64)
        len_r_0 = user_data[0].len_r_0
        len_r_1 = user_data[0].len_r_1
        r = nb.carray(address_as_void_pointer(user_data[0].address_r),(len_r_0, len_r_1), dtype=np.float64)
        len_x = user_data[0].len_x
        x = nb.carray(address_as_void_pointer(user_data[0].address_x),(len_x,), dtype=np.float64)
        
        # then we call the jitted rhs function, passing in data
        jitted_cr(t, u, du, l, r, x) 
    return wrapped

# loss function (= MRSE + lam * ||x||1)
@nb.njit(parallel = True)
def loss_log(x, funcptr_cr, l, r, d, z0_full, lam):
        
    args_cr = np.array((l.ctypes.data,
                 l.shape[0], 
                 l.shape[1], 
                 r.ctypes.data,
                 r.shape[0], 
                 r.shape[1], 
                 x.ctypes.data, 
                 x.shape[0],
                 d.ctypes.data,
                 d.shape[0], 
                 d.shape[1], 
                 d.shape[2]),dtype='int64')
    
    sol = np.empty((d.shape[0], l.shape[1], d.shape[2]), dtype = 'float64')
    usol = np.empty((d.shape[2], l.shape[1]), dtype = 'float64')
    t_eval = np.empty(d.shape[2], dtype = 'float64')
    success = True
    loss: float = 0.0
    residues = np.empty(d.shape[0], dtype = 'float64')
        
    for i in nb.prange(d.shape[0]):
        t_eval = d[i,0,:].astype('float64')
        usol, success = lsoda(funcptr_cr, z0_full[i, :], t_eval, data = args_cr)
        sol[i,:,:] = usol.T
        residues[i] = np.sum((d[i,1:3,:] - sol[i,0:2,:])**2) / np.sum((d[i,1:3,:]-np.mean(d[i,1:3,:]))**2)
    
    loss = 1/residues.size * np.sum(residues) + lam * np.sum(x)
    return loss
