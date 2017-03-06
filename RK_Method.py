import numpy as np
import sympy as sy
import matplotlib.pyplot as plt

def str2sym(var_list):
    var_list = var_list.split(' ')
    var_store = list(np.zeros(len(var_list)))
    for n in range(len(var_list)):
        var_store [n] = sy.symbols(var_list[n])
    return var_store
static = str2sym(static)
def num_insertor(sym_func, var, val):
    dummy_sym = sym_func
    for n in range(len(var)):
        dummy_sym = dummy_sym.subs(var[n], val[n])
    sym_func = dummy_sym
    return sym_func
equation = sy.sympify(equation)
for k in range(len(equation)):
    equation[k] = num_insertor (equation[k], static, static_init)

def RK_Operation(rktype, equation, d_val, val_d, d_val_init, val_d_init, h, incr):
    def str2sym(var_list):
        var_list = var_list.split(' ')
        var_store = list(np.zeros(len(var_list)))
        for n in range(len(var_list)):
            var_store [n] = sy.symbols(var_list[n])
        return var_store

    def num_storage(var_store ,var_init, incr):
        num_store = np.zeros([len(var_store),incr])
        num_store[0:len(var_store),0] = var_init
        return num_store

    def prg_print(prg, incr):
        k = list(range(1,10))
        d = prg * 10 / incr
        if d in k:
            m = (k.index(d) + 1) * 10
            print('Progress ',m,'%')

    def zeros_trim(num):
        a = len(np.trim_zeros(num[0,:]))
        b = len(num[0,:])
        num = np.delete(num,list(range(a,b)),axis=1)
        return num

    def num_insertor(sym_func, var, val):
        dummy_sym = sym_func
        for n in range(len(var)):
            dummy_sym = dummy_sym.subs(var[n], val[n])
        sym_func = dummy_sym
        return sym_func

    def RKtype(rktype):
        if RKtype == "Classic":
            a = np.zeros([4,4])
            a[1,0] = 0.5
            a[2,1] = 0.5
            a[3,2] = 1.
            b = [1/6., 1/3., 1/3., 1/6.]
            c = [0., 0.5, 0.5, 1.]
            dim = len(b)
            k_as = np.zeros([len(ODE_func), dim])
        elif RKtype == 'Euler':
            a = np.ones([1,1])
            b = [1]
            c = [1]
            dim = len(b)
            k_as = np.zeros([len(ODE_func), dim])
        elif RKtype == '3/8Rule':
            a = np.zeros([4,4])
            a[1,0] = 1/3.
            a[2,0],a[2,1] = -1/3.,1
            a[3,0],a[3,1],a[3,2] = 1.,-1.,1.
            b = [1/8., 3/8., 3/8., 1/8.]
            c = [0., 1/3., 2/3., 1.]
            dim = len(b)
            k_as = np.zeros([len(ODE_func), dim])
        elif RKtype =='Heun':
            a = np.zeros([2,2])
            a[1,0] = 0.5
            b = [0,1]
            c = [0, 0.5]
            dim = len(b)
            k_as = np.zeros([len(ODE_func), dim])
        else:
            a = np.zeros([4,4])
            a[1,0] = 0.5
            a[2,1] = 0.5
            a[3,2] = 1.
            b = [1/6., 1/3., 1/3., 1/6.]
            c = [0., 0.5, 0.5, 1.]
            dim = len(b)
        return a,b,c,dim

    def RKtypeadapt(rktype):
        if rktype == "CashKarp":
            a = np.zeros([6,6])
            a[1,0] = 1/5.
            a[2,0:2] = 3/40., 9/40.
            a[3,0:3] = 3/10.,-9/10.,6/5
            a[4,0:4] = -11/54.,5/2.,-70/27.,35/27.
            a[5,0:5] = 1631/55296., 175/512., 575/13824., 44275/110592., 253/4096.
            b1 = [37/378., 0., 250/621., 125/594., 0., 512/1771.]
            b2 = [2825/27648., 0., 18575/48384., 13525/55296., 277/14336., 1/4.]
            b_diff = [-0.004293774801587311, 0.0, 0.018668586093857853, -0.034155026830808066, -0.019321986607142856, 0.03910220214568039]
            c = [0., 1/5., 3/10., 3/5., 1., 7/8]
            dim = len(c)
        return a, b1, b2, b_diff, c, dim

    def iteration(equation ,d_val, val_d, h, incr, a, b, c, dim):
        for j in range(1,incr): #LOOP for Whole RK Operation
            k_as = np.zeros([len(equation),dim])
            for k in range(dim): #LOOP for every K in RK Opearation
                for l in range(len(equation)): #LOOP FOR K CALCULATION
                    sum_in_k = np.zeros(len(equation))
                    for m in range (dim): #LOOP FOR K SUMMATION WITH B
                        for o in range(len(equation)):
                            sum_in_k[o] += a[k,m] * k_as[o,m]
                    subs1 = num_insertor(equation[l],d_val,d_val_num[:,j-1] + sum_in_k)
                    print(subs1)
                    subs2 = num_insertor(subs1,val_d,val_d_num[:,j-1] + c[k] * h)
                    k_as[l,k] = subs2 * h
            for k in range(dim): #Generating new K with B
                k_as[:,k] = k_as[:,k] * b[k]
            for n in range(len(equation)): #Inserting new value to storage
                d_val_num[n,j] = d_val_num [n,j-1] + sum(k_as[n,:])
                val_d_num[n,j] = val_d_num [n,j-1] + h
        return d_val_num, val_d_num

    def iteration_adapt(equation,d_val,val_d,h,incr, d_val_num, val_d_num,a,b1,b2,b_diff,c,dim):
        print(a)
        minimun_err = 10**-4
        iter_end = h * incr
        h_hist = np.zeros([len(equation),incr])
        delta = np.zeros([len(equation),dim])
        j = 1 #counter
        h_list = np.zeros(len(equation))
        for n in range(len(equation)):
            h_list[n] = h
            h_hist[n,0] = h
        while iter_end > 0:
            conditional = 0
            j += 1
            k_as = np.zeros([len(equation),dim])
            for k in range(dim):
                for l in range(len(equation)):
                    sum_in_k = np.zeros(len(equation))
                    for m in range (dim):
                        for o in range(len(equation)):
                            sum_in_k[o] += a[k,m] * k_as[o,m]
                    subs1 = num_insertor(equation[l],d_val,d_val_num[:,j-1] + sum_in_k)
                    print(subs1)
                    subs2 = num_insertor(subs1,val_d,val_d_num[:,j-1] + c[k] * h_list[l])
                    k_as[l,k] = subs2 * h_list[l]
            for k in range(dim):
                k_as[:,k] = k_as[:,k] * b1[k]
                delta[:,k]  = k_as[:,k] * b_diff[k]
            for n in range(len(equation)):
                d_val_num[n,j] = d_val_num [n,j-1] + sum(k_as[n,:])
                val_d_num[n,j] = val_d_num [n,j-1] + h_list[n]
            err = np.zeros(len(equation))
            for n in range(len(equation)):
                err[n] = sum(delta[n,:])
            for n in range(len(equation)):
                if min(err[n]) >= minimun_err:
                    h_list[n] *= abs(minimun_err/min(err[n])) ** 0.2
                else:
                    h_list[n] *= abs(minimun_err/min(err[n])) ** 0.25
            h_hist[j,:] = h_list
            if j  > j - incr/10.:
                print('Adding more array for storing number... ')
                d_val_num = np.concatenate((d_val_num,np.zeros([len(equation),int(incr/10)])),axis = 1)
                val_d_num = np.concatenate((val_d_num,np.zeros([len(equation),int(incr/10)])),axis = 1)
            for n in range(len(equation)):
                if sum(h_hist [n,:]) == iter_end:
                    conditional = 1
            if conditional == 1:
                break
        d_val_num = zeros_trim(d_val_num)
        val_d_num = zeros_trim(val_d_num)
        return d_val_num, val_d_num

    print('Symboling Variables... ')
    equation = sy.sympify(equation)
    d_val = str2sym(d_val)
    val_d = str2sym(val_d)
    print('Creating Numeric Storage... ')
    d_val_num = num_storage(d_val, d_val_init, incr)
    val_d_num = num_storage(val_d, val_d_init, incr)
    if rktype == "CashKarp":
        print ('Initialize ',rktype,'... ')
        a,b1,b2,b_diff,c,dim = RKtypeadapt(rktype)
        print ('Starting Iteration... ')
        d_val_num, val_d_num = iteration_adapt(equation, d_val, val_d, h, incr, d_val_num, val_d_num, a, b1, b2, b_diff, c, dim)
        print ('Iteration finished')
    else:
        print ('Initialize ',rktype,'... ')
        a,b,c,dim = RKtype(rktype)
        print ('Starting Iteration... ')
        d_val_num, val_d_num = iteration(equation, d_val, val_d, h, incr, a, b, c, dim)
        print ('Iteration finished')
    return d_val_num, val_d_num
