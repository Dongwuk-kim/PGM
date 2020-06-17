import pandas as pd
import numpy as np
from scipy import special
from ypstruct import structure
from numpy import linalg as LA
import json
from numba.experimental import jitclass
from numba import jit
from numba import double, int32, float32, int64
#from numba import jit


spec = [
    ('data_u', int32[:]),
    ('data_i', int32[:]),  
    ('data_v', int64[:]),             
    ('rows', int32),
    ('cols', int32),
    ('test_u', int32[:]),
    ('test_i', int32[:]),
    ('test_v', float32[:]),
    ('latent_k', int32),
    ('alpha', float32),
    ('beta', float32),
    ('R', int32),
    ('normal_loc', float32),
    ('normal_var', float32),
    ('gamma_m', float32[:,:]),
    ('eps_plus_m', float32[:,:]),
    ('eps_minus_m', float32[:,:]),
    ('lambda_m', float32[:,:]),
    ('a_m', float32[:,:]),
    ('b_m', float32[:,:]),
    ('p_m', float32[:,:]),
    ('q_m', float32[:,:]),
    ('r_plus_v', float32[:,:]),
    ('r_minus_v', float32[:,:])
]

@jit(nopython=True, fastmath=True)
def dg(x):
    r = 0
    while x<=5:
        r -= 1/x
        x += 1
    f = 1/(x*x)
    t = f*(-1/12.0 + f*(1/120.0 + f*(-1/252.0 + f*(1/240.0 + f*(-1/132.0
        + f*(691/32760.0 + f*(-1/12.0 + f*3617/8160.0)))))))
    return r + np.log(x) - 0.5/x + t

@jitclass(spec)
class BPM_MatrixFactorization :
    def __init__(self, problem, params):
        #Problem information
        self.data_u = problem.data_m.row
        self.data_i = problem.data_m.col
        self.data_v = problem.data_m.data

        self.test_u = problem.test_m.row
        self.test_i = problem.test_m.col
        self.test_v = problem.test_m.data
        
        self.rows = problem.rows
        self.cols = problem.cols

        #Parmeters
        self.latent_k = params.latent_k
        self.alpha = params.alpha
        self.beta = params.beta
        self.R = params.R
        self.normal_loc = params.normal_loc
        self.normal_var = params.normal_var

        #Initializing
        self.gamma_m = np.zeros((self.rows, self.latent_k), dtype=np.float32)
        self.eps_plus_m = np.zeros((self.cols, self.latent_k), dtype=np.float32)
        self.eps_minus_m = np.zeros((self.cols, self.latent_k), dtype=np.float32)
        
        self.lambda_m = np.zeros((self.rows, self.cols, self.latent_k), dtype=np.float32)
        self.a_m = np.zeros((self.rows, self.latent_k), dtype=np.float32)
        self.b_m = np.zeros((self.cols, self.latent_k), dtype=np.float32)
        self.p_m = np.zeros((self.rows, self.cols), dtype=np.float32)
        self.q_m = np.zeros((self.rows, self.cols), dtype=np.float32)
        
        '''
        self.r_plus_v = np.zeros(len(self.data_v), dtype=np.float32)
        self.r_minus_v = np.zeros(len(self.data_v), dtype=np.float32)
        '''
        #Calculate r_plus & r_minus matrix        
        self.r_plus_v = self.data_v.copy()
        self.r_plus_v = self.r_plus_v - 1

        self.r_minus_v = self.data_v.copy()
        self.r_minus_v = 5 - self.r_minus_v
 
    def gen_random_gamma(self) :
        for user in range(self.rows) :
            random_arr = np.random.uniform(0,1,self.latent_k)
            random_arr /= np.sum(random_arr)
            self.gamma_m[user,:] = random_arr
    
    def gen_random_epslion(self) :
        for item in range(self.cols) :
            random_arr = np.random.normal(self.normal_loc,self.normal_var,self.latent_k)
            random_arr_2 = np.random.normal(self.normal_loc,self.normal_var,self.latent_k)

            random_arr += self.beta
            random_arr_2 += self.beta

            random_arr = np.where( random_arr <0, 0, random_arr )
            random_arr_2 = np.where( random_arr_2 <0, 0, random_arr_2 )
            self.eps_plus_m[item,:] = random_arr
            self.eps_minus_m[item,:] = random_arr_2
    
    def update_gamma(self) :
        sum_lambda = np.zeros((self.rows, self.latent_k))
        for idx in range(len(self.data_v)) :
            u = self.data_u[idx]
            i = self.data_i[idx]
            for k in range(self.latent_k) :
                sum_lambda[u,k] += self.lambda_m[u, i, k]
        self.gamma_m = self.alpha + sum_lambda

    def update_epslion(self):
        sum_lambda_mul_r_plus = np.zeros((self.cols, self.latent_k))
        sum_lambda_mul_r_minus = np.zeros((self.cols, self.latent_k))
        for idx in range(len(self.data_v)) :
            u = self.data_u[idx]
            i = self.data_i[idx]
            for k in range(self.latent_k) :
                sum_lambda_mul_r_plus[i,k] += self.lambda_m[u, i, k]*self.r_plus_v[idx]
                sum_lambda_mul_r_minus[i,k] += self.lambda_m[u, i, k]*self.r_minus_v[idx]
        self.eps_plus_m = self.beta + sum_lambda_mul_r_plus
        self.eps_minus_m = self.beta + sum_lambda_mul_r_minus

    def update_lambda(self):
        for idx in range(len(self.data_v)) :
            u = self.data_u[idx]
            i = self.data_i[idx]
            for k in range(self.latent_k):
                self.lambda_m[u,i,k] = np.exp(dg(self.gamma_m[u,k]) + self.r_plus_v[idx] * dg(self.eps_plus_m[i,k]) + \
                                                self.r_minus_v[idx] * dg(self.eps_minus_m[i,k])- \
                                                    self.R * dg(self.eps_plus_m[i,k]+self.eps_minus_m[i,k]))
            self.lambda_m[u,i,:] =  self.lambda_m[u,i,:] / np.sum(self.lambda_m[u,i,:])
    
    def update_a_m(self) :
        for user in range(self.rows) :
            self.a_m[user, :] = self.gamma_m[user, :] / np.sum(self.gamma_m[user, :])
    
    def update_b_m(self) :
        for item in range(self.cols) :
            self.b_m[item, :] = self.eps_plus_m[item, :] / (self.eps_minus_m[item, :] + self.eps_plus_m[item, :])

    def update_p_m(self) :
        for u in range(self.rows):
            for i in range(self.cols):
                for k in range(self.latent_k):
                    self.p_m[u, i] += self.a_m[u, k] + self.b_m[i, k]

    def update_q_m(self) : 
        self.q_m[:, :] = np.ceil(self.p_m * 5)

    def cal_MAE(self) :
        t=0
        x=0
        for idx in range(len(self.test_v)):
            u = self.test_u[idx]
            i = self.test_i[idx]
            t += np.abs(self.q_m[u,i] - self.test_v[idx])
            x += 1
        return t/x
    
    def cal_CMAE(self) :
        t=0
        x=0
        for idx in range(len(self.test_v)):
            u = self.test_u[idx]
            i = self.test_i[idx]
            if self.test_v[idx] < 4 or self.q_m[u,i] < 4 :
                continue
            x += 1
            t += np.abs(self.q_m[u,i] - self.test_v[idx])
        if x == 0 :
            return ("no_case")

        return t/x

    def cal_zero_one_loss(self) :
        t=0
        x=0
        for idx in range(len(self.test_v)):
            u = self.test_u[idx]
            i = self.test_i[idx]
            x +=1
            if self.test_v[idx] < 4 and self.q_m[u,i] < 4 :
                continue
            if self.test_v[idx] >= 4 and self.q_m[u,i] >= 4 :
                continue
            t +=1
        return t/x

    def cal_rmse(self) :
        t=0
        x=0
        for idx in range(len(self.test_v)):
            u = self.test_u[idx]
            i = self.test_i[idx]
            x +=1
            t += (self.q_m[u,i] - self.test_v[idx])**2
        return np.sqrt(t/x)
    


def fit(problem, params) :
    bpm_MatrixFactorization = BPM_MatrixFactorization(problem, params)

    #initialize gamma
    bpm_MatrixFactorization.gen_random_gamma()
    bpm_MatrixFactorization.gen_random_epslion()

    #define output
    outputs = structure()

    #repeat until maxiteration
    matrix_norm_list = []
    cmae_list = []
    rmse_list = []
    mae_list = []
    zero_one_list = []
    for iter in range(problem.maxiter) :

        print("{}_iteration computing".format(iter))
        bpm_MatrixFactorization.update_lambda()
        bpm_MatrixFactorization.update_gamma()
        bpm_MatrixFactorization.update_epslion()

        #update a & b matrix
        bpm_MatrixFactorization.update_a_m()
        bpm_MatrixFactorization.update_b_m()

        #update p & q matrix
        bpm_MatrixFactorization.update_p_m()
        bpm_MatrixFactorization.update_q_m()

        if iter != 0 :
            matrix_norm  = LA.norm(pre_q_m - bpm_MatrixFactorization.q_m)
            print("Matrix norm :",matrix_norm)
        else :
            matrix_norm = 999
        
        matrix_norm_list.append(matrix_norm)
        cmae_list.append(bpm_MatrixFactorization.cal_CMAE())
        mae_list.append(bpm_MatrixFactorization.cal_MAE())
        zero_one_list.append(bpm_MatrixFactorization.cal_zero_one_loss())
        rmse_list.append(bpm_MatrixFactorization.cal_rmse())
        pre_q_m = bpm_MatrixFactorization.q_m.copy()

        if matrix_norm < 1 :
            break

    summary_dic ={}
    summary_dic["F_norm"] = matrix_norm_list
    summary_dic["CMAE"] = cmae_list
    summary_dic["01_loss"] = zero_one_list
    summary_dic['RMSE'] = rmse_list
    
    '''
    with open("Beta_{}_k_{}_summary_dic.json".format(params.beta,params.latent_k), "w") as json_file:
        json.dump(summary_dic, json_file)
    '''
    

    outputs.a_m = bpm_MatrixFactorization.a_m
    outputs.b_m = bpm_MatrixFactorization.b_m
    outputs.lambda_m = bpm_MatrixFactorization.lambda_m
    outputs.gamma_m = bpm_MatrixFactorization.gamma_m
    outputs.p_m = bpm_MatrixFactorization.p_m  # added
    outputs.q_m = bpm_MatrixFactorization.q_m  # added
    outputs.MAE = bpm_MatrixFactorization.cal_MAE()
    outputs.CMAE = bpm_MatrixFactorization.cal_CMAE()
    outputs.zero_one_loss = bpm_MatrixFactorization.cal_zero_one_loss()
    outputs.params = params

    return outputs

    




