import pandas as pd
import numpy as np
from scipy import special
from ypstruct import structure
from numpy import linalg as LA
def dg(x) :
    return special.digamma(x)

class BPM_MatrixFactorization :

    def __init__(self, problem, params):
            
        #Problem information
        self.data_m = problem.data_m
        self.rows = problem.rows
        self.cols = problem.cols
        self.test_m = problem.test_m
            
        #Parmeters
        self.latent_k = params.latent_k
        self.alpha = params.alpha
        self.beta = params.beta
        self.R = params.R
        self.normal_loc = params.normal_loc
        self.normal_var = params.normal_var

        #Initializing
        #check(frobenus norm . l2 norm, maxeliment )
        self.gamma_m = np.zeros((self.rows, self.latent_k))
        self.eps_plus_m = np.zeros((self.cols, self.latent_k))
        self.eps_minus_m = np.zeros((self.cols, self.latent_k))
        
        self.lambda_m = np.zeros((self.rows, self.cols, self.latent_k))
        self.a_m = np.zeros((self.rows, self.latent_k))
        self.b_m = np.zeros((self.cols, self.latent_k))
        self.p_m = np.zeros((self.rows, self.cols))
        self.q_m = np.zeros((self.rows, self.cols))

        #Calculate r_plus & r_minus matrix        
        self.r_plus_m = problem.data_m.copy()
        self.r_plus_m.data[:] = self.r_plus_m.data - 1
        self.r_minus_m = problem.data_m.copy()
        self.r_minus_m.data[:] =  5- self.r_minus_m.data
        """
        self.r_plus_m = problem.data_m.copy()
        self.r_plus_m.data[:] = (self.R/5)* self.r_plus_m.data
        self.r_minus_m = problem.data_m.copy()
        self.r_minus_m.data[:] =  self.R - (self.R/5)*self.r_minus_m.data
        """


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
        for idx in range(len(self.data_m.data)) :
            u = self.data_m.row[idx]
            i = self.data_m.col[idx]
            for k in range(self.latent_k) :
                sum_lambda[u,k] += self.lambda_m[u, i, k]
        self.gamma_m = self.alpha + sum_lambda

    def update_epslion(self):
        sum_lambda_mul_r_plus = np.zeros((self.cols, self.latent_k))
        sum_lambda_mul_r_minus = np.zeros((self.cols, self.latent_k))
        for idx in range(len(self.data_m.data)) :
            u = self.data_m.row[idx]
            i = self.data_m.col[idx]
            for k in range(self.latent_k) :
                sum_lambda_mul_r_plus[i,k] += self.lambda_m[u, i, k]*self.r_plus_m.data[idx]
                sum_lambda_mul_r_minus[i,k] += self.lambda_m[u, i, k]*self.r_minus_m.data[idx]
        self.eps_plus_m = self.beta + sum_lambda_mul_r_plus
        self.eps_minus_m = self.beta + sum_lambda_mul_r_minus

    def update_lambda(self):
        for idx in range(len(self.data_m.data)) :
            u = self.data_m.row[idx]
            i = self.data_m.col[idx]
            for k in range(self.latent_k):
                self.lambda_m[u,i,k] = np.exp(dg(self.gamma_m[u,k]) + self.r_plus_m.data[idx] * dg(self.eps_plus_m[i,k]) + \
                                                self.r_minus_m.data[idx] * dg(self.eps_minus_m[i,k])- \
                                                    self.R * dg(self.eps_plus_m[i,k]+self.eps_minus_m[i,k]))
            self.lambda_m[u,i,:] =  self.lambda_m[u,i,:] / np.sum(self.lambda_m[u,i,:])

    def update_a_m(self) :
        for user in range(self.rows) :
            self.a_m[user, :] = self.gamma_m[user, :] / np.sum(self.gamma_m[user, :])
    
    def update_b_m(self) :
        for item in range(self.cols) :
            self.b_m[item, :] = self.eps_plus_m[item, :] / (self.eps_minus_m[item, :] + self.eps_plus_m[item, :])

    def update_p_m(self) :
        # editted: 05/26, minwoo
        self.p_m[:, :] = np.einsum("uk,ik->ui", self.a_m, self.b_m)

    def update_q_m(self) : 
        # editted: 05/26, minwoo
        self.q_m[:, :] = np.ceil(self.p_m * 5)

    def cal_MAE(self) :
        t=0
        x=0
        for idx in range(len(self.test_m.data)):
            u = self.test_m.row[idx]
            i = self.test_m.col[idx]
            t += np.abs(self.q_m[u,i] - self.test_m.data[idx])
            x += 1
        return t/x
    
    def cal_CMAE(self) :
        t=0
        x=0
        for idx in range(len(self.test_m.data)):
            u = self.test_m.row[idx]
            i = self.test_m.col[idx]
            if self.test_m.data[idx] < 4 or self.q_m[u,i] < 4 :
                continue
            x += 1
            t += np.abs(self.q_m[u,i] - self.test_m.data[idx])
        if x == 0 :
            return ("no_case")

        return t/x

    def cal_zero_one_loss(self) :
        t=0
        x=0
        for idx in range(len(self.test_m.data)):
            u = self.test_m.row[idx]
            i = self.test_m.col[idx]
            x +=1
            if self.test_m.data[idx] < 4 and self.q_m[u,i] < 4 :
                continue
            if self.test_m.data[idx] >= 4 and self.q_m[u,i] >= 4 :
                continue
            t +=1
        return t/x

def fit(problem, params) :
    bpm_MatrixFactorization = BPM_MatrixFactorization(problem, params)


    #initialize gamma
    bpm_MatrixFactorization.gen_random_gamma()
    bpm_MatrixFactorization.gen_random_epslion()

    #define output
    outputs = structure()

    #repeat until maxiteration
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
            print("Matrix norm :",LA.norm(pre_q_m - bpm_MatrixFactorization.q_m))

        pre_q_m = bpm_MatrixFactorization.q_m.copy()


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

    




