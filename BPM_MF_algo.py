import pandas as pd
import numpy as np
from scipy import special
from ypstruct import structure

def dg(x) :
    return special.digamma(x)

class BPM_MatrixFactorization :

    def __init__(self, problem, params):
            
        #Problem information
        self.data_m = problem.data_m
        self.rows = problem.rows
        self.cols = problem.cols
        self.latent_k = problem.latent_k
            
        #Parmeters
        self.alpha = params.alpha
        self.beta = params.beta
        self.R = params.R
        self.normal_loc = params.normal_loc
        self.normal_var = params.normal_var

        #Initializing
        self.gamma_m = np.zeros((self.rows, self.latent_k))
        self.lambda_m = np.zeros((self.rows, self.cols, self.latent_k))
        self.eps_plus_m = np.zeros((self.cols, self.latent_k))
        self.eps_minus_m = np.zeros((self.cols, self.latent_k))
        self.a_m = np.zeros((self.rows, self.latent_k))
        self.b_m = np.zeros((self.cols, self.latent_k))

        #Calculate r_plus & r_minus matrix        
        self.r_plus_m = problem.data_m.copy()
        self.r_plus_m.data[:] = (self.R/5)* self.r_plus_m.data
        self.r_minus_m = problem.data_m.copy()
        self.r_minus_m.data[:] =  self.R - (self.R/5)*self.r_minus_m.data

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
      
    def update_a_m(self) :
        for user in range(self.rows) :
            self.a_m[user, :] = self.gamma_m[user, :] / np.sum(self.gamma_m[user, :])
    
    def update_b_m(self) :
        for item in range(self.cols) :
            self.b_m[item, :] = self.eps_plus_m[item, :] / (self.eps_minus_m[item, :] + self.eps_plus_m[item, :])
    
def run(problem, params) :
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

    outputs.a_m = bpm_MatrixFactorization.a_m
    outputs.b_m = bpm_MatrixFactorization.b_m
    outputs.lambda_m = bpm_MatrixFactorization.lambda_m
    outputs.gamma_m = bpm_MatrixFactorization.gamma_m

    return outputs
    




