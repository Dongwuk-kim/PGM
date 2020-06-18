import pandas as pd
import numpy as np
from ypstruct import structure
from scipy.sparse import coo_matrix
from collections import Counter
import BPM_MF_algo
import BPM_MF_algo_hybrid
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
import json



#Import data
colnames = ['userId', 'movieId', 'rating', 'timeStamp']
data_df = pd.read_csv('E:/DW/확률그래프모델/project/ml-100k/u.data',sep='\t', 
                        names = colnames, header = None)
# data_df = pd.read_csv('E:/DW/확률그래프모델/project/ml-25m/ratings.csv')
#rating_df = data_df.pivot(index='userId', columns='movieId', values='rating')

userId_list = Counter(data_df['userId'])
movieId_list = Counter(data_df['movieId'])
ratings_list = Counter(data_df['rating'])

userId_arr  = data_df['userId'].values.copy()
movieId_arr = data_df['movieId'].values.copy()
rating_arr = data_df['rating'].values.copy()

#coordinate transformation minus 1
userId_arr -= 1
movieId_arr -= 1

#problem setting
problem = structure()
problem.maxiter = 100
problem.rows = max(userId_arr)+1
problem.cols = max(movieId_arr)+1
problem.dir = 'test_result/'



X_train, X_test = train_test_split(data_df, test_size=0.2, random_state=42)

#Get Test Result


userId_tr  = X_train['userId'].values
movieId_tr = X_train['movieId'].values
rating_tr = X_train['rating'].values

userId_vd  = X_test['userId'].values
movieId_vd = X_test['movieId'].values
rating_vd = X_test['rating'].values
    
userId_tr -= 1
movieId_tr -= 1

userId_vd -= 1
movieId_vd -= 1

sparse_train_m = coo_matrix((rating_tr, (userId_tr, movieId_tr)), \
                shape=(problem.rows, problem.cols))

sparse_validation_m = coo_matrix((rating_vd, (userId_vd, movieId_vd)), \
                shape=(problem.rows, problem.cols))

problem.data_m = sparse_train_m
problem.test_m = sparse_validation_m

#Parameters setting
params = structure()
params.alpha = 0.8
params.beta = 5
params.R = 4
params.normal_loc = 0
params.normal_var = 0.5
params.latent_k = 5

#run algorithm
bpm = BPM_MF_algo_hybrid.fit(problem,params)

#print result
print("{}_th latent ".format(params.latent_k), "test MAE : ", bpm.MAE)
print("{}_th latent ".format(params.latent_k), "test_th CMAE:", bpm.CMAE)
print("{}_th latent ".format(params.latent_k), "test_th 0_1_loss : ", bpm.zero_one_loss)
print("{}_th latent ".format(params.latent_k), "test_th RMSE :", bpm.RMSE)


