import pandas as pd
import numpy as np
from ypstruct import structure
from scipy.sparse import coo_matrix
from collections import Counter
import BPM_MF_algo


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
problem.maxiter = 5
problem.rows = max(userId_arr)+1
problem.cols = max(movieId_arr)+1

sparse_user_item_m = coo_matrix((rating_arr, (userId_arr, movieId_arr)), \
                     shape=(problem.rows, problem.cols))

problem.data_m = sparse_user_item_m
problem.latent_k = 10

#Parameters setting
params = structure()
params.alpha = 0.2
params.beta = 5
params.R = 4
params.normal_loc = 0
params.normal_var = 0.5

#run algorithm
output = BPM_MF_algo.run(problem,params )
