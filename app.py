import pandas as pd
import numpy as np
from ypstruct import structure
from scipy.sparse import coo_matrix
from collections import Counter
import BPM_MF_algo
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
problem.maxiter = 250
problem.rows = max(userId_arr)+1
problem.cols = max(movieId_arr)+1


X_train, X_test = train_test_split(data_df, test_size=0.2, random_state=42)

#Cross validation setting
cv = KFold(5, shuffle=True, random_state=42)
cv_output = []
for k in range(5) :
    #rmse
    temp_list = []
    for i, (idx_train, idx_validation) in enumerate(cv.split(X_train)):
        print("{}_th cv computing".format(i))
        df_train = data_df.iloc[idx_train]
        df_validation = data_df.iloc[idx_validation]

        userId_tr  = df_train['userId'].values
        movieId_tr = df_train['movieId'].values
        rating_tr = df_train['rating'].values

        userId_vd  = df_validation['userId'].values
        movieId_vd = df_validation['movieId'].values
        rating_vd = df_validation['rating'].values

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
        params.alpha = 0.2
        params.beta = 5
        params.R = 4
        params.normal_loc = 0
        params.normal_var = 0.5
        params.latent_k = 5*(k+1)

        #run algorithm
        temp_list.append(BPM_MF_algo.fit(problem,params))

        #print result
        print("{}_th latent ".format(params.latent_k), "cv_{}_th MAE :".format(i), temp_list[i].MAE)
        print("{}_th latent ".format(params.latent_k), "cv_{}_th CMAE :".format(i), temp_list[i].CMAE)
        print("{}_th latent ".format(params.latent_k), "cv_{}_th 0_1_loss :".format(i), temp_list[i].zero_one_loss)

 
