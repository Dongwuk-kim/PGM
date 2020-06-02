import pandas as pd
import numpy as np
from ypstruct import structure
from scipy.sparse import coo_matrix
from collections import Counter
from utils_steam import steam_data
from PMF import PMF
import BPM_MF_algo


# Import data
#### Google Travel Review Ratings ####

userId_arr, reviewId_arr, rating_arr = steam_data()

#coordinate transformation minus 1

# userId_arr -= 1
# reviewId_arr -= 1

#problem setting

problem = structure()
problem.maxiter = 5
problem.rows = max(userId_arr) + 1
problem.cols = max(reviewId_arr) + 1

sparse_user_item_m = coo_matrix((rating_arr, (userId_arr, reviewId_arr)), \
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

BPM_MF_algo.run(problem,params)
