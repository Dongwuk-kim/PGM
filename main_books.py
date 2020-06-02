import pandas as pd
import numpy as np
import random
from ypstruct import structure
from scipy.sparse import coo_matrix
from collections import Counter
from utils_books import books_data
from PMF import PMF
import BPM_MF_algo


# Import data
#### Google Travel Review Ratings ####

userId_arr, reviewId_arr, rating_arr = books_data()

from copy import deepcopy as dc

num_val = int( len(userId_arr) * 0.2 )

seeds = 2020

random.seed(seeds)
np.random.seed(seeds)

users = dc(userId_arr)
books = dc(reviewId_arr)
ratings = dc(rating_arr)

# train_user = []
# train_game = []
# train_rating = []

# val_user = []
# val_game = []
# val_rating = []

# random indexing

random.shuffle(users)
random.shuffle(books)
random.shuffle(ratings)

# train_user, train_game, train_rating 
train_tuple = (np.array(users[num_val:]), np.array(books[num_val:]), np.array(ratings[num_val:]) )

# val_user, val_game, val_rating 
val_tuple = ( np.array(users[:num_val]), np.array(books[:num_val]), np.array(ratings[:num_val]) )

#coordinate transformation minus 1

# userId_arr -= 1
# reviewId_arr -= 1

#problem setting

algorithm = ["BPM", "PMF"]
choose = algorithm[0]

if choose == "BPM" :

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

elif choose == "PMF" :
    
    pmf = PMF()
    
    output = pmf.fit(train_tuple, val_tuple)