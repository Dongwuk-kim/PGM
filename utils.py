import os
import numpy as np
import random
from scipy.stats import norm as norm


def books_data() :
    
#     reviewId_arr = np.load("books_review/books.npy")
    reviewId_arr = np.load("review data/book/booksId.npy")
    userId_arr = np.load("review data/book/users.npy")
    rating_arr = np.load("review data/book/ratings.npy")
    
    #### use only 200000 samples from 960529 samples ####
    
    seeds = 2020
    random.seed(seeds)
    np.random.seed(seeds)

    from copy import deepcopy

    users = deepcopy(userId_arr)
    items = deepcopy(reviewId_arr)
    ratings = deepcopy(rating_arr)

    random.shuffle(users)
    random.shuffle(items)
    random.shuffle(ratings)

    new_tuple = ( users[:200000], items[:200000], ratings[:200000] )
    
    new_userId_arr = new_tuple[0]
    new_reviewId_arr = new_tuple[1]
    new_rating_arr = new_tuple[2]
    
    assert len(new_userId_arr) == len(new_reviewId_arr)
    assert len(new_userId_arr) == len(new_rating_arr)
    
    return new_userId_arr, new_reviewId_arr, new_rating_arr


def steam_data() :
    
#     userId_arr = np.load("steam_review/users.npy")
    reviewId_arr = np.load("review data/game/games.npy")
    userId_arr = np.load("review data/game/users_sortedID.npy")
    rating_arr = np.load("review data/game/ratings.npy")
    
    assert len(userId_arr) == len(reviewId_arr)
    assert len(userId_arr) == len(rating_arr)
    
    return userId_arr, reviewId_arr, rating_arr


def movie_data() :
    
    reviewId_arr = np.load("review data/movie/movies.npy")
    userId_arr = np.load("review data/movie/users.npy")
    rating_arr = np.load("review data/movie/ratings.npy")
    
    assert len(userId_arr) == len(reviewId_arr)
    assert len(userId_arr) == len(rating_arr)
    
    return userId_arr, reviewId_arr, rating_arr