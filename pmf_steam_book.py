import pandas as pd
import numpy as np
import random

from surprise import SVD
from surprise import NormalPredictor
from surprise import Dataset
from surprise import Reader
from surprise import accuracy
from surprise.model_selection import cross_validate
from surprise.model_selection import train_test_split
from utils_steam import steam_data
from utils_books import books_data

"""
Calling data (steam games, books)
"""

datanames = ("steam", "book")

for name in datanames :

    if name == "steam" :
        userID, itemID, rating = steam_data()
    elif name == "book" :
        userID, itemID, rating = books_data()

    """
    Split train / test dataset = 80% : 20%
    """

    num_val = int( len(userID) * 0.2 )

    seeds = 2020

    random.seed(seeds)
    np.random.seed(seeds)

    from copy import deepcopy

    users = deepcopy(userID)
    items = deepcopy(itemID)
    ratings = deepcopy(rating)

    random.shuffle(users)
    random.shuffle(items)
    random.shuffle(ratings)

    train_tuple = ( users[num_val:].tolist(), items[num_val:].tolist(), ratings[num_val:].tolist() )
    test_tuple = ( users[:num_val].tolist(), items[:num_val].tolist(), ratings[:num_val].tolist() )

    testset = []
    for i in range(len(test_tuple[0])) :
        testset.append( ( test_tuple[0][i], test_tuple[1][i], float(test_tuple[2][i]) ) )


    """
    Creation of the dataframe
    """

    train_ratings_dict = {'userID': train_tuple[0],
                          'itemID': train_tuple[1],
                          'rating': train_tuple[2]}

    df_train = pd.DataFrame(train_ratings_dict)

    reader = Reader(rating_scale = (1,5))

    traindata = Dataset.load_from_df(df_train[['userID', 'itemID', 'rating']], reader)

    trainset = traindata.build_full_trainset()


    model = SVD(biased = False)

    print("============ Dataset : {} fitting ... ============".format(name))

    model.fit(trainset)
    preds = model.test(testset)

    print("==================  testing ... ==================")
    
    accuracy.mae(preds)

    cross_validate(model, traindata, measures = ['MAE'], cv = 5, verbose = True)

