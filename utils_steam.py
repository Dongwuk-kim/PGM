import os
import numpy as np
from scipy.stats import norm as norm


def game2id (games) :

    idx = []
    gameNames = []
    
    games = list(set(games))
    for i, game in enumerate(games) :
        if game not in idx :
            idx.append(i)
            gameNames.append(game)
    
    assert len(idx) == len(gameNames)
    
    return idx, gameNames


def user2id (users) :
    
    ids = []
    
    sorted_users = np.sort(list(set(users)))
    numberings = np.arange(len(sorted_users))
    
    for user in users :
        for i, sort in enumerate(sorted_users) :
            if user == sort :
                ids.append(numberings[i]+1)
                
    return ids


def time2rating (playedTimes, reviewId_arr) :
    
    total_ratings = [0 for _ in range(len(playedTimes))]
    
    for id_games in range(0, max(reviewId_arr) + 1) :
        playedTime = []
        matching_ids = []
        for idx, reviewid in enumerate(reviewId_arr) : 
            if id_games == reviewid :
                matching_ids.append(idx)
                playedTime.append(playedTimes[idx])
        
        playedTime = np.array(playedTime, dtype = float)
        matching_ids = np.array(matching_ids, dtype = int)
        
        mean = np.mean(playedTime)
        median = np.median(playedTime)
        std = np.std(playedTime)
        playedTime = ( playedTime - max(mean, median) ) / ( std + 1e-12)
        Z = norm(loc = 0, scale = 1)
               
        for k, played in enumerate(playedTime) :    
            
            average_rating = []
            
            if played == 0.0 :
                average_rating.append(3)
            if played < max( Z.ppf(0.2), np.quantile(playedTime, 0.2) ) :
                average_rating.append(1)
            if played < max( Z.ppf(0.4), np.quantile(playedTime, 0.4) ) :
                average_rating.append(2)
            if played < max( Z.ppf(0.6), np.quantile(playedTime, 0.6) ) :
                average_rating.append(3)
            if played < max( Z.ppf(0.8), np.quantile(playedTime, 0.8) ) :
                average_rating.append(4)
            if played <= max( Z.ppf(1.0), np.quantile(playedTime, 1.0) ) :
                average_rating.append(5)
            
            total_ratings[matching_ids[k]] = int(np.min(average_rating))
            
    assert len(total_ratings) == len(playedTimes)
    
    return total_ratings

            
def preprocess():
    
    raw = open("steam_review/steam-200k.csv")
    lines = raw.readlines()
    raw.close()

    userId_arr = []
    reviewId_arr = []
    rating_arr = []
    
    userId = []
    games = []
    playedTimes = []
    
    for line in lines :
        
        line = line.replace(',"', ',')
        line = line.replace('",', ',')
        
        data = line.split(",")

        if data[-3] == "play" :
            userId.append(data[0])
            games.append(data[1])
            playedTimes.append(data[-2])
    
    # Game id matching procedure #
    
    gameids, gamenames = game2id(games)
    
    for game in games :
        for gameid in gameids :
            if gamenames[gameid] == game :
                reviewId_arr.append(gameid)
                
#     ratings_arr = time2rating(playedTimes, reviewId_arr)        
       
    return userId, reviewId_arr
        
def steam_data() :
    
#     userId_arr, reviewId_arr = preprocess()
    
#     userId_arr = np.array(userId_arr, dtype = int)
#     reviewId_arr = np.array(reviewId_arr, dtype = int)

#     userId_arr = np.load("steam_review/users.npy")
    reviewId_arr = np.load("steam_review/games.npy")
    userId_arr = np.load("steam_review/users_sortedID.npy")
    rating_arr = np.load("steam_review/ratings.npy")
    
    assert len(userId_arr) == len(reviewId_arr)
    assert len(userId_arr) == len(rating_arr)
    
    return userId_arr, reviewId_arr, rating_arr