import os
import numpy as np
from scipy.stats import norm as norm
   
    
def book2id (books) :

    idx = []
    bookNames = []
    
    books = list(set(books))
    for i, book in enumerate(books) :
        if book not in idx :
            idx.append(i)
            bookNames.append(book)
    
    assert len(idx) == len(bookNames)
    
    return idx, bookNames
    
    
    
def preprocess():
    
    raw = open("books_review/books_review.csv")    
    lines = raw.readlines()
    raw.close()

    userId = []
    books = []
    ratings = []
    
    for line in lines :
        
        line = line.replace("\n", "")
        line = line.replace('"""', "")
        data = line.split(",")
        
        if data[2].isdigit() and data[1].isdigit() :

            userId.append(data[0])
            books.append(data[1])
            ratings.append(data[2])
     
    
    userId = np.array(userId, dtype = int)
    books = np.array(books, dtype = int)
    ratings = np.array(ratings, dtype = int)
    
    
    bookids, booknames = book2id(books)
    
    for book in books :
        for bookid in bookids :
            if booknames[bookid] == book :
                reviewId_arr.append(bookid)
    
    
    for i, rating in enumerate(ratings) : 
        
        ratings[i] = round((4 * rating / 11) + 1, 0)
    
    return userId, books, ratings
        
def books_data() :
    
    reviewId_arr = np.load("books_review/books.npy")
    userId_arr = np.load("books_review/users.npy")
    rating_arr = np.load("books_review/ratings.npy")
    
    assert len(userId_arr) == len(reviewId_arr)
    assert len(userId_arr) == len(rating_arr)
    
    return userId_arr, reviewId_arr, rating_arr