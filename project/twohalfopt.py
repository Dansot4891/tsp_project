import random
import numpy
import itertools 
import math

# 도시 개수
CITIES = 17
table = []

def getDistance(c):
    dist = 0
    
    for i in range(CITIES-1):
        dist += table[c[i]][c[i+1]]
    dist += table[c[CITIES-1]][c[0]]

    return dist

def two_opt(c1):
    fit = getDistance(c1)
    c3 = c1.copy()
    for i in itertools.combinations((c1), 2):
        c2 = c1.copy()
        start, end = i[0], i[1]
        c2[start:end] = reversed(c1[start:end])
        if getDistance(c2) < fit:
            fit = getDistance(c2)
            c3 = c2.copy()
    return c3             

def three_opt(c):
    c3 = c
    dist = getDistance(c)

    for i, j, k in itertools.combinations(range(1, CITIES - 2), 3):
        new_c = c[:i] + c[i:j][::-1] + c[j:k][::-1] + c[k:]
        new_dist = getDistance(new_c)
        if new_dist < dist:
            dist = new_dist
            c3 = new_c

    return c3

def twohalf_opt(c):
    c1 = two_opt(c)
    c2 = three_opt(c)
    if(getDistance(c1) < getDistance(c)):
        return c1
    elif(getDistance(c2)< getDistance(c)):
        return c2
    else:
        return c