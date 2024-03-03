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