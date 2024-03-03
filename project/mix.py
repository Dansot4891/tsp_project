#!/usr/bin/env python
# coding: utf-8

# In[ ]:


def mix_initialize():
    p = []
    start = list(range(SIZE))
    
    random.shuffle(start)
        
    for i in range(SIZE):
        current = start[i]
        unvisited = list(range(SIZE))
        unvisited.remove(current)
        individual = [current]

        while unvisited:
            # 다음 도시 선택
            next_node = min(unvisited, key=lambda node: table[current][node])
            unvisited.remove(next_node)
            individual.append(next_node)
            current = next_node

        p.append(individual)
    
    return p

