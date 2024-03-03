#!/usr/bin/env python
# coding: utf-8

# In[ ]:


def inversion_mutation(individual, mutation_rate):
    # 뮤테이션 확률을 체크
    if random.random() < mutation_rate:
        # 랜덤하게 두 위치 선택
        index1, index2 = random.sample(range(len(individual)), 2)
        index1, index2 = min(index1, index2), max(index1, index2)

        # 선택한 범위 내의 유전자를 역전시킴
        individual[index1:index2+1] = reversed(individual[index1:index2+1])

    return individual    

