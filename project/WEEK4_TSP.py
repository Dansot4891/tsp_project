import random
import numpy
import itertools
import math
import pandas
import os
import time
# 도시 개수
CITIES = 48
# population에 포함된 chromosome의 개수
SIZE = 400
# 세대
generation = 1
G_SIZE = 3000


# 도시 간 거리
table = [[]]
# population
p = [[]]
# fitness
f = [0 for i in range(SIZE)]


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

# 파일로부터 도시 간 거리 읽어와서 반환
def getTable(filename):
    f = open(filename, 'r')
    table = [[0 for col in range(CITIES)] for row in range(CITIES)]

    for i in range(CITIES):
        table[i] = list(map(float, f.readline().split()))

    return table
# ^getTable^


# 초기 population 반환 (order-based)
def getPopulation():
    p = [[col for col in range(CITIES)] for row in range(SIZE)]

    for i in range(SIZE):
        random.shuffle(p[i])

    return p
# ^getPopulation^


# chromosome의 fitness(도시 순회 시 총 거리) 반환
def getDistance(c):
    dist = 0

    for i in range(CITIES-1):
        dist += table[c[i]][c[i+1]]
    dist += table[c[CITIES-1]][c[0]]

    return dist
# ^getDistance^


# 가장 좋은 fitenss인(짧은 거리인) chromosome의 index 반환
def getBestIdx():
    minDist = numpy.finfo(numpy.float64).max
    idx = -1

    for i in range(SIZE):
        if(f[i] <= minDist):
            minDist = f[i]
            idx = i

    return idx
# ^getBestIdx^


# 가장 나쁜 fitness인(긴 거리인) chromosome의 index 반환
def getWorstIdx():
    maxDist = 0
    idx = -1

    for i in range(SIZE):
        if(f[i] >= maxDist):
            maxDist = f[i]
            idx = i

    return idx
# ^getWorstIdx^


#rouletteWheel
def rouletteWheel():
    #(Cw-Ci) + (Cw-Cb)/(k-1) , k = 3
    Cw = max(f)
    Cb = min(f)
    sum_fit = sum([(Cw-fit)+(Cw-Cb)/2 for fit in f])
    point = random.uniform(0,sum_fit)
    current_fit = 0

    for i in range(SIZE):
        current_fit += (Cw-f[i])+(Cw-Cb)/2
        if point<=current_fit:
            return i
# ^rouletteWheel^

# tournamentSelection
def tournamentSelection(i1, i2 , rate=0.5):
    if(random.random() > rate):
        return i1 if f[i1] <= f[i2] else i2
    else:
        return i2 if f[i1] <= f[i2] else i1
# ^tournamentSelection^


# orderCrossOver
def orderCrossover(c1, c2, l=CITIES//3, r=2*CITIES//3):
    c = [0 for i in range(CITIES)]
    isVisited = [False for i in range(CITIES)]

    for i in range(l, r+1, 1):
        c[i] = c1[i]
        isVisited[c1[i]] = True

    for i in c2:
        if(isVisited[i] == False):
            r += 1
            c[r % CITIES] = i
    return c
# ^orderCrossover^


# cycleCrossover(chromosome1, chromosome2)
def cycleCrossover(c1, c2):
    # 반환할 offspring
    offspring = [-1 for i in range(CITIES)]

    # c1의 index 저장
    index = [-1 for i in range(CITIES)]
    for i in range(CITIES):
        index[c1[i]] = i

    # chromosome1에서 떼어올지 chromosome2에서 떼어올지 여부
    flag = True

    # cycle 되었는지 여부 저장
    isVisited = [False for i in range(CITIES)]

    for i in range(CITIES):
        if(offspring[i] == -1):
            temp = i

            while True:
                if(flag == True):
                    offspring[temp] = c1[temp] ### chromosome1
                else:
                    offspring[temp] = c2[temp] ### chromosome2

                isVisited[temp] = True
                temp = index[c2[temp]]

                if(isVisited[temp] == True):
                    break

            flag = not flag

    return offspring
# ^cycleCrossover^


# pmx(chromosome1, chromosome2, index1, index2)
def pmx(c1, c2,i1 = random.randint(1, CITIES//3-2), i2=random.randint(CITIES//3, 2*CITIES//3-2)):
    # 반환할 offspring
    offspring = [-1 for i in range(CITIES)]

    # c1의 index 저장
    index = [-1 for i in range(CITIES)]
    for i in range(CITIES):
        index[c1[i]] = i

    # chromosome1의 부분을 떼어다가 그대로 붙여넣기
    isVisited = [False for i in range(CITIES)]
    for i in range(i1, i2+1):
        offspring[i] = c1[i]
        isVisited[c1[i]] = True

    # chromosome2 중복 처리
    for i in range(0, CITIES):
        if not i1 <= i <= i2:  # chromosome1에서 붙여넣은 부분이 아니라면
            temp = c2[i]
            while isVisited[temp]:  # 중복이 아니게 될 때까지
                temp = c2[index[temp]]
            offspring[i] = temp

    return offspring
# ^pmx^


# edgeRecombinationCrossover(chromosome1, chromosome2)
def edgeRecombinationCrossover(c1, c2):
    offspring = [-1 for i in range(CITIES)]

    table = [set() for i in range(CITIES)] ### 인접 도시 저장
    c1.append(c1[0])
    c2.append(c2[0])
    for i in range(CITIES):
        table[c1[i]].add(c1[i+1])
        table[c1[i+1]].add(c1[i])
        table[c2[i]].add(c2[i+1])
        table[c2[i+1]].add(c2[i])

    size = [0 for i in range(CITIES)] ### 각 도시의 이웃 개수 저장
    for i in range(CITIES):
        size[i] = len(table[i])

    isVisited = [False for i in range(CITIES)] ### 방문여부를 True, False로 저장
    unvisited_list = [i for i in range(CITIES)] ### 미방문여부를 list로 저장

    c1.pop(0)
    c2.pop(0)
    x = c1[0] if random.random() >= 0.5 else c2[0] ### 부모 중 랜덤하게 골라 맨 처음 노드 저장
    for i in range(CITIES):
        isVisited[x] = True
        unvisited_list.remove(x)
        offspring[i] = x

        # offspring이 완성되었으면
        if(i == CITIES-1):
            return offspring

        # 인접 도시 리스트에서 x 삭제
        for i in range(CITIES):
            if(isVisited[i] == False):
                if(x in table[i]):
                    table[i].discard(x)
                    size[i] -= 1

        # x의 이웃한 도시가 남아있지 않다면
        if(len(table[x]) == 0):
            # 방문하지 않은 도시들 중 랜덤으로 선택해 x에 저장
            x = random.choice(unvisited_list)
        # 아니라면
        else:
            # x의 이웃한 도시 중 가장 적은 인접 도시를 가진 도시를 뽑아 x에 저장 (2개 이상이라면 랜덤)
            neighbors = [[] for i in range(5)]
            for city in table[x]:
                neighbors[size[city]].append(city)
            for city in range(5):
                if(len(neighbors[city]) > 0):
                    x = random.choice(neighbors[city])
                    break
# ^edgeRecombinationCrossover^

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

#################LK##################
def getCycle(c): # 인접도시 리스트에서 염색체 추출
    city= []
    selected_city2 = 0  #출발 도시
    selected_city1 = c[0][0]   #도착 도시
    city.append(selected_city1)
    while True:
        try:
            selected_city = [x for x in c[selected_city1] if x != selected_city2][0] # 출발 도시 할당 x 도착하고 다시 출발
            city.append(selected_city)
            selected_city2 = selected_city1
            selected_city1 = selected_city
        except IndexError:
            break
        if selected_city == 0:
            return city
def LK(c):
    while True:
        c1 = [[] for i in range(CITIES)] ### 인접 도시 저장
        for i in range(CITIES-1):
            c1[c[i]].append(c[i+1])
            c1[c[i+1]].append(c[i])
        c1[c[CITIES-1]].append(c[0])
        c1[c[0]].append(c[CITIES-1])

        lock_lst = [False for i in range(CITIES)]

        first = random.randint(0,CITIES-2)
        lock_lst[first] = True # 시작 노드 Lock
        start = c1[first][0]   # 시작 노드의 인접노드에서 시작
        c1[start].remove(first) # 인접노드 삭제
        c1[first].remove(start) # 인접노드 삭제

        while True:
            edge_fit = [table[start][i] for i in range(CITIES)] # 시작 도시에서 모든 도시 거리
            best = 100000
            end = -1
            for i in range(CITIES):# 가장 가까운 도시구하기
                if lock_lst[i]==False and i not in c1[start]: #Lock  되지 않고 시작노드의 인접노드가 아님
                    if 0<edge_fit[i]<best:
                        best = edge_fit[i]
                        end = i
            if end == -1:#start가 Lock 되지 않음
                c1[first].append(start)
                c1[start].append(first)
                c1 = getCycle(c1)
                c_fit = getDistance(c)
                c1_fit = getDistance(c1)
                if c_fit<=c1_fit:  # 더 좋은 염색체 선택
                    return c
                else:
                    c = c1
                    break
            c1[start].append(end)  #시작 도시의 인접노드에 도착도시 추가
            c1[end].append(start)
            lock_lst[end]=True     #도착 도시 Lock
            F = True
            for i in c1[end]:  #추가한 도시와 인접도시들을 순회 / 싸이클 삭제하는것임
                if i == start: #출발 도시는 빼야됨
                    pass
                selected_city2 = end  #출발 도시
                selected_city1 = i    #도착 도시
                while F:
                    try:
                        selected_city = [x for x in c1[selected_city1] if x != selected_city2][0] # 출발 도시 할당 x 도착하고 다시 출발
                        selected_city2 = selected_city1
                        selected_city1 = selected_city
                    except IndexError: # 사이클이 아니라 더이상 진행 못하고 에러 발생
                        break
                    if selected_city == end: # 사이클이라면 삭제
                        F = False
                        c1[i].remove(end)
                        c1[end].remove(i)
                        start = i
                        break

            if all(lock_lst):
                c1[first].append(start)
                c1[start].append(first)
                c1 = getCycle(c1)
                c_fit = getDistance(c)
                c1_fit = getDistance(c1)
                if c_fit<=c1_fit:  # 더 좋은 염색체 선택
                    return c
                else:
                    c = c1
                    break
##########################LK############################

def swap_mutate(individual,mutation_rate=0.05):
    if random.random() < mutation_rate:
        i, j = random.sample(range(len(individual)), 2)
        individual[i], individual[j] = individual[j], individual[i]

def inversion_mutation(individual, mutation_rate=0.05):
    # 뮤테이션 확률을 체크
    if random.random() < mutation_rate:
        # 랜덤하게 두 위치 선택
        index1, index2 = random.sample(range(len(individual)), 2)
        index1, index2 = min(index1, index2), max(index1, index2)

        # 선택한 범위 내의 유전자를 역전시킴
        individual[index1:index2+1] = reversed(individual[index1:index2+1])

    return individual
# 대치

## 안정상태 GA
def replaceWorst():
    worstIdx = getWorstIdx()
    p[worstIdx] = offspring
    f[worstIdx] = getDistance(p[worstIdx])

def replaceParent(parent_lst):
    parent = random.choice(parent_lst)
    print(parent)
    p[parent] = offspring
    f[parent] = getDistance(p[parent])


## 세대 GA
def replaceWorst_for_generation(n):
    p_f_lst = [[p[i],f[i],i] for i in range(SIZE)]
    p_f_lst = sorted(p_f_lst, key=lambda x: x[1], reverse=True)
    for i in range(n):
        worstIdx = p_f_lst[i][2]
        p[worstIdx] = off_lst[i]
        f[worstIdx] = getDistance(p[worstIdx])


def replace_save_best(n):
    p_f_lst = [[p[i],f[i],i] for i in range(SIZE)]
    p_f_lst = sorted(p_f_lst, key=lambda x: x[1], reverse=True)
    p_f_lst_notgood = random.sample(p_f_lst[:SIZE//2],n)
    for i in range(n):
        worstIdx = p_f_lst_notgood[i][2]
        p[worstIdx] = off_lst[i]
        f[worstIdx] = getDistance(p[worstIdx])



table = getTable('input48.txt')

p = getPopulation()

for i in range(SIZE):
    p[i] = LK(p[i])
    f[i] = getDistance(p[i])

def stop_condition(dist):
  if dist <= 33700 : return False
  else : return True

for i in range(SIZE):
    p[i] = LK(p[i])
    f[i] = getDistance(p[i])


start = time.time()
while generation <= G_SIZE:
    mutate_weight = generation/G_SIZE*4/10 # 뮤테이션 가중치 0.4 선형 증가
    tournament_weight = generation/G_SIZE*5/10 # 토너먼트 셀렉션 가중치 0~0.5 선형 증가 => 처음엔 무조건 좋은거뽑고 나중엔 점점 랜덤하게
    generation_percent = 1+int(SIZE*generation/G_SIZE*2/10) # generation 비율 증가 1%~21%, generation비율 너무 높거나 SIZE 너무 작으면 오류 발생
    off_lst = []
    parent_lst = []
    if generation < 3*G_SIZE//10:
        local = LK
    # elif generation < 2*G_SIZE//10:
    #     local = two_opt
    else:
        local = three_opt

    child = random.sample(range(SIZE),generation_percent*4) #4배 뽑아야 토너먼트해서 1개 만듬
    for i in range(generation_percent):
        i1,i2 = tournamentSelection(child[i],child[i+1], tournament_weight), tournamentSelection(child[i+2],child[i+3] , tournament_weight )
        parent_lst.append([i1,i2])
        offspring = edgeRecombinationCrossover(p[i1], p[i2])
        inversion_mutation(offspring, mutate_weight)

        offspring = local(offspring)
        off_lst.append(offspring)
    replace_save_best(generation_percent)

    generation += 1
    if generation%300==0:
        end = time.time()

        # 최종 출력
        print("시간 :", end - start)
        print("최단 경로 : ", p[getBestIdx()])
        print("세대 평균 : ", sum(f)//SIZE)
        print("최단 거리 : ", f[getBestIdx()])
        print("세대수 : ", generation)

for i in range(SIZE):
    p[i] = three_opt(p[i])
    f[i] = getDistance(p[i])
# main 종료

# 최종 출력
end = time.time()

# 최종 출력
print("시간 :", end - start)
print("최단 경로 : ", p[getBestIdx()])
print("세대 평균 : ", sum(f)//SIZE)
print("최단 거리 : ", f[getBestIdx()])
print("세대수 : ", generation)


