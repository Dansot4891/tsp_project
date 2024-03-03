# 테스트용 상수
CITIES = 10

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
                
            flag = ~flag
                
    return offspring
# ^cycleCrossover^

# 테스트용 출력
print(cycleCrossover([8,7,1,0,6,3,4,9,5,2], [0,2,4,3,1,5,6,7,8,9]))
