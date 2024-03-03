# 테스트용 상수
CITIES = 17

# pmx(chromosome1, chromosome2, index1, index2)
def pmx(c1, c2, i1, i2):
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
# def pmx(c1, c2, i1, i2):
#     # 반환할 offspring
#     offspring = [-1 for i in range(CITIES)]
    
#     # c1의 index 저장
#     index = [-1 for i in range(CITIES)]
#     for i in range(CITIES):
#         index[c1[i]] = i

#     # chromosome1의 부분을 떼어다가 그대로 붙여넣기
#     isVisited = [False for i in range(CITIES)]
#     for i in range(i1, i2+1, 1):
#         offspring[i] = c1[i]
#         isVisited[c1[i]] = True

#     # chromosome2 중복 처리
#     for i in range(0, CITIES, 1):
#         temp = i
#         if (i1 <= i & i <= i2) == False: ### chromosome1에서 붙여넣은 부분이 아니라면
#             if isVisited[i] == True: ### 중복이라면
#                 while isVisited[c2[temp]] == True: ### 중복이 아니게 될 때까지
#                     temp = index[temp]
#             offspring[i] = c2[temp]
                
#     return offspring
# ^pmx^

# 테스트용 출력
print(pmx([14, 3, 8, 16, 9, 7, 12, 6, 2, 0, 5, 15, 13, 4, 11, 10, 1], [16, 8, 3, 12, 6, 0, 14, 5, 2, 10, 9, 11, 7, 15, 1, 4, 13], 3, 5))
