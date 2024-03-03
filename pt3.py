inputorder = input("Order-based : ")
locus = []
order = []

for i in range(0,10):
    order.append(inputorder[i])

int_order = list(map(int, order))


for i in range(0,10):
    k = int_order.index(i) + 1
    if (k==10):
        k = 0
    elif(k==11):
        k = 0
    locus.append(int_order[k])

int_locus = list(map(str, locus))
result = ''.join(s for s in int_locus)

print("Locus_based : " + result)
