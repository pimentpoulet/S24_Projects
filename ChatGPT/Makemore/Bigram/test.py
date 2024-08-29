
dick = {}
for i in range(2):
    print(dick.get((1,0),0))
    dick[(1,0)] = dick.get((1,0),0) + 1
    print(dick)
