"""Get the subset whichi size is k"""
"""2020/4/28   wrote by Cmz"""
from itertools import combinations

def subsets(nbrs,k):
    s=combinations(nbrs,k)
    sub=[]
    for i in s:
        sub.append(list(i))
    return sub

"""example"""
# y=[]
# for i in range(4):
#     z = subset(x,i)
#     print(z)
#     for j in z:
#         y.append(j)
# print(y)
# print(len(y))