import math
from scipy.stats import chisquare
from matplotlib import pyplot as plt

n = 64
p = 0.5

expected = []
expected2 = []
for i in range(n+1):
    x = i
    binomial1 = math.factorial(n) / (math.factorial(x) * (math.factorial(n - x)))
    
    binomial2 = math.pow(2,64)
    
    bino = binomial1 / binomial2
    
    c = bino
    expected.append(c)

print(expected)
a = []
for i in range (1,65):
    a.append(i)
    
