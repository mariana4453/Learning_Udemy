import sys

import numpy as np

# creating numpy arrays
list = [1, 2, 3, 4, 5]
print(type(list))

array = np.array(list)
print(type(array))

dimensional_list = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
array_dim = np.array(dimensional_list)
print(array_dim)

# creating range
range_np = np.arange(10, 20)
print(range_np)

# reshape
print(range_np.reshape(2, 5))


# creating lin space, 3rd parameter num - shows how many evenly spaced numbers over interval
linspace = np.linspace(0, 15, 4)
print(linspace)

# zeros; number of rows first and number of columns second
zeros = np.zeros(5)
print(zeros)

dim_zeroes = np.zeros((2, 3))
print(dim_zeroes)

ones = np.ones((5, 3))
print(ones)

# random numbers
print(np.random.rand(1, 2))

# standard normal distribution
print(np.random.randn(10))

# random integers - 2 random integers btw 0 and 10
print(np.random.randint(0, 10, 2))
print(np.random.randint(0, 10, (1, 4)))         # 1 row and 4 columns

rand_int = np.random.randint(1, 101, 10)
print(rand_int)

print(rand_int.max())
print(rand_int.argmax())

print(rand_int.min())
print(rand_int.argmin())

print(rand_int.shape)

# with seed - set particular random set numbers
np.random.seed(42)
print(np.random.rand(4))
