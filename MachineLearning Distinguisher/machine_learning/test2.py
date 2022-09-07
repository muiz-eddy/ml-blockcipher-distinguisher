import numpy as np

a = np.arange(5)

# a is printed.
print("a is:")
print(a)

# The array is saved in the file npfile.npy
np.save('npfile', a)

print("The array is saved in the file npfile.npy")