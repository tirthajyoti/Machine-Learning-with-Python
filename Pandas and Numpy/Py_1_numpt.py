import numpy as np


a = np.arange(20)
print(a)
print(a[8:12])
print(a[1:100:3]) 


cosvalue = np.cos(4)
print("Cos value" ,cosvalue)
sinvalue = np.sin(4)
print("Cos value" ,sinvalue)


print(np.random.rand())
print(np.random.ranf())


print("\n1-D array")
a = np.array([1,2,3,4,5,6])
print(a)

print("\n1-D array with arange function to interval ")
a1 = np.arange(2,12,3)
print(a1)

print("\n1-D array with linspace ")
a2 = np.linspace(1,10,4)
print(a2)


print("\n2-D array")
b = np.array([[1,2,3],[4,5,6]])
print(b)

print("\n2-D array with ones element ")
b1 = np.ones((2,2))
print(b1)

print("\n2-D array with random array ")
b2 = np.random.rand(2,2)
print(b2)

print("\n2-D array with identity")
b3 = np.identity(2)
print(b3)


print("\n3-D array")
c = np.array([[[1,2,3],
               [4,5,6]],
              [[7,8,9],
               [10,11,12]]])
print(c)

print("\n3-D array with zeros ")
c1 = np.zeros((2,3,4))
print(c1)


print("B\n", np.linspace(1.0, 3.0, num=5, retstep=True), "\n")



arr = np.array([[14, 70], [42, 60]], 
                 dtype = np.float64)

print(c.dtype)

print("\nAddition of Array elements: ")
print(np.sum(arr))

print("\nSquare root of Array1 elements: ")
print(np.sqrt(arr))

print("\nTranspose of Array: ")
print(arr.T)


# Printing type of arr object
print("Array is of type: ", type(arr))
 
# Printing array dimensions (axes)
m = np.random.rand(3,2)
print("No. of dimensions : ", m.ndim)
 
# Printing shape of array
print("Shape of array: ", arr.shape)
 
# Printing size (total number of elements) of array
print("Size of array: ", arr.size)
 
# Printing type of elements in array
print("Array stores elements of type: ", arr.dtype)  # complex
