import numpy as np

# def numpy_sum(n):
#     x = np.arange(n)
#     y = np.arange(n)   ** 2
#     z = x + y
#     return z
#
#
# a=numpy_sum(3)
# print(a)

# the minimum dimension is 2
arr1 = np.array([1, 23, 4, 56, 6, 1, 2, 3], ndmin=2)
# print(arr1)
arr1.shape = (2, 4)
# print(arr1)
arr2 = np.array([12, 3, 4, 56, 6], dtype=np.float32)
# print(arr2.dtype)

# start:0, end:20, num:6
arr3 = np.linspace(0, 20, 6)
# print(arr3) # [ 0.  4.  8. 12. 16. 20.]
arr4 = np.linspace(0, 20, 6, endpoint=0)
# print(arr4) # [ 0.          3.33333333  6.66666667 10.         13.33333333 16.66666667]


arr5 = np.eye(3)
# print(arr5)
"""
output:
    [[1. 0. 0.]
    [0. 1. 0.]
    [0. 0. 1.]
"""

arr6 = np.diag((1, 2, 3))
# print(arr6)
"""
output:
    [[1 0 0]
    [0 2 0]
    [0 0 3]]
"""

arr7 = np.asarray([(1, 2, 3), (4, 5, 6)])
# print(arr7)
"""
output:
    [[1 2 3]
     [4 5 6]]
"""

float_arr = np.array([1.1, 2.1, 2.3, 4.5])
# dtype: the num is set dtype
# print(float_arr.dtype)
int_arr = float_arr.astype(np.int32)
# print(int_arr, int_arr.dtype)   # [1 2 2 4] int32
str_arr = np.array(['1.2', '2.3', '3.3', '3.4'], dtype=np.string_)
float_arr2 = str_arr.astype(dtype=np.float64)
# print(float_arr2)   #[1.2 2.3 3.3 3.4]

# array slice
arr8 = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
# dim:1
# arr[start:end:step]
b1 = arr8[1:11:2]
b2 = arr8[:5]
b3 = arr8[5:]
# print(b1, b2, b3)
b4 = arr8[1:3]
arr8[1:3] = 20, 30
# print(b4, arr8)
arr9 = np.array([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9], [0, 1, 2, 3, 4]])
a = arr9[0, 3]
# print(a)
c1 = arr9[:2, 1:3]
"""
[[1 2]
[6 7]]
"""
c2 = arr9[1:, 2:5]  # [[7 8 9]]
c3 = arr9[..., 1:3]
"""
[[1 2]
[6 7]
[1 2]]
"""
# print(c3)

d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
# d1 = d[[(0, 1, 2), (0, 1, 0)]]  # [1 5 7]
# print(d1)

arr10 = np.arange(9).reshape(3, 3)
arr_list = arr10.tolist()
# print(arr_list)

arr_a = np.arange(6).reshape(2, 3)
arr_b = np.arange(3)
# print(arr_a,arr_b)
"""
[[0 1 2]
 [3 4 5]] [0 1 2]
"""
# print(np.dot(arr_a,arr_b))  # [ 5 14]

# arr11 = np.arange(30).reshape(3, 5, 2)
# print(arr11.tolist())


arr12 = np.arange(12).reshape(3, 4)
# print(arr12)
# axis=0表示按列，axis=1表示按行 none:all
arr12_mean1 = np.mean(arr12)
arr12_mean2 = np.mean(arr12, axis=0)
arr12_mean3 = np.mean(arr12, axis=1)
# print(arr12_mean1, arr12_mean2, arr12_mean3)
"""
output:
5.5 [4. 5. 6. 7.] [1.5 5.5 9.5]
"""

arr13 = np.array([1, 2, 1, 4,2, 6, 7, 8, 6, 10, 11, 12])
# 去重
arr_u=np.unique(arr13,return_index=True,return_inverse=True)
# print(arr_u)

arr14=np.arange(12).reshape(3,4)
print(arr14)
maxindex=np.argmax(arr14)
print(maxindex)
maxindex0=np.argmax(arr14,axis=1)
print(maxindex0)