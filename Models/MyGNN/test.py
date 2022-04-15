import numpy as np

# a = np.random.uniform(0, 1, (32, 50, 100))
# b = np.random.uniform(0, 1, (100, 3))
# res = np.einsum('bij,jk->bik', a, b) # tf.matmul(x, y)
# print(res.shape)

a = np.random.uniform(0, 1, (32, 50, 3))
b = np.random.uniform(0, 1, (3000, 3000))
c = np.matmul(b, a)
print(c.shape)