import numpy as np
import numpy.random as rd

# problem 1

# loss = 0
# for i in range(10000):
#     X = np.random.uniform()
#     # 1
#     if X <= 0.5:
#         Y = np.random.binomial(1, 0.6)
#         if Y != 1:
#             loss += 1
#     else:
#         Y = np.random.binomial(1, 0.4)
#         if Y == 1:
#             loss += 1

# print(loss/10000)


# problem 2

nb_test = 1000000
X = rd.binomial(20, 0.2, nb_test) + 1
Y = rd.binomial(3 ** X, 0.5, nb_test)
loss = np.where(3 ** X * 0.5 != Y, (3 ** X * 0.5 - Y) ** 2, 0).mean()
print(loss)