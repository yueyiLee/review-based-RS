# -*- coding: utf-8 -*-
'this is the implementation of the paper "Matrix Techniques for RS" for zhihu specification article'
__author__ = 'Siwei Fantastic'

from numpy import *
from numpy import random
import matplotlib.pyplot as plt

# load data to build real rating matrix
file_name = 'user_item_rating.txt'
rating_matrix = loadtxt(file_name, dtype=bytes).astype(float)
user_num = rating_matrix.shape[0]
item_num = rating_matrix.shape[1]

# initialize user and item matrix with random float between -1 and 1(not included)
feature_num = 2
user_matrix = random.random_sample((user_num, feature_num))
item_matrix = random.random_sample((item_num, feature_num))


def sgd(data_matrix, user, item, alpha, lam, iter_num):

    for j in range(iter_num):
        for u in range(data_matrix.shape[0]):
            for i in range(data_matrix.shape[1]):
                if data_matrix[u][i] != 0:
                    e_ui = data_matrix[u][i] - sum(user[u,:] * item[i,:])
                    user[u,:] += alpha * (e_ui * item[i,:] - lam * user[u,:])
                    item[i,:] += alpha * (e_ui * user[u,:] - lam * item[i,:])
    return user, item

user, item = sgd(rating_matrix, user_matrix, item_matrix, 0.001, 0.1, 1000)
filter_matrix_entry = rating_matrix <= 0
matrix_predict = dot(user, item.transpose())

# filter the ratings that are already rated
matrix_predict_filtered = matrix_predict * filter_matrix_entry
# save matrix_predict and matrix_predict_filtered to the files respectively
# and make every element correct to two decimal places
savetxt('matrix_predict.txt', matrix_predict, fmt='%.2f')
savetxt('matrix_predict_filtered.txt', matrix_predict_filtered, fmt='%.2f')

# MF for new comer
# randomly initialize the new comer's rate
new_comer_Eric = random.randint(0, 6, size=100)
rating_matrix_new = vstack((rating_matrix, new_comer_Eric))
user_matrix_new = 2 * random.random_sample((user_num+1, feature_num)) - 1
item_matrix_new = 2 * random.random_sample((item_num, feature_num)) - 1
user_new, item_new = sgd(rating_matrix_new, user_matrix_new, item_matrix_new, 0.01, 0.1, 100)
print('The latent feature of new comer(user) is : ', user_new[-1, :])


# let us put bias into the model
# get the overall average rating
def get_miu(data_matrix):

    non_zero_num = 0
    non_zero_sum = 0
    for i in range(data_matrix.shape[0]):
        for j in range(data_matrix.shape[1]):
            if data_matrix[i][j] != 0:
                non_zero_num += 1
                non_zero_sum += data_matrix[i][j]
    return non_zero_sum/non_zero_num, non_zero_num


# the sgd funtion with bias, note that bu and bi shoulde also be trained!(very important)
def sgd_bias(data_matrix, user, item, alpha, lam, iter_num, miu):

    b_u = [1] * rating_matrix.shape[0]
    b_i = [1] * rating_matrix.shape[1]
    for j in range(iter_num):
        for u in range(data_matrix.shape[0]):
            for i in range(data_matrix.shape[1]):
                if data_matrix[u][i] != 0:
                    b_ui = b_u[u] + b_i[i] + miu
                    e_ui = data_matrix[u][i] - b_ui - sum(user[u,:] * item[i,:])
                    user[u,:] += alpha * (e_ui * item[i,:] - lam * user[u,:])
                    item[i,:] += alpha * (e_ui * user[u,:] - lam * item[i,:])
                    b_u[u] += alpha * (e_ui - lam * b_u[u])
                    b_i[i] += alpha * (e_ui - lam * b_i[i])
    return user, item, b_u, b_i

miu, non_zero_num = get_miu(rating_matrix)
print(miu, non_zero_num)
print('the sparse rate of rating matrix is : %.3f' % (non_zero_num/2500))


user_bias, item_bias, b_u, b_i = sgd_bias(rating_matrix, user_matrix, item_matrix, 0.001, 0.1, 1000, miu)
print(user_bias)

# visualize user and item feature
plt.plot(item_bias[:, 0], item_bias[:, 1], 'b*')
plt.plot(user_bias[:, 0], user_bias[:, 1], 'yo')
plt.show()

# calculate MSE
def cal_MSE(data_matrix, predict_matrix, non_zero_num):
    filter_matrix_entry = data_matrix > 0
    predict_matrix_filtered = predict_matrix * filter_matrix_entry
    diff_matrix = (predict_matrix_filtered - data_matrix) * (predict_matrix_filtered - data_matrix)
    mse = (1/non_zero_num) * (diff_matrix.sum())
    return mse

# get the right predictive matrix with bias
matrix_predict_bias = dot(user_bias, item_bias.transpose())
for u in range(matrix_predict_bias.shape[0]):
    for i in range(matrix_predict_bias.shape[1]):
        matrix_predict_bias[u][i] += (miu + b_u[u] + b_i[i])

print(matrix_predict, matrix_predict_bias)
mse = cal_MSE(rating_matrix, matrix_predict, non_zero_num)
mse_bias = cal_MSE(rating_matrix, matrix_predict_bias, non_zero_num)
print('the MSE of the matrix factorization is %.4f' % mse)
print('the MSE of the matrix factorization with considering bias is %.4f' % mse_bias)


