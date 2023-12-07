import cv2
import cv2 as cv
import matplotlib.pylab as plt
import numpy as np

#Gray scale image
img = cv.imread('white_cat.jpeg')
gray_image = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('img_original', gray_image)
original_image = cv.resize(gray_image, (500, 500))
matrix_image = cv.resize(gray_image, (500, 500))
# matrix_image = matrix_image.reshape((matrix_image.shape[0], matrix_image.shape[1]**2))

#Chuẩn hóa hình ảnh
mean_value = np.average(matrix_image, axis=0)
matrix_normalized = matrix_image - np.average(matrix_image, axis=0)

#Tìm ma trận hiệp phương sai
covariance_matrix = np.cov(matrix_normalized, rowvar=False)

#Tính toán vector riêng và giá trị riêng của ma trận hiệp phương sai
eigen_values, eigen_vector = np.linalg.eig(covariance_matrix)

sorted_index = np.argsort(eigen_values)[::-1]
sorted_values = eigen_values[sorted_index]
sorted_eigenvectors = eigen_vector[:, sorted_index]

n_components = 500
eigenvector_subset = sorted_eigenvectors[:,0:n_components]
print(eigenvector_subset)
#Chuyển đổi dữ liệu hình ảnh sang cơ sở mới
matrix_image = np.dot(np.transpose(eigenvector_subset), matrix_normalized)
matrix_image = matrix_image
reconstructed_image = np.matmul(eigenvector_subset, matrix_image)
reconstructed_image.shape
reconstructed_image = reconstructed_image.real
# reconstructed_image = np.reshape(matrix_image, original_image)
cv.imshow('img', reconstructed_image)
print(np.array_equal(reconstructed_image, matrix_normalized))
cv.waitKey(0)
