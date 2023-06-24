import sys
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


# calculate covariance matrix of input matrix X with shape (D, N) that has D dimension and N samples
def calculate_covariance_matrix(X):
    # calculate mean of data
    mean = np.mean(X, axis=1)
    # calculate centered data matrix
    X_centered = X - mean[:, None]
    # compute covariance matrix with centered data matrix
    N = X.shape[1]
    covariance_matrix = np.dot(X_centered, X_centered.T) / N
    return covariance_matrix

# calculate eigenvector and eigenvalue of given matrix
def calculate_eigenvalue_eigenvector(X):
    eig_val, eig_vec = np.linalg.eig(X)
    return eig_val, eig_vec

# compute principal component analysis of given matrix X with 
# shape (D, N) with D dimension and N samples to k principal components
def pca(X, k):
    # calculate covariance matrix of X
    print('calculating covariance matrix')
    covariance_matrix = calculate_covariance_matrix(X)
    # calculate eigenvalue and eigenvector of covariance matrix
    print('calculating eigenvalue and eigenvector')
    eigenvalue, eigenvector = calculate_eigenvalue_eigenvector(covariance_matrix)
    # sort eigenvalue and eigenvector in descending order
    print('sorting eigenvalue and eigenvector')
    eigenvalue_sorted_idx = np.argsort(eigenvalue)[::-1]
    # eigenvalue_sorted = eigenvalue[eigenvalue_sorted_idx]
    eigenvector_sorted = eigenvector[:, eigenvalue_sorted_idx]
    print('calculating projection')
    # calculate projection matrix
    projection_matrix = eigenvector_sorted[:, :k].T
    # calculate projected data matrix
    mean = np.mean(X, axis=1)
    projected_data = np.dot(projection_matrix, X-mean[:, None])
    # return projected data matrix and projection matrix
    return projected_data, projection_matrix


# get input parameter k from command line, default 100
k = 100
if len(sys.argv) > 1:
    k = int(sys.argv[1])
print(f'k = {k}')

# get train data from command line
train_data_path = 'train_data.csv'
if len(sys.argv) > 2:
    train_data_path = sys.argv[1]

# load training data
train_data = np.loadtxt(train_data_path, delimiter=',')

# compute eigenface of train data with projection dimension k
projected_data, projection_matrix = pca(train_data.T, k)

# save eigenface to csv file
np.savetxt(f'projected_data_{k}.csv', projected_data, delimiter=',')

# save projection matrix to csv file
np.savetxt(f'projection_matrix_{k}.csv', projection_matrix, delimiter=',', fmt='%.4e %+.4ej')

print('saved projected_data.csv and projection_matrix.csv')

if k>=8:
    # plot first 4, last 4 eigenface
    fig, axes = plt.subplots(2, 4, figsize=(8, 5))
    for i in range(4):
        axes[0][i].imshow(projection_matrix[i,:].reshape(64, 64).astype(np.float64), cmap='gray')
        axes[1][i].imshow(projection_matrix[-i-1,:].reshape(64, 64).astype(np.float64), cmap='gray')
        axes[0, i].axis('off')
        axes[1, i].axis('off')
    plt.suptitle('First 4 and Last 4 Eigenfaces')
    plt.show()
else:
    # plot all eigenface
    fig, axes = plt.subplots(2, 4, figsize=(8, 4))
    for i in range(k):
        axes[i//4][i%4].imshow(projection_matrix[i,:].reshape(64, 64).astype(np.float64), cmap='gray')
    for i in range(4):
        axes[0, i].axis('off')
        axes[1, i].axis('off')
    plt.suptitle(f'First {k} Eigenfaces')
    plt.show()