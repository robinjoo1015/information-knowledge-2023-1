import sys
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')



# get input parameter k from command line, default 100
k = 100
if len(sys.argv) > 1:
    k = int(sys.argv[1])
print(f'k = {k}')

# get train data from command line
train_data_path = 'train_data.csv'
if len(sys.argv) > 2:
    train_data_path = sys.argv[1]

# get test data from command line
test_data_path = 'test_data.csv'
if len(sys.argv) > 3:
    test_data_path = sys.argv[3]

# get train target from command line
train_target_path = 'train_target.csv'
if len(sys.argv) > 4:
    train_target_path = sys.argv[4]

# get test target from command line
test_target_path = 'test_target.csv'
if len(sys.argv) > 5:
    test_target_path = sys.argv[5]



print('==================== TRAINING PHASE ====================')


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
    # calculate projection matrix
    projection_matrix = eigenvector_sorted[:, :k].T
    # calculate projected data matrix
    print('calculating projection')
    mean = np.mean(X, axis=1)
    projected_data = np.dot(projection_matrix, X-mean[:, None])
    # return projected data matrix and projection matrix
    return projected_data, projection_matrix


# load training data
print('loading training data')
train_data = np.loadtxt(train_data_path, delimiter=',')


# compute eigenface of train data with projection dimension k
projected_data, projection_matrix = pca(train_data.T, k)


# save projected data and projection matrix to csv file
np.savetxt(f'projected_data_{k}.csv', projected_data, delimiter=',')
np.savetxt(f'projection_matrix_{k}.csv', projection_matrix, delimiter=',')
print(f'saved projected_data_{k}.csv and projection_matrix_{k}.csv')

if k>=8:
    # plot first 4, last 4 eigenface
    fig, axes = plt.subplots(2, 4, figsize=(8, 5))
    for i in range(4):
        axes[0][i].imshow(projection_matrix[i,:].reshape(64, 64).astype(np.float64), cmap='gray')
        axes[1][i].imshow(projection_matrix[-i-1,:].reshape(64, 64).astype(np.float64), cmap='gray')
        axes[0, i].axis('off')
        axes[1, i].axis('off')
    plt.suptitle(f'First 4 and Last 4 Eigenfaces of k={k}')
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



print('==================== TESTING PHASE ====================')


# calculate projection of data X with projection matrix
def projection(X, projection_matrix):
    print('calculating projection')
    return np.dot(projection_matrix, X)

# calculate euclidean distance between two vectors
def euclidean_distance(x, y):
    return np.linalg.norm(x - y)

# calculate prediction label of test data
def predict_label(test_data, train_data, train_label):
    print('predicting label')
    # initialize prediction label
    test_prediction_label = np.zeros(test_data.shape[1], dtype=np.int64)
    for i in range(test_data.shape[1]):
        # calculate euclidean distances between test data and train data
        distance = np.array([euclidean_distance(test_data[:,i], train_data[:,j]) for j in range(train_data.shape[1])])
        # get prediction label by finding the index of minimum distance
        test_prediction_label[i] = train_label[np.argmin(distance)]
    return test_prediction_label

# calculate accuracy of prediction
def calculate_accuracy(prediction_label, ground_truth_label):
    print('calculating accuracy')
    return np.sum(prediction_label == ground_truth_label) / ground_truth_label.shape[0]


# load test data
print('loading test data')
test_data = np.loadtxt(test_data_path, delimiter=',')

# load ground truth label of train and test data
print('loading train and test label')
train_label = np.loadtxt(train_target_path, delimiter=',', dtype=np.int64)
test_label = np.loadtxt(test_target_path, delimiter=',', dtype=np.int64)


# calculate mean face
mean_face = np.mean(train_data, axis=0)

# subtract mean face from test data
test_data_centered = test_data.T - mean_face[:, None]

# project test data to k eigenface subspace
projected_test_data = projection(test_data_centered, projection_matrix)

# predict label of test data
prediction_label = predict_label(projected_test_data, projected_data, train_label)

# calculate accuracy of prediction
accuracy = calculate_accuracy(prediction_label, test_label)
print('Accuracy:', accuracy)


# save prediction label to csv file
np.savetxt(f'prediction_label_{k}.csv', prediction_label, delimiter=',', fmt='%d')
print(f'saved prediction_label_{k}.csv')