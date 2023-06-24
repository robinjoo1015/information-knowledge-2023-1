import sys
import numpy as np


def projection(X, projection_matrix):
    return np.dot(projection_matrix, X)

# calculate euclidean distance between two vectors
def euclidean_distance(x, y):
    return np.linalg.norm(x - y)

# calculate prediction label of test data
def predict_label(test_data, train_data, train_label):
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
    return np.sum(prediction_label == ground_truth_label) / ground_truth_label.shape[0]

# get test data from command line
test_data_path = 'test_data.csv'
if len(sys.argv) > 1:
    test_data_path = sys.argv[1]

# get projection matrix from command line
projection_matrix_path = 'projection_matrix_100.csv'
if len(sys.argv) > 2:
    projection_matrix_path = sys.argv[2]

# get projected data from command line
projected_data_path = 'projected_data_100.csv'
if len(sys.argv) > 3:
    projected_data_path = sys.argv[3]

# get train target from command line
train_target_path = 'train_target.csv'
if len(sys.argv) > 4:
    train_target_path = sys.argv[4]

# get test target from command line
test_target_path = 'test_target.csv'
if len(sys.argv) > 5:
    test_target_path = sys.argv[5]

# get mean face data from command line
mean_face_path = 'mean_face.csv'
if len(sys.argv) > 6:
    mean_face_path = sys.argv[6]

# load test data
test_data = np.loadtxt(test_data_path, delimiter=',')

# load mean face data
mean_face = np.loadtxt(mean_face_path, delimiter=',', dtype=np.float64)

# load projection matrix and projected train data
projection_matrix = np.loadtxt(projection_matrix_path, delimiter=',', dtype=np.complex128)
projected_train_data = np.loadtxt(projected_data_path, delimiter=',', dtype=np.complex128)

# load ground truth label of train and test data
train_label = np.loadtxt(train_target_path, delimiter=',', dtype=np.int64)
test_label = np.loadtxt(test_target_path, delimiter=',', dtype=np.int64)

print(projection_matrix[0])

# subtract mean face from test data
test_data_centered = test_data.T - mean_face[:, None]

# project test data to k eigenface subspace
projected_test_data = projection(test_data.T - mean_face[:, None], projection_matrix)

# predict label of test data
prediction_label = predict_label(projected_test_data, projected_train_data, train_label)

# calculate accuracy of prediction
accuracy = calculate_accuracy(prediction_label, test_label)
print(accuracy)

# save prediction label to csv file
np.savetxt('prediction_label.csv', prediction_label, delimiter=',', fmt='%d')