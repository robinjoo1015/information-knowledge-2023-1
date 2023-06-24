#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import library
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity

# Read csv data file
pd_csv = pd.read_csv('./CC GENERAL.csv')

# Remove unnecessary column and blank rows
data = pd_csv.drop(columns=['CUST_ID']).dropna().to_numpy()

# Get column names
column_names = pd_csv.columns[1:]

# Apply log to skewed columns
data_log = data
data_log[:,0] = np.log1p(data_log[:,0])
data_log[:,2] = np.log1p(data_log[:,2])
data_log[:,3] = np.log1p(data_log[:,3])
data_log[:,4] = np.log1p(data_log[:,4])
data_log[:,5] = np.log1p(data_log[:,5])
data_log[:,10] = np.log1p(data_log[:,10])
data_log[:,11] = np.log1p(data_log[:,11])
data_log[:,12] = np.log1p(data_log[:,12])
data_log[:,13] = np.log1p(data_log[:,13])
data_log[:,14] = np.log1p(data_log[:,14])

# Apply standardization
data_log_standardized = (data_log - np.mean(data_log, axis=0)) / np.std(data_log, axis=0)


# In[15]:


def euclidean_distance(point1, point2):
    return np.linalg.norm(point1 - point2)


# In[3]:


def intra_cluster_distance(data_scaled, center, data_label):
    return np.mean([euclidean_distance(data_scaled[i], center[data_label[i]]) for i in range(data_scaled.shape[0])])


# In[4]:


def intra_inter_variation(data_scaled, center, data_label):
    intra_variation = intra_cluster_distance(data_scaled, center, data_label)
    inter_variation = np.mean([
        euclidean_distance(center[i], center[j]) 
        for i in range(0, center.shape[0]-1) 
        for j in range(i, center.shape[0])
    ])
    return intra_variation/inter_variation


# In[ ]:


# Parameters
k=6
num_epoch=50
num_trial=10
data_num = data_log_standardized.shape[0]
data_dim = data_log_standardized.shape[1]

# Array for saving each trial results
medoid_trial = np.zeros((num_trial, k), dtype=np.float64)
data_label_trial = np.zeros((num_trial, data_num), dtype=np.int32)
distance_trial = []
intra_inter_variation_trial = np.zeros(num_trial, dtype=np.float64)

# Calculate distance matrix
start_time = time.time()
distance_matrix = np.zeros((data_num, data_num), dtype=np.float64)
for i in range(0, data_num-1):
    for j in range(i+1, data_num):
        distance_matrix[i, j] = euclidean_distance(data_log_standardized[i], data_log_standardized[j])
distance_matrix = distance_matrix + distance_matrix.T
print(f'Distance matrix calculation time: {time.time()-start_time}s')

# 10 trials
for trial in range(num_trial):
    
    # Time check
    start_time = time.time()
    
    # Initialize medoids
    medoid = np.full(k, fill_value=-1, dtype=np.int32)
    medoid_history = np.zeros((num_epoch, k), dtype=np.int32)
    
    # Initialize label of each data
    data_label = np.zeros((data_log_standardized.shape[0]), dtype=np.int32)
    
    # Select first medoid as random
    medoid[0] = np.random.randint(data_num)
    
    # Initialize distance from data to each medoid
    distance_medoid = distance_matrix[medoid[0],:]
    distance_sum = np.sum(distance_medoid)
    distance_record = []
    
    # Greedily select other medoids
    for m in range(k):
        distance_medoid_temp = distance_medoid
        distance_sum_temp = distance_sum
        data_label_temp = data_label
        medoid_temp = 0
        for i in range(data_num):
            if i in medoid:
                continue
            distance_new = np.min(np.array([distance_medoid, distance_matrix[i,:]]).T, axis=1)
            if np.sum(distance_new)<distance_sum_temp:
                data_label_temp = np.argmin(np.array([distance_medoid, distance_matrix[i,:]]).T, axis=1)
                distance_medoid_temp = distance_new
                distance_sum_temp = np.sum(distance_new)
                medoid_temp = i
        medoid[m] = medoid_temp
        distance_medoid = distance_medoid_temp
        distance_sum = distance_sum_temp
        data_label[np.where(data_label_temp == 1)[0]] = m
    
    distance_record.append(distance_sum)
    
    medoid_new = medoid
    for epoch in range(num_epoch):
        isbreak = False
        for m in range(k):
            distance_medoid_temp = distance_medoid
            distance_sum_temp = distance_sum
            data_label_temp = data_label
            medoid_temp = 0
            for i in range(data_num):
                if data_label[i]!=m:
                    continue
                if i==medoid[m]:
                    continue
                distance_new = np.min(np.array([distance_medoid, distance_matrix[i,:]]).T, axis=1)
                if np.sum(distance_new) < distance_sum_temp:
                    data_label_temp = np.argmin(np.array([distance_medoid, distance_matrix[i,:]]).T)
                    distance_medoid_temp = distance_new
                    distance_sum_temp = np.sum(distance_new)
                    medoid_temp = i
            if distance_sum_temp<distance_sum:
                medoid[m] = medoid_temp
                distance_medoid = distance_medoid_temp
                distance_sum = distance_sum_temp
                data_label[np.where(data_label_temp == 1)[0]] = m
                distance_record.append(distance_sum)
            else:
                isbreak = True
                break
        if isbreak:
            break
            
    medoid_trial[trial] = medoid
    data_label_trial[trial] = data_label
    distance_trial.append(distance_record)
    intra_inter_variation_trial[trial] =     intra_inter_variation(data_log_standardized, data_log_standardized[medoid], data_label)
    print(f'Trial #{trial} execution time: {time.time()-start_time}s')


# In[47]:


plt.figure(figsize=(20,12))
for t in range(num_trial):
    plt.plot(
        np.arange(len(distance_trial[t])), 
        distance_trial[t], 
        'o-', label=f'trial #{t}: {intra_inter_variation_trial[t]}'
    )
plt.legend()
plt.show()


# In[ ]:





# In[49]:


# Select best trial
best_trial = np.argmin(intra_inter_variation_trial)


# In[50]:


# 4*4 plot
fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(20,20))
for c in range(data_dim-1):
    
    linspace = np.linspace(np.min(data_log_standardized[:,c+1]), np.max(data_log_standardized[:,c+1]), 1000)
    for l in range(k):
        kde = KernelDensity(
            bandwidth=linspace[100]-linspace[0], 
            kernel='gaussian'
        ).fit(
            data_log_standardized[np.where(data_label_trial[best_trial]==l)[0], c+1].reshape(-1,1)
        )
        
        logprob = kde.score_samples(linspace[:, None])
        axes[c//4][c%4].plot(linspace, np.exp(logprob), label=l)
    axes[c//4][c%4].set_title(f'{column_names[c+1]}')
plt.show()


# Remaining 1 plot
fig = plt.figure(figsize=(12,8))
ax = plt.subplot()

linspace = np.linspace(np.min(data_log_standardized[:,0]), np.max(data_log_standardized[:,0]), 1000)
for l in range(k):
    kde = KernelDensity(
        bandwidth=linspace[100]-linspace[0], 
        kernel='gaussian'
    ).fit(
        data_log_standardized[np.where(data_label_trial[best_trial]==l)[0], 0].reshape(-1,1)
    )

    logprob = kde.score_samples(linspace[:, None])
    ax.plot(linspace, np.exp(logprob), label=f'Cluster #{l}')
    
ax.set_title(f'{column_names[0]}')

box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

# Put a legend to the right of the current axis
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

plt.show()


# In[ ]:




