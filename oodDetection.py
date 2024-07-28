import gc
import numpy as np
from scipy.spatial.distance import cdist
from scipy.spatial.distance import mahalanobis
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import IncrementalPCA, PCA

featureDir = '/fs/scratch/sgh_cr_bcai_dl_cluster_users/archive/CULaneFeature/'

print("Norm")

train_feat = np.memmap(featureDir + "trainFeature.mmap", dtype=np.float32, mode='r', shape=(88880, 460800))
test_feat = np.memmap(featureDir + "testFeature.mmap", dtype=np.float32, mode='r', shape=(34680, 460800))

train_feat_norm = np.memmap(featureDir + "trainFeatureNormNBatch.mmap", dtype=np.float32, mode='w+', shape=(88880, 460800))
test_feat_norm = np.memmap(featureDir + "testFeatureNormNBatch.mmap", dtype=np.float32, mode='w+', shape=(34680, 460800))

scaler = StandardScaler()

train_feat_norm[:] = scaler.fit_transform(train_feat)

del train_feat
del train_feat_norm
gc.collect()

np.save(featureDir + "scaler_mean.npy", scaler.mean_)
np.save(featureDir + "scaler_scale.npy", scaler.scale_)

mean = np.load(featureDir + "scaler_mean.npy")
scale = np.load(featureDir + "scaler_scale.npy")

scaler = StandardScaler()
scaler.mean_ = mean
scaler.scale_ = scale

test_feat_norm[:] = scaler.transform(c)

del test_feat
del test_feat_norm
del scaler
del mean
del scale
gc.collect()

print("PCA")

# PCA

# n_components = 400
# pca = PCA(n_components=n_components)

# test_feat_norm = np.memmap(featureDir + "testFeatureNorm.mmap", dtype=np.float32, mode='r', shape=(34680, 460800))
# train_feat_norm = np.memmap(featureDir + "trainFeatureNorm.mmap", dtype=np.float32, mode='r', shape=(88880, 460800))

# test_feat_norm_pca = np.memmap(featureDir + "testFeatureNormPCANbatch.mmap", dtype=np.float32, mode='w+', shape=(34680, 400))
# train_feat_norm_pca = np.memmap(featureDir + "trainFeatureNormPCANbatch.mmap", dtype=np.float32, mode='w+', shape=(88880, 400))

# pca.fit(train_feat_norm)

# train_feat_norm_pca[:] = pca.transform(train_feat_norm)
# del train_feat_norm
# del train_feat_norm_pca
# gc.collect()

# test_feat_norm_pca[:] = pca.transform(test_feat_norm)
# del pca
# del n_components
# del test_feat_norm
# del test_feat_norm_pca
# gc.collect()

# IncrementalPCA

n_components = 400
ipca = IncrementalPCA(n_components=n_components)

test_feat_norm = np.memmap(featureDir + "testFeatureNorm.mmap", dtype=np.float32, mode='r', shape=(34680, 460800))
train_feat_norm = np.memmap(featureDir + "trainFeatureNorm.mmap", dtype=np.float32, mode='r', shape=(88880, 460800))

test_feat_norm_pca = np.memmap(featureDir + "testFeatureNormPCA.mmap", dtype=np.float32, mode='w+', shape=(34680, 400))
train_feat_norm_pca = np.memmap(featureDir + "trainFeatureNormPCA.mmap", dtype=np.float32, mode='w+', shape=(88880, 400))

batch_size = 400
train_num_samples = train_feat_norm.shape[0]
train_num_batches = train_num_samples // batch_size
test_num_samples = test_feat_norm.shape[0]
test_num_batches = test_num_samples // batch_size

for i in range(train_num_batches):
    start_idx = i * batch_size
    end_idx = (i + 1) * batch_size
    batch = train_feat_norm[start_idx:end_idx]
    ipca.partial_fit(batch)

for i in range(train_num_samples):
    train_feat_norm_pca[i] = ipca.transform(train_feat_norm[i].reshape(1,-1))

del train_feat_norm
del train_feat_norm_pca
gc.collect()

for i in range(test_num_samples):
    test_feat_norm_pca[i] = ipca.transform(test_feat_norm[i].reshape(1,-1))

del test_feat_norm
del test_feat_norm_pca
del n_components
del ipca
del batch_size
del train_num_samples
del train_num_batches
del test_num_samples
del test_num_batches
gc.collect()

print("Mahalanobis Distance")

test_feat_norm_pca = np.memmap(featureDir + "testFeatureNormPCA.mmap", dtype=np.float32, mode='r', shape=(34680, 400))
train_feat_norm_pca = np.memmap(featureDir + "trainFeatureNormPCA.mmap", dtype=np.float32, mode='r', shape=(88880, 400))
covariance_matrix_inv = np.memmap(featureDir + "trainFeatureNormPCACovInvD.mmap", dtype=np.float32, mode='w+', shape=(400, 400))
distances = np.memmap(featureDir + "testTrainDistanceD.mmap", dtype=np.float32, mode='w+', shape=(34680, 88880))

covariance_matrix_inv[:] = np.linalg.inv(np.cov(train_feat_norm_pca, rowvar=False))

distances[:] = cdist(test_feat_norm_pca, train_feat_norm_pca, 'mahalanobis', VI=covariance_matrix_inv)

k = 200
k_smallest_distances_indices = np.argpartition(distances, k, axis=1)[:, :k]

k_smallest_distances = np.take_along_axis(distances, k_smallest_distances_indices, axis=1)

sum_of_k_smallest_distances = np.sum(k_smallest_distances, axis=1)

threshold = 10000
percentage = 5

# Find the indices above threshold
indices_above_threshold = np.where(sum_of_k_smallest_distances > threshold)[0]
np.savetxt(featureDir + "indices_above_threshold.txt", indices_above_threshold, fmt="%d")

# Find the indices corresponding to the smallest 5% sums
num_biggest_indices = int(len(sum_of_k_smallest_distances) * percentage / 100)
biggest_indices = np.argpartition(-sum_of_k_smallest_distances, num_biggest_indices)[:num_biggest_indices]
np.savetxt(featureDir + "indices_5_percentD.txt", biggest_indices, fmt="%d")


del train_feat_norm_pca
del test_feat_norm_pca
del covariance_matrix_inv
del distances
del k
del k_smallest_distances_indices
del sum_of_k_smallest_distances
del threshold
del percentage
del indices_above_threshold
del num_biggest_indices
del biggest_indices
gc.collect()

print("Find Ood Images")

line_numbers_file = "/fs/scratch/sgh_cr_bcai_dl_cluster_users/archive/CULaneFeature/indices_5_percent.txt" # indices_above_threshold.txt
original_file = "/home/ulc2sgh/laneDetectionFeature/RESA_ROOT/data/CULane/list/test.txt"
output_file = "/home/ulc2sgh/laneDetectionFeature/RESA_ROOT/data/CULane/list/output.txt"

with open(line_numbers_file, "r") as f:
    line_numbers = f.readlines()

with open(original_file, "r") as f:
    original_lines = f.readlines()

output_lines = []

for line_num in line_numbers:
    index = int(line_num.strip())
    if 0 <= index < len(original_lines):
        output_lines.append(original_lines[index])

with open(output_file, "w") as f:
    f.writelines(output_lines)

testFile = '/home/ulc2sgh/laneDetectionFeature/RESA_ROOT/data/CULane/list/test_split/'
first_file_path = '/home/ulc2sgh/laneDetectionFeature/RESA_ROOT/data/CULane/list/output.txt'
other_files_paths = [testFile+'test0_normal.txt', testFile+'test1_crowd.txt', testFile+'test2_hlight.txt', testFile+'test3_shadow.txt', testFile+'test4_noline.txt', testFile+'test5_arrow.txt', testFile+'test6_curve.txt', testFile+'test7_cross.txt', testFile+'test8_night.txt']

first_file_contents = set(open(first_file_path, 'r', encoding='utf-8').read().splitlines())

for i, other_file_path in enumerate(other_files_paths):

    other_file_contents = set(open(other_file_path, 'r', encoding='utf-8').read().splitlines())

    intersection = first_file_contents.intersection(other_file_contents)

    with open(f'intersection_result_{i}.txt', 'w', encoding='utf-8') as output_file:
        output_file.write('\n'.join(intersection))

print("END")