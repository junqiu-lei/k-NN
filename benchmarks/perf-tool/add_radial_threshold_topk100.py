import h5py
import numpy as np
import os
import sys
import shutil

def calculate_distances(test_query, train_docs, engine_type, distance_metric='l2_squared'):
    if distance_metric == 'l2_squared':
        distances = np.sum((train_docs - test_query) ** 2, axis=1)
    elif distance_metric == 'cosine':
        norm_test = np.linalg.norm(test_query)
        norms_train = np.linalg.norm(train_docs, axis=1)
        distances = 1 - (np.dot(train_docs, test_query) / (norms_train * norm_test))
    elif distance_metric == 'inner_product':
        # distances = np.dot(train_docs, test_query)

        if engine_type == 'faiss':
            distances = -np.dot(train_docs, test_query)
        elif engine_type == 'lucene':
            distances = np.dot(train_docs, test_query)
    else:
        raise ValueError("Unsupported distance metric")
    return distances

def add_thresholds(source_file_path, output_file_path, space_type, engine_type):
    # Generate new file path by appending "_with_thresholds" before the file extension
    base, ext = os.path.splitext(source_file_path)

    # Copy the original file to the new file path
    shutil.copy2(source_file_path, output_file_path)

    try:
        with h5py.File(output_file_path, 'a') as hdf5_file:  # Open the new file in append mode
            # Ensure required dataset 'distances' is present
            # if 'distances' not in hdf5_file.keys():
            #     print("Required dataset 'distances' not found in the file.")
            #     return

            # Load the 'distances' dataset
            # distances = hdf5_file['distances'][()]
            train_docs = hdf5_file['train'][()]
            test_queries = hdf5_file['test'][()]
            neighbors = hdf5_file['neighbors'][()]

            # read the 100th neighbor id for each query
            neighbor_ids = neighbors[:, 99]
            print(neighbor_ids)
            print(len(neighbor_ids))
            i = 0
            distances = []
            scores = []
            for neighbor_id in neighbor_ids:
                # print(f"neighbor_id: {neighbor_id}")
                # print(f"test_queries: {test_queries}")
                test_query = test_queries[i]
                if space_type == 'l2_squared':
                    distance = np.sum((train_docs[neighbor_id] - test_queries) ** 2, axis=1)
                    score = 1 / (1 + distance)
                elif space_type == 'cosine':
                    norm_test = np.linalg.norm(test_queries[i])
                    norms_train = np.linalg.norm(train_docs[neighbor_id], axis=1)
                    distance = 1 - (np.dot(train_docs[neighbor_id], test_queries[i]) / (norms_train * norm_test))
                    score = (2 - distance) / 2
                elif space_type == 'inner_product':
                    if engine_type == 'lucene':
                        distance = np.dot(train_docs[neighbor_id], test_queries[i])
                        if distance > 0:
                            score = 1 + distance
                        else:
                            score = 1 / (1 - distance)
                    else:
                        distance = -np.dot(train_docs[neighbor_id], test_queries[i])
                        if distance >= 0:
                            score = 1 / (1 + distance)
                        else:
                            score = 1 - distance
                else:
                    raise ValueError("Unsupported distance metric")
                distances.append(distance)
                scores.append(score)
                i += 1

                # print(f"distance: {distance}")
            # print(f"distances: {distances}")
            print(f"len(distances): {len(distances)}")
            print(f"median: {np.median(distances)}")
            print(f"mean: {np.mean(distances)}")
            print(f"max: {np.max(distances)}")
            print(f"min: {np.min(distances)}")
            print(f"std: {np.std(distances)}")
            print(f"len(scores): {len(scores)}")
            print(f"median: {np.median(scores)}")
            print(f"mean: {np.mean(scores)}")
            print(f"max: {np.max(scores)}")
            print(f"min: {np.min(scores)}")

            hdf5_file.create_dataset('max_distance_topk100', data=distances)
            hdf5_file.create_dataset('min_score_topk100', data=scores)

            print(f"Datasets 'max_distance_topk100' and 'min_score_topk100' added successfully to {output_file_path}.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python script.py <path_to_hdf5_file> <path_to_hdf5_output_file> <space_type> <engine_type>")
    else:
        source_file_path = sys.argv[1]
        output_file_path = sys.argv[2]
        space_type = sys.argv[3]
        engine_type = sys.argv[4]
        add_thresholds(source_file_path, output_file_path, space_type, engine_type)
