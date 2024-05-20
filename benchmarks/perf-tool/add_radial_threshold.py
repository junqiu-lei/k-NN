import h5py
import numpy as np
import os
import sys
from datetime import datetime

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

def calculate_scores(test_queries, train_docs, distance_metric='l2_squared'):
    distances = calculate_distances(test_queries, train_docs, distance_metric)
    if distance_metric == 'l2_squared':
        scores = 1 / (1 + distances)
    elif distance_metric == 'cosine':
        scores = (2 - distances) / 2
    else:
        raise ValueError("Unsupported distance metric")
    return scores

def add_threshold_dataset(input_file_path, output_file_path, max_distance, engine_type, distance_metric='l2_squared', max_length=10000):
    with h5py.File(input_file_path, 'r') as input_hdf5, h5py.File(output_file_path, 'w') as output_hdf5:
        if 'train' not in input_hdf5.keys() or 'test' not in input_hdf5.keys():
            raise ValueError("The input file must contain 'train' and 'test' datasets.")

        # Copy existing datasets all fields except the ones we are going to modify
        for key in input_hdf5.keys():
            input_hdf5.copy(key, output_hdf5)

        train_docs = input_hdf5['train'][()]
        test_queries = input_hdf5['test'][()]

        # Use first 100 for test purposes, as indicated in your snippet
        # test_queries = test_queries[:100]

        padded_data = np.full((len(test_queries), max_length), -1, dtype=int)  # Using -1 for padding

        avarage_num_neighbors = 0
        median_num_neighbors = 0
        min_num_neighbors = 0
        max_num_neighbors = 0
        num_zero_neighbors = 0

        for i, test_query in enumerate(test_queries):
            distances = calculate_distances(test_query, train_docs, engine_type, distance_metric)
            print(f"Query target {i} distances calculated.")
            print(f"distances: {distances}")
            # if engine_type == 'lucene':
            #     # max_distance = -max_distance
            #     distances = -distances
            within_threshold_ids = np.where(distances <= max_distance)[0]



            # print(f"within_threshold_ids: {within_threshold_ids}")
            print(f"total number of neighbors within threshold: {len(within_threshold_ids)}")
            # if engine_type == 'lucene' and distance_metric == 'inner_product':
            #     sorted_ids = within_threshold_ids[np.argsort(-distances[within_threshold_ids])][:max_length]
            # else:
            sorted_ids = within_threshold_ids[np.argsort(distances[within_threshold_ids])][:max_length]
            padded_data[i, :len(sorted_ids)] = sorted_ids
            print(f"Query target {i} done.")

            avarage_num_neighbors += len(sorted_ids)
            if i == 0:
                median_num_neighbors = len(sorted_ids)
            else:
                if len(sorted_ids) < min_num_neighbors:
                    min_num_neighbors = len(sorted_ids)
                if len(sorted_ids) > max_num_neighbors:
                    max_num_neighbors = len(sorted_ids)
            if i == 0:
                min_num_neighbors = len(sorted_ids)
                max_num_neighbors = len(sorted_ids)
            else:
                if len(sorted_ids) < min_num_neighbors:
                    min_num_neighbors = len(sorted_ids)
                if len(sorted_ids) > max_num_neighbors:
                    max_num_neighbors = len(sorted_ids)
            if len(sorted_ids) == 0:
                num_zero_neighbors += 1

        avarage_num_neighbors /= len(test_queries)
        print(f"max_distance: {max_distance}")
        print(f"avarage_num_neighbors: {avarage_num_neighbors}")
        print(f"median_num_neighbors: {median_num_neighbors}")
        print(f"min_num_neighbors: {min_num_neighbors}")
        print(f"max_num_neighbors: {max_num_neighbors}")
        print(f"num_zero_neighbors: {num_zero_neighbors}")

        # Create a new dataset with padded data
        # dataset_name = 'max_distance_{:.2f}'.format(max_distance).replace('.', '_') + '_padded'
        dataset_name = 'max_distance_neighbors'
        output_hdf5.create_dataset(dataset_name, data=padded_data)

        print(f"Dataset '{dataset_name}' added successfully to {output_file_path}.")

if __name__ == "__main__":
    if len(sys.argv) != 6:
        print("Usage: python add_radial_threshold.py <input_hdf5_file> <output_hdf5_file> <max_distance> <distance_metric> <engine_type>")
    else:
        input_file_path = sys.argv[1]
        output_file_path = sys.argv[2]
        max_distance = float(sys.argv[3])
        distance_metric = sys.argv[4]
        engine_type = sys.argv[5]
        add_threshold_dataset(input_file_path, output_file_path, max_distance, engine_type, distance_metric)
