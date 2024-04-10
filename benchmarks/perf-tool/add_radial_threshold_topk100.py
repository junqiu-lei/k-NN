import h5py
import numpy as np
import os
import sys
import shutil

def add_thresholds(source_file_path, output_file_path, space_type):
    # Generate new file path by appending "_with_thresholds" before the file extension
    base, ext = os.path.splitext(source_file_path)

    # Copy the original file to the new file path
    shutil.copy2(source_file_path, output_file_path)

    try:
        with h5py.File(output_file_path, 'a') as hdf5_file:  # Open the new file in append mode
            # Ensure required dataset 'distances' is present
            if 'distances' not in hdf5_file.keys():
                print("Required dataset 'distances' not found in the file.")
                return

            # Load the 'distances' dataset
            distances = hdf5_file['distances'][()]

            max_distances = []
            min_scores = []

            if space_type == 'cosine':
                # Calculate the cosine similarity for the distances
                max_distances = 1 - np.min(distances, axis=1)
                min_scores = (2 - max_distances) / 2

            elif space_type == 'l2_squared':
                # Calculate the L2 squared (Euclidean squared) distance for the distances
                max_distances = np.square(distances)
                min_scores = 1 / (1 + max_distances)

            else:
                print(f"Unsupported space type: {space_type}. Supported types are 'cosine' and 'l2_squared'.")
                return

            # Create new datasets in the new file
            hdf5_file.create_dataset('max_distance_topk100', data=max_distances)
            hdf5_file.create_dataset('min_score_topk100', data=min_scores)

            print(f"Datasets 'max_distance_topk100' and 'min_score_topk100' added successfully to {output_file_path}.")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python script.py <path_to_hdf5_file> <path_to_hdf5_output_file> <space_type>")
    else:
        source_file_path = sys.argv[1]
        output_file_path = sys.argv[2]
        space_type = sys.argv[3]
        add_thresholds(source_file_path, output_file_path, space_type)
