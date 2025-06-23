import os
import cv2
import numpy as np
import pandas as pd
import argparse

def process_images(folder_path):
    # Two lists to store rows for each resolution.
    data_112 = []
    data_56 = []
    
    # Iterate over the subfolders: "Benign" and "Malicious"
    for label_name in ['Benign', 'Malicious']:
        # Assign label: 0 for benign, 1 for malicious
        label = 0 if label_name.lower() == 'benign' else 1
        subfolder = os.path.join(folder_path, label_name)
        
        # Process each image in the subfolder.
        for file_name in os.listdir(subfolder):
            if file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
                file_path = os.path.join(subfolder, file_name)
                # Read the image in grayscale mode.
                img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    print(f"Warning: Unable to read image {file_path}. Skipping.")
                    continue
                
                # Process based on image size.
                if img.shape == (112, 112):
                    # Normalize the pixel values to the range [0, 1].
                    img_norm = img.astype(np.float32) / 255.0
                    # Flatten the image into a 1D vector.
                    features = img_norm.flatten()
                    row = [file_name] + features.tolist() + [label]
                    data_112.append(row)
                elif img.shape == (56, 56):
                    img_norm = img.astype(np.float32) / 255.0
                    features = img_norm.flatten()
                    row = [file_name] + features.tolist() + [label]
                    data_56.append(row)
                else:
                    print(f"Skipping {file_path} due to unexpected size: {img.shape}")
                    
    return data_112, data_56

def save_to_csv(data, resolution, output_file):
    # Determine the number of features from the image resolution.
    if resolution == 112:
        num_features = 112 * 112
    elif resolution == 56:
        num_features = 56 * 56
    else:
        raise ValueError("Unsupported resolution")
        
    # Create column names: 'filename', then f0, f1, ..., f{num_features-1}, and 'label'
    columns = ['filename'] + [f"f{i}" for i in range(num_features)] + ['label']
    df = pd.DataFrame(data, columns=columns)
    # Remove any rows with null values (shouldn't happen, but just in case).
    df.dropna(inplace=True)
    # Save the DataFrame to CSV.
    df.to_csv(output_file, index=False)
    print(f"Saved {len(df)} rows to {output_file}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert images to CSV with normalized features.')
    parser.add_argument('--folder', type=str, required=True,
                        help='Path to the main folder containing "Benign" and "Malicious" subfolders.')
    args = parser.parse_args()
    folder_path = args.folder

    data_112, data_56 = process_images(folder_path)
    
    # Save two CSV files: one for 112x112 images and one for 56x56 images.
    save_to_csv(data_112, 112, 'features_112.csv')
    save_to_csv(data_56, 56, 'features_56.csv')
