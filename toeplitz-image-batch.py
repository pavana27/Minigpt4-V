import os
import numpy as np
import pandas as pd
from scipy.linalg import toeplitz
from skimage.transform import resize
import matplotlib.pyplot as plt

window_size = 224  # Number of samples per window
stride = 224  # Stride for the sliding window
output_dir = '/Users/pavana/Desktop/CREMA/HR-signal/'

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Function to create a Toeplitz matrix from a signal window
def create_toeplitz_matrix(signal_window):
    # Create the Toeplitz matrix
    c = signal_window
    r = np.zeros(len(signal_window))
    toeplitz_matrix = toeplitz(c, r)
    return toeplitz_matrix

# Read the breath signal input CSV file
#input_file_path = '/Users/pavana/Desktop/Student-Enagagement/Datasets/Daisee/breathing_signals.csv'
input_file_path = '/Users/pavana/Desktop/crema-HR.csv'
# Read the heart rate signal input CSV file
#input_file_path = '/Users/pavana/Desktop/Student-Enagagement/Datasets/Daisee/hr_signals.csv'
df = pd.read_csv(input_file_path)

# Process each column in the DataFrame
for column in df.columns:
    #change this to hr_signal in case of HR signal file or retain the same variable
    br_signal = df[column].values

    # Collect Toeplitz matrices in a list
    toeplitz_matrices = []

    # Segment the signal into windows and create Toeplitz matrices
    for i in range(0, len(br_signal) - window_size + 1, stride):
        signal_window = br_signal[i:i + window_size]
        print(f"Processing window starting at index {i}, length of signal_window: {len(signal_window)}")
        try:
            toeplitz_matrix = create_toeplitz_matrix(signal_window)
            toeplitz_matrices.append(toeplitz_matrix)
        except ValueError as e:
            print(e)

    # Concatenate all Toeplitz matrices into a single large matrix
    if toeplitz_matrices:
        large_matrix = np.concatenate(toeplitz_matrices, axis=0)
        
        # Rescale the large matrix to 224x224
        rescaled_matrix = resize(large_matrix, (224, 224), anti_aliasing=True)
        
        # Save the rescaled matrix as a grayscale image
        output_image_path = os.path.join(output_dir, f'{column}_toeplitz.png')
        #comment it out for l;arger batch processing
        """
        plt.imshow(rescaled_matrix, cmap='gray')
        plt.axis('off')
        plt.savefig(output_image_path, bbox_inches='tight', pad_inches=0)
        plt.close()
        """
        print(f'Rescaled Toeplitz image for column {column} saved to {output_image_path}')
    else:
        print(f"No Toeplitz matrices to concatenate for column {column}.")