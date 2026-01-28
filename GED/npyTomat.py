## Convert the .npy EEG files to .mat files
import numpy as np
from scipy.io import savemat
import scipy.io as sio
import os


def npy_to_mat(npy_file, mat_file):
    data = np.load(npy_file,allow_pickle=True).item()  # Load the .npy file
    savemat(mat_file, {'data': data})  # Save to .mat file

npy_to_mat('G:\Resting state\sub-01\sub-01\ses-01\eeg_resting_state_1.npy', 'G:\Resting state\sub-01\sub-01\ses-03')
npy_to_mat('G:\Resting state\sub-01\sub-01\ses-01\eeg_resting_state_1.npy', 'G:\Resting state\sub-01\sub-01\ses-04')
npy_to_mat('G:\Resting state\sub-07\sub-07\ses-07\eeg_resting_state_1.npy', 'G:\Resting state\sub-07\sub-07\ses-02')
npy_to_mat('G:\Resting state\sub-07\sub-07\ses-07\eeg_resting_state_1.npy', 'G:\Resting state\sub-07\sub-07\ses-03')
npy_to_mat('G:\Resting state\sub-08\sub-08\ses-08\eeg_resting_state_1.npy', 'G:\Resting state\sub-08\sub-08\ses-04')
npy_to_mat('G:\Resting state\sub-08\sub-08\ses-08\eeg_resting_state_1.npy', 'G:\Resting state\sub-08\sub-08\ses-02')
## convert the data similarly for all sessions in these subjects


def convert_mat_field_to_npy(mat_file_path, field_name, output_npy_path=None):
    """
    Extract a specific field from a .mat file and save it as a .npy file.
    
    Parameters:
    -----------
    mat_file_path : str
        Path to the .mat file
    field_name : str
        Name of the field to extract
    output_npy_path : str, optional
        Path where to save the .npy file. If None, uses the same path as mat_file
        but with .npy extension
        
    Returns:
    --------
    numpy.ndarray
        The extracted field data
    """
    # Load the .mat file
    mat_data = sio.loadmat(mat_file_path)
    
    # Check if the field exists
    if field_name not in mat_data:
        raise KeyError(f"Field '{field_name}' not found in the .mat file")
    
    # Extract the field
    field_data = mat_data[field_name]
    
    # Check if the shape is 20x63
    shape = field_data.shape
    print(f"Extracted field shape: {shape}")
    
    if shape != (20, 63):
        print(f"Warning: The extracted field has shape {shape}, not the expected (20, 63)")
    
    # Check if data type is double
    if field_data.dtype != np.float64:
        print(f"Warning: The extracted field has dtype {field_data.dtype}, not np.float64 (double)")
        # Convert to double if needed
        field_data = field_data.astype(np.float64)
    
    # Define output path if not provided
    if output_npy_path is None:
        base_name = os.path.splitext(mat_file_path)[0]
        output_npy_path = f"{base_name}_{field_name}.npy"
    
    # Save as .npy file
    np.save(output_npy_path, field_data)
    print(f"Field '{field_name}' successfully saved to {output_npy_path}")
    
    return field_data
