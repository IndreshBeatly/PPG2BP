import numpy as np
import os
import csv

def convert_npz_to_csv(npz_file_path: str, output_dir: str):
    """
    Convert all arrays in an .npz file to CSV files.

    Args:
        npz_file_path (str): Path to the .npz file.
        output_dir (str): Directory where CSV files will be saved.
    """
    # Load the .npz file
    try:
        data = np.load(npz_file_path, allow_pickle=True)
    except Exception as e:
        print(f"Error loading .npz file: {e}")
        return

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Iterate through all arrays in the .npz file
    for key in data.files:
        array = data[key]
        output_file = os.path.join(output_dir, f"{key}.csv")

        try:
            with open(output_file, 'w', newline='') as f:
                writer = csv.writer(f)

                # 0‐dimensional (scalar)
                if array.ndim == 0:
                    # array.item() extracts the scalar value
                    writer.writerow([array.item()])

                # 1‐dimensional
                elif array.ndim == 1:
                    writer.writerow(array.tolist())

                # 2‐dimensional or higher
                else:
                    # For any array with ndim >= 2, write each row separately
                    for row in array:
                        writer.writerow(row.tolist())

            print(f"Saved {key} to {output_file}")

        except Exception as e:
            print(f"Error saving {key} to CSV: {e}")


if __name__ == "__main__":
    # Example usage (change these paths as needed):
    npz_file_path = "C:/Users/Intern/Desktop/PPG2BPmodel/signals.npz"
    output_dir    = "C:/Users/Intern/Desktop/PPG2BPmodel"

    convert_npz_to_csv(npz_file_path, output_dir)
