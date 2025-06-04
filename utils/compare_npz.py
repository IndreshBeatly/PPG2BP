import os
import numpy as np

def compare_npz_files(file1, file2, atol=1e-5, rtol=1e-3):
    """
    Compare two .npz files and return a dictionary of differences.
    """
    try:
        d1 = np.load(file1)
        d2 = np.load(file2)
    except Exception as e:
        return {"error": str(e)}

    all_keys = set(d1.files) | set(d2.files)
    report = {}

    for k in all_keys:
        if k not in d1.files:
            report[k] = " Missing in file1"
        elif k not in d2.files:
            report[k] = " Missing in file2"
        else:
            arr1, arr2 = d1[k], d2[k]
            if arr1.shape != arr2.shape:
                report[k] = f" Shape mismatch {arr1.shape} vs {arr2.shape}"
            elif not np.allclose(arr1, arr2, atol=atol, rtol=rtol):
                max_diff = np.max(np.abs(arr1 - arr2))
                report[k] = f" Max difference = {max_diff:.6f}"
            else:
                report[k] = " Match"

    return report

def compare_patient_steps(manual_dir, pipeline_dir):
    """
    Compare all substep .npz files in two directories for the same patient.
    """
    step_files = sorted(f for f in os.listdir(manual_dir) if f.endswith(".npz"))

    print(f"\n📋 Comparing patient folders:")
    print(f"   Manual  → {manual_dir}")
    print(f"   Pipeline→ {pipeline_dir}\n")

    for step_file in step_files:
        manual_path = os.path.join(manual_dir, step_file)
        pipeline_path = os.path.join(pipeline_dir, step_file)

        print(f" Comparing {step_file}...")
        if not os.path.exists(pipeline_path):
            print(f"   Missing in pipeline: {step_file}")
            continue

        report = compare_npz_files(manual_path, pipeline_path)
        for k, result in report.items():
            print(f"    • {k:<15}: {result}")
        print()

if __name__ == "__main__":
    # Replace these with your actual paths
    manual_dir = "manual_outputs/1"
    pipeline_dir = "tmp_pipeline/1"

    compare_patient_steps(manual_dir, pipeline_dir)
