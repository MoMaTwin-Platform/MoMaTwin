import os
import json
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import sys

def analyze_multiple_directories(directories, output_dir="action_stats"):
    all_dim_values = defaultdict(list)
    total_trajectories = 0
    valid_trajectories = 0
    all_nan_entries = defaultdict(set)
    
    for directory in directories:
        print(f"Processing directory: {directory}")
        dim_values, total, valid, nan_entries = process_directory(directory)
        total_trajectories += total
        valid_trajectories += valid
        
        # Merge dimension values
        for dim, values in dim_values.items():
            all_dim_values[dim].extend(values)
        
        # Merge NaN entry records
        for dim, trajs in nan_entries.items():
            all_nan_entries[dim].update(trajs)
    
    # Generate combined statistics
    print(f"Total trajectories across all directories: {total_trajectories}")
    print(f"Valid trajectories across all directories: {valid_trajectories}")
    
    # Calculate min/max for each dimension
    min_max_values = {}
    for dim, values in all_dim_values.items():
        values_array = np.array(values)
        min_max_values[dim] = {
            "min": float(np.min(values_array)),
            "max": float(np.max(values_array)),
            "mean": float(np.mean(values_array)),
            "std": float(np.std(values_array))
        }
    
    # Generate distribution histograms
    os.makedirs(output_dir, exist_ok=True)
    
    for dim, values in all_dim_values.items():
        plt.figure(figsize=(10, 6))
        plt.hist(values, bins=50)
        plt.title(f"Distribution of {dim}")
        plt.xlabel("Value")
        plt.ylabel("Frequency")
        plt.savefig(os.path.join(output_dir, f"{dim}_distribution.png"))
        plt.close()
    
    # Save min/max values to JSON
    with open(os.path.join(output_dir, "min_max_stats.json"), 'w') as f:
        json.dump(min_max_values, f, indent=2)
    
    # Save NaN entries to a report file
    nan_report = {}
    for dim, trajs in all_nan_entries.items():
        nan_report[dim] = list(trajs)
    
    with open(os.path.join(output_dir, "nan_report.json"), 'w') as f:
        json.dump(nan_report, f, indent=2)
    
    # Print summary of NaN values
    if all_nan_entries:
        print("\nDimensions with NaN values:")
        for dim, trajs in all_nan_entries.items():
            print(f"  - {dim}: {len(trajs)} trajectories")
        print(f"Detailed NaN report saved to {os.path.join(output_dir, 'nan_report.json')}")
    
    print(f"Combined statistics saved to {output_dir}")
    return min_max_values

def process_directory(base_path):
    # Load report.json to identify excluded trajectories
    report_path = os.path.join(base_path, "report.json")
    with open(report_path, 'r') as f:
        report = json.load(f)
    
    # Create a set of excluded trajectories
    excluded = set()
    for category in ["out_of_3_sigma", "rot_out_of_range", "vague_sample", 
                    "No_action", "gripper_fast_oscillations", "Switch_Over_Range"]:
        excluded.update(report.get(category, []))
    
    # Get valid trajectories (all samples minus excluded ones)
    all_samples = set(report.get("sample_name", []))
    valid_trajectories = all_samples - excluded
    
    print(f"  - Total trajectories in directory: {len(all_samples)}")
    print(f"  - Valid trajectories in directory: {len(valid_trajectories)}")
    
    # Dictionaries to store stats
    dimensions = None
    dim_values = defaultdict(list)
    nan_entries = defaultdict(set)  # To track NaN values
    
    # Process each valid trajectory
    for traj in valid_trajectories:
        traj_folder = os.path.join(base_path, traj)
        traj_file = os.path.join(traj_folder, f"{traj}.json")
        traj_path = os.path.join(base_path, traj)
        
        try:
            with open(traj_file, 'r') as f:
                data = json.load(f)
            
            # Extract all dimensions from first trajectory if not already done
            if dimensions is None:
                dimensions = extract_dimensions(data["data"][0])
            
            # Collect values for each dimension
            for frame_idx, frame in enumerate(data["data"]):
                for dim, path in dimensions.items():
                    value = get_nested_value(frame, path)
                    if isinstance(value, list):
                        for i, v in enumerate(value):
                            dim_name = f"{dim}_{i}"
                            # Check for NaN
                            if isinstance(v, (int, float)) and np.isnan(v):
                                nan_entries[dim_name].add(f"{traj_path} (frame {frame_idx})")
                            else:
                                dim_values[dim_name].append(v)
                    else:
                        # Check for NaN
                        if isinstance(value, (int, float)) and np.isnan(value):
                            nan_entries[dim].add(f"{traj_path} (frame {frame_idx})")
                        else:
                            dim_values[dim].append(value)
        except Exception as e:
            print(f"    Error processing {traj}: {e}")
    
    # Report dimensions with NaN values for this directory
    if nan_entries:
        print(f"  - Found NaN values in {len(nan_entries)} dimensions in this directory")
    
    return dim_values, len(all_samples), len(valid_trajectories), nan_entries

def extract_dimensions(frame_data):
    """Extract all dimensions from a single frame."""
    dimensions = {}
    
    def extract_recursive(data, path=None):
        if path is None:
            path = []
        
        if isinstance(data, dict):
            for k, v in data.items():
                extract_recursive(v, path + [k])
        elif isinstance(data, list):
            # If it's a list of numbers, consider it as a dimension
            if all(isinstance(item, (int, float)) or np.isnan(item) for item in data):
                dimensions["/".join(path)] = path
            else:
                for i, item in enumerate(data):
                    extract_recursive(item, path + [str(i)])
        else:
            # It's a leaf node, consider it as a dimension
            dimensions["/".join(path)] = path
    
    extract_recursive(frame_data)
    return dimensions

def get_nested_value(data, path):
    """Get value from nested dictionary using path list."""
    for key in path:
        data = data[key]
    return data

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python script.py <output_dir> <dir1> [<dir2> <dir3> ...]")
        sys.exit(1)
    
    output_dir = sys.argv[1]
    directories = sys.argv[2:]
    analyze_multiple_directories(directories, output_dir)