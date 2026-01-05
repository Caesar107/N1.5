#!/usr/bin/env python
"""
Convert native LIBERO spatial data to training format.

This script converts the native LIBERO data format (with quaternions) to the format
expected by the training pipeline (with Euler angles).
"""

import argparse
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation
from tqdm import tqdm


def quat_to_euler(quat_wxyz):
    """Convert quaternion (w, x, y, z) to Euler angles (roll, pitch, yaw) in radians."""
    # scipy uses xyzw order
    quat_xyzw = np.array([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]])
    r = Rotation.from_quat(quat_xyzw)
    euler = r.as_euler('xyz', degrees=False)
    return euler


def convert_episode(parquet_path, output_path):
    """Convert a single episode from native format to training format."""
    df = pd.read_parquet(parquet_path)
    
    # Convert state: gripper_qpos (2) + ee_pos (3) + ee_quat (4) -> xyz (3) + rpy (3) + gripper (2)
    new_states = []
    for state in df['observation.state']:
        gripper_qpos = state[0:2]  # 2 dim
        ee_pos = state[2:5]  # 3 dim (x, y, z)
        ee_quat = state[5:9]  # 4 dim (w, x, y, z)
        
        # Convert quaternion to Euler angles
        euler = quat_to_euler(ee_quat)
        
        # New state: x, y, z, roll, pitch, yaw, gripper (2 dim)
        new_state = np.concatenate([ee_pos, euler, gripper_qpos])
        new_states.append(new_state)
    
    # Update state column
    df['observation.state'] = new_states
    
    # Action is already in the correct format (ee_delta 6 dim + gripper 1 dim)
    # No conversion needed for action
    
    # Save converted episode
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)


def recompute_stats(output_dir):
    """Recompute statistics for the converted dataset."""
    print("Recomputing statistics for converted data...")
    data_dir = output_dir / 'data'
    
    all_states = []
    all_actions = []
    all_timestamps = []
    all_task_indices = []
    all_episode_indices = []
    all_indices = []
    all_rewards = []
    all_dones = []
    
    idx = 0
    for chunk_dir in sorted(data_dir.glob('chunk-*')):
        for parquet_file in sorted(chunk_dir.glob('episode_*.parquet')):
            df = pd.read_parquet(parquet_file)
            states = np.array([s for s in df['observation.state']])
            actions = np.array([a for a in df['action']])
            all_states.append(states)
            all_actions.append(actions)
            
            if 'timestamp' in df.columns:
                all_timestamps.extend(df['timestamp'].tolist())
            if 'task_index' in df.columns:
                all_task_indices.extend(df['task_index'].tolist())
            if 'episode_index' in df.columns:
                all_episode_indices.extend(df['episode_index'].tolist())
            if 'next.reward' in df.columns:
                all_rewards.extend(df['next.reward'].tolist())
            if 'next.done' in df.columns:
                all_dones.extend(df['next.done'].tolist())
            
            for _ in range(len(df)):
                all_indices.append(idx)
                idx += 1
    
    all_states = np.vstack(all_states)
    all_actions = np.vstack(all_actions)
    
    def compute_stats(data):
        if len(data.shape) == 1:
            data = data.reshape(-1, 1)
        return {
            'mean': data.mean(axis=0).tolist(),
            'std': data.std(axis=0).tolist(),
            'min': data.min(axis=0).tolist(),
            'max': data.max(axis=0).tolist(),
            'q01': np.percentile(data, 1, axis=0).tolist(),
            'q99': np.percentile(data, 99, axis=0).tolist()
        }
    
    new_stats = {
        'observation.state': compute_stats(all_states),
        'action': compute_stats(all_actions),
    }
    
    if all_timestamps:
        new_stats['timestamp'] = compute_stats(np.array(all_timestamps))
    if all_task_indices:
        new_stats['task_index'] = compute_stats(np.array(all_task_indices))
        new_stats['annotation.human.action.task_description'] = compute_stats(np.array(all_task_indices))
    if all_episode_indices:
        new_stats['episode_index'] = compute_stats(np.array(all_episode_indices))
    if all_indices:
        new_stats['index'] = compute_stats(np.array(all_indices))
        new_stats['vlm_hidden_state_index'] = compute_stats(np.array(all_indices))
    if all_rewards:
        new_stats['next.reward'] = compute_stats(np.array(all_rewards))
    if all_dones:
        new_stats['next.done'] = compute_stats(np.array(all_dones))
    
    new_stats['annotation.human.validity'] = {
        'mean': [1.0], 'std': [0.0], 'min': [1.0], 'max': [1.0], 'q01': [1.0], 'q99': [1.0]
    }
    
    stats_path = output_dir / 'meta' / 'stats.json'
    with open(stats_path, 'w') as f:
        json.dump(new_stats, f, indent=4)
    
    print(f"Updated stats.json: state dim={len(new_stats['observation.state']['mean'])}, action dim={len(new_stats['action']['mean'])}")


def convert_dataset(input_dir, output_dir):
    """Convert entire dataset from native format to training format."""
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    
    # Copy meta files
    print("Copying meta files...")
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copytree(input_dir / 'meta', output_dir / 'meta', dirs_exist_ok=True)
    shutil.copytree(input_dir / 'videos', output_dir / 'videos', dirs_exist_ok=True)
    
    # Update modality.json
    print("Updating modality.json...")
    modality_path = output_dir / 'meta' / 'modality.json'
    with open(modality_path, 'r') as f:
        modality = json.load(f)
    
    # Update to match training format
    modality['state'] = {
        "x": {"start": 0, "end": 1},
        "y": {"start": 1, "end": 2},
        "z": {"start": 2, "end": 3},
        "roll": {"start": 3, "end": 4},
        "pitch": {"start": 4, "end": 5},
        "yaw": {"start": 5, "end": 6},
        "gripper": {"start": 6, "end": 8}
    }
    modality['action'] = {
        "x": {"start": 0, "end": 1},
        "y": {"start": 1, "end": 2},
        "z": {"start": 2, "end": 3},
        "roll": {"start": 3, "end": 4},
        "pitch": {"start": 4, "end": 5},
        "yaw": {"start": 5, "end": 6},
        "gripper": {"start": 6, "end": 7}
    }
    modality['video'] = {
        "image": {"original_key": "observation.images.agentview"},
        "wrist_image": {"original_key": "observation.images.wrist"}
    }
    modality['annotation'] = {
        "human.action.task_description": {"original_key": "annotation.human.action.task_description"}
    }
    
    with open(modality_path, 'w') as f:
        json.dump(modality, f, indent=4)
    
    # Convert data files
    print("Converting data files...")
    data_dir = input_dir / 'data'
    output_data_dir = output_dir / 'data'
    output_data_dir.mkdir(parents=True, exist_ok=True)
    
    for chunk_dir in sorted(data_dir.glob('chunk-*')):
        chunk_name = chunk_dir.name
        output_chunk_dir = output_data_dir / chunk_name
        output_chunk_dir.mkdir(parents=True, exist_ok=True)
        
        parquet_files = sorted(chunk_dir.glob('episode_*.parquet'))
        for parquet_file in tqdm(parquet_files, desc=f"Converting {chunk_name}"):
            output_file = output_chunk_dir / parquet_file.name
            convert_episode(parquet_file, output_file)
    
    # Recompute statistics for converted data
    recompute_stats(output_dir)
    
    print(f"Conversion complete! Output directory: {output_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert native LIBERO data to training format')
    parser.add_argument('--input_dir', type=str, required=True,
                       help='Input directory containing native LIBERO data')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for converted data')
    args = parser.parse_args()
    
    convert_dataset(args.input_dir, args.output_dir)

