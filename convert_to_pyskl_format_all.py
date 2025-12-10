import pickle
import numpy as np
import pandas as pd
import json
import os
from collections import defaultdict

def load_3dyoga_json(json_path):
    """load 3D Yoga JSON file to get pose labels and split info"""
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Create a mapping from pose names to labels
    pose_to_label = {}
    label_counter = 0
    video_info = {}  # sequence_id -> (pose_name, label, split)
    
    for pose_entry in data:
        pose_name = pose_entry['pose']
        if pose_name not in pose_to_label:
            pose_to_label[pose_name] = label_counter
            label_counter += 1
        
        for instance in pose_entry['instances']:
            sequence_id = str(instance['sequence_id'])
            video_info[sequence_id] = {
                'pose': pose_name,
                'label': pose_to_label[pose_name],
                'split': instance['split']
            }
    
    return video_info, pose_to_label

def load_skeleton_data(parquet_path, sequence_id, target_keypoints=25):
    """Load skeleton data from Parquet file and map to target keypoints format"""
    parquet_file = os.path.join(parquet_path, f"{sequence_id}.parquet")
    if not os.path.exists(parquet_file):
        return None
    
    df = pd.read_parquet(parquet_file)
    
    # Get total number of frames
    total_frames = df['frame'].max() + 1
    
    # Get original number of keypoints
    orig_keypoints = len(df['landmark_index'].unique())
    
    # Map MediaPipe's 33 keypoints to NTU RGB+D's 25 keypoints
    def map_to_ntu_keypoints(orig_data, orig_kp_count, target_kp_count):
        """
        Map original keypoints to target keypoints format
        Implement the specific mapping logic based on the keypoint extraction method you use
        """
        if orig_kp_count == target_kp_count:
            return orig_data
        
        # Map 33 keypoints to 25 keypoints
        mapping = {
            0: 3,   # head
            11: 4,  # left shoulder
            12: 8,  # right shoulder
            13: 5,  # left elbow
            14: 9,  # right elbow
            15: 7,  # left hand
            16: 11, # right hand
            19: 21, # left hand tip
            20: 23, # right hand tip
            21: 22, # left thumb
            22: 24, # right thumb
            23: 12, # left hip
            24: 16, # right hip
            25: 13, # left knee
            26: 17, # right knee
            27: 14, # left ankle
            28: 18, # right ankle
            31: 15, # left foot
            32: 19  # right foot
        }
        
        mapped_data = np.zeros((1, total_frames, target_kp_count, 3))
        
        for orig_idx, target_idx in mapping.items():
            if orig_idx < orig_kp_count:
                mapped_data[0, :, target_idx, :] = orig_data[0, :, orig_idx, :]
        
        # fine tune the mapping relation
        mapped_data[0, :, 0, :] = (orig_data[0, :, 23, :] + orig_data[0, :, 24, :]) / 2.0 # base of spine
        mapped_data[0, :, 20, :] = (orig_data[0, :, 11, :] + orig_data[0, :, 12, :]) / 2.0 # spine
        mapped_data[0, :, 1, :] = (mapped_data[0, :, 0, :] + mapped_data[0, :, 20, :]) / 2.0 # middle of spine
        mapped_data[0, :, 2, :] = (2 * mapped_data[0, :, 20, :] + mapped_data[0, :, 3, :]) / 3.0 # neck
        
        return mapped_data
    
    # Reorganize data to [M, T, V, C] format
    keypoints = np.zeros((1, total_frames, orig_keypoints, 3))
    
    for frame in range(total_frames):
        frame_data = df[df['frame'] == frame]
        for _, row in frame_data.iterrows():
            landmark_idx = row['landmark_index']
            keypoints[0, frame, landmark_idx, 0] = row['x']
            keypoints[0, frame, landmark_idx, 1] = row['y']
            keypoints[0, frame, landmark_idx, 2] = row['z']
    
    # Map to target keypoints count
    keypoints = map_to_ntu_keypoints(keypoints, orig_keypoints, target_keypoints)
    
    return keypoints, total_frames

def convert_to_pyskl_format(json_path, parquet_path, output_path, num_keypoints=25):
    """Convert 3D Yoga data to a single PySKL format pkl file"""
    
    # Load original data information
    video_info, pose_to_label = load_3dyoga_json(json_path)
    
    # Initialize PySKL data structure
    pyskl_data = {
        'split': defaultdict(list),
        'annotations': []
    }
    
    # Process each video
    processed_count = 0
    for sequence_id, info in video_info.items():
        print(f"Processing {sequence_id}...")
        
        # Load skeleton data
        skeleton_data = load_skeleton_data(parquet_path, sequence_id, num_keypoints)
        if skeleton_data is None:
            print(f"Warning: No skeleton data for {sequence_id}")
            continue
            
        keypoints, total_frames = skeleton_data
        
        # Create annotation entry
        annotation = {
            'frame_dir': sequence_id,
            'total_frames': total_frames,
            # 'img_shape': (1080, 1920),  # Adjust according to actual video dimensions
            # 'original_shape': (1080, 1920),
            'label': info['label'],  # Integer label
            'keypoint': keypoints.astype(np.float32),
            # 'keypoint_score': np.ones((1, total_frames, num_keypoints), dtype=np.float32)
        }
        
        pyskl_data['annotations'].append(annotation)
        pyskl_data['split'][info['split']].append(sequence_id)
        processed_count += 1
    
    # Save label mapping information in the pkl file (as an additional field)
    pyskl_data['label_mapping'] = {v: k for k, v in pose_to_label.items()}
    pyskl_data['num_classes'] = len(pose_to_label)
    
    # Save as a single pickle file
    with open(output_path, 'wb') as f:
        pickle.dump(pyskl_data, f)
    
    # Save label mapping separately
    label_mapping = {v: k for k, v in pose_to_label.items()}
    with open(os.path.join(os.path.dirname(output_path), 'label_mapping.pkl'), 'wb') as f:
        pickle.dump(label_mapping, f)
    
    print(f"Conversion completed! Processed {processed_count} samples")
    print(f"Training set: {len(pyskl_data['split']['train'])} samples")
    print(f"Test set: {len(pyskl_data['split']['test'])} samples")
    print(f"Number of classes: {pyskl_data['num_classes']}")
    
    return pyskl_data

if __name__ == "__main__":
    convert_to_pyskl_format(
        json_path='/root/autodl-tmp/EECS442 project/3DYoga90/data/3DYoga90.json',
        parquet_path='/root/autodl-tmp/test_skeletons',  # Output of skeleton_generator.py
        output_path='/root/autodl-tmp/test.pkl',
        num_keypoints=25  # Adjust according to the keypoint format you use
    )