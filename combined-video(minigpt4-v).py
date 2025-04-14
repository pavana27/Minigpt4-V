#this program combines 45 video frames and 45 T-imeages into single video
#the annotation is done on these genrated videos
import cv2
import os
import glob
import numpy as np

def extract_video_frames(video_path, num_frames=45):
    """
    Extracts 45 evenly spaced frames from a video.

    Args:
        video_path (str): Path to the video file.
        num_frames (int): Number of frames to extract.

    Returns:
        list: A list of extracted frames.
    """
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames < num_frames:
        print(f"Skipping {video_path}: Not enough frames.")
        cap.release()
        return []

    frame_indices = [int(i * total_frames / num_frames) for i in range(num_frames)]
    frames = []

    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames.append(frame)

    cap.release()
    return frames

def create_toeplitz_frames(toeplitz_image_path, num_frames=45):
    """
    Duplicates a single Toeplitz frame to match the number of video frames.

    Args:
        toeplitz_image_path (str): Path to the Toeplitz reference image.
        num_frames (int): Number of duplicate Toeplitz frames to create.

    Returns:
        list: A list of replicated Toeplitz frames.
    """
    toeplitz_frame = cv2.imread(toeplitz_image_path)
    if toeplitz_frame is None:
        print(f"Error: Toeplitz image {toeplitz_image_path} not found.")
        return []

    toeplitz_frames = [toeplitz_frame] * num_frames
    return toeplitz_frames

def save_combined_video(video_frames, toeplitz_frames, output_video_path, fps=10):
    """
    Combines 45 video frames and 45 Toeplitz frames into a single video.

    Args:
        video_frames (list): List of video frames.
        toeplitz_frames (list): List of Toeplitz frames.
        output_video_path (str): Path to save the combined video.
        fps (int): Frames per second for the output video.
    """
    if not video_frames or not toeplitz_frames:
        print(f"Skipping {output_video_path}: No frames available.")
        return

    height, width, _ = video_frames[0].shape
    out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    for frame in video_frames + toeplitz_frames:
        out.write(frame)

    out.release()
    print(f"Saved: {output_video_path}")

def process_all_videos(dataset_folder, toeplitz_image_path, output_folder):
    """
    Processes all videos in a dataset folder, extracts 45 frames, 
    replicates 45 Toeplitz frames, and creates a new video.

    Args:
        dataset_folder (str): Path to the folder containing videos.
        toeplitz_image_path (str): Path to the Toeplitz reference image.
        output_folder (str): Directory to save the combined videos.
    """
    os.makedirs(output_folder, exist_ok=True)
    video_files = glob.glob(os.path.join(dataset_folder, "*.mp4"))  # Modify for other formats if needed

    for video_path in video_files:
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        output_video_path = os.path.join(output_folder, f"{video_name}_combined.mp4")

        print(f"Processing {video_path}...")

        video_frames = extract_video_frames(video_path, num_frames=45)
        toeplitz_frames = create_toeplitz_frames(toeplitz_image_path, num_frames=45)

        save_combined_video(video_frames, toeplitz_frames, output_video_path)

# Example usage
dataset_folder = "video_dataset"  # Folder containing all videos
toeplitz_image_path = "toeplitz_frame.jpg"  # Reference Toeplitz image
#these videos are used for annotation using the automated annotation script
output_folder = "output_videos"  # Output folder for combined videos used in annotation

process_all_videos(dataset_folder, toeplitz_image_path, output_folder)
