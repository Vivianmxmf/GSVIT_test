import os
import cv2
import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Initialize frame counter and list to hold frames
# Loop to read frames from video
# Resize the frame to the target size
def read_video(n_frames=None, video_loc=None, target_size=(244, 244)):
    i = 0
    frames = []
    cap = cv2.VideoCapture(video_loc)
    if n_frames is None:
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(n_frames)
    while cap.isOpened() and i < n_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, target_size)  # Add resizing here
        frames.append(frame)
        i += 1
    cap.release()
    return np.array(frames)


class SurgicalDataset(Dataset):
    def __init__(self, root, is_train=True, n_frames_input=1, n_frames_output=1, transform=None, batch_size=128, predict_change=False, finetune=False, target_size=(14, 14)):
        super(SurgicalDataset, self).__init__()
        self.root = root
        self.videos = os.listdir(root)
        self.is_train = is_train
        self.n_frames_input = n_frames_input
        self.n_frames_output = n_frames_output
        self.n_frames_total = self.n_frames_input + self.n_frames_output ## Total number of frames per clip
        self.transform = transform
        self.predict_change = predict_change
        self.batch_size = batch_size
        self.target_size = target_size
        self.clips = self.generate_dataset()

    def generate_dataset(self):
        clips = []
        for video in self.videos:
            video_path = os.path.join(self.root, video)
            video_frames = read_video(video_loc=video_path, target_size=self.target_size)
            if len(video_frames) >= self.n_frames_total:
                for i in range(0, len(video_frames) - self.n_frames_total + 1, self.n_frames_total):
                    clip = video_frames[i:i + self.n_frames_total] # Extract a clip of frames
                    clips.append(clip)
        return clips

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, idx):
        clip = self.clips[idx]
        inp = clip[:self.n_frames_input] / 255.0
        out = clip[self.n_frames_input:self.n_frames_total] / 255.0
        inp = torch.tensor(inp, dtype=torch.float32).permute(0, 3, 1, 2)
        out = torch.tensor(out, dtype=torch.float32).permute(0, 3, 1, 2)
        return inp, out

def load_data(root, batch_size=16, n_frames_input=1, n_frames_output=1, num_workers=4, is_train=True, pin_memory=True, target_size=(14, 14)):
    dataset = SurgicalDataset(root=root, is_train=is_train, n_frames_input=n_frames_input, n_frames_output=n_frames_output, batch_size=batch_size, target_size=target_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=is_train, num_workers=num_workers, pin_memory=pin_memory)
    return dataloader

if __name__ == "__main__":
    # Update this path to point to your directory with video files
    data_root = "Training_video"
    dataloader_train = load_data(root=data_root, batch_size=16, is_train=True)

    for inputs, outputs in dataloader_train:
        print(inputs.shape, outputs.shape)
        break
