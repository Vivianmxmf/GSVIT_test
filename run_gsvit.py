import random
import os
import cv2
import torch
import numpy as np
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import ToPILImage
from EfficientViT.classification.model.build import EfficientViT_M5
from dataloader_surgical import load_data

class EfficientViT(nn.Module):
    def __init__(self, in_size, predict_change=False):
        super(EfficientViT, self).__init__()
        self.predict_change = predict_change
        self.evit = EfficientViT_M5(pretrained='efficientvit_m5')
        # remove the classification head
        self.evit = torch.nn.Sequential(*list(self.evit.children())[:-1])

    def forward(self, x):
        out = self.evit(x)
        decoded = self.decoder.forward(out)
        return decoded

def process_inputs(images):
    # flip color channels
    tmp = images[:, 0, :, :].clone()
    images[:, 0, :, :] = images[:, 2, :, :]
    images[:, 2, :, :] = tmp
    return images

if __name__ == "__main__":
    np.random.seed(0)
    torch.manual_seed(0)

    batch_size = 16  # Set to anything
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    class GSViT(nn.Module):
        def __init__(self):
            super().__init__()
            gsvit = EfficientViT(in_size=batch_size)
            state_dict = torch.load("GSViT.pkl")
            # Remove unexpected keys from the state_dict
            state_dict = {k: v for k, v in state_dict.items() if "decoder" not in k}
            gsvit.load_state_dict(state_dict, strict=False)
            self.gsvit = gsvit.to(device)
            
        def forward(self, x):
            x = process_inputs(x)  # Flip color channels
            return self.gsvit(x)

    # Initialize model
    gsvit = GSViT().to(device)
    
    # Load data
    data_root = "Training_video"  # Ensure this path points to your video directory
    dataloader_train = load_data(root=data_root, batch_size=batch_size, is_train=True)

    # Example input
    example_input = torch.randn(batch_size, 3, 224, 224).to(device)  # Example input size
    output = gsvit(example_input)
    print(output.shape)
    
    # Training loop
    optimizer = torch.optim.Adam(gsvit.parameters(), lr=1e-4)
    criterion = nn.MSELoss()  # Assuming mean squared error loss for simplicity

    for epoch in range(10):  # Number of epochs
        gsvit.train()  # Set model to training mode
        for inputs, targets in dataloader_train:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = gsvit(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            print(f"Epoch [{epoch+1}/10], Loss: {loss.item()}")

    # Evaluation loop (if needed)
    # dataloader_val = load_data(root=data_root, batch_size=16, is_train=False)
    # gsvit.eval()  # Set model to evaluation mode
    # with torch.no_grad():
    #     for inputs, targets in dataloader_val:
    #         inputs, targets = inputs.to(device), targets.to(device)
    #         outputs = gsvit(inputs)
    #         loss = criterion(outputs, targets)
    #         print(f"Validation Loss: {loss.item()}")
