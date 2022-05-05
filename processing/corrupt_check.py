# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 19:46:05 2022

@author: mehmoodyar.baig
"""
#THis code is to check if the video is corrupted or not..
#If the video is corrupted delete the video.
import glob
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import face_recognition


#Check if the file is corrupted or not
def validate_video(vid_path,train_transforms):
      transform = train_transforms
      count = 20
      video_path = vid_path
      frames = []
      a = int(100/count)
      first_frame = np.random.randint(0,a)
      temp_video = video_path.split('/')[-1]
      for i,frame in enumerate(frame_extract(video_path)):
        frames.append(transform(frame))
        if(len(frames) == count):
          break
      frames = torch.stack(frames)
      frames = frames[:count]
      return frames
  
#extract a frame from from video
def frame_extract(path):
  vidObj = cv2.VideoCapture(path) 
  success = 1
  while success:
      success, image = vidObj.read()
      if success:
          yield image

im_size = 112
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

train_transforms = transforms.Compose([
                                        transforms.ToPILImage(),
                                        transforms.Resize((im_size,im_size)),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean,std)])
video_file  =  glob.glob('C:/Users/mehmoodyar.baig/Desktop/deepfake detection/dataset/cropped/Celeb_faces/*.mp4')
video_file += glob.glob('C:/Users/mehmoodyar.baig/Desktop/deepfake detection/dataset/cropped/DFDC_faces/*.mp4')
video_file += glob.glob('C:/Users/mehmoodyar.baig/Desktop/deepfake detection/dataset/cropped/FF_faces/*.mp4')

print("Total no of videos :" , len(video_file))
print(video_file)
count = 0;
for i in video_file:
  try:
    count+=1
    validate_video(i,train_transforms)
  except:
    print("Number of video processed: " , count ," Remaining : " , (len(video_file) - count))
    print("Corrupted video is : " , i)
    continue
print((len(video_file) - count))