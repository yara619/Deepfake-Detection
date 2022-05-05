# -*- coding: utf-8 -*-
"""
Created on Sun Apr  3 17:53:54 2022

@author: mehmoodyar.baig
"""
#To get the average frame count
import json
import glob
import numpy as np
import cv2
import copy
#change the path accordingly
video_files = glob.glob('C:/Users/mehmoodyar.baig/Desktop/deepfake detection/dataset/videos/data_fake/DF/*.mp4')
video_files +=  glob.glob('C:/Users/mehmoodyar.baig/Desktop/deepfake detection/dataset/videos/data_fake/F2F/*.mp4')
video_files +=  glob.glob('C:/Users/mehmoodyar.baig/Desktop/deepfake detection/dataset/videos/data_fake/FS/*.mp4')
video_files +=  glob.glob('C:/Users/mehmoodyar.baig/Desktop/deepfake detection/dataset/videos/data_fake/NT/*.mp4')
frame_count = []
for video_file in video_files:
  cap = cv2.VideoCapture(video_file)
  if(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))<150):
    video_files.remove(video_file)
    continue
  frame_count.append(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
print("frames" , frame_count)
print("Total number of videos: " , len(frame_count))
print('Average frame per video:',np.mean(frame_count))

# to extract frame
def frame_extract(path):
  vidObj = cv2.VideoCapture(path)
  success = 1
  while success:
      success, image = vidObj.read()
      if success:
          yield image


import os
import cv2
import face_recognition
import tqdm


# process the frames
def create_face_videos(path_list,out_dir):
  already_present_count =  glob.glob(out_dir+'*.mp4')
  print("No of videos already present " , len(already_present_count))
  for path in tqdm.tqdm(path_list):
    out_path = os.path.join(out_dir,path.split('/')[-1])
    file_exists = glob.glob(out_path)
    print(out_path)
    if (len(file_exists) != 0):
        print("File Already exists: " , out_path)
        continue
    frames = []
    out = cv2.VideoWriter(out_path,cv2.VideoWriter_fourcc('M','J','P','G'), 30, (112,112))
    for idx,frame in enumerate(frame_extract(path)):
      print(idx)  
      if(idx <= 150):
        frames.append(frame)
        if(len(frames) == 4):     
            faces = face_recognition.batch_face_locations(frames)
            for i,face in enumerate(faces):
                if(len(face) != 0):
                    top,right,bottom,left = face[0]
                try:
                    out.write(cv2.resize(frames[i][top:bottom,left:right,:],(112,112)))
                except:
                    pass
            frames = []
    try:
      del top,right,bottom,left
    except:
      pass
    out.release()

