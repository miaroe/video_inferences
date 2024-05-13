import pandas as pd
import os

def get_frame_label_dict(full_video_path):
    frame_label_dict = {}
    labels_path = os.path.join(full_video_path, 'labels.csv')
    # read the csv file using pandas
    df = pd.read_csv(labels_path, sep=';')

    # create a dictionary to store the mappings
    for index, row in df.iterrows():
        frame = (row['path']).split('\\')[-1]
        frame_index = frame.replace('.png', '').split('_')[-1]
        frame_label_dict[frame_index] = row['label']

    return frame_label_dict

def get_frame_label_dict_modified(full_video_path):
    frame_label_dict = {}
    labels_path = os.path.join(full_video_path, 'labels.csv')
    # read the csv file using pandas
    df = pd.read_csv(labels_path, sep=',')

    # create a dictionary to store the mappings
    for index, row in df.iterrows():
        frame_index = (row['frame_number'])
        frame_label_dict[frame_index] = row['label']

    return frame_label_dict

