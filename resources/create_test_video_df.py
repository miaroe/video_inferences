import os
import pandas as pd


test_path = '/Users/miarodde/Documents/sintef/ebus-ai/video_inferences/data/test_sequence'

def create_test_video_df():
    """Create a test video from the test data. The test video is used in the FAST pipeline."""
    test_video_df = []

    for patient in os.listdir(test_path):
        patient_path = os.path.join(test_path, patient)
        for label in os.listdir(patient_path):
            label_path = os.path.join(patient_path, label)
            for frame in sorted(os.listdir(label_path)):
                frame_path = os.path.join(label_path, frame)
                frame_nr = frame.split('_')[1].split('.')[0]
                test_video_df.append({'frame_nr': frame_nr, 'patient': patient, 'label': label, 'frame_path': frame_path})

    test_video_df = pd.DataFrame(test_video_df)
    test_video_df.to_csv(os.path.join(test_path, 'test_video_df.csv'), index=False)
    print('test_video_df', test_video_df)

#create_test_video_df()

def change_frame_paths():
    """Change the frame paths in the test_video_df to match the new file structure."""
    test_video_df = pd.read_csv('/Users/miarodde/Documents/sintef/ebus-ai/video_inferences/data/EBUS_Aalesund_full_videos_uncropped/Patient_20240423-102422/Sequence_001/labels.csv', sep=';')
    # change the path in front of the frame nr to local path
    test_video_df['path'] = test_video_df['path'].apply(lambda x: '/Users/miarodde/Documents/sintef/ebus-ai/video_inferences/data/EBUS_Aalesund_full_videos_uncropped/Patient_20240423-102422/Sequence_001/' + x.split('\\')[-1])
    test_video_df.to_csv('/Users/miarodde/Documents/sintef/ebus-ai/video_inferences/data/EBUS_Aalesund_full_videos_uncropped/Patient_20240423-102422/Sequence_001/labels_local.csv', index=False, sep=';')

change_frame_paths()


