import os
import cv2
import re

test_ds_path = '/Users/miarodde/Documents/sintef/ebus-ai/video_inferences/data/test_sequence_with_zero'

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [atoi(c) for c in re.split(r'(\d+)', text)]
def view_test_station(test_station):
    for patient in os.listdir(test_ds_path):
        if patient != 'EBUS_Levanger_with_consent_Patient_20231129-091812':
            continue
        patient_path = os.path.join(test_ds_path, patient)
        for station in os.listdir(patient_path):
            station_path = os.path.join(patient_path, station)
            if station == test_station:
                print(station_path)
                frames = os.listdir(station_path)
                frames.sort(key=natural_keys)
                for frame in frames:
                    print(frame)
                    img = cv2.imread(os.path.join(station_path, frame))
                    cv2.putText(img, f'Patient: {patient}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                (255, 117, 24), 2, cv2.LINE_AA)
                    cv2.putText(img, f'Frame: {frame}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                (255, 117, 24), 2, cv2.LINE_AA)
                    cv2.imshow('image', img)
                    if cv2.waitKey(200) & 0xFF == ord('q'):
                        break

view_test_station('10L')






