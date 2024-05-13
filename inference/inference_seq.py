import sys
import os
sys.path.append(os.path.abspath('/Users/miarodde/Documents/sintef/ebus-ai/video_inferences/')) # Add the path to the root of the project to run the script from the terminal

import cv2
import numpy as np
import tensorflow as tf
import time
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

from tensorflow.keras.preprocessing.image import img_to_array
from resources.create_model import create_model
from resources.stations_config import get_stations_config
from sklearn.metrics import confusion_matrix, classification_report

video_df_path = '/Users/miarodde/Documents/sintef/ebus-ai/video_inferences/data/test_sequence/test_video_df.csv'
model_path = '/Users/miarodde/Documents/sintef/ebus-ai/video_inferences/models/sequence/22:33:14/best_model'


# Function to preprocess images
def preprocess_frame(image_path):
    frame = tf.keras.utils.load_img(image_path, color_mode='rgb', target_size=(224, 224))
    frame = img_to_array(frame)
    frame = tf.cast(frame, tf.float32)
    return frame

# Function to get predictions for a sequence of images
def get_predictions(sequence, model):
    sequence = np.array(sequence)  # Convert list to np.array
    sequence = np.expand_dims(sequence, axis=0)  # Model expects batch dimension
    predictions = model.predict(sequence)
    return predictions[0]

def load_stateful_model(model_path):
    model = tf.keras.models.load_model(model_path)
    stateful_model = create_model(instance_size=(224, 224, 3), num_stations=8, stateful=True)
    stateful_model.set_weights(model.get_weights())
    model.reset_states()
    return stateful_model

def load_model(model_path):
    model = tf.keras.models.load_model(model_path)
    return model


class PlotPred:

    def __init__(self, pred_values, labels):
        plt.ion()
        self.pred_values = pred_values
        self.labels = labels
        self.colors = ['green' if val >= 0.5 else 'yellow' if val >= 0.3 else 'grey' for val in pred_values]
        plt.style.use('dark_background')
        self.fig, self.ax = plt.subplots(figsize=(8,4))

    def update(self, pred_values, labels):
        self.pred_values = pred_values
        self.labels = labels
        self.colors = ['green' if val >= 0.5 else 'yellow' if val >= 0.3 else 'grey' for val in pred_values]
        plt.cla()
        plt.bar(self.labels, self.pred_values, color=self.colors)
        plt.subplots_adjust(bottom=0.2)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        plt.ylim(0, 1)
        plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
        plt.title('Predicted stations', size=20)

        for ticklabel, color in zip(self.ax.get_xticklabels(), self.colors):
            ticklabel.set_color(color)  # Set color individually for each tick label

        # remove box around plot
        self.ax.spines['top'].set_visible(False)
        self.ax.spines['right'].set_visible(False)

        # set tight layout
        plt.tight_layout()

        plt.draw()
        #plt.pause(0.001)



# Main loop to stream images as video
def stream_video(video_df_path, model_path, stations_config, sequence_length=10):
    sequence = []
    last_label = ''
    prediction = ''
    stations = list(stations_config.keys())
    print(stations)

    # for statistics
    num_correct = 0
    num_total = 0
    predictions = []
    true_labels = []

    model = load_stateful_model(model_path)
    #model = load_model(model_path)

    df = pd.read_csv(video_df_path, sep=',')

    # initialize plot class for live plotting
    plot = PlotPred([0.1]*8, stations)

    # Loop through the rows of the df
    for index, row in df.iterrows():
        loop_start_time = time.time()  # Record the start time of the loop
        frame_path = row['frame_path']
        label = row['label']
        patient = row['patient']
        original_frame = cv2.imread(frame_path)  # Read the original frame for display
        frame = preprocess_frame(frame_path)  # Preprocess for the model

        if label != last_label:
            #model.reset_states()  # Reset states if the label changes
            sequence = []  # Clear the sequence if the label changes
        last_label = label

        sequence.append(frame)  # Add the frame to the sequence

        if len(sequence) == sequence_length:
            # Get predictions for the current sequence
            pred = get_predictions(sequence, model)
            prediction = stations[np.argmax(pred)]
            sequence = []
            # remove first element in sequence
            #sequence.pop(0)

            # Plot the predictions
            plot.update(pred, stations)

            # for statistics
            if prediction == label:
                num_correct += 1
            num_total += 1
            predictions.append(prediction)
            true_labels.append(label)
            '''
            if prediction == '4R' and label == '10R':
                print(f'Prediction: {prediction}, True label: {label}, Patient: {patient}')
                print('frame_path:', frame_path)
                print('-'*50)

            if prediction == '10R' and label == '4R':
                print(f'Prediction: {prediction}, True label: {label}, Patient: {patient}')
                print('frame_path:', frame_path)
                print('-'*50)
            '''
        cv2.putText(original_frame, f'Prediction: {prediction}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(original_frame, f'True label: {label}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        #cv2.putText(original_frame, f'Patient: {patient}', (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Inference', original_frame)

        loop_end_time = time.time()  # Record the end time of the loop
        loop_duration = (loop_end_time - loop_start_time) * 1000  # Convert to milliseconds
        print(f"Processing Time: {loop_duration:.2f} ms")

        # Calculate remaining time to wait
        wait_time = max(200 - loop_duration, 1)  # Ensure there's at least a minimal wait time
        print(f"Wait Time: {wait_time:.2f} ms")
        if cv2.waitKey(int(wait_time)) & 0xFF == ord('q'):  # Adjust wait time based on processing duration
            break

    plt.ioff()
    cv2.destroyAllWindows()

    print(f'Accuracy: {num_correct/num_total:.2f}')
    print(classification_report(true_labels, predictions, digits=3, target_names=list(stations_config.keys()), labels=stations))
    print(confusion_matrix(true_labels, predictions, labels=stations))


'''
    stateful model, with model.reset_states()
    Accuracy: 0.66
              precision    recall  f1-score   support
    
          4L       1.00      0.05      0.09        22
          4R       0.78      0.69      0.73        45
          7L       0.62      0.45      0.52        29
          7R       0.73      0.73      0.73        45
         10L       0.67      0.98      0.80        44
         10R       0.78      0.75      0.77        68
         11L       0.43      0.53      0.48        36
         11R       0.52      0.64      0.57        39
    
    accuracy                           0.66       328
    macro avg       0.69      0.60      0.59       328
    weighted avg       0.69      0.66      0.64       328
    
    [[43  1  0  0  0  0  0  0]
    [ 0 51  1  8  0  3  1  4]
    [ 4  0 19 13  0  0  0  0]
    [ 0  3 11 25  0  0  0  0]
    [12  2  0  0  1  5  0  2]
    [ 3  4  3  0  0 31  1  3]
    [ 1  0  9  2  0  1 13  3]
    [ 1  4  1  0  0  0  6 33]] 43+51+19+25+1+31+16+39=225/328=0.6859756097560976
    
    Prediction: 10R, True label: 4R, Patient: EBUS_Levanger_with_consent_Patient_20231107-093258 02:13
    Prediction: 10R, True label: 4R, Patient: EBUS_Levanger_Patient_022  05:59
    Prediction: 4R, True label: 10R, Patient: EBUS_Levanger_Patient_024 07:20 - 07:48
    Prediction: 4R, True label: 10R, Patient: EBUS_Levanger_Patient_024
    Prediction: 10R, True label: 4R, Patient: EBUS_Levanger_Patient_024 08:38
    Prediction: 4R, True label: 10R, Patient: EBUS_Levanger_with_consent_Patient_20231129-091812 10:26 - 10:39
    Prediction: 4R, True label: 10R, Patient: EBUS_Levanger_with_consent_Patient_20231129-091812
    
    
    stateful model, without model.reset_states():
    Accuracy: 0.59
    
              precision    recall  f1-score   support
        
        4L      0.677     0.955     0.792        44
        4R      0.754     0.676     0.713        68
        7L      0.400     0.500     0.444        36
        7R      0.469     0.590     0.523        39
        10L      0.000     0.000     0.000        22
        10R      0.651     0.622     0.636        45
        11L      0.462     0.414     0.436        29
        11R      0.619     0.578     0.598        45
        
        accuracy                          0.595       328
        macro avg      0.504     0.542     0.518       328
        weighted avg      0.562     0.595     0.573       328
        
        [[42  1  0  0  0  1  0  0]
        [ 0 46  2  9  0  4  2  5]
        [ 6  0 18 12  0  0  0  0]
        [ 0  3 10 23  0  3  0  0]
        [10  3  0  0  0  7  0  2]
        [ 3  4  3  0  0 28  1  6]
        [ 0  0  9  5  0  0 12  3]
        [ 1  4  3  0  0  0 11 26]]
        
    stateless model:
    Accuracy: 0.57
          precision    recall  f1-score   support
    
      4L      0.621     0.932     0.745        44
      4R      0.732     0.603     0.661        68
      7L      0.429     0.500     0.462        36
      7R      0.431     0.564     0.489        39
     10L      0.500     0.045     0.083        22
     10R      0.700     0.622     0.659        45
     11L      0.424     0.483     0.452        29
     11R      0.605     0.511     0.554        45
    
    accuracy                          0.573       328
    macro avg      0.555     0.533     0.513       328
    weighted avg      0.584     0.573     0.558       328
    
    [[41  1  0  1  0  0  1  0]
    [ 2 41  3  9  0  3  3  7]
    [ 3  1 18 14  0  0  0  0]
    [ 2  3 11 22  1  0  0  0]
    [12  2  0  0  1  5  1  1]
    [ 3  5  2  1  0 28  1  5]
    [ 1  0  7  4  0  1 14  2]
    [ 2  3  1  0  0  3 13 23]]
    
    stateful model, with model.reset_states() and predictions on every new frame:
    Accuracy: 0.60
              precision    recall  f1-score   support
    
          4L      0.656     0.945     0.774       434
          4R      0.763     0.592     0.667       689
          7L      0.452     0.513     0.480       355
          7R      0.446     0.601     0.512       388
         10L      0.478     0.050     0.091       220
         10R      0.630     0.663     0.646       442
         11L      0.522     0.381     0.440       281
         11R      0.626     0.667     0.646       466
    
    accuracy                          0.597      3275
    macro avg      0.572     0.551     0.532      3275
    weighted avg      0.600     0.597     0.579      3275
    
    [[410   7   0   5   1   0   7   4]
    [ 18 408  20  95   4  58  14  72]
    [ 26  10 182 137   0   0   0   0]
    [  9  21  98 233   7  13   6   1]
    [112  11   0   4  11  67   4  11]
    [ 21  44  24  10   0 293   8  42]
    [ 11   0  68  32   0   7 107  56]
    [ 18  34  11   6   0  27  59 311]]
    
'''



if __name__ == '__main__':
    stream_video(video_df_path, model_path, get_stations_config(3))

