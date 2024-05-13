import sys
import os

sys.path.append(os.path.abspath(
    '/Users/miarodde/Documents/sintef/ebus-ai/video_inferences/'))  # Add the path to the root of the project to run the script from the terminal

import cv2
import numpy as np
import tensorflow as tf
import time
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

from resources.create_model import create_model
from resources.stations_config import get_stations_config
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.metrics import Precision, Recall
from sklearn.metrics import accuracy_score, precision_score, recall_score
from resources.confusion_matrix import confusion_matrix_and_report

# video_df_path = '/Users/miarodde/Documents/sintef/ebus-ai/video_inferences/data/test_sequence_full_video/Sequence_001/labels_local.csv'

video_df_paths = ['/Users/miarodde/Documents/sintef/ebus-ai/video_inferences/data/EBUS_Aalesund_full_videos_uncropped/Patient_20240419-102838/Sequence_001/labels_local.csv',
                  '/Users/miarodde/Documents/sintef/ebus-ai/video_inferences/data/EBUS_Aalesund_full_videos_uncropped/Patient_20240423-102422/Sequence_001/labels_local.csv']


classification_model_path = '/Users/miarodde/Documents/sintef/ebus-ai/video_inferences/models/sequence/12:07:18/best_model'

plt.style.use('dark_background')


# Function to preprocess images
def preprocess_class_frame(frame_path):
    frame = tf.keras.utils.load_img(frame_path, color_mode='rgb', target_size=None)
    frame = np.array(frame)
    frame = frame[105:635, 585:1335]
    frame = tf.cast(frame, tf.float32)
    frame = tf.image.resize(frame, [224, 224], method='nearest')
    frame = frame / 127.5 - 1
    return frame


# Function to get predictions for a sequence of images
def get_predictions(sequence, model):
    sequence = np.array(sequence)  # Convert list to np.array
    sequence = np.expand_dims(sequence, axis=0)  # Model expects batch dimension
    predictions = model.predict(sequence)
    return predictions[0]


def load_stateful_model(model_path):

    stateful_model = create_model(instance_size=(224, 224, 3), num_stations=8, stateful=True)
    stateful_model.compile(loss=tf.keras.losses.CategoricalCrossentropy(), optimizer='adam',
                  metrics=['accuracy', Precision(), Recall()])
    stateful_model.load_weights(model_path).expect_partial()
    stateful_model.reset_states()
    return stateful_model


def load_stateless_model(model_path):
    stateless_model = create_model(instance_size=(224, 224, 3), num_stations=8, stateful=False)
    stateless_model.compile(loss=tf.keras.losses.CategoricalCrossentropy(), optimizer='adam',
                           metrics=['accuracy', Precision(), Recall()])
    stateless_model.load_weights(model_path).expect_partial()
    return stateless_model
class PlotPred:

    def __init__(self):
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(10, 6))

    def update(self, pred_values, labels):
        self.ax.clear()
        self.pred_values = pred_values
        self.labels = labels
        self.colors = ['green' if val >= 0.5 else 'yellow' if val >= 0.3 else 'grey' for val in pred_values]

        self.ax.bar(self.labels, self.pred_values, color=self.colors)
        self.ax.set_ylim(0, 1)
        self.ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
        self.ax.set_title('Predicted stations', size=20)
        self.ax.tick_params(axis='x', labelsize=22)
        self.ax.tick_params(axis='y', labelsize=18)

        for ticklabel, color in zip(self.ax.get_xticklabels(), self.colors):
            ticklabel.set_color(color)  # Set color individually for each tick label

        # remove box around plot
        self.ax.spines['top'].set_visible(False)
        self.ax.spines['right'].set_visible(False)

        # set tight layout
        self.fig.tight_layout()
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()


# Main loop to stream images as video
def stream_video(video_df_paths, class_model_path, stations_config, sequence_length=10):
    stations = list(stations_config.keys())
    print(stations)

    # for statistics
    num_correct = 0
    num_total = 0
    predictions = []
    true_labels = []

    # seg_model = tf.keras.models.load_model(seg_model_path)
    class_model = load_stateless_model(class_model_path)

    # seg_plot = PlotSeg()
    plot = PlotPred()
    for video_df_path in video_df_paths:
        sequence = []
        df = pd.read_csv(video_df_path, sep=';')
        last_label = ''

        # Loop through the rows of the df
        for index, row in df.iterrows():
            #loop_start_time = time.time()  # Record the start time of the loop
            frame_path = row['path']
            label = row['label']
            print(f'True label: {label}')
            original_frame_gray = cv2.imread(frame_path, cv2.IMREAD_GRAYSCALE)
            original_frame_gray = original_frame_gray[110:725, 515:1400]

            class_frame = preprocess_class_frame(frame_path)  # Preprocess for the model
            sequence.append(class_frame)  # Add the frame to the sequence
            cv2.imshow('frame', original_frame_gray)

            if label != last_label: class_model.reset_states()  # Reset states if the label changes
            last_label = label

            if len(sequence) == sequence_length:
                # Get predictions for the current sequence
                pred = get_predictions(sequence, class_model)
                prediction = stations[np.argmax(pred)]
                # if prediction == '0':
                #    print(prediction)
                #    class_model.reset_states()
                sequence = []  # Clear the sequence
                #sequence.pop(0)  # remove first element in sequence

                # Plot the predictions
                plot.update(pred, stations)

                # for statistics
                if label != '0':  # and prediction != '0'
                    if prediction == label:
                        num_correct += 1
                    num_total += 1
                    predictions.append(prediction)
                    true_labels.append(label)

            #loop_end_time = time.time()  # Record the end time of the loop
            #loop_duration = (loop_end_time - loop_start_time) * 1000  # Convert to milliseconds
            #print(f"Processing Time: {loop_duration:.2f} ms")

            # Calculate remaining time to wait
            # wait_time = max(200 - loop_duration, 1)  # Ensure there's at least a minimal wait time
            # print(f"Wait Time: {wait_time:.2f} ms")
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    plt.ioff()
    stations = list(get_stations_config(3).keys())

    print(f'Accuracy: {num_correct / num_total:.2f}')
    print('accuracy:', accuracy_score(true_labels, predictions))
    print('precision:', precision_score(true_labels, predictions, average='weighted'))
    print('recall:', recall_score(true_labels, predictions, average='weighted'))
    print(classification_report(true_labels, predictions, digits=3, target_names=stations,
                                labels=stations))
    print(confusion_matrix(true_labels, predictions, labels=stations))

    # get number from true labels config
    true_labels = [stations_config[label] for label in true_labels]
    predictions = [stations_config[pred] for pred in predictions]
    confusion_matrix_and_report(true_labels, predictions, 8,
                                stations,
                                '/Users/miarodde/Documents/sintef/ebus-ai/video_inferences/inference/figures/Aalesund/',
                                '12:07:18_stateful_reset_weights_')


if __name__ == '__main__':
    stream_video(video_df_paths, classification_model_path, get_stations_config(3))



'''
Accuracy: 0.33
accuracy: 0.3275862068965517
precision: 0.411524500907441
recall: 0.3275862068965517


              precision    recall  f1-score   support

          4L      0.500     0.211     0.296        19
          4R      0.579     1.000     0.733        11
          7L      0.000     0.000     0.000         0
          7R      1.000     0.500     0.667         8
         10L      0.000     0.000     0.000         4
         10R      0.000     0.000     0.000         0
         11L      0.000     0.000     0.000        16
         11R      0.000     0.000     0.000         0

   micro avg      0.328     0.328     0.328        58
   macro avg      0.260     0.214     0.212        58
weighted avg      0.412     0.328     0.328        58

[[ 4  8  0  0  0  0  1  6]
 [ 0 11  0  0  0  0  0  0]
 [ 0  0  0  0  0  0  0  0]
 [ 0  0  0  4  0  2  0  2]
 [ 4  0  0  0  0  0  0  0]
 [ 0  0  0  0  0  0  0  0]
 [ 0  0  0  0  0  0  0 16]
 [ 0  0  0  0  0  0  0  0]]
 '''

'''
21:02:32 - Accuracy: 0.22
16:08:35 - Accuracy: 0.33
22:33:14 - Accuracy: 0.40
'''

