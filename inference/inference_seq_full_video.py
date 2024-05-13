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
from matplotlib.colors import LinearSegmentedColormap
from skimage.transform import resize
from matplotlib.patches import Patch
from tensorflow.keras.metrics import Precision, Recall
from sklearn.metrics import accuracy_score, precision_score, recall_score
from resources.confusion_matrix import confusion_matrix_and_report

#video_df_path = '/Users/miarodde/Documents/sintef/ebus-ai/video_inferences/data/test_sequence_full_video/Sequence_001/labels_local.csv'

video_df_paths = ['/Users/miarodde/Documents/sintef/ebus-ai/video_inferences/data/test_sequence_full_videos/Patient_001/Sequence_001/labels_local.csv',
                  '/Users/miarodde/Documents/sintef/ebus-ai/video_inferences/data/test_sequence_full_videos/Patient_022/Sequence_001/labels_local.csv',
                  '/Users/miarodde/Documents/sintef/ebus-ai/video_inferences/data/test_sequence_full_videos/Patient_024/Sequence_001/labels_local.csv',
                  '/Users/miarodde/Documents/sintef/ebus-ai/video_inferences/data/test_sequence_full_videos/Patient_20231107-093258/Sequence_001/labels_local.csv',
                  '/Users/miarodde/Documents/sintef/ebus-ai/video_inferences/data/test_sequence_full_videos/Patient_20231129-091812/Sequence_001/labels_local.csv']

#classification_model_path = '/Users/miarodde/Documents/sintef/ebus-ai/video_inferences/models/sequence/with_0/best_model'
classification_model_path = '/Users/miarodde/Documents/sintef/ebus-ai/video_inferences/models/sequence/12:07:18/best_model'
segmentation_model_path = '/Users/miarodde/Documents/sintef/ebus-ai/video_inferences/models/segmentation/best_model'

plt.style.use('dark_background')

# Function to preprocess images
def preprocess_class_frame(frame_path):
    frame = tf.keras.utils.load_img(frame_path, color_mode='rgb', target_size=None)
    frame = np.array(frame)
    frame = frame[100:1035, 530:1658]
    frame = tf.cast(frame, tf.float32)
    frame = tf.image.resize(frame, [224, 224], method='nearest')
    frame = frame / 127.5 - 1
    return frame

def preprocess_seg_frame(frame):
    frame = (frame[..., None] / 255.0).astype(np.float32)
    frame = resize(frame, output_shape=(256, 256), preserve_range=True, anti_aliasing=False)
    frame = np.expand_dims(frame, axis=0)
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
    #model = tf.keras.models.load_model(model_path)
    #stateful_model = create_model(instance_size=(224, 224, 3), num_stations=8, stateful=True)
    #stateful_model.set_weights(model.get_weights())
    #stateful_model.reset_states()
    return stateful_model

def load_stateless_model(model_path):
    stateless_model = create_model(instance_size=(224, 224, 3), num_stations=8, stateful=False)
    stateless_model.compile(loss=tf.keras.losses.CategoricalCrossentropy(), optimizer='adam',
                           metrics=['accuracy', Precision(), Recall()])
    stateless_model.load_weights(model_path).expect_partial()
    return stateless_model

class PlotSeg:

    def __init__(self):
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(9, 6))


    def update(self, frame, mask):
        self.ax.clear()

        c_invalid = (0, 0, 0)
        colors = [(0.2, 0.2, 0.2),  # dark gray = background
                  (0.55, 0.4, 0.85),  # purple = lymph nodes
                  (0.76, 0.1, 0.05)]  # red   = blood vessels

        label_cmap = LinearSegmentedColormap.from_list('label_map', colors, N=3)
        label_cmap.set_bad(color=c_invalid, alpha=0)  # set invalid (nan) colors to be transparent
        image_cmap = plt.cm.get_cmap('gray')
        image_cmap.set_bad(color=c_invalid)

        # Resize mask to cropped frame size
        mask_resized = resize(mask, (935, 1128), preserve_range=True, order=0).astype(mask.dtype)

        #full_size_mask = np.zeros((1080, 1920), dtype=mask.dtype)
        #full_size_mask[100:1035, 530:1658] = mask_resized

        #frame_mask = self.get_img_ultrasound_sector_mask()
        #frame = np.ma.array(frame, mask=~frame_mask) # mask out everything outside the ultrasound sector

        mask_resized = np.ma.masked_less_equal(mask_resized, 0) # mask out background

        self.ax.set_title('Segmentation masks', size=20)
        self.ax.imshow(frame, cmap=image_cmap)
        self.ax.imshow(mask_resized, cmap=label_cmap, interpolation='nearest', alpha=0.4, vmin=0, vmax=2)
        # self.ax.text(0.05, 0.95, f'True label: {self.true_label}', transform=self.ax.transAxes, fontsize=12,
        #             verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5))

        # Define legend patches
        legend_patches = [
            Patch(color=colors[1], label='Lymph node'),
            Patch(color=colors[2], label='Blood vessel'),
        ]

        # Add the legend to the plot
        self.ax.legend(handles=legend_patches, loc='upper left', fontsize=12, frameon=True, edgecolor='black')

        self.ax.axis('off')
        self.fig.tight_layout()
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()


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
def stream_video(video_df_paths, class_model_path, seg_model_path, stations_config, sequence_length=10):
    stations = list(stations_config.keys())


    # for statistics
    num_correct = 0
    num_total = 0
    predictions = []
    true_labels = []

    #seg_model = tf.keras.models.load_model(seg_model_path)
    class_model = load_stateful_model(class_model_path)

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
            original_frame_gray = original_frame_gray[100:1035, 530:1658]

            class_frame = preprocess_class_frame(frame_path)  # Preprocess for the model
            sequence.append(class_frame)  # Add the frame to the sequence
            cv2.imshow('frame', original_frame_gray)


            #seg_frame = preprocess_seg_frame(original_frame_gray)  # Preprocess for the model
            #seg_pred = seg_model.predict(seg_frame)[0]
            #seg_pred = np.argmax(seg_pred, axis=-1)
            #seg_plot.update(original_frame_gray, seg_pred)

            #if label != last_label: class_model.reset_states()  # Reset states if the label changes
            #last_label = label

            if len(sequence) == sequence_length:
                # Get predictions for the current sequence
                pred = get_predictions(sequence, class_model)
                prediction = stations[np.argmax(pred)]
                #if prediction == '0':
                #    print(prediction)
                #    class_model.reset_states()
                #sequence = []  # Clear the sequence
                sequence.pop(0)  # remove first element in sequence

                # Plot the predictions
                plot.update(pred, stations)

                # for statistics
                if label != '0': #and prediction != '0'
                    if prediction == label:
                        num_correct += 1
                    num_total += 1
                    predictions.append(prediction)
                    true_labels.append(label)


            #loop_end_time = time.time()  # Record the end time of the loop
            #loop_duration = (loop_end_time - loop_start_time) * 1000  # Convert to milliseconds
            #print(f"Processing Time: {loop_duration:.2f} ms")

            # Calculate remaining time to wait
            #wait_time = max(200 - loop_duration, 1)  # Ensure there's at least a minimal wait time
            #print(f"Wait Time: {wait_time:.2f} ms")
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    plt.ioff()
    stations = list(stations_config.keys())

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
                                '/Users/miarodde/Documents/sintef/ebus-ai/video_inferences/inference/figures/', '12:07:18_stateful_pop(0)_')


# This code streams the full videos from the test set and predicts a station for every seq_length frames (inference setup)
if __name__ == '__main__':
    stream_video(video_df_paths, classification_model_path, segmentation_model_path, get_stations_config(3))


'''
---------------------------------------------------- NEW ----------------------------------------------------
/Users/miarodde/Documents/sintef/ebus-ai/video_inferences/models/sequence/12:07:18 and reset_weights at change of label
Accuracy: 0.68
accuracy: 0.6846846846846847
precision: 0.6319432690947179
recall: 0.6846846846846847

              precision    recall  f1-score   support

          4L      0.712     0.955     0.816        44
          4R      0.728     0.971     0.832        69
          7L      0.545     0.649     0.593        37
          7R      0.581     0.462     0.514        39
         10L      0.000     0.000     0.000        24
         10R      0.700     0.795     0.745        44
         11L      0.667     0.357     0.465        28
         11R      0.762     0.667     0.711        48

    accuracy                          0.685       333
   macro avg      0.587     0.607     0.584       333
weighted avg      0.632     0.685     0.646       333

[[42  2  0  0  0  0  0  0]
 [ 2 67  0  0  0  0  0  0]
 [ 0  0 24 13  0  0  0  0]
 [ 0  3 17 18  0  1  0  0]
 [11  1  0  0  0 11  0  1]
 [ 0  9  0  0  0 35  0  0]
 [ 4  0  3  0  0  2 10  9]
 [ 0 10  0  0  0  1  5 32]]

/Users/miarodde/Documents/sintef/ebus-ai/video_inferences/models/sequence/12:07:18 stateful
Accuracy: 0.71
accuracy: 0.7117117117117117
precision: 0.6736555929946383
recall: 0.7117117117117117
              precision    recall  f1-score   support

          4L      0.754     0.977     0.851        44
          4R      0.791     0.986     0.877        69
          7L      0.500     0.892     0.641        37
          7R      0.667     0.205     0.314        39
         10L      0.000     0.000     0.000        24
         10R      0.691     0.864     0.768        44
         11L      0.733     0.393     0.512        28
         11R      0.857     0.750     0.800        48

    accuracy                          0.712       333
   macro avg      0.624     0.633     0.595       333
weighted avg      0.674     0.712     0.662       333

[[43  1  0  0  0  0  0  0]
 [ 0 68  0  0  0  1  0  0]
 [ 0  0 33  4  0  0  0  0]
 [ 0  0 29  8  0  0  0  2]
 [11  1  0  0  0 11  0  1]
 [ 0  6  0  0  0 38  0  0]
 [ 3  4  4  0  0  3 11  3]
 [ 0  6  0  0  0  2  4 36]]
 
  using pop(0) on /Users/miarodde/Documents/sintef/ebus-ai/video_inferences/models/sequence/12:07:18
 
 Accuracy: 0.62
accuracy: 0.6204511278195489
 precision: 0.5943887766191167
recall: 0.6204511278195489
               precision    recall  f1-score   support

          4L      0.723     0.934     0.815       439
          4R      0.602     0.905     0.723       696
          7L      0.483     0.811     0.606       360
          7R      0.672     0.198     0.306       393
         10L      0.000     0.000     0.000       233
         10R      0.606     0.626     0.616       447
         11L      0.681     0.329     0.443       286
         11R      0.714     0.592     0.647       471

    accuracy                          0.620      3325
   macro avg      0.560     0.550     0.520      3325
weighted avg      0.594     0.620     0.573      3325

[[410  23   3   3   0   0   0   0]
 [  7 630   8  10   0  28   1  12]
 [  0  45 292  14   0   0   0   9]
 [  0  66 240  78   0   7   0   2]
 [107  26   0   0   0  98   0   2]
 [  0 110   7   5   0 280   0  45]
 [ 16  73  38   4   0  19  94  42]
 [ 27  74  16   2   0  30  43 279]]

/Users/miarodde/Documents/sintef/ebus-ai/video_inferences/models/sequence/16:08:35/best_model
Accuracy: 0.63
accuracy: 0.6306306306306306
precision: 0.5941941086464413
recall: 0.6306306306306306
              precision    recall  f1-score   support

          4L      0.784     0.909     0.842        44
          4R      0.667     0.899     0.765        69
          7L      0.542     0.351     0.426        37
          7R      0.545     0.769     0.638        39
         10L      0.000     0.000     0.000        24
         10R      0.727     0.364     0.485        44
         11L      0.538     0.500     0.519        28
         11R      0.603     0.729     0.660        48

    accuracy                          0.631       333
   macro avg      0.551     0.565     0.542       333
weighted avg      0.594     0.631     0.595       333

[[40  0  0  0  4  0  0  0]
 [ 0 62  0  0  0  0  0  7]
 [ 0  0 13 22  0  0  2  0]
 [ 0  0  9 30  0  0  0  0]
 [11  2  0  0  0  4  0  7]
 [ 0 28  0  0  0 16  0  0]
 [ 0  0  2  3  0  0 14  9]
 [ 0  1  0  0  0  2 10 35]]
 
 

 (22:33:14)
[[40  0  4  0  0  0  0  0]
 [ 0 54  1  5  0  1  0  8]
 [ 0  0 23 14  0  0  0  0]
 [ 0  1 10 27  0  0  0  1]
 [ 8  0  0  4  0  8  1  3]
 [ 0  5  0  0  0 34  0  5]
 [ 0  0  0  1  0  8 12  7]
 [ 0  0  0  0  0  0  4 44]] #258  0,7747747748
 
 
 /Users/miarodde/Documents/sintef/ebus-ai/video_inferences/models/sequence/16:08:35/best_model and reset_weights at change of label
Accuracy: 0.64
accuracy: 0.6426426426426426
precision: 0.6253984142873031
recall: 0.6426426426426426
              precision    recall  f1-score   support

          4L      0.722     0.886     0.796        44
          4R      0.750     0.870     0.805        69
          7L      0.519     0.378     0.438        37
          7R      0.583     0.718     0.644        39
         10L      0.375     0.125     0.188        24
         10R      0.771     0.614     0.684        44
         11L      0.407     0.393     0.400        28
         11R      0.593     0.667     0.627        48

    accuracy                          0.643       333
   macro avg      0.590     0.581     0.573       333
weighted avg      0.625     0.643     0.624       333

[[39  0  0  0  5  0  0  0]
 [ 1 60  0  0  0  2  0  6]
 [ 0  0 14 20  0  0  3  0]
 [ 0  1  9 28  0  1  0  0]
 [11  1  0  0  3  2  1  6]
 [ 0 17  0  0  0 27  0  0]
 [ 3  0  4  0  0  0 11 10]
 [ 0  1  0  0  0  3 12 32]]
 
 
 
 using pop(0) on /Users/miarodde/Documents/sintef/ebus-ai/video_inferences/models/sequence/16:08:35/best_model
Accuracy: 0.61
accuracy: 0.6117293233082707
precision: 0.6184352124365984
recall: 0.6117293233082707
              precision    recall  f1-score   support

          4L      0.730     0.918     0.813       439
          4R      0.696     0.759     0.726       696
          7L      0.456     0.361     0.403       360
          7R      0.519     0.654     0.579       393
         10L      0.432     0.163     0.237       233
         10R      0.923     0.539     0.681       447
         11L      0.398     0.423     0.410       286
         11R      0.544     0.671     0.601       471

    accuracy                          0.612      3325
   macro avg      0.587     0.561     0.556      3325
weighted avg      0.618     0.612     0.600      3325

[[403   0   0   0  36   0   0   0]
 [ 12 528   0  42   3   6  23  82]
 [  0   0 130 184   0   0  46   0]
 [ 12   9 115 257   0   0   0   0]
 [ 91  30   0   0  38   8  10  56]
 [  9 165   0   0   0 241   2  30]
 [ 13   3  40  12   0   0 121  97]
 [ 12  24   0   0  11   6 102 316]]
 
 
 
with 0 and reset_weights only when loading model: /Users/miarodde/Documents/sintef/ebus-ai/video_inferences/models/sequence/with_0/best_model
Accuracy: 0.54
accuracy: 0.5405405405405406
precision: 0.6197431583494919
recall: 0.5405405405405406
              precision    recall  f1-score   support

           0      0.000     0.000     0.000         0
          4L      0.935     0.977     0.956        44
          4R      1.000     0.377     0.547        69
          7L      0.333     0.027     0.050        37
          7R      0.553     0.538     0.545        39
         10L      0.000     0.000     0.000        24
         10R      0.462     0.977     0.628        44
         11L      0.462     0.214     0.293        28
         11R      0.606     0.833     0.702        48

    accuracy                          0.541       333
   macro avg      0.483     0.438     0.413       333
weighted avg      0.620     0.541     0.518       333

[[ 0  0  0  0  0  0  0  0  0]
 [ 1 43  0  0  0  0  0  0  0]
 [ 0  0 26  0  0  0 39  0  4]
 [ 8  0  0  1 17  0  0  5  6]
 [15  0  0  2 21  0  0  0  1]
 [14  3  0  0  0  0  3  1  3]
 [ 1  0  0  0  0  0 43  0  0]
 [ 9  0  0  0  0  0  1  6 12]
 [ 0  0  0  0  0  0  7  1 40]]
 
 with 0 and reset_weights at predicted '0': /Users/miarodde/Documents/sintef/ebus-ai/video_inferences/models/sequence/with_0/best_model
Accuracy: 0.53
accuracy: 0.5315315315315315
precision: 0.5696267696267695
recall: 0.5315315315315315
              precision    recall  f1-score   support

           0      0.000     0.000     0.000         0
          4L      0.917     1.000     0.957        44
          4R      0.886     0.449     0.596        69
          7L      0.400     0.595     0.478        37
          7R      0.500     0.282     0.361        39
         10L      0.000     0.000     0.000        24
         10R      0.419     1.000     0.591        44
         11L      0.250     0.214     0.231        28
         11R      0.594     0.396     0.475        48

    accuracy                          0.532       333
   macro avg      0.441     0.437     0.410       333
weighted avg      0.570     0.532     0.511       333

[[ 0  0  0  0  0  0  0  0  0]
 [ 0 44  0  0  0  0  0  0  0]
 [ 0  0 31  0  0  0 38  0  0]
 [ 0  0  4 22 11  0  0  0  0]
 [ 0  0  0 28 11  0  0  0  0]
 [ 7  4  0  0  0  0  7  0  6]
 [ 0  0  0  0  0  0 44  0  0]
 [ 5  0  0  5  0  0  5  6  7]
 [ 0  0  0  0  0  0 11 18 19]]
 
 
/Users/miarodde/Documents/sintef/ebus-ai/video_inferences/models/sequence/09:22:05
Accuracy: 0.63
accuracy: 0.6306306306306306
precision: 0.6404031529714589
recall: 0.6306306306306306
              precision    recall  f1-score   support

          4L      0.597     0.977     0.741        44
          4R      0.705     0.797     0.748        69
          7L      0.508     0.838     0.633        37
          7R      0.818     0.231     0.360        39
         10L      0.000     0.000     0.000        24
         10R      0.667     0.682     0.674        44
         11L      1.000     0.036     0.069        28
         11R      0.631     0.854     0.726        48

    accuracy                          0.631       333
   macro avg      0.616     0.552     0.494       333
weighted avg      0.640     0.631     0.565       333

[[43  1  0  0  0  0  0  0]
 [13 55  0  0  0  0  0  1]
 [ 3  1 31  2  0  0  0  0]
 [ 3  1 25  9  0  0  0  1]
 [10  2  1  0  0  9  0  2]
 [ 0 11  0  0  0 30  0  3]
 [ 0  0  4  0  0  6  1 17]
 
 
/Users/miarodde/Documents/sintef/ebus-ai/video_inferences/models/sequence/21:02:32
Accuracy: 0.55
              precision    recall  f1-score   support

          4L      0.800     1.000     0.889        44
          4R      0.600     1.000     0.750        69
          7L      0.402     1.000     0.574        37
          7R      0.000     0.000     0.000        39
         10L      0.000     0.000     0.000        24
         10R      0.417     0.114     0.179        44
         11L      0.407     0.393     0.400        28
         11R      0.600     0.375     0.462        48

    accuracy                          0.553       333
   macro avg      0.403     0.485     0.407       333
weighted avg      0.451     0.553     0.460       333

[[44  0  0  0  0  0  0  0]
 [ 0 69  0  0  0  0  0  0]
 [ 0  0 37  0  0  0  0  0]
 [ 0  0 39  0  0  0  0  0]
 [11  0  4  2  0  0  0  7]
 [ 0 39  0  0  0  5  0  0]
 [ 0  0 12  0  0  0 11  5]
 [ 0  7  0  0  0  7 16 18]]

'''

'''
---------------------------------------------------- OLD ----------------------------------------------------
stateful model, without model.reset_states():
Accuracy: 0.71
              precision    recall  f1-score   support
    
          4L      1.000     1.000     1.000         7
          4R      0.917     0.917     0.917        12
          7L      0.778     1.000     0.875         7
          7R      1.000     0.500     0.667         6
         10L      0.000     0.000     0.000         7
         10R      0.688     0.846     0.759        13
         11L      0.286     0.500     0.364         4
         11R      0.500     0.600     0.545        10
    
    accuracy                          0.712        66
    macro avg      0.646     0.670     0.641        66
    weighted avg      0.675     0.712     0.680        66
    
    [[ 7  0  0  0  0  0  0  0]
    [ 0 11  0  0  0  0  0  1]
    [ 0  0  7  0  0  0  0  0]
    [ 0  1  2  3  0  0  0  0]
    [ 0  0  0  0  0  5  1  1]
    [ 0  0  0  0  0 11  0  2]
    [ 0  0  0  0  0  0  2  2]
    [ 0  0  0  0  0  0  4  6]]
    
stateful model, with model.reset_states():
    Accuracy: 0.73

              precision    recall  f1-score   support
    
          4L      0.778     1.000     0.875         7
          4R      0.857     1.000     0.923        12
          7L      0.875     1.000     0.933         7
          7R      1.000     0.500     0.667         6
         10L      0.000     0.000     0.000         7
         10R      0.688     0.846     0.759        13
         11L      0.333     0.500     0.400         4
         11R      0.600     0.600     0.600        10
    
    accuracy                          0.727        66
    macro avg      0.641     0.681     0.645        66
    weighted avg      0.669     0.727     0.685        66
    
    [[ 7  0  0  0  0  0  0  0]
    [ 0 12  0  0  0  0  0  0]
    [ 0  0  7  0  0  0  0  0]
    [ 0  2  1  3  0  0  0  0]
    [ 2  0  0  0  0  5  0  0]
    [ 0  0  0  0  0 11  0  2]
    [ 0  0  0  0  0  0  2  2]
    [ 0  0  0  0  0  0  4  6]] 7+12+7+3+11+4+10 = 54/66 = 0.8181818181818182
    
    
With zero station and stateful and model.reset_states():
Accuracy: 0.63
              precision    recall  f1-score   support

           0      0.980     0.641     0.775       153
          4L      1.000     0.714     0.833         7
          4R      0.467     0.583     0.519        12
          7L      0.700     1.000     0.824         7
          7R      1.000     0.667     0.800         6
         10L      0.000     0.000     0.000         7
         10R      0.250     0.769     0.377        13
         11L      0.154     0.500     0.235         4
         11R      0.167     0.500     0.250        10

    accuracy                          0.630       219
   macro avg      0.524     0.597     0.513       219
weighted avg      0.817     0.630     0.683       219

[[98  0  8  2  0  1 23  7 14]
 [ 0  5  0  0  0  1  0  0  1]
 [ 0  0  7  0  0  0  0  0  5]
 [ 0  0  0  7  0  0  0  0  0]
 [ 1  0  0  1  4  0  0  0  0]
 [ 1  0  0  0  0  0  6  0  0]
 [ 0  0  0  0  0  0 10  0  3]
 [ 0  0  0  0  0  0  0  2  2]
 [ 0  0  0  0  0  0  1  4  5]]
 
 
With zero station and stateful and model.reset_states(), 64 samples:
Accuracy: 0.62
          precision    recall  f1-score   support

      4L      1.000     0.714     0.833         7
      4R      1.000     0.583     0.737        12
      7L      0.875     1.000     0.933         7
      7R      1.000     0.800     0.889         5
     10L      0.000     0.000     0.000         6
     10R      0.588     0.769     0.667        13
     11L      0.333     0.500     0.400         4
     11R      0.312     0.500     0.385        10

accuracy                          0.625        64
macro avg      0.639     0.608     0.605        64
weighted avg      0.660     0.625     0.621        64

[[ 5  0  0  0  1  0  0  1]
[ 0  7  0  0  0  0  0  5]
[ 0  0  7  0  0  0  0  0]
[ 0  0  1  4  0  0  0  0]
[ 0  0  0  0  0  6  0  0]
[ 0  0  0  0  0 10  0  3]
[ 0  0  0  0  0  0  2  2]
[ 0  0  0  0  0  1  4  5]]


------------------------------------------------------------------------------------------------------


all five full videos from test set, zero model, stateful model, model.reset_states() at predicted zero:
Accuracy: 0.60

              precision    recall  f1-score   support

           0      0.000     0.000     0.000         0
          4L      0.894     0.955     0.923        44
          4R      0.778     0.493     0.603        71
          7L      0.518     0.784     0.624        37
          7R      0.619     0.333     0.433        39
         10L      0.000     0.000     0.000        23
         10R      0.647     0.717     0.680        46
         11L      0.400     0.483     0.438        29
         11R      0.621     0.766     0.686        47

    accuracy                          0.601       336
   macro avg      0.497     0.503     0.487       336
weighted avg      0.620     0.601     0.594       336

[[ 0  0  0  0  0  0  0  0  0]
 [ 0 42  0  0  0  1  0  0  1]
 [ 6  0 35  2  0  0  4 13 11]
 [ 0  0  0 29  8  0  0  0  0]
 [ 2  0  1 22 13  0  0  1  0]
 [ 7  5  0  0  0  0 11  0  0]
 [ 2  0  5  0  0  0 33  1  5]
 [ 5  0  0  3  0  0  2 14  5]
 [ 0  0  4  0  0  0  1  6 36]]
 
all five full videos from test set, normal model (22:33:14), stateful model, model.reset_states() only when loaded:
Accuracy: 0.70
              precision    recall  f1-score   support

          4L      0.833     0.909     0.870        44
          4R      0.900     0.783     0.837        69
          7L      0.605     0.622     0.613        37
          7R      0.529     0.692     0.600        39
         10L      0.000     0.000     0.000        24
         10R      0.667     0.773     0.716        44
         11L      0.706     0.429     0.533        28
         11R      0.647     0.917     0.759        48

    accuracy                          0.703       333
   macro avg      0.611     0.640     0.616       333
weighted avg      0.667     0.703     0.676       333

[[40  0  4  0  0  0  0  0]
 [ 0 54  1  5  0  1  0  8]
 [ 0  0 23 14  0  0  0  0]
 [ 0  1 10 27  0  0  0  1]
 [ 8  0  0  4  0  8  1  3]
 [ 0  5  0  0  0 34  0  5]
 [ 0  0  0  1  0  8 12  7]
 [ 0  0  0  0  0  0  4 44]]
 
all five full videos from test set, normal model (22:33:14), stateful model, reset states every time label changes: (reset sequences for new patient --> three less sequences)
Accuracy: 0.70
               precision    recall  f1-score   support

          4L      0.780     0.886     0.830        44
          4R      0.836     0.812     0.824        69
          7L      0.605     0.622     0.613        37
          7R      0.509     0.692     0.587        39
         10L      0.000     0.000     0.000        24
         10R      0.720     0.818     0.766        44
         11L      0.706     0.429     0.533        28
         11R      0.690     0.833     0.755        48

    accuracy                          0.700       333
   macro avg      0.606     0.636     0.613       333
weighted avg      0.657     0.700     0.672       333

[[39  0  4  0  0  0  0  1]
 [ 0 56  0  5  0  1  0  7]
 [ 0  0 23 14  0  0  0  0]
 [ 0  2  9 27  0  0  0  1]
 [11  0  0  6  0  6  1  0]
 [ 0  5  0  0  0 36  0  3]
 [ 0  0  2  1  0  7 12  6]
 [ 0  4  0  0  0  0  4 40]]
 
 
all five full videos from test set, normal model (22:33:14), stateless model: (reset sequences for new patient --> three less sequences)
 Accuracy: 0.61
              precision    recall  f1-score   support

          4L      0.722     0.886     0.796        44
          4R      0.742     0.667     0.702        69
          7L      0.512     0.568     0.538        37
          7R      0.453     0.615     0.522        39
         10L      0.000     0.000     0.000        24
         10R      0.694     0.568     0.625        44
         11L      0.516     0.571     0.542        28
         11R      0.554     0.646     0.596        48

    accuracy                          0.607       333
   macro avg      0.524     0.565     0.540       333
weighted avg      0.574     0.607     0.586       333

[[39  0  3  0  0  0  1  1]
 [ 0 46  2  9  0  1  1 10]
 [ 0  1 21 14  0  0  0  1]
 [ 1  2 11 24  0  0  0  1]
 [12  0  0  3  0  4  3  2]
 [ 1 10  3  0  0 25  0  5]
 [ 0  0  1  3  0  3 16  5]
 [ 1  3  0  0  0  3 10 31]]
 
all five full videos from test set, normal model (09:26:50), stateful model, model.reset_states() only when loaded:
Accuracy: 0.68
              precision    recall  f1-score   support

          4L      0.808     0.955     0.875        44
          4R      0.884     0.884     0.884        69
          7L      0.469     0.622     0.535        37
          7R      0.545     0.615     0.578        39
         10L      0.000     0.000     0.000        24
         10R      0.556     0.795     0.654        44
         11L      0.667     0.214     0.324        28
         11R      0.766     0.750     0.758        48

    accuracy                          0.682       333
   macro avg      0.587     0.604     0.576       333
weighted avg      0.646     0.682     0.649       333

[[42  0  0  0  0  2  0  0]
 [ 0 61  1  1  0  4  0  2]
 [ 0  0 23 14  0  0  0  0]
 [ 0  0 14 24  0  1  0  0]
 [ 9  0  2  1  0 10  0  2]
 [ 1  8  0  0  0 35  0  0]
 [ 0  0  9  4  0  2  6  7]
 [ 0  0  0  0  0  9  3 36]]
 

'''

