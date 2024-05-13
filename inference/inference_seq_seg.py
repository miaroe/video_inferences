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

from tensorflow.keras.preprocessing.image import img_to_array
from resources.create_model import create_model
from resources.stations_config import get_stations_config
from sklearn.metrics import confusion_matrix, classification_report
from skimage.morphology.convex_hull import grid_points_in_poly
from matplotlib.colors import LinearSegmentedColormap
from skimage.transform import resize
from matplotlib.patches import Patch

video_df_path = '/Users/miarodde/Documents/sintef/ebus-ai/video_inferences/data/test_sequence/test_video_df.csv'
classification_model_path = '/Users/miarodde/Documents/sintef/ebus-ai/video_inferences/models/sequence/22:33:14/best_model'
segmentation_model_path = '/Users/miarodde/Documents/sintef/ebus-ai/video_inferences/models/segmentation/best_model'

plt.style.use('dark_background')


# Function to preprocess images
def preprocess_class_frame(image_path):
    frame = tf.keras.utils.load_img(image_path, color_mode='rgb', target_size=(224, 224))
    frame = img_to_array(frame)
    frame = tf.cast(frame, tf.float32)
    return frame

def preprocess_seg_frame(frame):
    frame = cv2.imread(frame, cv2.IMREAD_GRAYSCALE)
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
    model = tf.keras.models.load_model(model_path)
    stateful_model = create_model(instance_size=(224, 224, 3), num_stations=8, stateful=True)
    stateful_model.set_weights(model.get_weights())
    return stateful_model

class PlotSeg:

        def __init__(self, frame, mask, true_label):
            plt.ion()
            self.fig, self.ax = plt.subplots(figsize=(9, 6))
            self.frame = frame
            self.mask = mask
            self.true_label = true_label


        # generates a mask for the ultrasound sector, returns a mask of same shape as the reference image shape with True values inside the sector and False outside
        def ultrasound_sector_mask(self, reference_image_shape, origin=None, sector_angle=None):

            if origin is None:
                origin = (-60, 1100)  # (-60, (525+1685)/2)  # (y, x) coordinates in pixels
            # plt.scatter(origin[1], origin[0], s=5)
            if sector_angle is None:
                sector_angle = np.pi / 3  # radians, pi/3 = 60deg

            # Points in clockwise order from top left
            # Top of sector
            p1 = (100, 1000)  # p1 = (75, 1010)
            p2 = (100, 1100)
            p3 = (100, 1190)  # p3 = (75, 1183)
            # Bottom of sector
            p4 = (820, 1658)
            p5 = (1035, 1658)
            p6 = (1035, 530)
            p7 = (820, 530)

            pts = (p1, p2, p3, p4, p5, p6, p7)
            conv_hull = grid_points_in_poly(shape=reference_image_shape, verts=pts).astype(dtype=bool)

            return conv_hull

        def update(self, frame, mask, true_label):
            self.ax.clear()
            self.frame = frame
            self.mask = mask
            self.true_label = true_label

            c_invalid = (0, 0, 0)
            colors = [(0.2, 0.2, 0.2),  # dark gray = background
                      (0.55, 0.4, 0.85),  # purple = lymph nodes
                      (0.76, 0.1, 0.05)]  # red   = blood vessels

            label_cmap = LinearSegmentedColormap.from_list('label_map', colors, N=3)
            label_cmap.set_bad(color=c_invalid, alpha=0)  # set invalid (nan) colors to be transparent
            image_cmap = plt.cm.get_cmap('gray')
            image_cmap.set_bad(color=c_invalid)

            # Resize mask to original frame size
            mask_resized = resize(self.mask, (935, 1128), preserve_range=True, order=0).astype(mask.dtype)

            frame_mask = self.ultrasound_sector_mask(reference_image_shape=(1080, 1920))[100:1035, 530:1658]
            frame = np.ma.array(self.frame, mask=~frame_mask)

            mask_resized = np.ma.masked_less_equal(mask_resized, 0)
            mask_resized = np.ma.array(mask_resized, mask=~frame_mask)

            self.ax.set_title('Segmentation masks', size=20)
            self.ax.imshow(frame, cmap=image_cmap)
            self.ax.imshow(mask_resized, cmap=label_cmap, interpolation='nearest', alpha=0.3, vmin=0, vmax=2)
            # set text at top right corner
            #self.ax.text(0.05, 0.95, f'True station: {self.true_label}', transform=self.ax.transAxes, fontsize=18,
            #                verticalalignment='top', color='white')

            # Define legend patches
            legend_patches = [
                Patch(color=colors[1], label='Lymph node'),
                Patch(color=colors[2], label='Blood vessel'),
            ]

            # Add the legend to the plot
            self.ax.legend(handles=legend_patches, loc='upper right', fontsize=12, frameon=True, edgecolor='black')

            self.ax.axis('off')
            self.fig.tight_layout()
            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()
            #plt.pause(0.001)

class PlotPred:

    def __init__(self, pred_values, labels):
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(10, 6))
        self.pred_values = pred_values
        self.labels = labels
        self.colors = ['green' if val >= 0.5 else 'yellow' if val >= 0.3 else 'grey' for val in pred_values]


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

        # plt.pause(0.001)


# Main loop to stream images as video
def stream_video(video_df_path, class_model_path, seg_model_path, stations_config, sequence_length=10):
    sequence = []
    last_label = ''
    stations = list(stations_config.keys())
    print(stations)

    # for statistics
    #num_correct = 0
    #num_total = 0
    #predictions = []
    #true_labels = []

    # Load the models
    seg_model = tf.keras.models.load_model(seg_model_path)
    class_model = load_stateful_model(class_model_path)
    class_model.reset_states()

    df = pd.read_csv(video_df_path, sep=',')

    # initialize plot class for live plotting
    class_plot = PlotPred([0.1] * 8, stations)
    # initialize plot class with empty image and mask
    seg_plot = PlotSeg(np.zeros((256, 256)), np.zeros((256, 256)), 'None')

    # Loop through the rows of the df
    for index, row in df.iterrows():
        loop_start_time = time.time()  # Record the start time of the loop
        frame_path = row['frame_path']
        label = row['label']
        patient = row['patient']
        print('patient:', patient)
        original_frame = cv2.imread(frame_path, cv2.IMREAD_GRAYSCALE)
        class_frame = preprocess_class_frame(frame_path)  # Preprocess for the model
        seg_frame = preprocess_seg_frame(frame_path)  # Preprocess for the model

        sequence.append(class_frame)  # Add the frame to the sequence

        if label != last_label: class_model.reset_states()  # Reset states if the label changes
        last_label = label

        # Predictions
        seg_pred = seg_model.predict(seg_frame)[0]
        seg_pred = np.argmax(seg_pred, axis=-1)
        seg_plot.update(original_frame, seg_pred, label)

        if len(sequence) == sequence_length:
            # Get predictions for the current sequence
            class_pred = get_predictions(sequence, class_model)
            prediction = stations[np.argmax(class_pred)]
            #sequence.pop(0)  # Remove the first frame from the sequence
            sequence = []

            # Plot the predictions
            class_plot.update(class_pred, stations)

            # for statistics
            #if prediction == label:
            #    num_correct += 1
            #num_total += 1
            #predictions.append(prediction)
            #true_labels.append(label)

        loop_end_time = time.time()  # Record the end time of the loop
        loop_duration = (loop_end_time - loop_start_time) * 1000  # Convert to milliseconds
        print(f"Processing Time: {loop_duration:.2f} ms")

        # Calculate remaining time to wait
        wait_time = max(200 - loop_duration, 1)  # Ensure there's at least a minimal wait time
        #print(f"Wait Time: {wait_time:.2f} ms")
        #if cv2.waitKey(int(wait_time)) & 0xFF == ord('q'):  # Adjust wait time based on processing duration
        #    break



    plt.ioff()
    #cv2.destroyAllWindows()

    #print(f'Accuracy: {num_correct / num_total:.2f}')
    #print(classification_report(true_labels, predictions, digits=3, target_names=list(stations_config.keys()),
    #                            labels=stations))
    #print(confusion_matrix(true_labels, predictions, labels=stations))



if __name__ == '__main__':
    stream_video(video_df_path, classification_model_path, segmentation_model_path, get_stations_config(3))

