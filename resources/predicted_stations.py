import matplotlib.pyplot as plt
import numpy as np
import io
import matplotlib.ticker as mtick
import os
import pandas as pd

plt.style.use('dark_background')

def get_frame_label_dict(full_video_path):
    frame_label_dict = {}
    labels_path = os.path.join(full_video_path, 'labels.csv')
    # read the csv file using pandas
    df = pd.read_csv(labels_path, sep=';')

    # create a dictionary to store the mappings
    for index, row in df.iterrows():
        frame = (row['path']).split('\\')[-1].replace('.png', '')
        frame_label_dict[frame] = row['label']

    return frame_label_dict


def plot_pred_stations(local_full_video_path, station_labels, pred, frame):
    fig = plt.figure(figsize=(10, 5))
    plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))

    # multiclass
    if len(station_labels) > 2:
        print('frame: ', frame)
        pred_max_i = np.argsort(pred)[::-1][:3]  # index of top three predictions
        frame_label_dict = get_frame_label_dict(local_full_video_path)
        frame = 'frame_' + str(frame)
        print('fast_pred: ', station_labels[np.argmax(pred)])
        print('fast_pred value: ', pred[np.argmax(pred)])
        for i in pred_max_i:
            plt.bar(station_labels[i], pred[i])
        plt.title(frame + '\n' + 'True label: ' + frame_label_dict[frame])

    # binary
    else:
        pos_prob = pred[0]
        neg_prob = 1.0 - pos_prob
        probs = np.array([pos_prob, neg_prob])
        pred_int = (probs >= 0.5).astype(int)

        print('probs: ', probs)  # these values are really low compared to the values from model.predict TODO fix this
        print('pred_int: ', pred_int)
        for i in pred_int:
            plt.bar(station_labels[i], probs[i])

    img_buf = io.BytesIO()
    plt.savefig(img_buf, format='png')

    # save img to numpy array
    img_arr = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img_arr = img_arr.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return img_arr


'''
pred = [0.04027496, 0.01737576, 0.0540364, 0.02898372, 0.810823, 0.02736586, 0.02114019]
plot_pred_stations(station_labels, pred)
'''