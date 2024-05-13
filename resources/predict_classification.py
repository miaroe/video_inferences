import shutil
import tensorflow as tf
import pickle
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from resources.stations_config import get_stations_config

dirname_test_df = '/mnt/EncryptedData1/LungNavigation/EBUS/ultrasound/baseline/Levanger_and_StOlavs/test_dirname_label_df.csv'
model_path = '/home/miaroe/workspace/lymph-node-classification/output/models/2023-11-21/10:51:52'
model_name = 'best_model'

# local_full_video_path = '/Users/miarodde/Documents/sintef/ebus-ai/EBUS_Levanger_full_videos/Patient_036/Sequence_001'
local_full_video_path = '/Users/miarodde/Documents/sintef/ebus-ai/baseline/Levanger_and_StOlavs/test/full_video'
# local_full_video_path = '/Users/miarodde/Documents/sintef/ebus-ai/EBUS_StOlavs_baseline_test/full_video'
# local_full_video_path = '/Users/miarodde/Documents/sintef/ebus-ai/Patient_036/full_video'
local_model_path = '/Users/miarodde/Documents/sintef/ebus-ai/videos_in_FAST/models/'
local_model_name = 'best_model'


# ------------------ Helper functions ------------------#

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


def rescale_image(image):
    image = tf.cast(image, tf.float32)
    image = image / 255.0
    return image


def preprocess_image(image):
    image = tf.cast(image, tf.float32)
    image = image[100:1035, 530:1658]
    image = tf.image.resize(image, [256, 256])
    image = image / 255.0
    return image


# ------------------ Server function ------------------#

def predict_classification_per_station(dirname_test_df, model_path, model_name, labels):
    model = tf.keras.models.load_model(os.path.join(model_path, model_name))
    dirname_test_df = pd.read_csv(dirname_test_df, sep=',')
    # for each row in the df, get the station and the dirname. Then for each dirname, loop through the images and predict.
    # Then take the average of the prediction arrays and assign the label to the station.
    # Then compare the label to the true label and calculate the accuracy.

    # create a df to store the predictions, including all columns from the original df
    station_predictions = []

    # loop through the df and get the dirname and station
    for index, row in dirname_test_df.iterrows():
        dirname = row['dirname']
        station = row['label']
        patient_id = row['patient_id']
        print('dirname:', dirname)

        prediction_per_image_arr = []
        num_correct_threshold = 0
        num_frames = 0

        # loop through the images in the dirname and predict
        for image_name in os.listdir(dirname):
            if image_name.endswith(".png"):
                img = tf.keras.utils.load_img(os.path.join(dirname, image_name), color_mode='rgb',
                                              target_size=None)
                img = preprocess_image(img)
                img_array = tf.keras.utils.img_to_array(img)
                img_array = tf.expand_dims(img_array, 0)

                prediction_per_image = model.predict(img_array)
                prediction_per_image_arr.append(prediction_per_image)
                num_frames += 1

        prediction_per_station_arr = np.mean(prediction_per_image_arr, axis=0)
        # print('prediction_per_station:', prediction_per_station_arr)
        prediction_value = np.round(100 * np.max(prediction_per_station_arr[0]), 2)
        # print('prediction_value:', prediction_value)
        prediction_station = labels[np.argmax(prediction_per_station_arr)]
        # print('prediction_station:', prediction_station)
        # print('station:', station)

        for prediction in prediction_per_image_arr:
            if np.max(prediction) > 0.5:
                prediction_station = labels[np.argmax(prediction)]
                if prediction_station == station:
                    num_correct_threshold += 1

        if prediction_station != station:
            print('prediction_station:', prediction_station)
            print('station:', station)
            print('prediction_value:', prediction_value)
            print('prediction_per_station:', prediction_per_station_arr)
            print('-----------------------------')

        # add the prediction to the station_predictions df
        station_predictions.append({'dirname': dirname, 'patient_id': patient_id, 'station': station,
                                    'prediction_values_arr': prediction_per_station_arr,
                                    'prediction_value': prediction_value,
                                    'prediction_station': prediction_station,
                                    'num_correct_threshold': num_correct_threshold / num_frames})

    station_predictions_df = pd.DataFrame(station_predictions)

    # save in model_path
    station_predictions_df.to_csv(os.path.join(model_path, 'station_predictions.csv'), index=False)

    print('num_correct:', len(
        station_predictions_df[station_predictions_df['station'] == station_predictions_df['prediction_station']]))
    print('num_total:', len(station_predictions_df))
    print('percent_correct:', len(station_predictions_df[station_predictions_df['station'] == station_predictions_df[
        'prediction_station']]) / len(station_predictions_df) * 100)


#predict_classification_per_station(dirname_test_df, model_path, model_name, labels=list(get_stations_config(3).keys()))


def compare_predictions_to_labels(model_path):
    station_predictions_df = pd.read_csv(os.path.join(model_path, 'station_predictions.csv'), sep=',')
    num_incorrect = 0
    num_total = 0
    for index, row in station_predictions_df.iterrows():
        if row['station'] != row['prediction_station']:
            num_incorrect += 1
            print('station:', row['station'])
            print('prediction_station:', row['prediction_station'])
            print('patient_id:', row['patient_id'])
            print('dirname:', row['dirname'])
            print('prediction_value:', row['prediction_value'])
            print('prediction_values_arr:', row['prediction_values_arr'])
            print('----------------')
        num_total += 1

    print('num_incorrect:', num_incorrect)
    print('num_total:', num_total)
    print('accuracy:', (1 - num_incorrect / num_total) * 100, '%')


# compare_predictions_to_labels(model_path)


# ------------------ Local functions ------------------#

def make_frame_pred_dict(video_path, model_path, model_name):
    model = tf.keras.models.load_model(os.path.join(model_path, model_name))
    frame_pred_dict = {}

    for image_name in os.listdir(video_path):
        if image_name.endswith(".png"):
            img = tf.keras.utils.load_img(os.path.join(video_path, image_name), color_mode='rgb',
                                          target_size=(256, 256))
            # img = tf.image.crop_to_bounding_box(img, 24, 71, 223, 150)  # Cropping the image to the region of interest
            img = rescale_image(img)
            img_array = tf.keras.utils.img_to_array(img)
            img_array = tf.expand_dims(img_array, 0)

            prediction = model.predict(img_array)
            image_index = image_name.split('.')[0]
            frame_pred_dict[image_index] = prediction
            # print(image_name, prediction) #frame_201.png [[8.8051502e-06 9.3291172e-05 1.3058403e-01 8.6508465e-01 3.9644584e-07 1.1566924e-05 6.9905451e-05 4.1473657e-03]]
    return frame_pred_dict


def plot_misclassified_images(video_path, model_path, model_name, labels):
    plt.style.use('ggplot')
    misclassified_images = []

    # get dictionaries with frame labels and predictions
    frame_pred_dict = make_frame_pred_dict(video_path, model_path, model_name)
    frame_label_dict = get_frame_label_dict_modified(video_path)
    print('number of frames in total:', len(frame_pred_dict))

    # create folder for misclassified images
    if not os.path.exists(os.path.join(video_path, 'misclassified_images')):
        os.makedirs(os.path.join(video_path, 'misclassified_images'))

    # create dataframe of misclassified images
    for image_name, prediction_arr in frame_pred_dict.items():
        prediction_arr = frame_pred_dict[image_name]
        prediction_label = labels[np.argmax(prediction_arr)]
        label = frame_label_dict[int(image_name)]
        if prediction_label != label:
            misclassified_images.append({'image_name': image_name,
                                         'label': label,
                                         'prediction_label': prediction_label,
                                         'prediction_value': np.round(100 * np.max(prediction_arr[0]), 2)})

    df = pd.DataFrame(misclassified_images)
    # total number of misclassified images
    print("total number of misclassified images:", len(df))
    # number of misclassified images per label
    print(df.groupby(['label']).size())
    # number of misclassified images per prediction label
    print(df.groupby(['prediction_label']).size())

    # create plot showing number of misclassified images per label
    df.groupby(['label']).size().plot(kind='bar')
    fig = plt.gcf()
    fig.subplots_adjust(bottom=0.2)
    plt.title('Number of misclassified images per label,' + ' Total: ' + str(len(df)))
    plt.xlabel('Label')
    plt.ylabel('Number of misclassified images')
    plt.savefig(os.path.join(video_path, 'misclassified_images', 'number_of_misclassified_images_per_label.png'))
    plt.clf()

    # df_filtered = df.loc[df['label'] != '0']
    # print("number of misclassified images that does not have label='0':", len(df_filtered))

    # calculate average prediction value for each label and plot
    df['prediction_value'] = df['prediction_value'].astype(float)
    df.groupby(['label'])['prediction_value'].mean().plot(kind='bar')
    fig = plt.gcf()
    fig.subplots_adjust(bottom=0.2)
    plt.title('Average prediction value per label, Total: ' + str(len(df)))
    plt.xlabel('Label')
    plt.ylabel('Average prediction value')
    plt.savefig(os.path.join(video_path, 'misclassified_images', 'average_prediction_value_per_label.png'))
    plt.clf()

    # save misclassified images to folder, including prediction label and prediction value
    for label in df['label'].unique():
        print(label)
        # create folder for each label
        dst = os.path.join(video_path, 'misclassified_images', label)
        if not os.path.exists(dst):
            os.makedirs(dst)

        # plot predicted labels for label
        print('plotting predicted labels for label', label)
        df_filtered = df.loc[df['label'] == label]
        df_filtered.groupby(['prediction_label']).size().plot(kind='bar')
        fig = plt.gcf()
        fig.subplots_adjust(bottom=0.2)
        plt.title('Number of misclassified images per predicted label for label ' + label + ', Total: ' + str(
            len(df_filtered)))
        plt.xlabel('Predicted label')
        plt.ylabel('Number of misclassified images')
        plt.savefig(os.path.join(dst, 'pred_labels_for_' + label + '.png'))
        plt.clf()

        # os.mkdir(os.path.join(video_path, 'misclassified_images', label))
        for image_name in df.loc[df['label'] == label]['image_name']:
            img = tf.io.read_file(os.path.join(video_path, f'{image_name}.png'))
            img = tf.image.decode_png(img, channels=3)
            # get prediction label from df_filtered
            prediction_label = df.loc[df['image_name'] == image_name]['prediction_label'].values[0]
            prediction_value = df.loc[df['image_name'] == image_name]['prediction_value'].values[0]
            # plot image, label and prediction label
            plt.figure()
            plt.imshow(img)
            plt.title(f'label: {label}, prediction_label: {prediction_label}, prediction_value: {prediction_value}')
            plt.axis("off")
            plt.savefig(os.path.join(video_path, 'misclassified_images', label, f'{image_name}.png'), dpi=300)
            plt.close()


plot_misclassified_images(video_path=local_full_video_path, model_path=local_model_path, model_name=local_model_name, labels=list(get_stations_config(3).keys()))


def save_dict_to_pickle(video_path):
    frame_pred_dict = make_frame_pred_dict(video_path)
    # save dict to file with pickle
    with open(os.path.join(local_full_video_path, "frame_pred_dict.pickle"), "wb") as f:
        pickle.dump(frame_pred_dict, f)


# save_dict_to_pickle(local_full_video_path)


def create_full_video_from_baseline_test(baseline_path):
    # create full video from all the folders in baseline_path and a labels.csv file from the folder name (label) and the frame number
    file_name_index = 0

    labels = []
    full_video_path = os.path.join(baseline_path, 'full_video')
    # create full video
    if not os.path.exists(full_video_path):
        os.makedirs(full_video_path)

    for folder in os.listdir(baseline_path):
        print(folder)
        if folder.endswith(".csv") or folder.endswith(".DS_Store"):
            print(folder)
            continue
        for image_name in os.listdir(os.path.join(baseline_path, folder)):
            if image_name.endswith(".png"):
                labels.append({'old_frame_number': image_name.split('.')[0],
                               'frame_number': file_name_index,
                               'label': folder})
                print(str(file_name_index) + '.png')
                shutil.copyfile(os.path.join(baseline_path, folder, image_name),
                                os.path.join(full_video_path, str(file_name_index) + '.png'))
                file_name_index += 1

    df = pd.DataFrame(labels)
    print(df)
    df.to_csv(os.path.join(full_video_path, 'labels.csv'), index=False)

# create_full_video_from_baseline_test('/Users/miarodde/Documents/sintef/ebus-ai/baseline/Levanger_and_StOlavs/test')
