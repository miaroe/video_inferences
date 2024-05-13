import fast
import numpy as np
import pickle
import os

from resources.predicted_stations import plot_pred_stations
from resources.stations_config import get_stations_config

video_path = '/Users/miarodde/Documents/sintef/ebus-ai/EBUS_Levanger_full_videos/Patient_036/Sequence_001'
model_path = '/Users/miarodde/Documents/sintef/ebus-ai/videos_in_FAST/models/'
model_name = 'best_model'


fast.Reporter.setGlobalReportMethod(fast.Reporter.COUT) # Uncomment to show debug info

#frame_pred_dict = pickle.load(open(os.path.join(local_full_video_path, "frame_pred_dict.pickle"), "rb"))
'''
station_labels = get_stations_config(station_config_nr)
for key in frame_pred_dict:
    pred_m = frame_pred_dict[key]
    pred_l = list(station_labels.keys())[np.argmax(pred_m)]
    print(f'frame: {key}, prediction: {pred_l}')
'''

class ClassificationToPlot(fast.PythonProcessObject):

    def __init__(self, data_path, labels=None, name=None):
        super().__init__()
        self.createInputPort(0)
        self.createOutputPort(0)
        self.frame = 0
        self.data_path = data_path

        if labels is not None:
            self.labels = labels
        if name is not None:
            self.name = name

    def execute(self):
        classification = self.getInputData(0)
        classification_arr = np.asarray(classification)
        print('classification_arr', classification_arr)

        #prediction_arr = frame_pred_dict[f'frame_{self.frame}.png']
        #prediction_label = self.labels[np.argmax(prediction_arr)]

        #print('prediction_arr', prediction_arr)
        #print('prediction_label', prediction_label)
        #print('prediction value', prediction_arr[0][np.argmax(prediction_arr)])

        img_arr = plot_pred_stations(self.data_path, self.labels, classification_arr, self.frame)
        fast_image = fast.Image.createFromArray(img_arr)

        self.addOutputData(0, fast_image)
        self.frame += 1


class ImageClassificationWindow(object):
    is_running = False

    def __init__(self, station_labels, data_path, model_path, model_name, framerate, seq_length=10):

        # Setup a FAST pipeline
        self.streamer = fast.ImageFileStreamer.create(os.path.join(data_path, 'frame_#.png'), loop=True, framerate=framerate)
        print('model: ', os.path.join(model_path, model_name))


        # Neural network model
        self.sequence = fast.ImagesToSequence.create(seq_length).connect(self.streamer)
        self.classification_model = fast.NeuralNetwork.create(os.path.join(model_path, model_name))
        self.classification_model.connect(0, self.streamer)
        print('station_labels', station_labels)

        # Classification (neural network output) to Plot
        self.station_classification_plot = ClassificationToPlot.create(name='Station', data_path=data_path, labels=list(station_labels.keys()))
        self.station_classification_plot.connect(0, self.classification_model, 0)

        # Renderers
        self.image_renderer = fast.ImageRenderer.create().connect(self.streamer)
        self.classification_renderer = fast.ImageRenderer.create()
        self.classification_renderer.connect(self.station_classification_plot)

        # Set up video window
        self.window = fast.DualViewWindow2D.create(
            width=800,
            height=900,
            bgcolor=fast.Color.Black(),
            verticalMode=True  # defaults to False
        )

        self.window.connectTop([self.classification_renderer])
        self.window.addRendererToTopView(self.classification_renderer)
        self.window.connectBottom([self.image_renderer])
        self.window.addRendererToBottomView(self.image_renderer)

        # Set up playback widget
        self.widget = fast.PlaybackWidget(streamer=self.streamer)
        self.window.connect(self.widget)

    def run(self):
        self.window.run()


def run_nn_image_classification(station_labels, data_path, model_path, model_name, framerate):

    fast_classification = ImageClassificationWindow(
            station_labels=station_labels,
            data_path=data_path,
            model_path=model_path,
            model_name=model_name,
            framerate=framerate
            )

    fast_classification.window.run()


run_nn_image_classification(station_labels=get_stations_config(3),
                            data_path=video_path,
                            model_path=model_path,
                            model_name=model_name,
                            framerate=5
                            )