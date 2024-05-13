import fast  # Must import FAST before rest of pyside2
import os
from time import sleep
from resources.frame_label_dict import get_frame_label_dict

full_video_path = '/Users/miarodde/Documents/sintef/ebus-ai/EBUS_Levanger_full_videos/Patient_008/Sequence_001'

class MyRandomAccessStreamer(fast.PythonRandomAccessStreamer):
    """
    A simple FAST random access streamer which runs in its own thread.
    By random access it is meant that it can move to any given frame index, thus
    facilitating playback with for instance PlaybackWidget.
    This streamer reads a series of MHD images on disk.
    This can be done easily with the ImageFileStreamer, but this is just an example.
    """
    def __init__(self, framerate):
        """
        Constructor, remember to create the output port here
        """
        super().__init__()
        self.createOutputPort(0)
        self.setFramerate(framerate)
        self.frame = 0

    def getNrOfFrames(self):
        """
        This function must return how many frames the streamer has.
        :return: nr of frames
        """
        return 1586

    def generateStream(self):
        """
        This method runs in its own thread.
        Run you streaming loop here.
        Remember to call self.addOutputData and self.frameAdded for each frame.
        If these calls return and exception, it means the streaming should stop, thus you need to exit
        your streaming loop.
        """

        path = os.path.join(full_video_path, 'frame_#.png')
        while not self.isStopped():
            # First, we need to check if this streaming is paused
            if self.getPause():
                self.waitForUnpause() # Wait for streamer to be unpaused
            pause = self.getPause() # Check whether to pause or not
            self.frame = self.getCurrentFrameIndex()

            print('Streaming', self.frame)

            # Read frame from disk
            importer = fast.ImageFileImporter.create(path.replace('#', str(self.frame)))
            image = importer.runAndGetOutputData()
            if self.frame == self.getNrOfFrames()-1: # If this is last frame, mark it as such
                image.setLastFrame('MyStreamer')

            if not pause:
                if self.getFramerate() > 0:
                    sleep(1.0/self.getFramerate()) # Sleep to give the requested framerate
                self.getCurrentFrameIndexAndUpdate() # Update the frame index to the next frame
            try:
                self.addOutputData(0, image)
                self.frameAdded() # Important to notify any listeners
            except:
                break


class FrameToLabel(fast.PythonProcessObject):

    def __init__(self):
        super().__init__()
        self.createInputPort(0)
        self.createOutputPort(0)
        self.frame_label_dict = get_frame_label_dict(full_video_path)


    def execute(self):
        # Get the label corresponding to the current frame
        input_image = self.getInputData()
        print('input_image: ', input_image)
        #frame = self.getCurrentFrameIndex()
        frame = 10
        print('frame: ', frame)
        label = self.frame_label_dict.get(str(frame), 'N/A')
        print('Label:', label)
        output_text = f'{label}'

        self.addOutputData(0, fast.Text.create(output_text, color=fast.Color.White()))


class StreamerWindow(object):
    is_running = False

    def __init__(self, station_labels, framerate):
        # Setup processing chain
        self.streamer = MyRandomAccessStreamer.create(framerate=framerate)
        self.streamer.setLooping(True)

        print('station_labels', station_labels)

        self.label = FrameToLabel.create()
        self.label.connect(self.streamer)

        # Renderers
        self.image_renderer = fast.ImageRenderer.create().connect(self.streamer)
        self.label_renderer = fast.TextRenderer.create(fontSize=48)
        self.label_renderer.connect(self.label)

        self.window = fast.SimpleWindow2D.create() \
            .connect([self.image_renderer, self.label_renderer])

        #self.window.addRenderer(self.label_renderer)

        # Set up playback widget
        self.widget = fast.PlaybackWidget(streamer=self.streamer)
        self.window.connect(self.widget)

    def run(self):
        self.window.run()


def run_streamer_with_label(station_labels, framerate):
    fast_streamer = StreamerWindow(
        station_labels=station_labels,
        framerate=framerate
    )

    fast_streamer.window.run()


def get_stations_config(station_config_nr):
    if station_config_nr == 1:  # binary classification
        return {
            '4L': 0,
            '7R': 1,
        }
    elif station_config_nr == 2:  # multiclass classification with almost balanced classes
        return {
            '4L': 0,
            '4R': 1,
            '7L': 2,
            '7R': 3,
            '10R': 4,
            '11R': 5
        }

    elif station_config_nr == 3:  # multiclass classification with unbalanced classes, deleted 7
        return {
            '4L': 0,
            '4R': 1,
            '7L': 2,
            '7R': 3,
            '10L': 4,
            '10R': 5,
            '11L': 6,
            '11R': 7
        }


run_streamer_with_label(station_labels=list(get_stations_config(3).keys()),
                        framerate=5
                        )
