# ================================================================================
# FAST random access streamer in python.
# A FAST streamer is a special process object which generates data asynchronously
# ================================================================================

import fast
import os
import numpy as np
from time import sleep
import pandas as pd
import cv2
from resources.frame_label_dict import get_frame_label_dict

# fast.Reporter.setGlobalReportMethod(fast.Reporter.COUT)

full_video_path = '/Users/miarodde/Documents/sintef/ebus-ai/EBUS_Levanger_full_videos/Patient_008/Sequence_001'

class MyRandomAccessStreamer(fast.PythonRandomAccessStreamer):
    """
    A simple FAST random access streamer which runs in its own thread.
    By random access it is meant that it can move to any given frame index, thus
    facilitating playback with for instance PlaybackWidget.
    This streamer reads a series of MHD images on disk.
    This can be done easily with the ImageFileStreamer, but this is just an example.
    """

    def __init__(self, full_video_path, framerate, nr_of_frames):
        """
        Constructor, remember to create the output port here
        """
        super().__init__()
        self.createOutputPort(0)
        self.setFramerate(framerate)
        self.full_video_path = full_video_path
        self.frame_label_dict = get_frame_label_dict(self.full_video_path)
        self.nr_of_frames = nr_of_frames

    def getNrOfFrames(self):
        """
        This function must return how many frames the streamer has.
        :return: nr of frames
        """
        return self.nr_of_frames

    def generateStream(self):
        """
        This method runs in its own thread.
        Run you streaming loop here.
        Remember to call self.addOutputData and self.frameAdded for each frame.
        If these calls return and exception, it means the streaming should stop, thus you need to exit
        your streaming loop.
        """

        path = os.path.join(self.full_video_path, 'frame_#.png')
        while not self.isStopped():
            # First, we need to check if this streaming is paused
            if self.getPause():
                self.waitForUnpause()  # Wait for streamer to be unpaused
            pause = self.getPause()  # Check whether to pause or not
            frame = self.getCurrentFrameIndex()

            print('Streaming', frame)

            # Read frame from disk
            importer = fast.ImageFileImporter.create(path.replace('#', str(frame)))
            image = importer.runAndGetOutputData()
            if frame == self.getNrOfFrames() - 1:  # If this is last frame, mark it as such
                image.setLastFrame('MyStreamer')

            # Generate text based on frame name
            frame_name = f'Frame_{frame}.png'
            label = self.frame_label_dict.get(str(frame), 'N/A')
            print('label', label)
            label_text = f'Label: {label}'

            # Convert the image to a NumPy array
            image_array = np.array(image)

            # Overlay text on the image using NumPy operations
            cv2.putText(image_array, frame_name, (10, 30), cv2.FONT_HERSHEY_DUPLEX, 1.2, (173, 216, 230), 2)
            cv2.putText(image_array, label_text, (10, 90), cv2.FONT_HERSHEY_DUPLEX, 1.2, (173, 216, 230), 2)

            # Convert the modified array back to an image
            composite_image = fast.Image.createFromArray(image_array)

            if not pause:
                if self.getFramerate() > 0:
                    sleep(1.0 / self.getFramerate())  # Sleep to give the requested framerate
                self.getCurrentFrameIndexAndUpdate()  # Update the frame index to the next frame
            try:
                self.addOutputData(0, composite_image)
                self.frameAdded()  # Important to notify any listeners
            except:
                break


def run_streamer_with_label(full_video_path, framerate, nr_of_frames):
    # Setup processing chain and run
    streamer = MyRandomAccessStreamer.create(full_video_path, framerate, nr_of_frames)
    streamer.setLooping(True)

    image_renderer = fast.ImageRenderer.create().connect(streamer)

    window = fast.SimpleWindow2D.create().connect(image_renderer)
    # set window size
    window.setWidth(1652)
    window.setHeight(990)
    widget = fast.PlaybackWidget(streamer)  # GUI widget for controlling playback
    window.addWidget(widget)
    window.run()


run_streamer_with_label(full_video_path=full_video_path,
                        framerate=5, nr_of_frames=1000)
