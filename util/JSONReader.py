import cv2
import json
import numpy as np
import os

class JSONReader:
    def __init__(self, root_dir: str, json: str, frame_rate: int):
        """
        :param json: path to json file
        :param translation_threshold: translation threshold on OX axis
        :param rotation_threshold: rotation threshold relative to OY axis
        :param time_penalty: time penalty for human intervention
        """
        self.root_dir = root_dir
        self.json = json
        self.frame_rate = frame_rate
        self._read_json()
        self.reset()

    def _read_json(self):
        # get data from json
        with open(os.path.join(self.root_dir, self.json)) as f:
            self.data = json.load(f)

        # get cameras
        self.center_camera = self.data['cameras'][0]
        # self.left_camera = self.data['cameras'][1]
        # self.right_camera = self.data['cameras'][2]

        # read locations list
        self.locations = self.data['locations']

    def reset(self):
        video_path = os.path.join(self.root_dir, self.json[:-5] + ".mov")
        self.center_capture = cv2.VideoCapture(video_path)
        self.frame_index = 0
        self.locations_index = 0

    @staticmethod
    def get_relative_course(prev_course, crt_course):
        a = crt_course - prev_course
        a = (a + 180) % 360 - 180
        return a

    def _get_closest_location(self, tp):
        return min(self.locations, key=lambda x: abs(x['timestamp'] - tp))

    def get_next_image(self):
        """
        :param predicted_course: predicted course by nn in degrees
        :return: augmented image corresponding to predicted course or empty np.array in case the video ended
        """
        ret, frame = self.center_capture.read()
        dt = 1. / self.frame_rate

        # check if the video ended
        if not ret:
            return np.array([]), None

        # read course and speed for previous frame
        location = self._get_closest_location(1000 * dt * self.frame_index + self.locations[0]['timestamp'])
        next_location = self._get_closest_location(1000 * dt * (self.frame_index + 1) + self.locations[0]['timestamp'])

        # compute relative course and save current course
        location['rel_course'] = JSONReader.get_relative_course(location['course'], next_location['course'])
        self.prev_course = location['course']

        # increase the frame index
        self.frame_index += 1
        return frame, location


if __name__ == "__main__":
    # initialize evaluator
    # check multiple parameters like time_penalty, distance threshold and angle threshold
    # in the original paper time_penalty was 6s
    json_reader = JSONReader("./test_data/0ba94a1ed2e0449c.json")
    predicted_course = 0.0

    # get first frame of the video
    frame, _, _ = json_reader.get_next_image()

    while True:
        # make prediction based on frame
        # predicted_course = 0.01 * np.random.randn(1)[0]
        predicted_course = -0.1 * np.random.rand(1)[0]

        # get next frame corresponding to current prediction
        frame, _, _ = json_reader.get_next_image()
        if frame.size == 0:
            break
