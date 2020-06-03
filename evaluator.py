import cv2
import json
import math
import numpy as np
import simulator
import steering
import PIL.Image as pil
import matplotlib.pyplot as plt
from tqdm import tqdm

class AugmentationEvaluator:
    def __init__(self, json, translation_threshold=1.5, rotation_threshold=0.2, time_penalty=6, frame_rate=3):
        """
        :param json: path to json file
        :param translation_threshold: translation threshold on OX axis
        :param rotation_threshold: rotation threshold relative to OY axis
        :param time_penalty: time penalty for human intervention
        """
        self.json = json
        self.translation_threshold = translation_threshold
        self.rotation_threshold = rotation_threshold
        self.time_penalty = time_penalty
        self.frame_rate = frame_rate

        self._read_json()
        self.reset()

        # initialize simulator
        self.simulator = simulator.Simulator(
            time_penalty    =self.time_penalty,
            distance_limit  =self.translation_threshold,
            angle_limit     =self.rotation_threshold
        )

        # set transformation matrix
        self.T = np.eye(3)

    def _read_json(self):
        # get data from json
        with open(self.json) as f:
            self.data = json.load(f)

        self.center_camera  = self.data['cameras'][0]
        self.locations      = self.data['locations']

    def reset(self):
        self.center_capture = cv2.VideoCapture(self.json[:-5] + ".mov")
        self.locations_index = 0
        self.trajectories = {
            "real_trajectory": [],
            "simulated_trajectory": []
        }
        self.interv_points = {
            "northing": [],
            "easting": [],
        }

    @staticmethod
    def get_steer(course, speed, dt, eps=1e-12):
        sgn         = np.sign(course)
        dist        = speed * dt
        R           = dist / (np.deg2rad(abs(course)) + eps)
        delta, _, _ = steering.get_delta_from_radius(R)
        steer       = sgn * steering.get_steer_from_delta(delta)
        return steer

    @staticmethod
    def get_course(steer, speed, dt):
        dist        = speed * dt
        delta       = steering.get_delta_from_steer(steer)
        R           = steering.get_radius_from_delta(delta)
        rad_course  = dist / R
        course      = np.rad2deg(rad_course)
        return course

    @staticmethod
    def get_relative_course(prev_course, crt_course):
        a = crt_course - prev_course
        a = (a + 180) % 360 - 180
        return a

    @staticmethod
    def get_rotation_matrix(course):
        rad_course = -np.deg2rad(course)
        R   = np.array([
            [np.cos(rad_course), -np.sin(rad_course), 0],
            [np.sin(rad_course), np.cos(rad_course), 0],
            [0, 0, 1]
        ])
        return R

    @staticmethod
    def get_translation_matrix(position):
        T       = np.eye(3)
        T[0, 2] = position[0]
        T[1, 2] = position[1]
        return T

    def _get_closest_location(self, tp):
        return min(self.locations, key=lambda x: abs(x['timestamp'] - tp))

    def get_statistics(self):
        return self.simulator.get_statistics()

    def get_trajectories(self):
        return self.trajectories

    def get_init_state(self):
        # capture first frame
        ret, self.prev_frame  = self.center_capture.read()
        ret, self.frame       = self.center_capture.read()
        self.frame_index      = 1

        self.prev_frame = cv2.resize(self.prev_frame[:320, ...], (512, 256))
        self.frame      = cv2.resize(self.frame[:320, ...], (512, 256))

        self.prev_sim_frame = self.prev_frame.copy()
        self.sim_frame      = self.frame.copy()

        # read course and speed for previous frame
        dt = 1 / self.frame_rate
        location        = self._get_closest_location(self.locations[0]['timestamp'])
        next_location   = self._get_closest_location(1000 * dt + self.locations[0]['timestamp'])
        speed           = next_location['speed']

        return self.prev_sim_frame[..., ::-1], self.sim_frame[..., ::-1], speed


    def get_next_state(self, predicted_course=0.):
        """
        :param predicted_course: predicted course by nn in degrees
        :return: augmented image corresponding to predicted course or empty np.array in case the video ended
        """
        ret, next_frame  = self.center_capture.read()
        dt               = 1 / self.frame_rate

        # check if the video ended
        if not ret:
            return np.array([]), np.array([]), None, None

        # crop car from the frame
        next_frame = next_frame[:320, :, :]
        next_frame = cv2.resize(next_frame, (512, 256))

        # read course and speed for previous frame
        location        = self._get_closest_location(1000 * dt * self.frame_index + self.locations[0]['timestamp'])
        next_location   = self._get_closest_location(1000 * dt * (self.frame_index + 1) + self.locations[0]['timestamp'])
        
        course      = location['course']
        next_course = next_location['course']
        speed       = location['speed']

        # compute relative course between the frame and next(frame)
        rel_course = AugmentationEvaluator.get_relative_course(course, next_course)

        # compute steering from course, speed, dt
        steer           = AugmentationEvaluator.get_steer(rel_course, speed, dt)
        predicted_steer = AugmentationEvaluator.get_steer(predicted_course, speed, dt)

        # run augmentator
        args = [next_frame, steer, speed, dt, predicted_steer]
        next_sim_frame, interv  = self.simulator.run(args)

        # trajectory related stuff
        # get real position
        real_position = np.array([
            location['easting'] - self.locations[0]['easting'],
            location['northing'] - self.locations[0]['northing']
        ])

        # compute simulated car position from relative one
        relative_position   = np.array([self.simulator.get_distance(), 0, 1])
        R                   = AugmentationEvaluator.get_rotation_matrix(course)
        simulated_poisition = real_position + np.dot(R, relative_position)[:-1]

        # append coordinates to trajectory dictionary
        self.trajectories["real_trajectory"].append(real_position)
        self.trajectories["simulated_trajectory"].append(simulated_poisition)
        
        # update ground truth frames
        self.prev_frame   = self.frame
        self.frame        = next_frame
        self.frame_index += 1

        # append intervention points
        if interv:
            self.interv_points["easting"].append(simulated_poisition[0] + self.locations[0]['easting'])
            self.interv_points["northing"].append(simulated_poisition[1] + self.locations[1]['northing'])

            self.sim_prev_frame  = self.prev_frame
            self.sim_frame       = self.frame
            return self.sim_prev_frame[..., ::-1], self.sim_frame[..., ::-1], speed, None

        # increase the frame index
        self.sim_prev_frame = self.sim_frame
        self.sim_frame      = next_sim_frame

        return self.sim_prev_frame[..., ::-1], self.sim_frame[..., ::-1], next_location['speed'], rel_course

    def get_autonomy(self):
        total_time = (self.data['endTime'] - self.data['startTime']) / 1000
        return self.simulator.get_autonomy(
            total_time=total_time
        )

    def get_video_length(self):
        return (self.data['endTime'] - self.data['startTime']) / 1000

    def get_number_interventions(self):
        return self.simulator.get_number_interventions()

    def get_intev_points(self):
        return self.interv_points



if __name__ == "__main__":
    # initialize evaluator
    # check multiple parameters like time_penalty, distance threshold and angle threshold
    # in the original paper time_penalty was 6s
    augm = AugmentationEvaluator("./test_data/1c820d64b4af4c85.json", time_penalty=6)
    predicted_course = 0.0

    # get first frame of the video
    frame = augm.get_next_image()

    # while True:
    for i in tqdm(range(100)):
        # make prediction based on frame
        # predicted_course = 0.01 * np.random.randn(1)[0]
        predicted_course = -0.1 * np.random.rand(1)[0]

        # get next frame corresponding to current prediction
        frame, _, _ = augm.get_next_image(predicted_course)
        if frame.size == 0:
            break

        # show augmented frmae
        cv2.imshow("Augmented frame", frame[:, :, ::-1])
        cv2.waitKey(33)

    # print autonomy and number of interventions
    print("Autonomy:", augm.get_autonomy())
    print("#Interventions:", augm.get_number_interventions())
