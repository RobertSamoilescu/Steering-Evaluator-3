import steering
import numpy as np
import math
from pipeline.utils import *
import PIL.Image as pil


class Simulator(object):
    def __init__(self, time_penalty=6, distance_limit=1.5, angle_limit=0.15):
        """
        Constructor
        :param time_penalty: time penalty for a deviation
        :param distance_limit: distance threshold when car deviated
        :param angle_limit: angle threshold when car deviated
        """
        # save args
        self.time_penalty = time_penalty
        self.distance_limit = distance_limit
        self.angle_limit = angle_limit
        self.avg_predicted_steer = None  # predicted steer exp moving avg

        # objecta for image transformation (translation & rotation)
        self.monodepth = Monodepth("monodepth", "./pipeline")
        self.inpaint = Inapint("inpaint", "./pipeline")
        self.transf = Transformation()

        # reset simulator
        self.reset()

    def reset(self):
        self.avg_predicted_steer = None
        # simulated car distance from ground truth
        self.distance = 0.0
        # simulate care angle from ground truth
        self.angle = 0.0
        # number of human intervention, used to compute accuracy
        self.number_interventions = 0
        # variable to simulate human intervention
        self.waiting_time = -1.0

        # initialize data collection for statistics
        self.statistics = {
            "distances": [[]],
            "angles": [[]]
        }

    def _pipeline(self, img: np.array, tx: float=0.0, ry: float=0.0):
        # convension
        tx, ry = -tx, -ry
        
        # transform image to tensor
        img = pil.fromarray(img)
        img = np.asarray(img)
        timg = img.transpose(2, 0, 1)
        timg = torch.tensor(timg).unsqueeze(0).float()
        timg = F.interpolate(timg, (256, 512))

        # predict depth
        tdisp = self.monodepth.forward(timg)
        tdepth = self.monodepth.get_depth(tdisp)

        # shift image
        timg, tproj_img, tvalid_pts = self.transf.forward(timg, tdepth, tx=tx, ry=ry)

        img = timg.squeeze(0).numpy().transpose(1, 2, 0)
        img = img.astype(np.uint32)
        proj_img = tproj_img.squeeze(0).numpy().transpose(1, 2, 0)
        proj_img = proj_img.astype(np.uint32)

        # inpaint
        _, toutput = self.inpaint.forward(tproj_img.float(), tvalid_pts.float())
        toutput = F.interpolate(toutput, (256, 512))
        output = toutput.squeeze(0).numpy().transpose(1, 2, 0)
        output = np.clip(255 * output, 0, 255).astype(np.uint8)
        return output

    def _add_penalty(self):
        """
        Increases number of interventions and resets the car position and orientation
        :param time_penalty: time length for human intervention
        :return: None
        """
        self.distance = 0.0
        self.angle = 0.0
        self.number_interventions += 1

        # add new statistic list for each key
        for key in self.statistics.keys():
            self.statistics[key].append([])

    def get_statistics(self):
        """
        Dictionary contains the relative translation and rotation of the simulated
        car relative to the simulated one for each intervention
        :return: dictionary of statistics
        """
        return self.statistics

    def get_distance(self):
        """
        Returns the distance of the simulated car relative to the real cal
        :return: translation ox
        """
        return self.distance

    def get_angle(self):
        """
        Returns the orientation angle of the simulated car relative to the real cal
        :return: orientation
        """
        return self.angle

    def get_autonomy(self, total_time):
        """
        :param total_time: video duration
        :return: the autonomy of the car
        """
        intervention_time = self.number_interventions * self.time_penalty
        return 1 - intervention_time / (total_time + intervention_time)

    def get_number_interventions(self):
        """
        Return the number of times that the simulated care pass the translation or rotation threshold
        :return: number of human intervention
        """
        return self.number_interventions

    def run(self, data):
        """
        :param data: [frame, steer, velocity, delta_time, predicted_steer]
        :param time_penalty: time length for human intervention (by default 6s)
        :return: simulated frame
        """
        eps = 1e-12
        frame, steer, velocity, delta_time, predicted_steer = data

        # add statistics
        self.statistics["distances"][-1].append(self.distance)
        self.statistics["angles"][-1].append(self.angle)

        # if cumulative distance is bigger than distance limit, this means the car deviated
        # so put car back on track, increase number of intervention, add waiting time
        if abs(self.distance) > self.distance_limit or abs(self.angle) > self.angle_limit:
            self._add_penalty()
            return frame, True

        # transform current frame
        # simulated_frame = self._pipeline(img=frame, tx=self.distance, ry=self.angle)

        # compute wheel angle and radius of the real car
        steer = eps if abs(steer) < eps else steer
        wheel_angle = steering.get_delta_from_steer(steer)
        R = steering.get_radius_from_delta(wheel_angle)

        # check is the simulated car is after circle's center
        # can't be simulated
        if self.distance > R > 0 or self.distance < R < 0:
            self._add_penalty()
            return frame, True

        # estimate position of the real car
        # the condition is to avoid dividing by zero when computing Bx,
        # thus Bx1 != Bx2 always
        alpha = velocity * delta_time / R
        assert -math.pi < alpha < math.pi, "Turns bigger than 180 are not allowed"

        if abs(alpha - math.pi/2) < eps:
            alpha = math.pi/2 - eps
        x = R * (1 - np.cos(alpha))
        y = R * np.sin(alpha)

        # compute line from new position to the center of the circle
        p1 = np.array([x, y, 1])
        p2 = np.array([R, 0, 1])
        d1 = np.cross(p1, p2)
        d1 /= np.linalg.norm(d1[0:2])

        # fitler predicted steer
        # self.avg_predicted_steer = predicted_steer if self.avg_predicted_steer is None \
        #     else .5 * self.avg_predicted_steer + 0.5 * predicted_steer
        self.avg_predicted_steer = predicted_steer


        # compute wheel angle and radius for simulated car
        self.avg_predicted_steer = eps if abs(self.avg_predicted_steer) < eps else self.avg_predicted_steer
        sim_wheel_angle = steering.get_delta_from_steer(self.avg_predicted_steer)
        sim_R = steering.get_radius_from_delta(sim_wheel_angle)

        # d2 = (a, b, c), where a * x + b * y + c = 0 with the above params
        # line perpendicular to car's orientation that passes through (distance, 0)
        d2 = np.zeros((3,))
        d2[0] = np.sin(self.angle)
        d2[1] = np.cos(self.angle)
        d2[2] = -d2[0] * self.distance

        # compute circle center (Cx, Cy) with radius sim_R that passes through (distance, 0)
        # we have the system
        # d2[0] * Cx + d2[1] * Cy + d[2] = 0
        # (Cx - cumulative_distance)**2 + Cy**2 = sim_R**2
        # from the first equation, and due to the fact that maximum angle is 90, we can divide by d2[1]
        # Cy = -d[0]/d[1] * Cx - d[2]/d[1]
        # using notation with m & n
        # Cy =  m * Cx + n
        m = -d2[0] / d2[1]
        n = -d2[2] / d2[1]

        # substituting in the second equation
        # (Cx - cumulative_d)**2 + (m * Cx + n)**2 = sim_R**2
        # we obtain the quadratic equation
        # (m**2 + 1) * Cx**2 + (-2cumulative_d + 2mn) * Cx + (cumulative_d**2 + n**2 - sim_R**2) = 0
        a = m**2 + 1
        b = 2 * (m * n - self.distance)
        c = n**2 + self.distance**2 - sim_R**2

        discriminant = b**2 - 4 * a * c
        Cx1 = (-b + np.sqrt(discriminant)) / (2 * a)
        Cx2 = (-b - np.sqrt(discriminant)) / (2 * a)

        Cx = max(Cx1, Cx2) if sim_wheel_angle >= 0 else min(Cx1, Cx2)
        Cy = m * Cx + n

        # compute the new position of the car, (Bx, By)
        # we constrain (Bx, By) to be on the d1 line
        # so we compute the intersection of the line d1 with the circle ((Cx, Cy), sim_R)
        a = d1[0]**2 + d1[1]**2
        b = -2 * d1[1]**2 * Cx + 2 * d1[0] * (d1[2] + d1[1] * Cy)
        c = d1[1]**2 * Cx**2 + (d1[2] + d1[1] * Cy)**2 - d1[1]**2 * sim_R**2
        discriminant = b**2 - 4 * a * c

        # check if no solution
        if discriminant < 0:
            # this means no solution, car is behind
            self._add_penalty()
            return frame, True

        Bx1 = (-b + np.sqrt(discriminant)) / (2 * a)
        Bx2 = (-b - np.sqrt(discriminant)) / (2 * a)
        sgn_R = 1 if R >= 0 else -1
        turn_sgn = 1 if sgn_R * (x - R) < 0 else -1

        # this formula holds if the car makes a turn smaller than 90 degrees (turn_sgn = 1)
        # and bigger than 90 degrees (turn_sgn = -1)
        Bx = turn_sgn * min(turn_sgn * Bx1, turn_sgn * Bx2) \
            if sim_wheel_angle > 0 else turn_sgn * max(turn_sgn * Bx1, turn_sgn * Bx2)
        By = (-d1[0] * Bx - d1[2]) / d1[1]

        # update distance
        sgn = 1 if np.cross(np.array([Bx, By]), np.array([x, y])) >= 0 else -1
        self.distance = sgn * np.sqrt((Bx - x)**2 + (By - y)**2)

        # update cumulative angle
        # vector from the center to the new position <Bx - Cx, By - Cy>
        # vector perpendicular to the one above v = <Cy - By, Bx - Cx>
        # make it point in the positive/negative OY direction sign(Bx-Cx) * v
        # normalize v = v / norm(v)
        # take dot product with normal vector of d1 to get cos of angle, and arccos to get the angle
        # angle = angle * sign(vx), v = <vx, vy>
        v1 = np.array([Cy - By, Bx - Cx])
        sgn = turn_sgn if v1[1] >= 0 else -turn_sgn
        v1 = sgn * v1 / np.linalg.norm(v1)

        v2 = np.array([d1[0], d1[1]])
        sgn = turn_sgn if v2[1] >= 0 else -turn_sgn
        v2 = sgn * v2 / np.linalg.norm(v2)

        sgn = 1 if np.cross(v1, v2) >= 0 else -1
        self.angle = sgn * np.arccos(np.clip(np.dot(v1, v2), -1, 1))

        simulated_frame = self._pipeline(img=frame, tx=self.distance, ry=self.angle)
        return simulated_frame, False
