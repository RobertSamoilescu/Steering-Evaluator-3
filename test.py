from models.nvidia_full import *
from models.resnet_full import *

from evaluator import *
from util.JSONReader import *
from util.io import * 
from util.maps import *
from util.transformation import *
from util.vis import *
from util.plots import *

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import os
import time
import argparse
import itertools
import pandas as pd
import pickle as pkl
from tqdm import tqdm
import PIL.Image as pil
import scipy.stats as stats

parser = argparse.ArgumentParser()
parser.add_argument('--begin', type=int, help="starting video index", default=0)
parser.add_argument('--end', type=int, help="ending video index", default=81)
parser.add_argument('--model', type=str, help='nvidia or resnet', default='nvidia')
parser.add_argument('--width', type=int, help='input image width', default=256)
parser.add_argument('--height', type=int, help='input image height', default=128)
parser.add_argument('--log_dir', type=str, help='logging dir', default='logs')
parser.add_argument('--vis_dir', type=str, help='checkpoints dir', default='snapshots')
parser.add_argument('--load_model', type=str, help='name of the model', default='default')
parser.add_argument('--use_rgb', action='store_true')
parser.add_argument('--use_speed', action='store_true')
parser.add_argument('--use_stacked', action='store_true')
parser.add_argument('--use_disp', action='store_true')
parser.add_argument('--use_depth', action='store_true')
parser.add_argument('--use_flow', action='store_true')
parser.add_argument('--use_balance', action='store_true')
parser.add_argument('--split_path', type=str, help='path to the test scenes file (test_scenes.txt)')
parser.add_argument('--data_path', type=str, help='path to the raw dataset directory (.mov & .json)')
args = parser.parse_args()

# set seed
torch.manual_seed(0)

# get available divice
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# define model
nbins = 401
experiment = ''

if args.model == "nvidia":
	model = NVIDIA(
	    no_outputs=nbins,
	    use_rgb=args.use_rgb, 
	    use_stacked=args.use_stacked, 
	    use_disp=args.use_disp, 
	    use_depth=args.use_depth, 
	    use_flow=args.use_flow, 
	    use_speed=args.use_speed
	).to(device)
	experiment += "nvidia"
else:
	model = RESNET(
	    no_outputs=nbins,
	    use_rgb=args.use_rgb, 
	    use_stacked=args.use_stacked, 
	    use_disp=args.use_disp, 
	    use_depth=args.use_depth, 
	    use_flow=args.use_flow, 
	    use_speed=args.use_speed
	).to(device)
	experiment += "resnet"

if args.use_rgb:
    experiment += "_rgb"
if args.use_speed:
    experiment += "_speed"
if args.use_stacked:
    experiment += "_stacked"
if args.use_depth:
    experiment += "_depth"
if args.use_disp:
    experiment += "_disp"
if args.use_flow:
    experiment += "_flow"
if args.use_balance:
    experiment += "_balance"

model = model.to(device)

# load model
path = os.path.join("ckpts", experiment, "ckpts", "default.pth")
load_ckpt(path, [('model', model)])
model.eval()


# construct logs dirs
if not os.path.exists(args.log_dir):
    os.mkdir(args.log_dir)

# construct gaussian distribution
def gaussian_distribution(mean=200.0, std=5, eps = 1e-6):
    x = np.arange(401)
    mean = np.clip(mean, 0, 400)

    # construct pdf
    pdf = np.exp(-0.5 * ((x - mean) / std)**2)
    pmf = pdf / (pdf.sum() + eps)
    return pmf


# processing frame
def normalize(img):
    return img / 255.

def unnormalize(img):
    return (img * 255).astype(np.uint8)

def process_frame(frame):
    # crop & normalize
    frame = normalize(frame)

    # transpose and change shape
    frame = np.transpose(frame, (2, 0, 1))
    frame = torch.tensor(frame).unsqueeze(0).float().cuda()
    frame = F.interpolate(frame, (args.height, args.width))
    return frame

# output smoothing
def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'same') / w

def get_course(output, smooth=True):
    if smooth:
        output = moving_average(output, 5)
        output /= output.sum()

    index = np.argmax(output).item()
    return (index - 200) / 10, output


def make_prediction(prev_frame, frame, speed):
    # preprocess data
    prev_frame = process_frame(prev_frame)
    frame = process_frame(frame)
    speed = torch.tensor([[speed]]).to(device)

    # make first prediction
    with torch.no_grad():
        # construct data package
        data = {
            "prev_img": prev_frame.to(device),
            "img": frame.to(device), 
            "speed": speed.to(device),
        }

        # make prediction based on frame
        toutput, disp, depth, flow = model(data)

        # process the logits and get the course as the argmax
        toutput = F.softmax(toutput, dim=1)
        output = toutput.reshape(toutput.shape[1]).cpu().numpy()
        course, output = get_course(output)
        toutput = torch.tensor(output.reshape(*toutput.shape)).to(device)

    return course, toutput, disp, depth, flow

# close loop evaluation
def test_video(path, time_penalty=6, translation_threshold=1.5, rotation_threshold=0.2, verbose=True):
    video_name = path.split("/")[-1][:-5]

    log_path = os.path.join(args.log_dir, experiment, video_name)
    if not os.path.exists(log_path):
        os.makedirs(log_path)
        os.makedirs(os.path.join(log_path, "imgs"))


    # buffers to store evaluation details
    real_courses = []
    predicted_courses = []
    predicted_course_distributions = []
    real_course_distributions = []
    
    # initialize evaluator
    # check multiple parameters like time_penalty, distance threshold and angle threshold
    # in the original paper time_penalty was 6s
    augm = AugmentationEvaluator(
        path,
        time_penalty=time_penalty,
        translation_threshold=translation_threshold,
        rotation_threshold=rotation_threshold
    )

    # get first two frames of the video and make a prediction
    prev_frame, frame, speed  = augm.get_init_state()
    pred_course, toutput, disp, depth, flow = make_prediction(prev_frame, frame, speed)
   
    with torch.no_grad():
        for idx in tqdm(itertools.count()):
            prev_frame, frame, speed, real_course = augm.get_next_state(pred_course)
            print(frame.shape)
            # video is done
            if frame.size == 0:
                break

            if real_course is not None:
                # distribution for the ground truth course
                real_course_distribution = gaussian_distribution(10 * (real_course + 20))
                real_course_distributions.append(real_course_distribution)

                # distribution for the predicted course
                predicted_course_distribution = gaussian_distribution(10 * (pred_course + 20))
                predicted_course_distributions.append(predicted_course_distribution)

                output = toutput.reshape(toutput.shape[1]).cpu().numpy()
                predicted_courses.append(output)
                real_courses.append(real_course)

                # construct full image
                real_course_distribution = torch.tensor(real_course_distribution).unsqueeze(0)
                imgs_path = os.path.join(args.log_dir, experiment, video_name, "imgs", str(idx).zfill(5) + ".png")
                full_img = visualisation(process_frame(prev_frame), disp, depth, flow,
                    real_course_distribution, toutput, 1, imgs_path).astype(np.uint8)

                # print and save courses
                if verbose:
                    print("Predicted Course: %.2f, Real Course: %.2f, Speed: %.2f" % (course, real_course, speed))
                    cv2.imshow("State", full_img[...,::-1])
                    cv2.waitKey(100)

            # make the next prediction
            pred_course, toutput, disp, depth, flow = make_prediction(prev_frame, frame, speed)
            
            # import matplotlib.pyplot as plt
            # img = np.concatenate((prev_frame, frame), axis=1)
            # plt.imshow(img)
            # plt.show()

    # get some statistics [mean distance till an intervention, mean angle till an intervention]
    statistics = augm.get_statistics()
    absolute_mean_distance, absolute_mean_angle, plot_dist_ang = plot_statistics(statistics)
    
    # save stats plot
    stats_path = os.path.join(args.log_dir, experiment, video_name, "stats.png")
    cv2.imwrite(stats_path, plot_dist_ang[...,::-1])

    # get car's trajectory
    trajectories = augm.get_trajectories()
    plot_traj = plot_trajectories(trajectories)

    # save trajectory
    traj_path = os.path.join(args.log_dir, experiment, video_name, "traj.png")
    cv2.imwrite(traj_path, plot_traj[...,::-1])

    # save all data
    data = {
        "real_courses": real_courses,
        "predicted_courses": predicted_courses, 
        "predicted_course_distributions": predicted_course_distributions,
        "real_course_distributions": real_course_distributions,
        "autonomy": augm.get_autonomy(),
        "num_interventions": augm.get_number_interventions(),
        "video_length": augm.get_video_length(),
        "intervention_coords": augm.get_intev_points(),
        "statistics": statistics,
        "trajectories": trajectories
    }

    data_path = os.path.join(args.log_dir, experiment, video_name, "data.pkl")
    with open(data_path, 'wb') as fout:
        pkl.dump(data, fout)


if __name__ == "__main__":
    with open(args.split_path, 'rt') as fin:
        files = fin.read()

    files = files.split("\n")
    files = [os.path.join(args.data_path, file + ".json")  for file in files]
    files = files[args.begin:args.end]

    # test video
    for file in files:
        test_video(file, verbose=False)
