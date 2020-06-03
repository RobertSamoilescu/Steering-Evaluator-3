import numpy as np
import PIL.Image as pil
import matplotlib
import matplotlib.pyplot as plt
# matplotlib.use('Agg')

import torch
import torch.nn.functional as F
from .vis_flow import *
import cv2

def fig2data(fig):
    """
    @brief Convert a Matplotlib figure to a 4D numpy array with RGBA channels and return it
    @param fig a matplotlib figure
    @return a numpy 3D array of RGBA values
    """
    # draw the renderer
    fig.canvas.draw()
 
    # Get the RGBA buffer from the figure
    w,h = fig.canvas.get_width_height()
    buf = np.fromstring (fig.canvas.tostring_rgb(), dtype=np.uint8)
    buf.shape = (w, h, 3) 
    return buf
 
def fig2img(fig, width, height):
    """
    @brief Convert a Matplotlib figure to a PIL Image in RGBA format and return it
    @param fig a matplotlib figure
    @return a Python Imaging Library ( PIL ) image
    """
    # put the figure pixmap into a numpy array
    buf = fig2data(fig)
    w, h, d = buf.shape
    img = pil.frombytes("RGB", (w, h), buf.tostring())
    img = img.resize((width, height))
    return img


HEIGHT = 256
WIDTH = 512

def plot_img(img: torch.tensor):
    plot = 255 * img.detach().cpu().numpy().transpose(1, 2, 0)
    plot = cv2.resize(plot, (WIDTH, HEIGHT))
    return plot

def plot_disp(disp: torch.tensor):
    figure = plt.figure(figsize=(10, 5))
    np_disp = disp.detach().cpu().squeeze(0).numpy()
    vmax = np.percentile(np_disp, 95)
    plt.imshow(np_disp, cmap="magma", vmax=vmax)
    plt.axis("off")
    plot = np.asarray(fig2img(figure, height=HEIGHT, width=WIDTH))
    plt.close(figure)
    return plot

def plot_depth(depth: torch.tensor):
    figure = plt.figure(figsize=(10, 5))
    np_depth = depth.detach().cpu().squeeze(0).numpy()
    vmax = np.percentile(np_depth, 95)
    plt.imshow(np_depth, cmap="gray_r")
    plt.axis("off")
    plot = np.asarray(fig2img(figure, height=HEIGHT, width=WIDTH))
    plt.close(figure)
    return plot

def plot_flow(flow: torch.tensor):
    figure = plt.figure(figsize=(10, 5))
    np_flow = flow.detach().cpu().numpy().transpose(1, 2, 0)
    np_flow = flow_to_color(np_flow)
    plt.axis("off")
    plt.imshow(np_flow)
    plot = np.asarray(fig2img(figure, height=HEIGHT, width=WIDTH))
    plt.close(figure)
    return plot

def plot_distr(softmax_output, course):
    figure = plt.figure()
    so = softmax_output.cpu().numpy()
    c = course.cpu().numpy()
    plt.plot(np.arange(so.shape[0]), so, label="pred")
    plt.plot(np.arange(c.shape[0]), c, label="gt")
    plt.axvline(x=200, color='red', linestyle='--')
    plt.legend()
    plot = np.asarray(fig2img(figure, height=HEIGHT, width=WIDTH))
    plt.close(figure)
    return plot


def visualisation(img, disp, depth, flow, course, softmax_output, num_vis, path):
    figs = []

    for j in range(num_vis):
        fig = []

        # plot image 
        np_img = plot_img(img[j])
        fig.append(np_img)

        # plot disp
        if disp is not None:
            np_disp = plot_disp(disp[j])
            fig.append(np_disp)

        # plot depth:
        if depth is not None:
            np_depth = plot_depth(depth[j])
            fig.append(np_depth)

        # plot flow
        if flow is not None:
            np_flow = plot_flow(flow[j])
            fig.append(np_flow)

        np_dist = plot_distr(softmax_output[j], course[j])
        fig.append(np_dist)

        fig = np.concatenate(fig, axis=1)
        figs.append(fig)

    snapshot = np.concatenate(figs, axis=0)
    pil_snapshots = pil.fromarray(snapshot.astype(np.uint8))
    pil_snapshots.save(path)
    return snapshot
