import numpy as np
import cv2
from util.JSONReader import *
from util.transformation import *

def read_coordinates(root_dir: str, jsons: list, verbose: bool=False):
    Ns, Es = [], []
    
    for json in jsons:
        json_reader = JSONReader(root_dir, json, frame_rate=3)
        crop = Crop()

        # get first frame of the video
        frame, _ = json_reader.get_next_image()

        while True:
            # get next frame corresponding to current prediction
            frame, location = json_reader.get_next_image()
            if frame.size == 0:
                break
                
            northing, easting = location['northing'], location['easting']
            Ns.append(northing)
            Es.append(easting)
   
    Ns = np.array(Ns)
    Es = np.array(Es)
    return Ns, Es

    
def create_map(Ns: np.array, Es: np.array, padding=200, radius=1, color=(0.5, 0.5, 0.5), verbose=False):
    # center data coordinates
    max_Ns, min_Ns = Ns.max(), Ns.min()
    max_Es, min_Es = Es.max(), Es.min()

    Ns = [int(n - min_Ns) for n in Ns]
    Es = [int(e - min_Es) for e in Es]
    
    coords = set(zip(Ns, Es))
    Ns, Es = list(zip(*coords))
    Ns, Es = np.array(Ns), np.array(Es)
    
    # create map
    height = Ns.max() + 2 * padding
    width = Es.max() + 2 * padding
    upb_map = np.zeros((height, width, 3))
    
    for i in range(Ns.size):
        upb_map = cv2.circle(
            upb_map, 
            (Es[i] + padding, Ns[i] + padding), 
            radius=radius, color=color, thickness=-1)
    
    kernel = np.array([
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1]
    ], dtype=np.uint8)
    upb_map = cv2.morphologyEx(upb_map, cv2.MORPH_CLOSE, kernel)
    
    if verbose:
        cv2.imshow("Map", upb_map)
        cv2.waitKey(0)
    
    upb_map = {
        "img": upb_map,
        "N_max": max_Ns,
        "N_min": min_Ns,
        "E_max": max_Es,
        "E_min": min_Es,
        "padding": padding
    }
    
    return upb_map

def plot_trajectory(Ns: np.array, Es: np.array, upb_map: dict, radius=1, color=(0.7, 0, 0), verbose=False):
    Ns = np.array([int(n - upb_map['N_min']) for n in Ns])
    Es = np.array([int(e - upb_map['E_min']) for e in Es])
    
    img = upb_map['img'].copy()
    padding =upb_map['padding']
    
    for i in range(Ns.size):
        img = cv2.circle(
            img,
            (Es[i] + padding, Ns[i] + padding),
            radius=radius, color=color, thickness=-1)
        
    if verbose:
        cv2.imshow("Map", img)
        cv2.waitKey(0)
    
    upb_map_clone = deepcopy(upb_map)
    upb_map_clone['img'] = img
    return upb_map_clone

def plot_point(N: float, E: float, upb_map: dict, radius=1, color=(0, 0, 0.7), verbose=False):
    N = int(N - upb_map['N_min'])
    E = int(E - upb_map['E_min'])
    
    img = upb_map['img'].copy()
    padding =upb_map['padding']
    
    img = cv2.circle(
        img,
        (E + padding, N + padding),
        radius=radius, color=color, thickness=-1)
        
    if verbose:
        cv2.imshow("Map", img)
        cv2.waitKey(0)
    
    upb_map_clone = deepcopy(upb_map)
    upb_map_clone['img'] = img
    return upb_map_clone

def rotate_map(img: np.array, course: float=0., verbose: bool=False):
    height, width = img.shape[0], img.shape[1]
    center = (height//2, width//2)
    
    M = cv2.getRotationMatrix2D(center, course, scale=1)
    img = cv2.warpAffine(img, M, (width, height))
    
    if verbose:
        cv2.imshow("Map", img)
        cv2.waitKey(0)
        
    return img

def crop_image(upb_map: dict, center: tuple, width=150, height=150, verbose=False):
    padding = upb_map['padding']
    width = min(width, 2 * padding)
    height = min(height, 2 * padding)
    
    N = int(center[0] - upb_map['N_min'] + padding)
    E = int(center[1] - upb_map['E_min'] + padding)
   
    img = upb_map['img'].copy()
    ly, ry = N - height // 2, N + height // 2
    lx, rx = E - width // 2, E + width // 2
    img = img[ly:ry, lx:rx]
    
    if verbose:
        cv2.imshow("Map", img)
        cv2.waitKey(0)
    
    return img


def get_rotation_matrix(course):
    rad_course = -np.deg2rad(course)
    R = np.array([
        [np.cos(rad_course), -np.sin(rad_course), 0],
        [np.sin(rad_course), np.cos(rad_course), 0],
        [0, 0, 1]
    ])
    return R