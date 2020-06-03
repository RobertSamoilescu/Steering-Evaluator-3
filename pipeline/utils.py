import numpy as np
import PIL.Image as pil
import os
import matplotlib.pyplot as plt

# monodepth
from .monodepth.inverse_warp import *
from .monodepth.depth_decoder import *
from .monodepth.layers import *
from .monodepth.pose_cnn import *
from .monodepth.pose_decoder import *
from .monodepth.resnet_encoder import *

# inpainting
from .inpaint.net import *
from .inpaint.io import *
from .inpaint.image import *

# flow
import pipeline.flow.encoder as ENC
import pipeline.flow.decoder as DEC
import pipeline.flow.layers as LYR

from .warp import *
from .vis import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Monodepth(object):
    def __init__(self, model_name: str="monodepth", root_dir: str="./pipeline"):
        self.model_name = model_name
        self.root_dir = root_dir
        self.intrinsic = np.array([
            [0.61, 0, 0.5],   # width
            [0, 1.22, 0.5],   # height
            [0, 0, 1]],
        dtype=np.float32)
        self.CAM_HEIGHT = 1.5
        
        encoder_path = os.path.join(root_dir, "models", model_name, "encoder.pth")
        depth_decoder_path = os.path.join(root_dir, "models", model_name, "depth.pth")

        # LOADING PRETRAINED MODEL
        self.encoder = ResnetEncoder(18, False)
        self.encoder = self.encoder.to(device)
        
        self.depth_decoder = DepthDecoder(num_ch_enc=self.encoder.num_ch_enc, scales=range(4))
        self.depth_decoder = self.depth_decoder.to(device)
        
        loaded_dict_enc = torch.load(encoder_path, map_location='cpu')
        filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in self.encoder.state_dict()}
        self.encoder.load_state_dict(filtered_dict_enc)

        loaded_dict = torch.load(depth_decoder_path, map_location='cpu')
        self.depth_decoder.load_state_dict(loaded_dict)

        self.encoder.eval()
        self.depth_decoder.eval();
        
        
    def forward(self, img: torch.tensor):
        """
        @param img: input image (RGB), [B, 3, H, W]
        :returns depth map[B, 1, H, W]
        """
        # normalize
        if img.max() > 1:
            img = img / 255.
        
        img = img.to(device)
        
        with torch.no_grad():
            features = self.encoder(img)
            outputs = self.depth_decoder(features)

        disp = outputs[("disp", 0)].cpu()
        return disp
    
    def get_depth(self, disp: torch.tensor):
        """
        @param disp: disparity map, [B, 1, H, W]
        :returns depth map
        """
        scaled_disp, depth_pred = disp_to_depth(disp, 0.1, 100.0)
        factor = self.get_factor(depth_pred)
        depth_pred *= factor
        depth_pred = torch.clamp(depth_pred, min=1e-3, max=80)
        return depth_pred
    
    def get_factor(self, depth: torch.tensor):
        """
        @param disp: depth map, [B, 1, H, W]
        :returns depth factor
        """
        batch_size, _, height, width = depth.shape
        
        # construct intrinsic camera matrix
        intrinsic = self.intrinsic.copy()
        intrinsic[0, :] *= width
        intrinsic[1, :] *= height
        intrinsic = torch.tensor(intrinsic).repeat(batch_size, 1, 1)

        # get camera coordinates
        cam_coords = pixel2cam(depth.squeeze(1), intrinsic.inverse())
        
        # get some samples from the ground, center of the image
        samples = cam_coords[:, 1, height-10:height, width//2 - 50:width//2 + 50]
        samples = samples.reshape(samples.shape[0], -1)
        
        # get the median
        median = samples.median(1)[0]
  
        # get depth factor
        factor = self.CAM_HEIGHT / median
        return factor.reshape(factor.shape, 1, 1, 1)


class Flow(object):
    def __init__(self, root_dir: str="./pipeline"):
        self.root_dir = root_dir
        self.intrinsic = np.array([
            [0.61, 0, 0.5, 0],   # width
            [0, 1.22, 0.5, 0],   # height
            [0, 0, 1, 0],
            [0, 0, 0, 1]],
        dtype=np.float32)

        self.WIDTH = 512
        self.HEIGHT = 256
        self.scale = np.array([
            [self.WIDTH, 0, 0, 0],
            [0, self.HEIGHT, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        encoder_path = os.path.join(root_dir, "models", "monodepth", "encoder.pth")
        depth_decoder_path = os.path.join(root_dir, "models", "monodepth", "depth.pth")
        pose_encoder_path = os.path.join(root_dir, "models", "monodepth", "pose_encoder.pth")
        pose_decoder_path = os.path.join(root_dir, "models", "monodepth", "pose.pth", )
        dflow_path = os.path.join(root_dir, "models", "flow", "default.pth")

        # LOADING PRETRAINED DEPTH MODEL
        self.encoder = ResnetEncoder(18, False)
        self.encoder = self.encoder.to(self.device)
        
        self.depth_decoder = DepthDecoder(num_ch_enc=self.encoder.num_ch_enc, scales=range(4))
        self.depth_decoder = self.depth_decoder.to(self.device)

        loaded_dict_enc = torch.load(encoder_path, map_location='cpu')
        filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in self.encoder.state_dict()}
        self.encoder.load_state_dict(filtered_dict_enc)

        loaded_dict = torch.load(depth_decoder_path, map_location='cpu')
        self.depth_decoder.load_state_dict(loaded_dict)

        # LOADING PRETRAINED POSE MODEL        
        self.pose_encoder = ResnetEncoder(18, False, num_input_images=2)
        self.pose_encoder = self.pose_encoder.to(self.device)

        self.pose_decoder = PoseDecoder(
            self.pose_encoder.num_ch_enc,
            num_input_features=1,
            num_frames_to_predict_for=2
        )
        self.pose_decoder = self.pose_decoder.to(self.device)

        # LOADING PRETRAINDE DYNAMIC FLOW MODEL
        self.dflow_encoder = ENC.ResnetEncoder(
            num_layers=18,
            pretrained=True,
            num_input_images=2
        )
        self.dflow_encoder.encoder.conv1 = nn.Conv2d(12, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.dflow_encoder = self.dflow_encoder.to(self.device)

        self.dflow_decoder = DEC.FlowDecoder(
            num_ch_enc=self.dflow_encoder.num_ch_enc,
            scales=range(4),
            num_output_channels=2,
            use_skips=True
        ).to(self.device)

        # declare ssim
        self.ssim = LYR.SSIM().to(self.device)

        loaded_dict = torch.load(pose_encoder_path, map_location='cpu')
        self.pose_encoder.load_state_dict(loaded_dict)
        
        loaded_dict = torch.load(pose_decoder_path, map_location='cpu')
        self.pose_decoder.load_state_dict(loaded_dict)

        loaded_dict = torch.load(dflow_path, map_location='cpu')
        self.dflow_encoder.load_state_dict(loaded_dict['encoder'])
        self.dflow_decoder.load_state_dict(loaded_dict['decoder'])

        self.encoder.eval()
        self.depth_decoder.eval()
        self.pose_decoder.eval()
        self.dflow_encoder.eval()
        self.dflow_decoder.eval()
        
    def get_pix_coords(self, prev_img: torch.tensor, img: torch.tensor, batch_size=1):
        # get disp
        with torch.no_grad():
            depth_features = self.encoder(img)
            depth_output = self.depth_decoder(depth_features)
            disp = depth_output[("disp", 0)]
            _, depth = disp_to_depth(disp, 0.1, 100.0)

        # get pose
        with torch.no_grad():
            input = torch.cat([prev_img, img], dim=1)
            pose_features = self.pose_encoder(input)
            pose_output = self.pose_decoder([pose_features])
            axisangle, translation = pose_output

        # get transformation between frames
        T = transformation_from_parameters(
            axisangle[:, 0], translation[:, 0], invert=True)
        K = torch.tensor(self.scale @ self.intrinsic)\
            .unsqueeze(0).repeat(batch_size, 1, 1).float()\
            .to(self.device)
        inv_K = K.inverse().float().to(self.device)


        # get camera coords
        backproject_depth = BackprojectDepth(batch_size, self.HEIGHT, self.WIDTH).to(self.device)
        cam_points = backproject_depth(depth, inv_K).float()
        
        # get pixel coords
        project_3d = Project3D(batch_size, self.HEIGHT, self.WIDTH)
        pix_coords = project_3d(cam_points, K, T).float()

        return pix_coords

    def get_rigid_flow(self, pix_coords, batch_size=1):
        # mesh grid 
        W = self.WIDTH
        H = self.HEIGHT
        B = batch_size

        xx = torch.arange(0, W).view(1,-1).repeat(H,1)
        yy = torch.arange(0, H).view(-1,1).repeat(1,W)
        xx = xx.view(1,H,W,1).repeat(B,1,1,1)
        yy = yy.view(1,H,W,1).repeat(B,1,1,1)
        grid = torch.cat((xx,yy),3).float()
        grid = grid.to(self.device)

        return pix_coords - grid


    def get_flow(self, prev_img: torch.tensor, img: torch.tensor, batch_size=1):
        # get rigid flow
        pix_coords = self.get_pix_coords(prev_img, img, batch_size)
        rflow = self.get_rigid_flow(pix_coords, batch_size)
        rflow = rflow.transpose(2, 3).transpose(1, 2)

        # wrap frame using rigid flow
        wimg = warp(prev_img, rflow)

        # comput error map
        ssim_loss = self.ssim(wimg, img).mean(1, True)
        l1_loss = torch.abs(wimg - img).mean(1, True)
        err_map = 0.85 * ssim_loss + 0.15 * l1_loss

        # get dynamic flow correction
        input = torch.cat([prev_img, img, wimg], dim=1)
        with torch.no_grad():
            enc_ouput = self.dflow_encoder(input, rflow, err_map)
            dec_output = self.dflow_decoder(enc_ouput)
            dflow = dec_output[('flow', 0)]

        flow = dflow + rflow
        return flow
        

class Inapint(object):
    def __init__(self, model_name: str="inpaint", root_dir: str="./pipeline"):
        self.model_name = model_name
        self.root_dir = root_dir
        self.inpaint = PConvUNet().cuda()
        start_iter = load_ckpt(
            os.path.join(root_dir, "models", model_name, "unet.pth"), 
            [('model', self.inpaint)],
            None
        )
        self.inpaint = self.inpaint.to(device)
        self.inpaint.eval()
        
    def forward(self, img: torch.tensor, mask: torch.tensor):
        """
        @param img: RGB image, [B, 3, H, W]
        @param mask: image mask, [B, 3, H, W]
        :returns torch.tensor [B, 3, H, W]
        """
        if img.max() > 1:
            img /= 255.
        
        img = normalize(img)
        
        # send to gpu
        img = img.to(device)
        mask = mask.to(device)
        
        with torch.no_grad():
            output, _ = self.inpaint(img, mask)
        
        output = output.cpu()
        output = unnormalize(output)
        
        img = img.cpu()
        img = unnormalize(img)
        mask = mask.cpu()
        
        return output, mask * img + (1 - mask) * output


class Transformation(object):
    def __init__(self):
        self.intrinsic = torch.tensor([
            [0.61, 0, 0.5, 0],   # width
            [0, 1.22, 0.5, 0],   # height
            [0, 0, 1, 0],
            [0, 0, 0, 1]], dtype=torch.float64).unsqueeze(0)
        # self.extrinsic = None 
        self. extrinsic = torch.tensor([
        	[1,  0, 0, 0.00],
        	[0, -1, 0, 1.65],
        	[0,  0, 1, 1.54],
        	[0 , 0, 0, 1.00]], dtype=torch.float64).unsqueeze(0).inverse()
    
    def forward(self, img: torch.tensor, depth: torch.tensor, tx: float = 0.0, ry: float = 0.0):
        """
        @param img: rgb image, [B, 3, H, W]
        @param depth: depth map, [B, 1, H, W]
        @param tx: translation Ox [m]
        @param ry: rotation Oy [rad]
        :returns projected image, mask of valid points
        """
        # casting
        img = img.double()
        depth = depth.double()
        
        batch_size, _, height, width = img.shape
        
        # modify intrinsic
        _, _, height, width = img.shape
        intrinsic = self.intrinsic.clone()
        intrinsic[:, 0] *= width
        intrinsic[:, 1] *= height
        
        # add pose
        pose = torch.zeros((batch_size, 6), dtype=torch.float64)
        pose[:, 0], pose[:, 4] = tx, ry
    
        # down sample
        down = nn.AvgPool2d(2)
        down_img = down(img)
        down_depth= down(depth)

        S = torch.tensor([
            [0.5, 0.0, 0.0, 0.0],
            [0.0, 0.5, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ], dtype=torch.double)
        intrinsic = torch.matmul(S, intrinsic)
    
        # apply perspective transformation
        projected_img, valid_points = forward_warp(
            img=down_img,
            depth=down_depth.squeeze(1), 
            pose=pose, 
            intrinsics=intrinsic[:, :3, :3].repeat(1, 1, 1),
            extrinsics=self.extrinsic
        )
        
        return down_img, projected_img, valid_points.repeat(1, 3, 1, 1)