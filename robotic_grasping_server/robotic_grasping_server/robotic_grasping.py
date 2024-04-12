import os
import time

import matplotlib.pyplot as plt
import numpy as np
import torch


from hardware.device import get_device
from inference.post_process import post_process_output
from utils.data.camera_data import CameraData
from utils.dataset_processing.grasp import detect_grasps
from utils.visualisation.plot import plot_grasp

from PIL import Image


class GraspGenerator:
    def __init__(self, saved_model_path='/robotic-grasping/trained-models/cornell-randsplit-rgbd-grconvnet3-drop1-ch32/epoch_19_iou_0.98', visualize=False):
        self.saved_model_path = saved_model_path

        self.saved_model_path = saved_model_path
        self.model = None
        self.device = None

        self.cam_data = CameraData(include_depth=True, include_rgb=True, output_size=360)

        if visualize:
            self.fig = plt.figure(figsize=(10, 10))
        else:
            self.fig = None

    def load_model(self):
        # monkey patching
        np.float = float
        print('Loading model... ')
        self.model = torch.load(self.saved_model_path, map_location=torch.device('cpu'))
        self.device = get_device(force_cpu=True)

    def generate(self, rgb, depth):
        x, depth_img, rgb_img = self.cam_data.get_data(rgb=rgb, depth=depth)

        # Predict the grasp pose using the saved model
        with torch.no_grad():
            xc = x.to(self.device)
            pred = self.model.predict(xc)

        q_img, ang_img, width_img = post_process_output(pred['pos'], pred['cos'], pred['sin'], pred['width'])
        grasps = detect_grasps(q_img, ang_img, width_img)
        for grasp in grasps:
            print(grasp.as_gr)


        if self.fig:
            plot_grasp(fig=self.fig, rgb_img=self.cam_data.get_rgb(rgb, False), grasps=grasps, save=True)
        
        return grasps

    def run_test(self):
        rgb = Image.open('shared/data/test_pair1_rgb.png')
        rgb = np.array(rgb)
        print(rgb.shape)
        depth = Image.open('shared/data/test_pair1_depth.png')

        depth = np.array(depth)
        depth = np.expand_dims(depth, axis=2)
        print(depth.shape)
        self.generate(rgb, depth)
    
    def run(self, rgb, depth):
        rgb = np.array(rgb)
        depth = np.array(depth)
        depth = np.expand_dims(depth, axis=2)
        grasps = self.generate(rgb, depth)
        grasp_dict = []
        print (grasps[0].as_gr)
        for grasp in grasps:
            r_bbox = [ [pt[0],pt[1]] for pt in grasp.as_gr.points]
            grasp_dict.append({"r_bbox": r_bbox})

        return grasp_dict

    
if __name__ == "__main__":
    np.float = float
    generator = GraspGenerator(
        saved_model_path='/robotic-grasping/trained-models/cornell-randsplit-rgbd-grconvnet3-drop1-ch32/epoch_19_iou_0.98',
        visualize=True
    )
    generator.load_model()
    generator.run_test()