import cv2
import numpy as np

import os
import shutil
import sys

import airsimdroneracingvae as airsim
# print(os.path.abspath(airsim.__file__))
from airsimdroneracingvae.types import Pose, Vector3r, Quaternionr
import airsimdroneracingvae.types
import airsimdroneracingvae.utils
import time

# import utils
curr_dir = os.path.dirname(os.path.abspath(__file__))
import_path = os.path.join(curr_dir, '..', '..')
sys.path.insert(0, import_path)
from racing_utils.paths import *
from racing_utils.geom_utils import randomGatePose, randomQuadPose

# GATE_YAW_RANGE = [-1, 1]  # world theta gate -- this range is btw -pi and +pi
GATE_YAW_RANGE = [-np.pi, np.pi]  # world theta gate
UAV_X_RANGE = [-30, 30] # world x quad
UAV_Y_RANGE = [-30, 30] # world y quad
UAV_Z_RANGE = [-2, -3] # world z quad

UAV_YAW_RANGE = [-np.pi, np.pi]  #[-eps, eps] [-np.pi/4, np.pi/4]
eps = np.pi/10.0  # 18 degrees
UAV_PITCH_RANGE = [-eps, eps]  #[-np.pi/4, np.pi/4]
UAV_ROLL_RANGE = [-eps, eps]  #[-np.pi/4, np.pi/4]

R_RANGE = [0.1, 20]  # in meters
correction = 0.85
CAM_FOV = 90.0*correction  # in degrees -- needs to be a bit smaller than 90 in fact because of cone vs. square


class PoseSampler:
    def __init__(self, num_samples, dataset_path, with_gate=True):
        self.num_gates = 0
        self.num_samples = num_samples
        self.base_path = dataset_path
        self.csv_path = os.path.join(self.base_path, 'gate_training_data.csv')
        self.curr_idx = 10000
        self.with_gate = with_gate
        self.client = airsim.MultirotorClient()

        self.client.confirmConnection()
        self.client.simLoadLevel('Qualifier_Tier_3')
        time.sleep(4)
        self.client = airsim.MultirotorClient()
        self.configureEnvironment()

    def update(self):
        '''
        convetion of names:
        p_a_b: pose of frame b relative to frame a
        t_a_b: translation vector from a to b
        q_a_b: rotation quaternion from a to b
        o: origin
        b: UAV body frame
        g: gate frame
        '''
        # create and set pose for the quad
        p_o_b, phi_base = randomQuadPose(UAV_X_RANGE, UAV_Y_RANGE, UAV_Z_RANGE, UAV_YAW_RANGE, UAV_PITCH_RANGE, UAV_ROLL_RANGE)
        self.client.simSetVehiclePose(p_o_b, True) # set pose of the drone (b) relative to the origin (o).
        
        # create and set gate pose relative to the quad
        p_o_g, r, theta, psi, phi_rel = randomGatePose(p_o_b, phi_base, R_RANGE, CAM_FOV, correction)
        # self.client.simSetObjectPose(self.tgt_name, p_o_g_new, True)
        
        if self.with_gate:
            self.num_gates += 1
            self.tgt_name = self.client.simSpawnObject("gate_" + str(self.num_gates), "CheckeredDroneGate16x16", p_o_g, 0.75)
            # self.client.simSetObjectPose(self.tgt_name, p_o_g, True)
            # self.client.plot_tf([p_o_g], duration=20.0)
        
        # request quad img from AirSim
        image_response = self.client.simGetImages([airsim.ImageRequest('0', airsim.ImageType.Scene, False, False)])[0]
        # save all the necessary information to file
        self.writeImgToFile(image_response)
        self.writePosToFile(r, theta, psi, phi_rel)
        self.curr_idx += 1

        for gate_object in self.client.simListSceneObjects(".*[Gg]ate.*"):
            time.sleep(0.05)
            self.client.simDestroyObject(gate_object)

    def configureEnvironment(self):
        for gate_object in self.client.simListSceneObjects(".*[Gg]ate.*"):
            self.client.simDestroyObject(gate_object)
            time.sleep(0.5)
        
        # self.tgt_name = self.client.simSpawnObject("gate", "CheckeredDroneGate16x16", Pose(position_val=Vector3r(0,0,15)), 0.75)
        # self.tgt_name = self.client.simSpawnObject("gate", "CheckeredGate16x16", Pose(position_val=Vector3r(0,0,15)))

        if os.path.exists(self.csv_path):
            self.file = open(self.csv_path, "a")
        else:
            self.file = open(self.csv_path, "w")

    # write image to file
    def writeImgToFile(self, image_response):
        if len(image_response.image_data_uint8) == image_response.width * image_response.height * 3:
            img1d = np.fromstring(image_response.image_data_uint8, dtype=np.uint8)  # get numpy array
            img_rgb = img1d.reshape(image_response.height, image_response.width, 3)  # reshape array to 4 channel image array H X W X 3
            cv2.imwrite(os.path.join(self.base_path, 'images', str(self.curr_idx).zfill(len(str(self.num_samples))) + '.png'), img_rgb)  # write to png
            # cv2.imwrite(os.path.join(self.base_path, 'images', str(self.curr_idx).zfill(len(str(9999))) + '.png'), img_rgb)  # write to png

        else:
            print('ERROR IN IMAGE SIZE -- NOT SUPPOSED TO HAPPEN')

    # writes your data to file
    def writePosToFile(self, r, theta, psi, phi_rel):
        data_string = '{0} {1} {2} {3}\n'.format(r, theta, psi, phi_rel)
        self.file.write(data_string)
