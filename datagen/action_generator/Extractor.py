from __future__ import division
import numpy as np
import random
import math
import time
from airsimdroneracingvae import client
import airsimdroneracingvae.utils
from airsimdroneracingvae.types import Pose, Vector3r, Quaternionr
import os, sys
import airsimdroneracingvae
import csv
import pandas as pd

import racing_utils.geom_utils

curr_dir = os.path.dirname(os.path.abspath(__file__))
import_path = os.path.join(curr_dir, '..', '..')
sys.path.insert(0, import_path)


def GetBasePoses(client):
    gate_names_sorted = sorted(client.simListSceneObjects("Gate.*"))
    list_poses = []
    for gate in gate_names_sorted:
        pose = client.simGetObjectPose(gate)
        row = [pose.position.x_val, pose.position.y_val, pose.position.z_val,
               pose.orientation.x_val, pose.orientation.y_val, pose.orientation.z_val,
               pose.orientation.w_val]
        list_poses.append(row)
    return list_poses

def poses_to_list(Gates):
    list_poses = []
    for gate in Gates:
        row = [gate.position.x_val, gate.position.y_val, gate.position.z_val,
               gate.orientation.x_val, gate.orientation.y_val, gate.orientation.z_val,
               gate.orientation.w_val]
        list_poses.append(row)
    return list_poses

def ReadGates(file):
    df = pd.read_csv(file, header=None)
    gate_poses = []
    for i in range(df.shape[0]):
        row = df.iloc[i]
        if row[2] > 0:
            row[2] = -row[2]
        pose = Pose(Vector3r(row[0],row[1],row[2]),Quaternionr(row[3],row[4],row[5],row[6]))
        gate_poses.append(pose)
    return gate_poses

#gate_poses = ReadGates('Qualifier_Tier_3.csv')
#print(gate_poses[0].orientation.w_val)


#p = GetBasePoses(self.client)
#np.savetxt(str(level_name) + ".csv",p, delimiter=", ", fmt='% s')


def DistortCheckeredGates(base_poses, min, max):
    distorted_poses = []
    for Gatepositions in base_poses:
        new_x = Gatepositions.position.x_val + random.uniform(min, max)
        new_y = Gatepositions.position.y_val + random.uniform(min, max)
        #new_y = Gatepositions.position.y_val
        new_z = Gatepositions.position.z_val + random.uniform(min, max)
        new_pose = Pose(Vector3r(new_x, new_y, new_z), Quaternionr(Gatepositions.orientation.x_val,
                                                                   Gatepositions.orientation.y_val,
                                                                   Gatepositions.orientation.z_val,
                                                                   Gatepositions.orientation.w_val))
        distorted_poses.append(new_pose)

    return distorted_poses

#gate_poses = ReadGates('interpolated.csv')
#distorted = DistortCheckeredGates(gate_poses,-3,3)
#np.savetxt('Distorted' + ".csv", distorted, delimiter=", ", fmt='% s')


def Get_Distances_between_Gates(Gates):
    Dists = []
    start_end = 0
    for idx, gate in enumerate(Gates):
        curr_gate = gate
        if idx == len(Gates) - 1:
            next_gate = Gates[0]
        else:
            next_gate = Gates[idx+1]

        x_diff = curr_gate.position.x_val - next_gate.position.x_val
        y_diff = curr_gate.position.y_val - next_gate.position.y_val
        z_diff = curr_gate.position.z_val - next_gate.position.z_val
        Dist = math.sqrt(x_diff**2 + y_diff**2 + z_diff**2)

        if idx == len(Gates) - 1:
            start_end = Dist
        else:
            Dists.append(Dist)
    Dists = np.array(Dists)
    return Dists, start_end

#gate_poses = ReadGates('Qualifier_Tier_3.csv')
#Dists, start_end = Get_Distances_between_Gates(gate_poses)
#print(Dists)
#print('max dist:', np.max(Dists))
#print('min dist:', np.min(Dists))
#print('average dist:', np.average(Dists))
#print('start_end dist: ', start_end)

def interpolate_positions(Gate1,Gate2,n):
    '''returns a vector of gates [Gate1, ......, Gate2]'''
    xs = racing_utils.geom_utils.interp_vector(Gate1.position.x_val, Gate2.position.x_val, 2 + n)
    ys = racing_utils.geom_utils.interp_vector(Gate1.position.y_val, Gate2.position.y_val, 2 + n)
    zs = racing_utils.geom_utils.interp_vector(Gate1.position.z_val, Gate2.position.z_val, 2 + n)
    orient_xs = racing_utils.geom_utils.interp_vector(Gate1.orientation.x_val, Gate2.orientation.x_val, 2 + n)
    orient_ys = racing_utils.geom_utils.interp_vector(Gate1.orientation.y_val, Gate2.orientation.y_val, 2 + n)
    orient_zs = racing_utils.geom_utils.interp_vector(Gate1.orientation.z_val, Gate2.orientation.z_val, 2 + n)
    orient_ws = racing_utils.geom_utils.interp_vector(Gate1.orientation.w_val, Gate2.orientation.w_val, 2 + n)
    Gates = []
    for i in range(n):
        gate = Pose(Vector3r(xs[i+1], ys[i+1], zs[i+1]), Quaternionr(orient_xs[i+1], orient_ys[i+1], orient_zs[i+1], orient_ws [i+1]))
        Gates.append(gate)
    return Gates

#gate_poses = ReadGates('Qualifier_Tier_3.csv')
#Gates = interpolate_positions(gate_poses[0], gate_poses[-1], 3)
#print(Gates)

def interpolate_race_gates(max_dist, Gates):
    Dists, start_end = Get_Distances_between_Gates(Gates)
    new_Gates = []
    for i, dist in enumerate(Dists):
        req_num_gates = math.ceil(dist/max_dist) - 1
        if req_num_gates == 0:
            new_Gates.append(Gates[i])
        else:
            interpolated_gates = interpolate_positions(Gates[i], Gates[i+1], req_num_gates)
            new_Gates.append(Gates[i])
            new_Gates.extend(interpolated_gates)
    new_Gates.append(Gates[-1])
    return new_Gates

#gate_poses = ReadGates('interpolated_8.csv')
#mid_gate = Pose(Vector3r(75, 0.25, -19.049999237060547), Quaternionr(0, 0, 1, 0.0))
#mid_gate2 = Pose(Vector3r(100, 0.25, -19.049999237060547), Quaternionr(0, 0, -0.40957579016685486, 0.7867885828018188))
#closing_poses = [gate_poses[-1], gate_poses[0]]
#Gates = interpolate_race_gates(8, closing_poses)
#p = poses_to_list(Gates[1:-1])
#print(p)
#np.savetxt(str('Closing_8') + ".csv", p, delimiter=", ", fmt='% s')

#gate_poses = ReadGates('interpolated.csv')
#Dists, start_end = Get_Distances_between_Gates(gate_poses)
#print('max dist:', np.max(Dists))
#print('min dist:', np.min(Dists))
#print('average dist:', np.average(Dists))
#print('start_end dist: ', start_end)

def name_the_gates(num_gates):
    names = []
    for i in range(num_gates):
        name = 'Gate_' + str(i)
        names.append(name)
    return names

#print(name_the_gates(6))

