from __future__ import division
import random
import math
import time
import numpy as np
import threading
import os, sys
import airsimdroneracingvae
import airsimdroneracingvae.types
import airsimdroneracingvae.utils
from pynput.keyboard import Key, Controller
keyboard = Controller()
key = "r"

# import utils
curr_dir = os.path.dirname(os.path.abspath(__file__))
import_path = os.path.join(curr_dir, '..', '..')
sys.path.insert(0, import_path)
import racing_utils
import Extractor

random.seed()

# DEFINE DATA GENERATION META PARAMETERS
gate_passed_threshold = 0.8
viz_traj = False
vel_max = 30.0
acc_max = 20.0
speed_through_gate = 2
num_of_labs = 30
record = True

class DroneRacingDataGenerator(object):
    def __init__(self,
                 drone_name,
                 odom_loop_rate_sec,
                 vel_max,
                 acc_max):
        self.odom_loop_rate_sec = odom_loop_rate_sec
        self.base_track = None
        self.track_gate_poses = None
        self.num_training_laps = None
        self.next_gate_idx = 0
        self.last_gate_passed = -1
        self.Dists = None
        self.velocities = []
        # should be same as settings.json
        self.gate_names = []
        self.drone_name = drone_name
        self.track_name = '/home/dell/Drone_Project/src/Autonomous-Drone-Racing-with-Airsim/datagen/action_generator/tracks/interpolated_8.csv'
        self.last_future = []
        self.start_end = 0.0
        self.passed = True
        self.tic = 0
        self.record_flag = 0
        self.timer_flag = 0
        self.lab = 0
        # training params
        self.vel_max = vel_max
        self.acc_max = acc_max
        self.traj_tracker_gains = airsimdroneracingvae.TrajectoryTrackerGains(kp_cross_track=5.0, kd_cross_track=0.0,
                                                           kp_vel_cross_track=3.0, kd_vel_cross_track=0.0,
                                                           kp_along_track=0.4, kd_along_track=0.0,
                                                           kp_vel_along_track=0.04, kd_vel_along_track=0.0,
                                                           kp_z_track=2.0, kd_z_track=0.0,
                                                           kp_vel_z=0.4, kd_vel_z=0.0,
                                                           kp_yaw=3.0, kd_yaw=0.1)
        # todo encapsulate in function
        self.client = airsimdroneracingvae.MultirotorClient()
        self.client.confirmConnection()
        #self.client.enableApiControl(True, vehicle_name=self.drone_name)
        time.sleep(0.05)

        # threading stuff
        self.got_odom = False
        self.is_velocity_thread_active = True
        self.is_expert_planner_controller_thread_active = True
        #self.expert_planner_controller_thread = threading.Thread(target=self.repeat_timer_expert, args=(
        #self.expert_planner_controller_callback, odom_loop_rate_sec))
        #self.velocity_calculator_thread = threading.Thread(target=self.repeat_timer_vel, args=(self.velocity_calculator_callback, odom_loop_rate_sec))

    def initialize_quadrotor(self):
        self.client.enableApiControl(True, vehicle_name=self.drone_name)
        time.sleep(0.01)
        self.client.armDisarm(True, vehicle_name=self.drone_name)
        time.sleep(0.01)
        self.client.setTrajectoryTrackerGains(self.traj_tracker_gains.to_list(),
                                              vehicle_name=self.drone_name)
        time.sleep(0.01)

        self.takeoff_with_moveOnSpline(x=56, y=7, z=-13, vel_max=self.vel_max, acc_max=self.acc_max)
        # self.client.rotateToYawAsync(-90, self.drone_name)
        time.sleep(2)

    def start_training_data_generator(self, level_name='Soccer_Field_Easy'):
        # Environment Initialization
        self.load_level(level_name)
        open_track = Extractor.ReadGates(self.track_name)
        closing = Extractor.ReadGates('/home/dell/Drone_Project/src/Autonomous-Drone-Racing-with-Airsim/datagen/action_generator/tracks/Closing_8.csv')
        self.base_track = open_track + closing
        self.track_gate_poses = Extractor.DistortCheckeredGates(self.base_track, -0.5, 0.5)
        self.gate_names = self.name_the_gates(len(self.track_gate_poses))
        self.Dists, self.start_end = Extractor.Get_Distances_between_Gates(self.track_gate_poses)
        #print(self.Dists)
        self.create_track('CheckeredDroneGate16x16', self.track_gate_poses, 1)
        # Drone Initialization
        self.initialize_quadrotor()

        # Generation
        #self.start_expert_planner_controller_thread()

        if record and self.record_flag == 0:
                keyboard.press(key)
                keyboard.release(key)
                self.record_flag = 1

        while self.lab <= num_of_labs:
            self.expert_planner_controller_callback()

        if record and self.lab > num_of_labs:
            keyboard.press(key)
            keyboard.release(key)

        #self.velocity_calculator_thread.start()

    def repeat_timer_expert(self, task, period):
        while self.is_expert_planner_controller_thread_active:
            task()
            time.sleep(period)

    def repeat_timer_vel(self, task, period):
        while self.is_velocity_thread_active:
            task()
            time.sleep(period)

    def load_level(self, level_name):
        ''' Loads an empty Environment '''
        self.client.simLoadLevel(level_name)
        time.sleep(2)
        racing_utils.AllGatesDestroyer(self.client)

    def create_track(self, gate_type, gate_poses, scale):
        ''' Creates a track '''
        for i, poses in enumerate(gate_poses):
            self.client.simSpawnObject(self.gate_names[i], gate_type,
                                       poses, scale)
            time.sleep(0.05)

    def name_the_gates(self,num_gates_track):
        names = []
        for i in range(num_gates_track):
            name = 'Gate_' + str(i)
            names.append(name)
        return names

    def takeoff_with_moveOnSpline(self, x, y, z, vel_max, acc_max):
        self.client.moveOnSplineAsync(path=[airsimdroneracingvae.Vector3r(x, y, z)],
                                      vel_max=vel_max, acc_max=acc_max,
                                      add_curr_odom_position_constraint=True,
                                      add_curr_odom_velocity_constraint=True,
                                      viz_traj=viz_traj,
                                      vehicle_name=self.drone_name).join()

    def expert_planner_controller_callback(self):

        if self.next_gate_idx == 1 and self.timer_flag == 0:
            self.tic = time.perf_counter()
            self.timer_flag = 1
            self.lab += 1

        if self.next_gate_idx == len(self.track_gate_poses):
            toc = time.perf_counter()
            print('Finished lab ' + str(self.lab))
            print('lab Time:', toc - self.tic)
            print('lab max velocity = ', np.max(self.velocities))
            print('lab average velocity = ', np.mean(self.velocities))
            self.last_gate_passed = -1
            self.next_gate_idx = 0
            self.timer_flag = 0
            self.track_gate_poses = Extractor.DistortCheckeredGates(self.base_track, -0.5, 0.5)
            for gate_idx in range(len(self.track_gate_poses)):
                self.client.simSetObjectPose(self.gate_names[gate_idx], self.track_gate_poses[gate_idx])

        if self.passed:
            self.fly_to_next_gate_with_moveOnSpline(self.next_gate_idx)

        self.passed = self.simGetLastGatePassed()



    def fly_to_next_gate_with_moveOnSpline(self, next_gate_idx):
        #if self.next_gate_idx == 6:
            #v_max = 4
        #else:
            #v_max = self.vel_max

        self.last_future.append(self.client.moveOnSplineVelConstraintsAsync([self.track_gate_poses[next_gate_idx].position],
        [self.get_gate_facing_vector_from_quaternion(self.track_gate_poses[next_gate_idx].orientation,
                                                     scale=speed_through_gate)],
                                                             vel_max=self.vel_max, acc_max=self.acc_max,
                                                             add_curr_odom_position_constraint=True,
                                                             add_curr_odom_velocity_constraint=True,
                                                             viz_traj=viz_traj,
                                                             vehicle_name=self.drone_name))

        velocity_com = self.client.simGetGroundTruthKinematics(self.drone_name).linear_velocity
        self.velocities.append(math.sqrt(velocity_com.x_val ** 2 + velocity_com.y_val ** 2 + velocity_com.z_val ** 2))

    def fly_to_next_gate_with_moveOnSpline_No_Deacc(self, next_gate_idx):
        self.last_future.append(
            self.client.moveOnSplineAsync([self.track_gate_poses[next_gate_idx].position],
                                                            vel_max=self.vel_max, acc_max=self.acc_max,
                                                            add_curr_odom_position_constraint=True,
                                                            add_curr_odom_velocity_constraint=True,
                                                            viz_traj=viz_traj,
                                                            vehicle_name=self.drone_name))

    def velocity_calculator_callback(self):
        velocity_com = self.client.simGetGroundTruthKinematics(self.drone_name).linear_velocity
        self.velocities.append(math.sqrt(velocity_com.x_val ** 2 + velocity_com.y_val ** 2 + velocity_com.z_val ** 2))
        print(self.velocities)

    def synchronize(self, next_gate_idx):
        if next_gate_idx == 0:
            time.sleep(2.2)
            print('yes')
        if self.track_name == 'Qualifier_Tier_3.csv':
            if next_gate_idx == 12 or next_gate_idx == 11:
                time.sleep(0.22 * self.Dists[next_gate_idx - 1])
            elif next_gate_idx == 8:
                time.sleep(0.45 * self.Dists[next_gate_idx - 1])
            else:
                time.sleep(0.26 * self.Dists[next_gate_idx - 1])
        else:
            sleep_time = (10**(self.Dists[next_gate_idx - 1]/2.5+1))*math.exp(-self.Dists[next_gate_idx - 1])
            print(sleep_time)
            time.sleep(sleep_time)

    def simGetLastGatePassed(self):
        drone_position = self.client.simGetGroundTruthKinematics().position
        dist_from_curr_gate = math.sqrt(
            (drone_position.x_val - self.track_gate_poses[self.next_gate_idx].position.x_val) ** 2
            + (drone_position.y_val - self.track_gate_poses[self.next_gate_idx].position.y_val) ** 2
            + (drone_position.z_val - self.track_gate_poses[self.next_gate_idx].position.z_val) ** 2)

        if dist_from_curr_gate < gate_passed_threshold:
            self.last_gate_passed += 1
            self.next_gate_idx += 1
            return True

    #def Finish(self):



    # maybe maintain a list of futures, or else unreal binary will crash if join() is not called at the end of script

    def join_all_pending_futures(self):
        for i in range(len(self.track_gate_poses)):
            self.last_future[i].join()


    def start_expert_planner_controller_thread(self):
        if not self.is_expert_planner_controller_thread_active:
            self.is_expert_planner_controller_thread_active = True
            self.expert_planner_controller_thread.start()
            print("Started expert_planner_controller thread")

    def stop_expert_planner_controller_thread(self):
        if self.is_expert_planner_controller_thread_active:
            self.is_expert_planner_controller_thread_active = False
            #self.expert_planner_controller_thread.join()
            print("Stopped expert_planner_controller thread")

    def set_num_training_laps(self, num_training_laps):
        self.num_training_laps = num_training_laps

    def get_gate_facing_vector_from_quaternion(self, airsim_quat, scale=1):
        # convert gate quaternion to rotation matrix.
        # ref: https://en.wikipedia.org/wiki/Rotation_matrix#Quaternion; https://www.lfd.uci.edu/~gohlke/code/transformations.py.html
        q = np.array([airsim_quat.w_val, airsim_quat.x_val, airsim_quat.y_val, airsim_quat.z_val], dtype=np.float64)
        n = np.dot(q, q)
        if n < np.finfo(float).eps:
            return airsimdroneracingvae.Vector3r(0.0, 1.0, 0.0)
        q *= np.sqrt(2.0 / n)
        q = np.outer(q, q)
        rotation_matrix = np.array([[1.0 - q[2, 2] - q[3, 3], q[1, 2] - q[3, 0], q[1, 3] + q[2, 0]],
                                    [q[1, 2] + q[3, 0], 1.0 - q[1, 1] - q[3, 3], q[2, 3] - q[1, 0]],
                                    [q[1, 3] - q[2, 0], q[2, 3] + q[1, 0], 1.0 - q[1, 1] - q[2, 2]]])
        gate_facing_vector = rotation_matrix[:, 1]
        return airsimdroneracingvae.Vector3r(scale * gate_facing_vector[0], scale * gate_facing_vector[1],
                               scale * gate_facing_vector[2])

    def __del__(self):
        print('deleted Object')

if __name__ == "__main__":
        drone_racing_datagenerator = DroneRacingDataGenerator(drone_name='drone_0',
                                                          odom_loop_rate_sec=0.005,
                                                          vel_max=vel_max,
                                                          acc_max=acc_max
                                                          )
        drone_racing_datagenerator.start_training_data_generator()