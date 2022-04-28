from time import sleep
from pose_sampler import *

num_samples = 250000
dataset_path = "/home/dell/Drone_Project/datasets/img_data_250K"

# check if output folder exists
if not os.path.isdir(dataset_path):
    os.makedirs(dataset_path)
    img_dir = os.path.join(dataset_path, 'images')
    os.makedirs(img_dir)
else:
    print("Error: path already exists.")
    flag= input("Enter Y/y to confirm folder deletion and recreation or any letter to abort: ")
    if flag == 'y' or flag == 'Y' :
        shutil.rmtree(dataset_path)
        os.makedirs(dataset_path)
        img_dir = os.path.join(dataset_path, 'images')
        os.makedirs(img_dir)
    else: exit()

pose_sampler = PoseSampler(num_samples, dataset_path)

for idx in range(pose_sampler.num_samples):
    pose_sampler.update()
    if idx % 100 == 0:
        print('Num samples: {}'.format(idx))
    # sleep(0.3)   #comment this out once you like your ranges of values
