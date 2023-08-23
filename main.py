# Poder ver la calidad de los frames
# Poder ver barra de calidad
# Calcular tiempo promedio
# Obtener todo el BAG
#/device_0/sensor_1/Color_0/image/data
#/device_0/sensor_0/Depth_0/image/data
#'sensor_msgs/Image'
# Obtener todo el BAG
# Aplicar metrica de calidad en todo el bag
# Reproducir el BAG mostrando la metrica de calidad
#

#Compare the time of segmentation with quality measure
import numpy as np
from rosbag.bag import Bag
import cv2
import datetime
import math
import time
import pywt
import pyrealsense2 as rs
from IQA import LAP_MOD
from bag_utils import BagReview
# from segmentation.predictor import predict_from_array
# from brisque.get_brisque_features import brisque
input_path = 'C:\\Users\\53588\\Desktop\\Tesis\\Project\\a.bag'
input_path2 = 'C:\\Users\\53588\\Desktop\\Tesis\\DFU-Measure\\20230511_111900.bag'
input_path3 = 'C:\\Users\\53588\\Desktop\\Tesis\\DFU-Measure\\20230817_163423.bag'


# def test_time(IQA_method_list, predictor_list,bag_path_list):
#     with open(f'data {str(datetime.now())}.txt','w') as f:
#         log = []
#         for path in bag_path_list:
#             frames = get_all_frames(path)
#             count = frames.shape[0]
#             log.append('+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+')
#             log.append(f'Bag: {path}')
#             log.append(f'Frame count: {count}')
#             index = 0
#             for IQA_method in IQA_method_list:
#                 start_time = time.time()
#                 for index in range(count):
#                     IQA_method(frames[index,:,:,:])
#                 all_images_time = time.time() - start_time
#                 log.append(f'IQA_{index + 1} time: {all_images_time}')
#                 index += 1
#             index = 0
#             for predictor in predictor_list:
#                 start_time = time.time()
#                 predictor(frames)
#                 all_images_time = time.time() - start_time
#                 log.append(f'Predictor_{index + 1} time: {all_images_time}')
#                 index += 1
#         f.writelines(log)


bag = BagReview(input_path2)
bag.review_depth_frames()
