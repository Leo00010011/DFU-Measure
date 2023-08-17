import open3d as o3d
import pyrealsense2 as rs
import numpy as np
from rosbag.bag import Bag
from rosbag.rosbag_main import compress_cmd, ProgressMeter
from rosbag import Compression
import cv2


def get_image_from_msg(msg):
    nparr = np.frombuffer(msg.data,dtype = np.uint8)
    nparr = nparr.reshape((msg.height,msg.width,3))
    return cv2.cvtColor(nparr, cv2.COLOR_RGB2BGR)

def get_depth_from_msg(msg):
    nparr = np.frombuffer(msg.data,dtype= np.uint16)
    nparr = nparr.reshape((msg.height,msg.width))
    return nparr

def show_img(img):
    cv2.imshow('cosa',img)
    code = cv2.waitKey()   
    print(code)

def get_all_frames(path):
    bag = Bag(path)
    count = bag.get_type_and_topic_info().topics['/device_0/sensor_1/Color_0/image/data'].message_count
    first = True
    result = None
    for index, msg in enumerate(bag.read_messages(topics = ['/device_0/sensor_1/Color_0/image/data'])):
        msg = msg.message
        if first:
            first = False
            result = np.empty(shape = (count,msg.height,msg.width,3),dtype=np.uint8)
        result[index,:,:,:] = get_image_from_msg(msg)
    return result

class ScoreToColorConv:
    def __init__(self):
        color1 = (0, 0, 255)     
        color2 = (0, 165, 255)      
        color3 = (0, 255, 255)      
        color4 = (0, 255, 165)   
        color5 = (0, 255, 0) 
        colorArr = np.array([[color1, color2, color3, color4, color5]], dtype=np.uint8)
        self.min = 0
        self.max = 100
        # resize lut to 256 (or more) values
        self.lut = cv2.resize(colorArr, (256,1), interpolation = cv2.INTER_LINEAR)
        
    def get_color(self,score):
        scl_score = int(((score - self.min)/(self.max - self.min)) * 255)

        return int(self.lut[0,scl_score,0]) ,int(self.lut[0,scl_score,1]) ,int(self.lut[0,scl_score,2]) 

def compress_bag(input_path, output_path):
    inbag = Bag(input_path)
    outbag = Bag(output_path, 'w')
    outbag.compression = Compression.BZ2
    
    meter = ProgressMeter(outbag.filename, inbag._uncompressed_size)
    total_bytes = 0
    for topic, msg, t, conn_header in inbag.read_messages(raw=True, return_connection_header=True):
        msg_type, serialized_bytes, md5sum, pos, pytype = msg
        outbag.write(topic, msg, t, raw=True, connection_header=conn_header)

        total_bytes += len(serialized_bytes) 
        meter.step(total_bytes)
    inbag.close()
    outbag.close()
    meter.finish()

def compress_bag(input_path, output_path):
    inbag = Bag(input_path)
    outbag = Bag(output_path, 'w')
    outbag.compression = Compression.NONE
    meter = ProgressMeter(outbag.filename, inbag._uncompressed_size)
    total_bytes = 0
    for topic, msg, t, conn_header in inbag.read_messages(raw=True, return_connection_header=True):
        msg_type, serialized_bytes, md5sum, pos, pytype = msg
        outbag.write(topic, msg, t, raw=True, connection_header=conn_header)

        total_bytes += len(serialized_bytes) 
        meter.step(total_bytes)
    inbag.close()
    outbag.close()
    meter.finish()
class RSReader:


    def start_camera(self):
        if not o3d.t.io.RealSenseSensor.list_devices():
            raise Exception("Camera unavailable")
        cfg = o3d.t.io.RealSenseSensorConfig(
            {'serial': '', 'color_format': 'RS2_FORMAT_RGB8', 'color_resolution': '1280,720', 'depth_resolution': '1280,720'})

        self.rs = o3d.t.io.RealSenseSensor()
        self.rs.init_sensor(cfg, 0, 'output/reconstruction/bagfile.bag')
        self.rs.start_capture(False)

    def get_frames(self):
        while True:
            im_rgbd = self.rs.capture_frame(True, True)
            color = np.array(im_rgbd.color)
            depth = np.array(im_rgbd.depth)
            yield color, depth

    def stop_camera(self):
        self.rs.stop_capture()

    def save_intrinsic(self):
        json_obj = str(self.rs.get_metadata())
        with open('output/reconstruction/intrinsic.json', 'w') as intrinsic_file:
            intrinsic_file.write(json_obj)

    def get_resolution(self):
        return (self.rs.get_metadata().height,
                self.rs.get_metadata().width)


class RSPlayback:
    '''
    To get frames as nparray first call start_camera and the use get_frames
    '''
    
    def __init__(self,path) -> None:
        self.path = path

    def start_camera(self):
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_device_from_file(self.path)
        self.profile = self.pipeline.start(self.config)
        self.stream_color = self.profile.get_stream(rs.stream.color)
        self.stream_depth = self.profile.get_stream(rs.stream.depth)
        self.device = self.profile.get_device()
        self.playback = self.device.as_playback()
        

    def get_frames(self):
        
        align = rs.align(rs.stream.color)
        while True:
            frames = self.pipeline.wait_for_frames()
            frames = align.process(frames)
            color_frame = frames.get_color_frame()
            color_frame = np.asanyarray(color_frame.get_data())
            depth_frame = frames.get_depth_frame()
            depth_frame = np.asanyarray(depth_frame.get_data())
            yield color_frame, depth_frame



    def stop_camera(self):
        self.pipeline.stop()

    def get_resolution(self):
        return (self.stream_color.as_video_stream_profile().intrinsics.height,
                self.stream_color.as_video_stream_profile().intrinsics.width)
        

    def save_intrinsic(self):
        bag_reader = o3d.t.io.RSBagReader()
        bag_reader.open(self.path)
        json_obj = str(bag_reader.metadata)
        with open('output/reconstruction/intrinsic.json', 'w') as intrinsic_file:
            intrinsic_file.write(json_obj)
        bag_reader.close()


    
