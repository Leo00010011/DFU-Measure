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
LEFT_ARROW = 2424832 
RIGHT_ARROW = 2555904 
SCAPE = 27
SPACE_BAR = 32
#Compare the time of segmentation with quality measure
import numpy as np
from rosbag.bag import Bag
import cv2
import datetime
import math
import time
from utils import get_all_frames, show_img, ScoreToColorConv
# from segmentation.predictor import predict_from_array
# from brisque.get_brisque_features import brisque
input_path = 'C:\\Users\\53588\\Desktop\\Tesis\\Project\\a.bag'

def test_time(IQA_method_list, predictor_list,bag_path_list):
    with open(f'data {str(datetime.now())}.txt','w') as f:
        log = []
        for path in bag_path_list:
            frames = get_all_frames(path)
            count = frames.shape[0]
            log.append('+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+')
            log.append(f'Bag: {path}')
            log.append(f'Frame count: {count}')
            index = 0
            for IQA_method in IQA_method_list:
                start_time = time.time()
                for index in range(count):
                    IQA_method(frames[index,:,:,:])
                all_images_time = time.time() - start_time
                log.append(f'IQA_{index + 1} time: {all_images_time}')
                index += 1
            index = 0
            for predictor in predictor_list:
                start_time = time.time()
                predictor(frames)
                all_images_time = time.time() - start_time
                log.append(f'Predictor_{index + 1} time: {all_images_time}')
                index += 1
        f.writelines(log)

def put_score_bar(scr_list,img,cnv : ScoreToColorConv = None,index = None,org = None,scale = None,height = None):
    h, w, _ = img.shape
    count = scr_list.shape[0]
    if not scale:
        scale = w/count
    scale = min(int(w/count),scale)

    if not height:
        height = scale

    if not org:
        org = (h - 4*scale,int(w%count/2))

    #putting scale bar
    number_of_scales = 6
    r_s = int(h/8)
    c_s = w - scale*4
    height_s = int((h/4)/number_of_scales)
    for i in range(number_of_scales):
        score = cnv.max - i/(number_of_scales-1)*(cnv.max - cnv.min)
        color = cnv.get_color(score)
        img[r_s + i*height_s:r_s + (i + 1)*height_s - scale,c_s:c_s + scale,:] = color
        cv2.putText(img,'%.2f' % score,(w - scale*20,r_s + int((i + 1/2)*height_s)), 0,.55, color,thickness=1,)
    

    r,c = org
    #putting index
    arrow_col = index*scale + int(scale/2)
    cv2.arrowedLine(img,(c + arrow_col,r - height),(c + arrow_col,r),(0,255,0))


    # putting score bar
    for i in range(count):
        if not cnv:
            img[r:r + height,c + i*scale:c + (i + 1)*scale,:] = scr_list[i]
        else:
            img[r:r + height,c + i*scale:c + (i + 1)*scale,:] = cnv.get_color(scr_list[i]) 
        
    

def review_frames(arr,IQA):
    cnv = ScoreToColorConv()
    scores = []
    for index in range(arr.shape[0]):
        scores.append(IQA(arr[index,:,:,:]))
    print('COMPUTED ALL SCORES')
    scores = np.array(scores)
    scores = (scores - scores.mean())/scores.std()
    min_value = scores.min()
    max_value = scores.max()
    cnv.min = min_value
    cnv.max = max_value
    index = 0
    count = arr.shape[0]
    while True:
        img = arr[index,:,:,:]
        color = cnv.get_color(scores[index])
        put_score_bar(scores,img,cnv,index)
        cv2.putText(img,'Score:%.2f' % scores[index],(0,30), 0,.75, color,thickness=2,)
        cv2.imshow(input_path,img)
        code = cv2.waitKeyEx()
        if code == LEFT_ARROW:
            index = max(index - 1, 0)
        elif code == RIGHT_ARROW:
            index = min(count - 1, index + 1)
        elif code == 27:
            break
        elif code == SPACE_BAR:
            pass
        else:
            print(code)

    
# LAP 4
def LAP_VAR(img):
    # Varianza del laplaciano
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur_map = cv2.Laplacian(img, cv2.CV_64F)
    score = np.var(blur_map)
    return score

# LAP 2
def LAP_MOD(img):
    # Suma del laplaciano modificado
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    x_kernel = np.array([[-1,2,-1]])
    y_kernel = np.transpose(x_kernel)
    x_val = np.absolute(cv2.filter2D(img,-1,x_kernel)).sum()
    y_val = np.absolute(cv2.filter2D(img,-1,y_kernel)).sum()
    return x_val + y_val

# LAP3
def LAP_DIAG(img):
    # suma del laplaciano con diagonales
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    x_kernel = np.array([[-1,2,-1]])
    y_kernel = np.transpose(x_kernel)
    x1_kernel = np.array([[0,0,1],
                          [0,2,0],
                          [1,0,0]])*math.sqrt(2)
    x2_kernel = np.array([[1,0,0],
                          [0,2,0],
                          [0,0,1]])*math.sqrt(2)
    x_val = np.absolute(cv2.filter2D(img,-1,x_kernel)).sum()
    y_val = np.absolute(cv2.filter2D(img,-1,y_kernel)).sum()
    x1_val = np.absolute(cv2.filter2D(img,-1,x1_kernel)).sum()
    x2_val = np.absolute(cv2.filter2D(img,-1,x2_kernel)).sum()
    return x_val + y_val + x1_val + x2_val

def EIG_SUM(img: np.ndarray):
    # suma de los k mayores valores propios de la imagen
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img - img.mean()
    h, w = img.shape
    cov = img.dot(np.transpose(img))/(h*w - 1)
    s = np.linalg.eigvalsh(cov)
    result = 0
    for i in range(1,11):
        result += s[-1*i]
    return result

def resize_decorator(f, h,w):
    def new_func(img):
        img = cv2.resize(img,(w,h))
        return f(img)
    return new_func
    
print('STARTED')
arr = get_all_frames(input_path)
print('ALL FRAMES LOADED')
review_frames(arr,EIG_SUM)
