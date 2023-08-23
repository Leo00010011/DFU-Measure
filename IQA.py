import numpy as np
import cv2
import math
import pywt


def resize_decorator(f, h,w):
    def new_func(img):
        img = cv2.resize(img,(w,h))
        return f(img)
    return new_func

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

#STA2
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

def WAV_COEF(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, (cH, cV, cD) = pywt.dwt2(img,'db6')
    return np.absolute(cH).sum() +  np.absolute(cV).sum() + np.absolute(cD).sum()

def WAV_VAR(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, (cH, cV, cD) = pywt.dwt2(img,'db6')
    return np.absolute(cH).var() +  np.absolute(cV).var() + np.absolute(cD).var()

def WAV_RAT(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    coef = pywt.wavedec2(img, 'db6', level=3)
    cA3,_,_,(cH,cV,cD) = coef
    Mh = (cH**2).sum() + (cV**2).sum() + (cD**2).sum()
    Ml = (cA3**2).sum()
    return Mh/Ml