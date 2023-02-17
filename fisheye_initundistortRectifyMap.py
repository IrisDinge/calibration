# -*- coding:UTF-8 -*-

# author:Jin.Ding2
# contact: ding.jin233@gamil.com
# datetime:2023/2/10 14:23

import numpy as np
import cv2 as cv
from tkinter.filedialog import askdirectory
import glob
from tkinter import *
import time
from PIL import Image
import json, json_numpy
import math


def FisheyeIntrinsic_par(row, column, raw_image_path):

    '''
    :param row:
    :param column:
    :param raw_image_path:
    :return:

    chessboard par: row column
    criteria: Chessboard Size, for instance size of each grid 30mm

    '''

    subpix_criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 27, 0.001)
    calibration_flags = cv.fisheye.CALIB_RECOMPUTE_EXTRINSIC+ cv.fisheye.CALIB_CHECK_COND + cv.fisheye.CALIB_FIX_SKEW


    objp = np.zeros((1, row * column, 3), np.float32)

    objp[0, :, :2] = np.mgrid[0:row, 0:column].T.reshape(-1, 2)

    # 用于存储所有图像对象点与图像点的矩阵
    objpoints = []  # 在真实世界中的 3d 点
    imgpoints = []  # 在图像平面中的 2d 点
    images = glob.glob(raw_image_path + '/*.jpg')

    _img_shape = None

    for fname in images:
        img = cv.imread(fname)
        if _img_shape == None:
            _img_shape = img.shape[:2]
            #image的长、宽
        else:
            assert _img_shape  == img.shape[:2]
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        cv.imshow('gray', gray)

        ret, corners = cv.findChessboardCorners(gray, (row, column), cv.CALIB_CB_ADAPTIVE_THRESH + cv.CALIB_CB_FAST_CHECK + cv.CALIB_CB_NORMALIZE_IMAGE)

        if ret is True:

            objpoints.append(objp)
            # image, corner, windowSize
            cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), subpix_criteria)
            imgpoints.append(corners)

            cv.drawChessboardCorners(img, (row, column), corners, ret)


    mtx 3x3 intrinsic matrix
    K distortion coefficients
    rvecs, tvecs
    '''
    N = len(objpoints)
    K = np.zeros((3, 3))
    D = np.zeros((4, 1))
    rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N)]
    tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N)]


    rms, _, _, _, _= cv.fisheye.calibrate(
        objpoints,
        imgpoints,
        gray.shape[::-1], # gray.shape的转置
        K,
        D,
        rvecs,
        tvecs,
        calibration_flags,
        (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 27, 1e-6)
    )
    DIM = _img_shape[::-1]

    print("Found " + str(N) + " valid images for calibration")
    print("Dim = " + str(_img_shape[::-1]))
    print("K = ", K)
    print("D =", D)
    return DIM, K, D


class NumpyArrayEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def undistortimage(DIM, K, D):

    # root = Tk()
    # root.withdraw()
    # test_image_path = askdirectory(title="Choose Test Image Folder")
    # result_image_path = askdirectory(title="Choose Result Image Folder")
    # root.destroy()


    #images = glob.glob('test_images' + '/*.jpg')

    result_image_path = 'undistorted_image'
    """
    R: rectification transformation in the object space: 3x3 1-channel, or vector: 3x1/1x3 or 1x1 3-channel  
    P: new camera intrinsic matrix(3x3) or new projection matrix (3x4)
    单目相机中，P = K
    size: Undistorted image size
    m1type: can be CV_32FC1 or CV_16SC2

    :return map1, map2
    """
    # balance = 1
    # dim2 = None
    # dim3 = None
    #
    img = cv.imread('test_images' + '/3300.jpg')
    # dim1 = img.shape[:2][::-1]
    # try:
    #     dim1[0]/dim1[1] == DIM[0]/DIM[1]
    # except:
    #     print("Undistorted images need to have same aspect ratio as the ones used in calibration")
    #
    # if not dim2:
    #     dim2 = dim1
    # if not dim3:
    #     dim3 = dim1
    # scaled_K = K * dim1[0] / DIM[0]
    # scaled_K[2][2] = 1.0
    R = np.eye(3)
    #双目
    #P = cv.fisheye.estimateNewCameraMatrixForUndistortRectify(scaled_K, D, dim2, R, balance=balance)
    map1, map2 = cv.fisheye.initUndistortRectifyMap(K, D, R, K, DIM, cv.CV_16SC2)
    undistorted_image = cv.remap(img, map1, map2, interpolation=cv.INTER_LINEAR, borderMode=cv.BORDER_CONSTANT)

    # data_x={
    #     'x': map1,
    #
    # }
    #
    # data_y={
    #
    #     'y': map2
    # }
    # data_json_x = json.dumps(data_x, cls=NumpyArrayEncoder)
    # data_json_y = json.dumps(data_y, cls=NumpyArrayEncoder)
    with open('map_x.json', 'a') as f:
        json.dump(map1, f, cls=NumpyArrayEncoder)
    with open('map_y.json', 'a') as f:
        json.dump(map2, f, cls=NumpyArrayEncoder)
    cv.imshow("undistorted", undistorted_image)
    cv.waitKey(6000)
    cv.destroyAllWindows()




def initUndistortRectifyMap(DIM, K, D):


    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    k1, k2, k3, k4 = D[:]
    R = np.eye(3, 3)
# 内参矩阵与旋转矩阵R成绩的逆矩阵!
    #Ar = cv.getDefaultNewCameraMatrix(K, DIM, True)
    P = np.dot(K, R)
    iR = np.linalg.inv(P)

    img = cv.imread('test_images' + '/3300.jpg')
    img_array = np.array(img)
    shape = img_array.shape
    matTilt = np.eye(3, 3)
    mapx = np.ones((shape[0], shape[1]), dtype = np.float32)
    mapy = np.ones((shape[0], shape[1]), dtype = np.float32)
    for i in range(0, shape[0]): #height
        #_x, _y, _w = 0, 0, 1
        # 利用逆矩阵iR将像素坐标(j,i)转换到相机坐标系(_x,_y,_w)
        # transform[XYW] = iR*transform[x y 1]
        _x = i*iR[0,1] + iR[0,2]
        _y = i*iR[1,1] + iR[1,2]
        _w = i*iR[2,1] + iR[2,2]

        for j in range(0, shape[1]): #width
            #遍历每个相机坐标位置
            # _x += iR[0, 0]
            # _y += iR[1, 0]
            # _w += iR[2, 0]
            x = _x*_w
            y = _y*_w
            r2 = x*x + y*y
            r = math.sqrt(r2)
            #k3, k4, k5, k6 = 0
            theta = math.atan(r)
            theta2 = theta * theta
            theta4 = theta2 * theta2
            theta6 = theta4 * theta2
            theta8 = theta4 * theta4
            theta_d =  theta*(1 + k1*theta2 + k2* theta4 + k3*theta6 + k4*theta8 )

            if r == 0:
                scale = 1
            else:
                scale = theta_d / r

            # kr = (1 + ((0*r2 + k2)*r2 + k1)*r2)/(1 + ((0*r2 + 0)*r2 + 0)*r2)
            # xd = (x*kr + p1*_2xy + p2*(r2 + 2*x2) + 0*r2+0*r2*r2)
            # yd = (y * kr + p1 * (r2 + 2 * y2) + p2 * _2xy + 0 * r2 + 0 * r2 * r2)
            u = fx*x*scale + cx
            v = fy*y*scale + cy



            mapx[i, j] = u
            mapy[i, j] = v

            _x +=iR[0, 0]
            _y +=iR[1, 0]
            _w +=iR[2, 0]


    map = np.stack((mapx, mapy), axis =2)

    # with open('remap.json', 'a') as f:
    #     json.dump(map, f, cls=NumpyArrayEncoder)
    # with open('remapx.json', 'a') as f:
    #     json.dump(mapx, f, cls=NumpyArrayEncoder)
    # with open('remapy.json', 'a') as f:
    #     json.dump(mapy, f, cls=NumpyArrayEncoder)

    undistorted_image = cv.remap(img, mapx, mapy, interpolation=cv.INTER_LINEAR, borderMode=cv.BORDER_CONSTANT)
    cv.imshow("undistorted", undistorted_image)
    cv.waitKey(60000)
    cv.destroyAllWindows()




if __name__ == "__main__":


    raw_image_path = r'raw_images'
    row = 5
    column = 8

    DIM, K, D = FisheyeIntrinsic_par(row, column, raw_image_path)
    undistortimage(DIM, K, D)
    initUndistortRectifyMap(DIM, K, D)
