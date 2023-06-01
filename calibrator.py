import cv2, time, numpy as np
from funcs import *

def calibrate(n, size, filepath=""):
    vid = cv2.VideoCapture(0)
    vid.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    obj = np.zeros((size*size,3), np.float32)
    obj[:,:2] = np.mgrid[0:size,0:size].T.reshape(-1,2)
    objs, pts = [], []
    print(f'collecting {n} chessboard images of size {size} by {size}. press q to end early . . .')
    while len(pts) < n:
        ret, checker = vid.read()
        ret, pt = cv2.findChessboardCorners(checker, (size,size))
        if ret:
            gray = cv2.cvtColor(checker, cv2.COLOR_BGR2GRAY)
            corners = cv2.cornerSubPix(gray,pt, (13,13), (-1,-1), criteria)
            pts.append(corners)
            objs.append(obj)
            checker = cv2.drawChessboardCorners(checker, (size,size), corners, True)
            print(len(pts))
            time.sleep(2)
        cv2.imshow('calibrate', imscale(checker, .6))
        cv2.waitKey(1)
    ret, mtx, dst, rvecs, tvecs = cv2.calibrateCamera(objs, pts, gray.shape, None, None)
    h,  w = gray.shape[:2]
    newmtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dst, (w,h), 0, (w,h))
    mean_error = 0
    for i in range(len(objs)):
        imgpoints2, _ = cv2.projectPoints(objs[i], rvecs[i], tvecs[i], mtx, dst)
        error = cv2.norm(pts[i], imgpoints2, cv2.NORM_L2)/len(imgpoints2)
        mean_error += error
    print(f"reprojection error is {mean_error}")
    cv2.destroyWindow('calibrate')
    print(mtx, dst, newmtx, sep='\n')
    if filepath != "":
        np.save(f"{filepath}mtx.npy", mtx)
        np.save(f"{filepath}dst.npy", dst)
        np.save(f"{filepath}newmtx.npy", newmtx)
    return mtx, dst, newmtx

def collectGoodImages(size, n, path):
    vid = cv2.VideoCapture(0)
    vid.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    num = 0
    while num<n:
        ret, checker = vid.read()
        ret, pt = cv2.findChessboardCorners(checker, (size,size))
        if ret:
            print(num)
            cv2.imwrite(f'{path}{num}.png', checker)
            num += 1
            gray = cv2.cvtColor(checker, cv2.COLOR_BGR2GRAY)
            corners = cv2.cornerSubPix(gray,pt, (13,13), (-1,-1), criteria)
            checker = cv2.drawChessboardCorners(checker, (size,size), corners, True)
            time.sleep(2)
        cv2.imshow('calibrate', imscale(checker, .6))
        cv2.waitKey(1)


#path = f"C:\\Users\\ek\\Desktop\\sdfghj\\puzzle\\"
#calibrate(30, 7, filepath=path)
#path = f"C:\\Users\\ek\\Desktop\\sdfghj\\puzzle\\testimgs\\dino\\checker"
#collectGoodImages(7, 1, path)














































