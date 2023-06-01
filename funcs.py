import math, time, random, cv2, numpy as np
from tqdm import tqdm
import pynput

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

blue, green, cyan, yellow, red, purple, endc = bcolors.OKBLUE, bcolors.OKGREEN, bcolors.OKCYAN, bcolors.WARNING, bcolors.FAIL, bcolors.HEADER, bcolors.ENDC

def imscale(img, s):
    try:
        w, h, d = np.shape(img)
    except:
        w, h = np.shape(img)
    assert not 0 in [w, h], "empty src image"
    return cv2.resize(img, (round(len(img[0])*s), round(len(img)*s)), interpolation=cv2.INTER_NEAREST)

import ctypes
PROCESS_PER_MONITOR_DPI_AWARE = 2
ctypes.windll.shcore.SetProcessDpiAwareness(PROCESS_PER_MONITOR_DPI_AWARE)
def onMove(x, y):
    global mousex
    global mousey
    mousex, mousey = x, y
    return None
def onClick(x, y, button, pressed):
    global isPressed
    isPressed = pressed
mouse = pynput.mouse.Listener(on_click=onClick, on_move=onMove)
mouse.start()
def choosePts(im, numPts, scale=1):
    global mousex, mousey, isPressed
    pts = []
    imh, imw = np.shape(im)
    disp = np.array(im, copy=True)
    isPressed = False
    while len(pts) < numPts:
        cv2.imshow("im", imscale(disp, scale))
        wpx, wpy, _, _ = cv2.getWindowImageRect('im')
        if isPressed:
            imx, imy = mousex-wpx, mousey-wpy
            if 0 < imx < imw and 0 < imy < imw:
                if [imx/scale, imy/scale] not in pts:
                    pts.append([imx/scale, imy/scale])
                    disp = cv2.circle(disp, [int(imx//scale), int(imy//scale)], radius=20, color=(110, 10, 255), thickness=3)
                    if len(pts) < numPts: print(f"choose {numPts-len(pts)} more points")
        cv2.waitKey(1)
    return np.array(pts)

def showMatch(pcList, pcs, sides, thickness=1, scale=1):
    pc1, pc2 = pcs
    p1, p2 = pcList[pc1], pcList[pc2]
    i1, i2 = sides
    s1, s2 = p1.sides[i1], p2.sides[i2]
    #s1, s2 = rotate(s1, math.pi/2), rotate(s2, math.pi/2)
    x, y, w, h = cv2.boundingRect(np.append(s1, s2, axis=0).astype(np.float32))
    origin = [x-70, y-70]
    s1, s2 = s1-origin, s2-origin
    im = np.zeros((h+100, w+100, 3), np.uint8)
    im = cv2.putText(im, f"{pc1}[{i1}]", (round(.1*w), 30), cv2.FONT_HERSHEY_DUPLEX, 1, color=(50, 250, 50), lineType=cv2.LINE_AA)
    im = cv2.putText(im, f"{pc2}[{i2}]", (round(.1*w), 70), cv2.FONT_HERSHEY_DUPLEX, 1, color=(50, 0, 250), lineType=cv2.LINE_AA)
    im = cv2.polylines(im, np.int32([s1]), False, (50, 250, 50), thickness) # green is s1
    im = cv2.polylines(im, np.int32([s2]), False, (50, 0, 250), thickness) # red is s2
    return imscale(im, scale)



def rotate(pos, angle, origin = (0, 0)):
    #print(pos, type(pos))
    rmat = np.matrix([[math.cos(angle),-math.sin(angle)],
                      [math.sin(angle), math.cos(angle)]])
    return np.array(np.dot(np.matrix(pos.astype(np.float64)-origin), rmat.T))+origin

def filter(arr, dThresh):
    n = np.array(arr, copy=True)
    mod = 1
    while mod:
        mod = False
        for i, e in enumerate(n):
            for j, f in enumerate(n):
                d = np.linalg.norm(e-f)
                if (d < dThresh) and (i != j):
                    mod = True
                    n = np.delete(n, j, axis=0)
                    #n[i] = ptavg(e, f)
                    n[i] = (e+f)/2
                    break
            if mod:
                break
    return n

def rectangles(img, posList, dim, weight=5, color=(90, 0, 255)):
    for i, pos in enumerate(posList):
        if type(dim) == tuple:
            cv2.rectangle(img, pos, (pos[0] + dim[0], pos[1] + dim[1]), color, weight)
        if type(dim) == list:
            cv2.rectangle(img, pos, (pos[0] + dim[i][0], pos[1] + dim[i][1]), color, weight)
    return img

def circles(img, pos, radius=20, color=(20, 120, 220), width=7):
    i = np.copy(img)
    for e in pos:
        x, y = round(e[0]), round(e[1])
        i = cv2.circle(img, (x, y), radius, color, width)
    return i
