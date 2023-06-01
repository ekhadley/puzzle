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

def correctPlacement(p, pos):
    idx = 10*pos[1]+pos[0]
    return (idx, p.cheat[idx])

def correctRank(p, pos, matches):
    idx, rot = correctPlacement(p, pos)
    for i, m in enumerate(matches):
        if (m[0] == idx) and (m[2] == rot): return i
    assert 0, "correct placement not found in matches. :/"

def isTruePairing(p, pcs, sides, cheat):
    pc1, pc2 = pcs
    r1, r2 = cheat[pc1], cheat[pc2]
    s1, s2 = sides
    #print(f"idx: {pc1, pc2}   rots: {r1, r2}")
    if pc1 == pc2:
        return False
    if (pc2-pc1 == 1) and (pc2%10 != 0):
        return (r1+3)%4==s1 and (r2+1)%4==s2
    if (pc2-pc1 == -1) and (pc1%10 != 0):
        return (r1+1)%4==s1 and (r2+3)%4==s2
    if pc2-pc1 == 10:
        return (r1+2)%4==s1 and r2==s2
    if pc2-pc1 == -10:
        return r1==s1 and (r2+2)%4==s2
    return False

def metric(p):
    comp, diff = [], 0
    cheat = findCheat(p)
    for p1 in tqdm(range(len(p.pcs)), desc=f"{cyan}evaluating true vs best scores. . .{endc}", ncols=100, unit="it"):
        for s1 in range(4):
            best = None
            true = None
            for p2 in range(len(p.pcs)):
                for s2 in range(4):
                    if p1 != p2:
                        fit = p.evalMatch((p1, p2), (s1, s2))
                        if best==None or fit < best[2]:
                            best = [(p1, p2), (s1, s2), round(fit, 5)]
                        if isTruePairing(p, (p1, p2), (s1, s2), cheat):
                            true = [(p1, p2), (s1, s2), round(fit, 5)]
            if true != None:
                if true[2] != 0: diff += (true[2]-best[2])/true[2]
                else: diff += 1e6
                comp.append([best, true])
    return comp

def metricInfo(comp, all=False, thresh=5):
    scores, bestscores, scales, incompatibles = [], [], [], []
    worstTrueScore, worstScale = comp[0], comp[0]
    numcorrect = 0
    for m in comp:
        sb, st = m[0][2], m[1][2]
        s = (st-sb)/st
        scores.append(m[1][2])
        bestscores.append(m[0][2])
        scales.append(s)
        if s == 0: numcorrect += 1
        if m[1][2] > worstTrueScore[1][2]: worstTrueScore = m
        if s > (worstScale[1][2]-worstScale[0][2])/worstScale[1][2]: worstScale = m
        if m[1][2] > thresh: incompatibles.append(m)
    if all: [print(e) for e in comp]
    worstScaleVal = (worstScale[1][2]-worstScale[0][2])/worstScale[1][2]
    print(f"{len(comp)} true matches:")
    print(f"{blue}avg {green}true{blue} score: {sum(scores)/len(scores):.5f}, avg {red}best{blue} score: {sum(bestscores)/len(bestscores):.5f}")
    print(f"{green}avg scale diff to best match: {sum(scales)/len(scales):.5f}")
    print(f"{purple}number of correct best fits: {numcorrect}/{len(comp)}")
    print(f"{cyan}worst true match (absolute): {worstTrueScore}")
    print(f"{yellow}worst true match (scale): {worstScale} with scale {worstScaleVal:.5f}")
    print(f"{red}true matches worse than {thresh}:")
    [print(e) for e in incompatibles]
    print(endc)

def findCheat(p):
    rots = [1]
    print("i must be dreaming. . .")
    sta = p.state.copy()
    for i in range(1, 60):
        pos = (i%10, math.floor(i/10))
        x, y = pos
        best, rot = None, None
        if (x==0 or x==9 or y==0 or y==5):
            for r in range(4):
                if p.validEdgePlacement(i, pos, r):
                    #print(r)
                    rot = r
        else:
            for r in range(4):
                s = p.evalPlacement(i, pos, r, state=sta)
                #print(s)
                if (best == None) or (s < best[1]):
                    best = [r, s]
                    rot = r
        sta.place(i, pos, rot, True)
        #p.showState()
        rots.append(rot)
    return rots

def placementFeedback(p, i, zz, poss, pos, rots, sc2, t):
    idx = 10*pos[1]+pos[0]
    cz, crot = idx, p.cheat[idx]
    sc1 = p.evalPlacement(cz, pos, crot)
    delta = round(time.time()-t, 5)
    if type(zz) == list:
        confirm = ""
        for j in range(len(zz)):
            if zz[j]==10*poss[j][1]+poss[j][0]: confirm += (f"{bcolors.OKGREEN} (CORRECT)")
            else: confirm += (f"{bcolors.FAIL} (INCORRECT)")
        print(f"{bcolors.OKCYAN}{i}: prediction: pcs:{zz} at {poss} with rotations {rots} and a score of {sc2:.5f} {confirm}{bcolors.WARNING} [{delta}s]")
    else:
        if zz==10*poss[1]+poss[0]: confirm = f"{bcolors.OKGREEN} (CORRECT)"
        else: confirm = f"{bcolors.FAIL} (INCORRECT)"
        print(f"{bcolors.OKCYAN}{i}: prediction: pcs:{zz} at {pos} with rotation {rots} and a score of {sc2:.5f} {confirm}{bcolors.WARNING} [{delta}s]")

def listSim(a, b, shift, returnDisp=False):
    a, b = np.array(a), np.array(b)
    if len(b) > len(a):
        a, b = b, a
    #ran = np.arange(0, len(b), len(b)/len(a)).round().astype(np.int32)
    ran = np.linspace(0, len(b), len(a), endpoint=False).round().astype(np.int32)
    b = b[ran]
    dist = np.sum((a-b)**2, axis=1)**.5
    avgDist = dist.sum()/len(dist)
    if shift:
        disp = np.sum(a-b, axis=0)/len(a)
        dispMag = math.sqrt(disp[0]**2 + disp[1]**2)
        if returnDisp: return avgDist - dispMag, disp
        return avgDist - dispMag
    return avgDist

def rotate(pos, angle, origin = (0, 0)):
    #print(pos, type(pos))
    rmat = np.matrix([[math.cos(angle),-math.sin(angle)],
                      [math.sin(angle), math.cos(angle)]])
    return np.array(np.dot(np.matrix(pos.astype(np.float64)-origin), rmat.T))+origin

def scaleImgSet(img, lower, upper, steps):
    inc = (upper-lower)/steps
    return [imscale(img, lower+inc*i) for i in range(steps+1)]

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

def choices(arr, n, out=[], head=True):
    if head:
        out = []
    out.append(arr)
    if len(arr) > n:
        for i, e in enumerate(arr):
            choices(np.delete(arr, i, 0), n, out, head=False)
    return np.unique([e for e in out if len(e) == n], axis=0)

def areaDiff(a, b):
    a1 = cv2.contourArea(a) if len(a) > 0 else 0
    a2 = cv2.contourArea(b) if len(b) > 0 else 0
    return abs(a1-a2)

'''
def match(target, query, retMap = False):
    map = cv2.matchTemplate(target, query, cv2.TM_SQDIFF_NORMED)
    minSim, maxSim, maxSimPos, minSimPos = cv2.minMaxLoc(map)
    return ((maxSimPos, map[maxSimPos[1]][maxSimPos[0]], query, map) if retMap else (maxSimPos, map[maxSimPos[1]][maxSimPos[0]], query))
'''

def splitImage(img, dim):
    sampledim = np.shape(img)
    subdims = (sampledim[0]//dim[1], sampledim[1]//dim[0])
    subs = []
    for j in range(0, dim[0]):
        for i in range(0, dim[1]):
            subs.append(img[subdims[0]*j:subdims[0]*(j+1), subdims[1]*i:subdims[1]*(i+1),])
    return subs

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
