from utils import *

straightSideTypes = {1:0 ,2:1, 3:2, 4:3, 5:4, 6:5, 12:6, 20:7, 10:8}
# 0 is an interior piece, 1-4 are edge pieces, 5-8 are corner pieces. The key values are hashes. see identifyTypes()

class pc:
    def __init__(self, im, loadPath="", savePath="", loadMinimal=True):
        self.base = im
        self.minimal = loadMinimal
        assert loadPath!="" or np.shape(self.base) != (), "no source image or extracted information provided. need one or both"
        #if undistort != None:
        #    mtx, dst, newmtx = undistort
        #    im = cv2.undistort(im, mtx, dst, None, newmtx)
        if loadPath == "":
            self.im = self.preprocess(im)
            self.edge = self.findContours()
            rect = cv2.boundingRect(self.edge)
            self.corners, self.centroid = self.findCorners(rect)
            self.shapes = self.segment()
            self.sideTypes, self.typeID, self.dists = self.identifyTypes()
            self.sides = self.normalizeEdges()
        else:
            assert os.path.isdir(loadPath), f"extracted information location: {loadPath} is not a valid directory"
            if not loadMinimal:
                self.edge = np.load(f"{loadPath}\\edge.npy")
                self.corners = np.load(f"{loadPath}\\corners.npy")
                self.shapes = [np.load(f"{loadPath}\\shape{i}.npy") for i in range(4)]
                centroid = self.corners[3]
                for i in range(3): centroid += self.corners[i]
                self.centroid = centroid/4
            self.sideTypes = np.load(f"{loadPath}\\sideTypes.npy")
            self.sides = [np.load(f"{loadPath}\\side{i}.npy") for i in range(4)]
            #self.sides = [np.load(f"{loadPath}\\side{i}.npy")[::5] for i in range(4)] # take only every 5th contour pt to make loss fucntion faster
            self.dists, tid = [], 1
            for i, t in enumerate(self.sideTypes):
                if t==0: tid *= i+2
                #self.dists.append(math.sqrt(np.sum((self.corners[i]-self.corners[(i+1)%4])**2)))
                self.dists.append(math.sqrt(np.sum((self.sides[i][0]-self.sides[i][-1])**2)))
                try:
                    self.typeID = straightSideTypes[tid]
                except KeyError:
                   assert 0, f"loaded bad edge types:{self.sideTypes}(typeID hash:{tid}) is not a known type of interor/edge piece"

        self.Knbrs = self.generateComparators()
        #self.compatibilityMap = self.generateCompatibleTypes()
        #self.correctedSides = self.rectify(checker, mtx, dst, newmtx)
        
        self.pairingInfo = (self.sides, self.sideTypes, self.dists, self.Knbrs)

        if savePath != "":
            if not os.path.isdir(savePath):
                os.makedirs(savePath)
            if not loadMinimal:
                np.save(f"{savePath}\\edge.npy", self.edge)
                np.save(f"{savePath}\\corners.npy", self.corners)
                for i in range(4): np.save(f"{savePath}\\shape{i}.npy", self.shapes[i])
            np.save(f"{savePath}\\sideTypes.npy", self.sideTypes)
            for i in range(4): np.save(f"{savePath}\\side{i}.npy", self.sides[i])

    def identifyTypes(self, numPts=100, extremePoints=15):
        types = np.array([0,0,0,0])
        dists = []
        tid = 1
        cx, cy = self.centroid
        for i, side in enumerate(self.shapes):
            q = np.linspace(0, len(side)-1, numPts).astype(np.int32)
            midpt = (side[0] + side[-1])/2
            dists.append( math.sqrt((side[0][0]-side[-1][0])**2 + (side[0][1]-side[-1][1])**2) ) 
            d2mid = math.sqrt((midpt[0]-cx)**2 + (midpt[1]-cy)**2)
            midangle = math.atan2(cx-midpt[0], cy-midpt[1])
            ds, iii = [], []
            for j, pt in enumerate(side[q]):
                d = math.sqrt((pt[0]-cx)**2 + (pt[1]-cy)**2)
                an = math.atan2(cx-pt[0], cy-pt[1]) - midangle
                #ds.append(d*math.cos(an) - d2mid)
                ds.append(d*math.cos(an))
                iii.append(j)
            m = np.mean(ds) 
            ds = [e-m for e in ds]
            if np.std(ds) < 35: tid *= i+2
            else:
                a = len(ds)//4
                extr = sorted(ds[a:3*a], key=lambda x: -abs(x))[:extremePoints]
                numpos = sum([1 for e in extr if e > 0])
                numneg = sum([1 for e in extr if e < 0])
                if numneg == 0: types[i] = 1
                elif numpos == 0: types[i] = -1
                elif numneg > 0 and numpos > 0: types[i] = 9
                else: assert 0, f"all extreme points at 0 distance???"
        try:
            typeID = straightSideTypes[tid]
        except KeyError:
            assert 0, f"detected edge types:{types}(typeID:{tid}) is not a known type of interor/edge piece. failure in straight side detection."
        return types, typeID, dists

    def generateCompatibleTypes(self):
        pass

    def findContours(self):
        contours, heirarchy = cv2.findContours(self.im, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contours = [e for e in contours if len(e) > 60]
        edge = contours[0]
        return edge

    def manualCornerSelect(self, rect, blurKernel=(9,9), subPixKernel=(50,50)): #used to repair the piecs whose corner detection did not work.
        croppedComponent = cv2.GaussianBlur(self.im, blurKernel, 0)
        criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 50, 0.001)
        pts = choosePts(croppedComponent, 4, scale=.5).astype(np.float32)
        fpts = len(filter(pts, 50))
        while fpts < 4:
            print(f"{red}you fucked up, reselect points, dont drag mouse{endc}")
            pts = choosePts(croppedComponent, 4, scale=.5).astype(np.float32)
            fpts = len(filter(pts, 15))
        corners = cv2.cornerSubPix(croppedComponent, pts, subPixKernel, (-1,-1), criteria)
        print(f"{cyan}{corners=}")
        self.cm = circles(croppedComponent, corners, width=2, radius=30)
        
        centroid = (0,0)
        for c in corners: centroid += c
        centroid /= 4
        return np.flip(cv2.convexHull(corners)[:,0], axis=0), centroid
        
    def findCorners(self, rect, numClusters=25, threshold=.35, blurKernel=(13,13), blockSize=65, ksize=7, k=.070, criteriaIter=50, subPixKernel=(80,80)):
        #rx, ry, w, h = rect
        #croppedComponent = cv2.GaussianBlur(gray, (11, 11), 0)[ry-10:ry+h+10,rx-10:rx+w+10]
        croppedComponent = cv2.GaussianBlur(self.im, blurKernel, 0)
        #cornerMap = cv2.cornerHarris(croppedComponent, 40, 7, .001) #dinopi
        cornerMap = cv2.cornerHarris(croppedComponent, blockSize=blockSize, ksize=ksize, k=k) #yeet
        ret, cm = cv2.threshold(cornerMap, threshold, 255, cv2.THRESH_BINARY)
        #for i in range(10):
        #    bin = cv2.erode(cm, np.ones((1, 1), np.uint8))
        #    bin = cv2.dilate(cm, np.ones((1, 1), np.uint8))
        y, x = np.where(cm==255)
        candidates = np.float32([[x[i], y[i]] for i in range(len(x))])
        criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 50, 0.001)
        compactness, labels, clusters = cv2.kmeans(candidates, numClusters, np.arange(numClusters), criteria, criteriaIter, cv2.KMEANS_RANDOM_CENTERS)
        centers = cv2.cornerSubPix(croppedComponent, clusters, subPixKernel, (-1,-1), criteria)
        centers = filter(clusters, 15) #filters centers which are on the same corner, averaging them
        #quads = choices(centers, 4) # produces all possible 4 choices of center
        
        if len(centers) < 4:
            cv2.imshow("im", imscale(self.im, .5))
            cv2.imshow("crn", imscale(cornerMap, .5))
            print(f"\n{red}Corner detection failed, zero candidate quadrilaterals")
            cv2.waitKey(0)
        
        hull = cv2.convexHull(centers)[:,0]
        if len(hull)==4: quads = [hull]
        else:
            numquads = 1
            dim = tuple([len(hull) for i in range(4)])
            quads = []
            for i in range(4): numquads *= len(hull)-i
            for i in range(numquads):
                j = np.unravel_index(i, dim)
                if len(set(j)) == 4:
                    quads.append([hull[e] for e in j])

        #self.cm = circles(cm, hull, width=2, radius=30)

        quads = np.array(quads)
        best = (quads[0], cv2.contourArea(quads[0]))
        for e in quads:
            area = cv2.contourArea(e)
            if area > best[1]:
                best = (e, area)
        #corners = cv2.cornerSubPix(croppedComponent, best[0], (50, 50), (-1,-1), criteria)
        corners = best[0]
        
        cv2.imshow("im", imscale(circles(self.base, corners, width=10, radius=30, color=(0, 255, 150)), 0.7))
        # show the autolocated corners, if user presses spacebar, continue with the function, otherwise call manualCornerSelect and return that
        if cv2.waitKey(0) != 32:
            return self.manualCornerSelect(rect)

        centroid = (0,0)
        for c in corners: centroid += c
        centroid /= 4
        return np.flip(cv2.convexHull(corners)[:,0], axis=0), centroid

    def preprocess(self, im, lower=15_000, upper=1_000_000, choice=0):
        assert np.shape(im) != (), f"base image was not provided, cannot preprocess"
        h, w, d = np.shape(im)
        center = (w/2, h/2)
        gray = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
        blur = cv2.GaussianBlur(gray, (3,3), 50)
        ret, bin = cv2.threshold(blur, 160, 255, cv2.THRESH_BINARY_INV)
        #bin = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 7, 1)
        for i in range(1):
            bin = cv2.erode(bin, np.ones((3, 3), np.uint8))
            bin = cv2.dilate(bin, np.ones((1, 1), np.uint8))
        
        labels, labelids, values, centroids = cv2.connectedComponentsWithStats(bin, 4, cv2.CV_32S)
        qualified = []
        for i, e in enumerate(values):
            mass, height, width, x, y = e[4], e[3], e[2], centroids[i][0], centroids[i][1] 
            if (lower < mass < upper) and (height<h and width<w) and (w/4)<x<(3*w)/4 and (h/4)<y<(3*h)/4:
            #if (height<h and width<w) and (w/4)<x<(3*w)/4 and (h/4)<y<(3*h)/4:
                d = ((x-center[0])**2 + (y-center[1])**2)
                qualified.append((i, d))
        assert len(qualified) != 0, f"connected component failed: no piece match.\nComponents found:\n{values}"
        qualified.sort(key=lambda x: x[1])
        component = (labelids == qualified[choice][0]).astype("uint8")*255
        return component

    def segment(self):
        closest = [0, 0, 0, 0]
        closestDists = [None, None, None, None]
        localArea = .05*np.max(np.shape(self.im))
        for i, p in enumerate(self.edge):
            for j, c in enumerate(self.corners):
                if (c[0]+localArea > p[0][0] > c[0]-localArea) and (c[1]+localArea > p[0][1] > c[1]-localArea):
                    d = np.sum((c-p)**2)
                    if closestDists[j]==None or d<closestDists[j]:
                        closest[j] = i
                        closestDists[j] = d
        if None in closestDists: return None 
        shapes = []
        for i, idx in enumerate(closest):
            if closest[i] < closest[(i+1)%4]:
                seg = self.edge[idx:closest[(i+1)%4],0].astype(np.float64)
            else:
                seg = np.vstack((self.edge[idx:-1,0],self.edge[0:closest[(i+1)%4],0])).astype(np.float64)
            shapes.append(seg)
        return shapes

    def normalizeEdges(self):
        edges = []
        for i, edge in enumerate(self.shapes):
            seg = np.array(edge)
            if self.sideTypes[i] == -1:
                seg = np.flipud(seg)
            seg -= seg[0]
            seg = rotate(seg, -math.atan2(seg[-1][1], seg[-1][0]))
            edges.append(seg)
        return edges

    def generateComparators(self):
        return [NearestNeighbors(n_neighbors=1, algorithm='kd_tree', metric='euclidean').fit(edge) for edge in self.sides]

    def correctPerspective(self):
        cornerMap = cv2.cornerHarris(self.backgr, 15, 5, .001)
        y, x = np.where(cornerMap>.5*np.max(cornerMap))
        candidates = np.float32(np.array([[x[i], y[i]] for i in range(len(x))]))
        candidates = filter(candidates, 30)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners = cv2.cornerSubPix(self.im, candidates, (15,15), (-1,-1), criteria)

        corners = cv2.convexHull(corners)
        h, w = np.shape(self.im)
        rect = np.array([[0, 0], [w, 0], [w, h], [0, h]], np.float32)
        mat = cv2.getPerspectiveTransform(corners, rect)
        shifted = [[], [], [], []]
        for i, side in enumerate(self.sides):
            for p in side:
                new =  [(mat[0][0]*p[0] + mat[0][1]*p[1] + mat[0][2])/(mat[2][0]*p[0] + mat[2][1]*p[1] + mat[2][2]),
                        (mat[1][0]*p[0] + mat[1][1]*p[1] + mat[1][2])/(mat[2][0]*p[0] + mat[2][1]*p[1] + mat[2][2])]
                shifted[i].append(new)
        self.warped = cv2.warpPerspective(self.base, mat, (w, h))
        '''
        for i, side in enumerate(shifted):
            self.warped = cv2.polylines(self.warped, np.int32([side]), False, (250-70*i, 150-50*i, 80*i), 2)
        '''
        return shifted

    def rectify(self, checker, mtx, dst, new, size=7):
        assert checker is not None, 'empty checkerboard image'
        checker = cv2.undistort(checker, mtx, dst, None, new)
        self.undist = cv2.undistort(self.base, mtx, dst, None, new)
        obj = np.zeros((size*size,3), np.float32)
        obj[:,:2] = np.mgrid[0:size,0:size].T.reshape(-1,2)
        ret1, pts = cv2.findChessboardCorners(checker, (size,size))
        assert ret1, 'Checkerboard corner detection failed D:'
        gray = cv2.cvtColor(checker, cv2.COLOR_BGR2GRAY)
        pts = cv2.cornerSubPix(gray,pts, (13,13), (-1,-1), (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
        #checker = cv2.imshow('c', cv2.drawChessboardCorners(checker, (size,size), pts, False))
        ret2, rvecs, tvecs = cv2.solvePnP(obj, pts, mtx, dst)
        assert ret2, 'Pose estimation failed'
        scale = 4000
        invnew = np.linalg.inv(new)
        rotation_mtx, jac = cv2.Rodrigues(rvecs)
        inverse_rotation_mtx = np.linalg.inv(rotation_mtx)
        #extrinsic_mtx = np.column_stack((rotation_mtx,tvecs))
        rectified = [[], [], [], []]
        for i, side in enumerate(self.sides):
            for pt in side:
                uv = scale*(np.matrix([pt[0], pt[1], 1]).T)
                xyz = np.matrix(invnew.dot(uv))
                xyz = inverse_rotation_mtx.dot(xyz-tvecs).T
                xyz = np.array(xyz)[0][0:2]
                rectified[i].append(xyz)
        #assert np.ndim(rectified[0]) == 2, f"output points are of shape {np.ndim(rectified[0])}: {rectified}"
        for i, side in enumerate(rectified):
            side = np.array(side)
            self.undist = cv2.polylines(self.undist, np.int32([side+[700,700]]), False, (250-70*i, 150-50*i, 80*i), 4)
        return rectified

    def show(self, base=True, scale=1, edges=True, corners=True, center=True, radius=19, thickness=3):
        assert not self.minimal, f"minimal information was loaded from file, cannot display image overlay"
        if self.base is None:
            self.im = self.preprocess(self.base)
        if base:
            mod = np.copy(self.base)
        else:
            mod = cv2.cvtColor(self.im, cv2.COLOR_GRAY2BGR)

        if edges:
            if len(self.shapes) > 0:
                for i, e in enumerate(self.shapes):
                    #mod = cv2.drawContours(mod, [e], -1, (250-50*i, 150-50*i, 80*i), 3)
                    mod = cv2.polylines(mod, np.int32([e]), False, (250-70*i, 150-50*i, 80*i), thickness)
                    #mod = circles(mod, [e[len(e)//2]], radius=radius, width=thickness, color=(250-70*i, 150-50*i, 80*i)) #midpoints
            else:
                #mod = cv2.drawContours(mod, [self.edge], -1, (250, 150, 0), 3)
                mod = cv2.polylines(mod, np.int32([self.shapes]), False, (250-70*i, 150-50*i, 80*i), thickness)
        if corners:
            mod = circles(mod, self.corners, radius=radius, width=thickness, color=(150, 30, 255))
        if center:
            mod = cv2.circle(mod, (round(self.centroid[0]), round(self.centroid[1])), radius, (130, 255, 50), thickness)
        h, w, d = np.shape(mod)
        #mod = cv2.circle(mod, (w//2, h//2), radius=30, color=(160, 255, 250), thickness=4)
        return imscale(mod, scale)

def makePcs(imgdir, num, load="", save=""):
    pcs = []
    if load != "": print(f"{yellow}loading extracted piece information from {load}{endc}")
    if save != "": print(f"{yellow}saving extracted piece information to {save}{endc}")
    for i in tqdm(range(num), desc=f"{green}collecting piece information{endc}", ncols=100, unit="pcs"):
        im = cv2.imread(f"{imgdir}\\{i}.png") if imgdir != "" else None
        loadpath = f"{load}\\{i}" if load != "" else ""
        savepath = f"{save}\\{i}" if save != "" else ""
        newp = pc(im, loadPath=loadpath, savePath=savepath)
        #cv2.imshow(f'pc', newp.show(scale=.6))
        #print(f"\n{green}{i}")
        #cv2.imshow(f'crnrs', imscale(newp.cm, .5))
        #cv2.waitKey(0)
        pcs.append(newp)
    return pcs

def checkPcs(pcs, shape):
    pw, ph = shape
    straightSides, edgePcs, cornerPcs, maleSides, femaleSides, otherSides = 0, 0, 0, 0, 0, 0
    correctNumEdgePcs, correctNumStraightSides = 2*(pw+ph-4), 2*(pw+ph)
    for p in tqdm(pcs, ncols=100, desc=f"{purple}checking edge type distribution in piece pool{endc}"):
        typeID, types = p.typeID, p.sideTypes
        if 0 < typeID < 5:
            edgePcs += 1
            straightSides += 1
        if 4 < typeID:
            cornerPcs += 1
            straightSides += 2
        for t in types:
            if t == 1: maleSides += 1
            if t == -1: femaleSides += 1
            if t == 9: otherSides += 1
    #print(straightSides, edgePcs, cornerPcs, maleSides, femaleSides, otherSides)
    #assert straightSides==correctNumStraightSides, f"{red}number of detected straight sides:{straightSides} not correct for puzzle of shape {shape}: {correctNumStraightSides}"
    #assert edgePcs==correctNumEdgePcs, f"{red}number of detected edge pieces:{edgePcs} does not match correct number for puzzle of shape {shape}: {correctNumEdgePcs}"
    #assert cornerPcs==4, f"{red}number of detected edge pieces is {cornerPcs}. should be 4 right?"
    #assert maleSides==femaleSides, f"{red}number of detected female sides ({femaleSides}) does not match number of male sides: ({maleSides})"
    #assert  otherSides%2==0, f"{red}odd number of type other. they should probably each have a pair i think"
    if straightSides != correctNumStraightSides: print(f"{red}number of detected straight sides:{straightSides} not correct for puzzle of shape {shape}: {correctNumStraightSides}")
    if edgePcs != correctNumEdgePcs: print(f"{red}number of detected edge pieces:{edgePcs} does not match correct number for puzzle of shape {shape}: {correctNumEdgePcs}")
    if cornerPcs != 4: print(f"{red}number of detected edge pieces is {cornerPcs}. should be 4 right?")
    if maleSides != femaleSides: print(f"{red}number of detected female sides ({femaleSides}) does not match number of male sides: ({maleSides})")
    if  otherSides%2 != 0: print(f"{red}odd number of type other. they should probably each have a pair i think")