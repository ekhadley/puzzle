from funcs import *
class boardState:
    def __init__(self, shape, numPcs, state=None):
        self.numPcs = numPcs
        self.shape = shape
        self.cost = 0
        if type(state) == type(None):
            w, h = self.shape
            self.pcState = -1*np.ones((h, w), dtype=np.int32)
            self.rotState = -1*np.ones((h, w), dtype=np.int32)
            self.borders = -1*np.ones((h, w, 4, 2), dtype=np.int32)
            self.placed = np.int32([])
            self.unplaced = np.linspace(0, numPcs-1, numPcs).astype(np.int32)
        elif type(state) == list:
            self.pcState = np.array(state[0], copy=True)
            self.rotState = np.array(state[1], copy = True)
            self.borders = np.array(state[2], copy = True)
            self.placed = np.array(state[3], copy=True)
            self.unplaced = np.array(state[4], copy=True)
        else: assert 0, f"unkonwn type provided to boardState initializer as parent state. expected boardState class, received:{type(state)}:\n{state}"
    
    def place(self, pz, pos, rot, keep=True):
        if keep:
            pcState = self.pcState
            rotState = self.rotState
            borders = self.borders
        else:
            pcState = np.array(self.pcState, copy=True)
            rotState = np.array(self.rotState, copy=True)
            borders = np.array(self.borders, copy=True)
        placed, unplaced = None, None
        if type(pz) == int:
            pz, pos, rot = [pz], [pos], [rot]
        elif type(pz) == list: pass
        else: assert 0, f"unknown type/s passed to place() function. received: {pz}({type(pz)}), {pos}({type(pos)}), and {rot}({type(rot)})"
        for i in range(len(pz)):
            #assert type(pos[i]) == tuple, f"huh??got: {pos}, so pos[{i}] is {pos[i]}"
            x, y = pos[i]
            pcState[y][x] = pz[i]
            rotState[y][x] = rot[i]
            borders = self.updateBorders(pz[i], pos[i], rot[i], inBorders=borders)
            if keep:
                self.unplaced = np.delete(self.unplaced, np.where(self.unplaced==pz[i]))
            else:
                if unplaced is None: unplaced = np.array(self.unplaced, copy=True)
                #unplaced = np.delete(unplaced, np.where(unplaced==pz[i]))
                unplaced = np.int32([e for e in unplaced if e != pz[i]])
        if keep:
            self.placed = np.append(self.placed, pz, axis=0)
        else:
            if type(placed)==type(None): placed = np.array(self.placed, copy=True)
            placed = np.append(placed, pz, axis=0)
        if not keep: return boardState(self.shape, self.numPcs, state=[pcState, rotState, borders, placed, unplaced])

    def updateBorders(self, pz, pos, rot, inBorders=None):
        borders = np.array(self.borders, copy=True) if type(inBorders)==type(None) else inBorders
        x, y = pos
        c = [(x,y+1), (x+1,y), (x,y-1), (x-1,y)]
        for direc, spot in enumerate(c):
            i, j = spot
            if 0<=i<self.shape[0] and 0<=j<self.shape[1]:
                borders[j][i][direc][0] = pz
                borders[j][i][direc][1] = (rot+direc-2)%4
        return borders

    def getPiece(self, pos):
        return self.pcState[pos[1],pos[0]]
    def getRotation(self, pos):
        return self.rotState[pos[1],pos[0]]
    def getBorders(self, pos, num=False):
        borders = self.borders[pos[1],pos[0]]
        if num:
            numb = 0
            for b in borders:
                if b[0] != -1: numb += 1
            return (borders, numb)
        return borders
    def numBorders(self, pos):
        bord = self.getBorders(pos)
        return np.sum(bord[:,0]!=-1)

    def perimeterPositions(self, prio=True): # use gradient in diff directions then add. should be faster but this aint gotta be that fast
        per = self.numBorderMask()
        spots = []
        if prio: 
            for i in range(4):
                where = np.where(per==i+1)
                spots.append([(where[1][s], where[0][s]) for s in range(len(where[0]))])
        else:
            where = np.nonzero(per)
            spots = [(where[1][s], where[0][s]) for s in range(len(where[0]))]
        return spots
    
    def numBorderMask(self):
        up, down, left, right = self.pcState[0:-1]!=-1, self.pcState[1:]!=-1, self.pcState[:,1:]!=-1, self.pcState[:,0:-1]!=-1
        mask = (self.pcState==-1).astype(np.int32)
        up = np.pad(up, ((1,0),(0,0)), constant_values=False).astype(np.int32)
        down = np.pad(down, ((0,1),(0,0)), constant_values=False).astype(np.int32)
        left = np.pad(left, ((0,0),(0,1)), constant_values=False).astype(np.int32)
        right = np.pad(right, ((0,0),(1,0)), constant_values=False).astype(np.int32)
        #per = ((up+down+left+right)*mask).astype(np.int32)
        per = (up+down+left+right)*mask
        return per
        
    def showState(self):
        c = [purple, red, blue, green, cyan, yellow, lime, orange, pink]
        r = ""
        for row in self.pcState:
            for e in row:
                if e != -1:
                    if e < 100:
                        r += random.choice(c)
                        r += (f" {e}  " if e < 10 else f" {e} ")
                    else:
                        r += f"{random.choice(c)} {e}"
                else:
                    r += gray + " x  "
            r += "\n"
        print(r + endc)
    
    def copy(self):
        return boardState(self.shape, self.numPcs, [np.array(self.pcState, copy=True), np.array(self.rotState, copy=True), np.array(self.borders, copy=True)])
    
    def canFit(self, spots, shape, rotation=True, reflection=True): #problem with lshapes: try: 
        assert type(shape)==list, f"input shape should be a list of tuple. instead got: {shape}"
        assert (0,0) in shape, "input shape must include (0, 0) and all relative positions to (0,0), if any"
        spots, shape = spots[:], shape[:]
        if type(shape)!=tuple:
            if any(x*y != 0 for x, y in shape): [spots.append(c) for c in self.perimCorners(spots)]
        relatives, fits = [], []
        for coordX, coordY in spots:
            relatives.append({(coordX+shapeX, coordY+shapeY) for shapeX, shapeY in shape})
            if rotation:
                relatives.append({(coordX+shapeY, coordY+shapeX) for shapeX, shapeY in shape})
            if reflection:
                relatives.append({(coordX+shapeX, coordY-shapeY) for shapeX, shapeY in shape})
                relatives.append({(coordX-shapeX, coordY+shapeY) for shapeX, shapeY in shape})
        spots = set(spots)
        for rel in relatives:
            lr = list(rel)
            if lr not in fits:
                if rel.issubset(spots):
                    fits.append(lr)
        return fits

    def perimCorners(self, spots):
        corn = []
        diags = [(-1,-1),(1,-1),(1,1),(-1,1)]
        orthos = [(-1,0),(0,-1),(1,0),(0,1)]
        for spx, spy in spots:
            for i, (dix, diy) in enumerate(diags):
                if (spx+dix, spy+diy) in spots:
                    ox1, oy1 = orthos[i]
                    ox2, oy2 = orthos[(i+1)%4]
                    c1, c2 = (spx+ox1, spy+oy1), (spx+ox2, spy+oy2)
                    if self.getPiece(c1)==-1: corn.append(c1)
                    if self.getPiece(c2)==-1: corn.append(c2)
        return list(set(corn))
    
    def __lt__(self, other):
        return self.cost < other.cost