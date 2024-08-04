from piece import *

def validPairing(types1, dists1, types2, dists2, first, second, lengthThresh):
    if types1[first] != 9 and types2[second] != 9:
        if types1[first] == types2[second]:
            #print(f"matching side types:{types1[first]=}, {types2[second]=}")
            return False
    gap = abs(dists1[first] - dists2[second]) 
    if gap > lengthThresh:
        #print(f"distance gap too large:{dists1[first]=}, {dists2[second]=}, {gap=}, {lengthThresh=}")
        return False
    if not validStraightSides(types1, types2, first, second):
        #print(f"invalid due to straight sides: {types1, first}, {types2, second}")
        return False
    return True

def evalMatch(pcList, pcIdxs, sideIdxs, store=None, lengthThresh=50):
    first, second = sideIdxs
    pcid1, pcid2 = pcIdxs
    if store is not None and (storedScore := store[pcid1][pcid2][first][second]) != -1: return storedScore # walrus!!!!!!!
    sides1, types1, dists1, Knbrs1 = pcList[pcid1].pairingInfo
    sides2, types2, dists2, Knbrs2 = pcList[pcid2].pairingInfo
    s2 = sides2[second]
    
    if validPairing(types1, dists1, types2, dists2, first, second, lengthThresh):
        distances, indices = Knbrs1[first].kneighbors(s2)
        fit = np.mean(distances)
    else:
        #print(bold, red, f"invalid pairing: {pcid1, pcid2, first, second}", endc)
        fit = 1e6
    if store is not None:
        store[pcid1][pcid2][first][second] = fit
        store[pcid2][pcid1][second][first] = fit
    return fit

def newStore(numPcs):
    return -1*np.ones((numPcs, numPcs, 4, 4)).astype(np.float64)

def evalPlacement(pcList, borders, stateShape, queryPiece, pos, rot, store=None): 
    #borders, numborders = state.getBorders(pos, num=True)
    cumscore, scorenum = 0, 0
    for borderSide, other in enumerate(borders):
        otherPc, otherSide = other[0], other[1]
        if otherPc != -1:
            querySide = (rot+borderSide)%4
            if validEdgePlacement(pcList, stateShape, queryPiece, pos, rot):
                fit = evalMatch(pcList, (queryPiece, otherPc), (querySide, otherSide), store=store)
                cumscore, scorenum = cumscore+fit, scorenum+1
            else:
                cumscore, scorenum = 1e6, scorenum+1
                break
    return cumscore/scorenum

def bestFit(pcList, state, pos, returnAll=False, store=None):
    best, allFits = None, []
    borders, numBorder = state.getBorders(pos, num=True)
    assert numBorder > 0, f"No match evaluation performed: 0 bordering edges at {pos} in state:\n{state.showState()}"
    stateShape = state.shape
    for pcIdx in state.unplaced:
        for rot in range(4):
            score = evalPlacement(pcList, borders, stateShape, pcIdx, pos, rot, store=store)
            if best == None or score < best[2]:
                best = (pcIdx, rot, score)
            allFits.append([int(pcIdx), pos, rot, score])
    if returnAll: return allFits
    return best

def validEdgePlacement(pcList, stateShape, pcidx, pos, rot):
    types = pcList[pcidx].sideTypes
    x, y = pos
    w, h = stateShape
    if x == 0 and types[(rot+1)%4] != 0:
        return False
    if y == 0 and types[rot] != 0:
        return False
    if y == h-1 and types[(rot+2)%4] != 0:
        return False
    if x == w-1 and types[(rot-1)%4] != 0:
        return False
    typeID = pcList[pcidx].typeID
    if typeID != 0 and (x*y!=0 and x!=w-1 and y!=h-1):
        return False
    if typeID < 5 and pos in [(0,0), (w-1,0), (w-1,h-1), (0,h-1)]:
        return False
    return True

def validStraightSides(types1, types2, first, second):
    num1, num2 = sum(types1==0), sum(types2==0)
    if num1==0 and num2==0:
        return True
    elif (types1[first]==0) or (types2[second]==0):
       return False
    elif num1>0 and num2>0: # if both pieces are edges/corners
        #check that straight sides have continuity when evaluating placement.
        if num1==2 and num2==2: return False
        return (types1[(first-1)%4]==0 and types2[(second+1)%4]==0) or (types1[(first+1)%4]==0 and types2[(second-1)%4]==0)
    elif num1>0 and num2==0: # when comparing interior pieces to edge/corner piece
        if sum(types1==0)==2: # all pieces with borders to a corner piece must be edge peices
            return False
        else: #if pc1 is an edge piece, not a corner piece, and pc2 is an interior peice:
            i = np.where(types1==0)[0] # i is the straight side index, ((i+2)%4) is the opposite side
            return first==((i+2)%4) # interior pieces only fit on the opposite side from the straight side
    elif num1==0 and num2>0: # switch case of the above one
        if sum(types2==0)==2:
            return False
        else:
            i = np.where(types2==0)[0]
            return second==((i+2)%4)

class aStarSolver():
    def __init__(self, initialState, pcs,  initialPlacement=None):
        self.pcs = pcs
        if initialPlacement is None: self.initialState = self.makeInitialPlacement(initialState)
        else:
            ipz, ipos, irot = initialPlacement
            self.initialState = initialState.place(ipz, ipos, irot, keep=False)

    def chooseInitialPlacement(self, state):
        pass

    def solve(self, printEvery=500, maxScore=100, maxRank=10, costScale=1, heuristicScale=1, horizontal=True):
        t = time.time()
        pcs = self.pcs
        matchStore = newStore(len(pcs))
        i, states = 0, []
        stateShape, best = self.initialState.shape, self.initialState
        heapq.heappush(states, (0.01, self.initialState))
        if horizontal: spots = [(i%stateShape[0], i//stateShape[1]) for i in range(len(pcs))]
        else: spots = [(i//stateShape[0], i%stateShape[1]) for i in range(len(pcs))]

        while len(best.unplaced) != 0:
            #nbrs = Neighbors(pcs, best, stateShape, spots[len(best.placed)], cutoffRank=maxRank, cutoffScore=maxScore, store=matchStore)
            nbrs = neighbors(pcs, best, cutoffScore=maxScore, cutoffRank=maxRank, store=matchStore, horizontal=horizontal)
            
            for nbr in nbrs:
                hScore = costToComplete(nbr)
                heapq.heappush(states, (costScale*nbr.cost + heuristicScale*hScore, nbr))
            fScore, best = heapq.heappop(states)
            
            if i%printEvery == 0:
                best.showState()
                print(f"expanded state with cost {red}{best.cost:.5f}{endc} (average cost/placement={red}{best.cost/len(best.placed):.10f}{endc}) [working on {purple}{len(states):,}{endc} states]({purple}{time.time()-t:.4f}s{endc})\n")
            i += 1
        print(f"solution found! total cost of the solution is {red}{best.cost:.4f}{endc}, ")
        print(f"with an average cost/placement of {red}{best.cost/len(best.placed):.4f}{endc}({purple}{time.time()-t:.4f}s{endc})")
        best.showState()
        return best

def Neighbors(pcList, state, stateShape, spot, cutoffRank=5, cutoffScore=100, store=None):
    #this one assumes starting piece is at (0,0), so uses precomputed positions to avoid perimeterPositions() call
    borders = state.getBorders(spot)
    allFits = []
    for pcIdx in state.unplaced:
        for rot in range(4):
            cumscore, scorenum = 0, 0
            for borderSide, other in enumerate(borders):
                otherPc, otherSide = other[0], other[1]
                if otherPc != -1:
                    querySide = (rot+borderSide)%4
                    if validEdgePlacement(pcList, stateShape, pcIdx, spot, rot):
                        print(2319408912348704123)
                        fit = evalMatch(pcList, (pcIdx, otherPc), (querySide, otherSide), store=store)
                        cumscore, scorenum = cumscore+fit, scorenum+1
                    else:
                        cumscore, scorenum = 1e6, scorenum+1
                        break
            score = cumscore/scorenum
            if score < cutoffScore: allFits.append([int(pcIdx), spot, rot, score])
    
    placements = sorted(allFits, key=lambda x: x[3])
    nbrs = []
    for placement in placements[:cutoffRank]:
        pz, pos, rot, score = placement
        if score < cutoffScore:
            nbr = state.place(pz, pos, rot, keep=False)
            nbr.cost = state.cost + score
            #nbr.cost = state.cost + score**2
            nbrs.append(nbr)
            #print(f"{cyan}{placement}")
        else: break
    return nbrs

def neighbors(pcList, state, cutoffScore, cutoffRank, store=None, horizontal=False): # looks for possible placements at only one perimeter vacancy
    placeableSpots = state.perimeterPositions(prio=True)
    if len(placeableSpots[3]) != 0: spots=placeableSpots[3]
    elif len(placeableSpots[2]) != 0: spots=placeableSpots[2]
    elif len(placeableSpots[1]) != 0: spots=placeableSpots[1]
    elif len(placeableSpots[0]) != 0: spots=placeableSpots[0]
    else: assert 0, f"no valid perimeter positions found. state: {state.showState()}"
    placements = bestFit(pcList, state, spots[0 if horizontal else -1], store=store, returnAll=True)
    placements.sort(key=lambda x: x[3])
    nbrs = []
    for placement in placements[:cutoffRank]:
        pz, pos, rot, score = placement
        if score < cutoffScore:
            nbr = state.place(pz, pos, rot, keep=False)
            nbr.cost = state.cost + score
            #nbr.cost = state.cost + score**2
            nbrs.append(nbr)
        else: break
    return nbrs

def neighbors_(pcList, state, cutoffRank=5, cutoffScore=1_000, store=None): #looks at possible palcements for every perimeter vacancy
    placeableSpots = state.perimeterPositions(prio=True)
    if len(placeableSpots[3]) != 0: spots=placeableSpots[3]
    elif len(placeableSpots[2]) != 0: spots=placeableSpots[2]
    elif len(placeableSpots[1]) != 0: spots=placeableSpots[1]
    elif len(placeableSpots[0]) != 0: spots=placeableSpots[0]
    else: assert 0, f"no valid perimeter positions found. state: {state.showState()}"
    nbrs = []
    for spot in spots:
        placements = bestFit(pcList, state, spot, store=store, returnAll=True)
        placements.sort(key=lambda x: x[3])
        if placements[0][3] > cutoffScore: return []
        for placement in placements[:cutoffRank]:
            pz, pos, rot, score = placement
            if score < cutoffScore:
                nbr = state.place(pz, pos, rot, keep=False)
                nbr.cost = state.cost + score
                nbrs.append(nbr)
    return nbrs

def costToComplete(state):
    return (state.cost/len(state.placed)) * len(state.unplaced)

def astar(initialState, pcs, cutoffRank=10, cutoffScore=100, printEvery=100):
    t = time.time()
    matchStore = newStore(len(pcs))
    states = []
    heapq.heappush(states, (0, initialState))
    best = initialState
    i = 0
    while len(best.unplaced) != 0:
        nbrs = neighbors(pcs, best, cutoffRank=cutoffRank, cutoffScore=cutoffScore, store=matchStore)
        for nbr in nbrs:
            hScore = 1*costToComplete(nbr)
            heapq.heappush(states, (nbr.cost + hScore, nbr))
        fScore, best = heapq.heappop(states)
        
        if i%printEvery == 0:
            best.showState()
            print(f"expanded state with cost {red}{best.cost:.5f}{endc} (average cost/placement={red}{best.cost/len(best.placed):.10f}{endc}) [working on {purple}{len(states):,}{endc} states]({purple}{time.time()-t:.4f}s{endc})\n")
        i += 1
    print(f"solution found! total cost of the solution is {red}{best.cost:.4f}{endc}, ")
    print(f"with an average cost/placement of {red}{best.cost/len(best.placed):.4f}{endc}({purple}{time.time()-t:.4f}s{endc})")
    best.showState()
    return best