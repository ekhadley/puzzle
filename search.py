from funcs import *
import piece, bState, puzl

imgdir = "C:\\Users\\ek\\Desktop\\testimgs\\yeet\\imgs"
infodir = "C:\\Users\\ek\\Desktop\\testimgs\\yeet\\extracted"
pw, ph = 18, 18
#pw, ph = 10, 6

pcs = piece.makePcs("", pw*ph, load=infodir)
#pcs = piece.makePcs(imgdir, pw*ph, load=infodir)
#pcs = piece.makePcs(imgdir, pw*ph, save=infodir)
#pcs = piece.makePcs(imgdir, pw*ph)

piece.checkPcs(pcs, (pw, ph))

inps = [(0,(0,0),0), (17,(17,0),1), (306,(0,17),0), (323,(17,17),0), (12, (12,0), 0)]
initial = bState.boardState((pw, ph), len(pcs))
solver = puzl.aStarSolver(initial, pcs, initialPlacement=inps[0])
best = solver.solve(printEvery=500, heuristicScale=1, maxRank=5, maxScore=15, horizontal=True)

# random idea: look at partial solves, correct and incorrect, and look for identifiable
# differences in distribution (of scores, rotations, idk) between true configs and false
# (greedier) ones. Could be similair to asuming correct config has time constant delta
# score per placement.
