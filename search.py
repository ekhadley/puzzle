from funcs import *
import piece, bState, puzl
from matplotlib import pyplot as plt

imgdir = "C:\\Users\\ek\\Desktop\\testimgs\\yeet\\imgs"
infodir = "C:\\Users\\ek\\Desktop\\testimgs\\yeet\\extracted"
pw, ph = 18, 18

pcs = piece.makePcs("", pw*ph, load=infodir)
#pcs = piece.makePcs(imgdir, pw*ph, load=infodir)
#pcs = piece.makePcs(imgdir, pw*ph, save=infodir)
#pcs = piece.makePcs(imgdir, 60)

piece.checkPcs(pcs, (pw, ph))

inps = [(0,(0,0),0), (17,(17,0),0), (306,(0,17),0), (323,(17,17),0)]
initial = bState.boardState((pw, ph), len(pcs))
solver = puzl.aStarSolver(initial, pcs, initialPlacement=inps[0])
best = solver.solve(printEvery=3, heuristicScale=.8, maxRank=3, maxScore=25)


while 1: cv2.waitKey(1)
#for i, z in enumerate(pcs):
#    cv2.imshow(f'pc{i}', z.show(base=True, scale=.5, thickness=2))
#    cv2.waitKey(1)
#cv2.waitKey(0)
