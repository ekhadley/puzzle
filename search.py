import piece
import bState
import puzl


puzzle_name = ['dinopi', 'yeet'][1]
imgdir = f"puzzle_imgs\\{puzzle_name}\\imgs"
infodir = f"puzzle_imgs\\{puzzle_name}\\extracted"
if puzzle_name == 'dinopi':
    pw, ph = 10, 6
    initial_placement = (0,(0,0),1)
elif puzzle_name == 'yeet':
    pw, ph = 18, 18
    initial_placement = (0,(0,0),0)
    #initial_placement = (17,(17,0),1)
    #initial_placement = (306,(0,17),0)
    #initial_placement = (323,(17,17),0)
else:
    raise ValueError("unrecognized puzzle name")

if __name__ == "__main__":
    pcs = piece.makePcs(imgdir, pw*ph, load=infodir) # this loads in the stored piece info
    #pcs = piece.makePcs(imgdir, pw*ph, save=infodir) # this will rewrite the saves with autoextracted piece info
    #pcs = piece.makePcs(imgdir, pw*ph) # this will load the images and autoextract piece info without saving
    piece.checkPcs(pcs, (pw, ph))

    initial = bState.boardState((pw, ph), len(pcs))
    solver = puzl.aStarSolver(initial, pcs, initialPlacement=initial_placement)
    best = solver.solve(printEvery=50, heuristicScale=1, maxRank=10, maxScore=100, horizontal=True)

    # random idea: look at partial solves, correct and incorrect, and look for identifiable
    # differences in distribution (of scores, rotations, idk) between true configs and false
    # (greedier) ones. Could be similair to asuming correct config has time constant delta
    # score per placement.
