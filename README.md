# Puzzle Solver
  This code is a project whose aim is to automatically extract features from pictures of puzzle pieces from, and then
use that information to solve the puzzle. Calibrator and cameraparam is code for getting information about the camera 
matrix and distortion, which is not currently corrected as of the most recent version, but the code is there. You will
find the image processing/info extraction code in piece.py. Most of the solving code is in puzl.py. bState is a class 
describes a particular board state, which are created and evaluated as part of the search algorithm. This code was 
sucessfully tested on two puzzles, a 10x6 and an 18x18. The automatic feature detectin worked for the 60 piece puzzle,
but even after a good maount of tuning and tweaking, I could not successfully detect features for all 324 pieces in 
the 18x18. In the end, there was about 10 pieces which I had to go in and manually correct. Once all data has been
properly extracted, a correct configuration using all pieces was found succesfully, starting with only a single  correct
placement from the user. The search algorithm is a basic A*, with options to do simple weighting of the heuristic and
cost function. There are several improvements that could be made to this project as of right now. First off, it
would obviously be preferable to make the automatic feature extraction more reliable. On the solving side though, there
are two things that come to mind: one is that I think there is error in the loss function or from perspective  distortion
which causes some correct pairings to have unusually poor scores. The second is that this implementation currently only
uses normal or simple weighted A*. I am quite confident that using some more advanced A* variant (or another kind of
search altogether) would yeild correct configurations faster.

## Image Processing
  There is only really one thing we need to begin evaluting how well two pieces fit together: the shape of the sides.
But getting this involves a few intermediate steps, and some extra information we find to make solving easier. A few 
example images from the puzzles I tested on are included in exampleims. This is the basic steps on what we extract
from our images, and package it all into a pc class member:
  - First we find the contour of the whole piece, after binarizing.
  - We find the 4 corners of the piece
  - Split the full contour at the spots which are closest to each corner.
  - Identify the type of each side (straight, male, female, other)
  - Normalize the edges, and generate a comparator class for the loss function

  The point of failure for the large, puzzle, and the only step that needed my intervention was the corner detection. The
algorithm I used to select points is convoluted, and includes many different parameters (magic numbers) whose correct values,
if they exist, are highly sensitive to the morphology of the pieces you are working with, and how you photograph them. I roughly
outline the corner selection procedure below:
  - Get a corner map with cv2's Harris Corner Detection.
  - Binarize the map with a hand picked threshold.
  - Find clusters of cornery pixels with K Means Clustering. These are our candidates
  - Use cornerSubPix on each candidate to find the most cornery part of the feature.
  - Filter candidates by distance so that we dont get multiple candidates describing the same feature
  - Take only points on the convex hull of the set of candidates. Corners are unlikely to be interior to any other set of points. (*)
  - Go through all possible 4 choices of points on the hull, and select the quadrilateral which has the greatest area.
  
  
  
  



















