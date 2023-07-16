# Puzzle Solver
  This code is a project whose aim is to automatically extract features from pictures of puzzle pieces, and then
use that information to solve the puzzle. Calibrator and cameraparam is code for getting information about the camera 
matrix and distortion, which is not currently corrected for as of the most recent version, but the code is there. You will
find the image processing/info extraction code in piece.py. Most of the solving code is in puzl.py. bState is a class 
which describes a particular board state, which are created and evaluated as part of the search algorithm. This code was 
sucessfully tested on two puzzles, a 10x6 and an 18x18. The automatic feature detection worked for the 60 piece puzzle,
but even after a good amount of tuning and tweaking, I could not successfully detect features for all 324 pieces in 
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
  - We find the 4 corners of the piece.
  - Split the full contour at the spots which are closest to each corner.
  - Identify the type of each side (straight, male, female, other).
  - Normalize the edges, and generate a comparator class for the loss function.
### Corner Detection
  The point of failure for the large, puzzle, and the only step that needed my intervention was the corner detection. The
algorithm I used to select points is convoluted, and includes many different parameters (magic numbers) whose correct values,
if they exist, are highly sensitive to the morphology of the pieces you are working with, and how you photograph them. You can
look through the piece.findCorners() function to see the procedure. Most of the values were hand tuned, and the current values
are about as good as I could do. Automatic tuning, or several iterative attempts to find corners could be implemented.
### Side Type Identification
  After finding the corners we identify the side types. This is done becuase while it is obvious (based on shape alone) that two
male pieces are not going to form a good match, if we identify explicit types on piece initialization, we avoid evaluating the 
fairly expensive loss function. We use some trigonometry to find, for each point in the side contour, how far it is from the
straight line from the first point to the end point. Straight sides have basically 0 distance, and a low variance. Male and female
sides have one strong peak, and therefore a high variance. Depending on if they extend towards or away from the center of the
puzzle piece, we classify them as male or female. If a side has multiple peaks, one towards and one away from the center, we classify
it as a weird type. Males only pair with females, but weird types pair with anything, as a catchall. 
### Normalization
  Once we have all 4 sides and their types, we normalize them, shifting the first point to [0,0] and rotating them to horizontal. We
take the normalized edges and make for each a KD Tree class from scikit's K Nearest Neighbors algorithm, which is the basis of the
loss function. Here we also pre calculate the distance from the first point in the edge to the last, as another step to weed out
incompatible matches instead of doing a KNN calculation.
## Loss Function
   Our loss function is quite simple. It takes in two edge shapes, belonging to 2 different pieces, and gives us a simple average of
distances from K Nearest Neighbor for each point in the contour array. We use this as a metric of how similair the shape of two edges
are, and therfore how well they would fit together. A more complex or fine tuned loss function is certainly possible, but this one was
conceptually simple and worked well. Before actually doing a KNN evaluation, we check a few things to see if the piece combination makes
sense. We check that, assuming neither piece is a weird type, that we only compare between male-female pairs. There are several considerations
with pairs where one or more has a straight side, mostly keeping edge continuity. We also check if the end to end distances are similair between
the two sides. If all of these checks pass, we go on to calculate an average nearest neighbors distance. Any time a score is calculated between
two sides, (valid or not) that value is stored in an array so that later requests of the loss between that pair is basically free. Invalid
pairings get a default score of a million. Typical values are between 1-10, where less than 3 is pretty good. The average loss over all pairs
in the correct configuration of both tested puzzles is about 2.50.

## Solving
  The solving process for both puzzles consists of an A* search of possible states. At each step, we find the "neighbors" to the current state.
That is, we select one of the positions on the board which borders some other piece. We prefer to choose those that border more than 1 piece,
becuase if a piece has good loss for two adjacent pieces, we can be more confident that it is a correct placement than when comparing to just one.
We look through all possible pieces (those that have not already been placed), placed in that spot with any rotation (4 possible rotations), and rank the piece/rotation combinations by lowest loss. We keep all of the placements which lie above some rank and loss value, these thresholds being
set by the user. Those that we choose to keep become the neighbors of the current state. A* needs two values to choose which state to explore further: a cost for the state, and the estimated cost from that state to the finish. The cost of a puzzle state is the sum of the losses for each
placement made. Essentially, the cost is the average loss per piece * the number of placed pieces. It's easy to guess the heuristic then: estimated cost for the rest of the puzzle = average loss per piece * number of UNplaced pieces. We assume the average losses in the future will be the same
as they have been in the past. In my testing, I have observed that this is usually not the case with states that have made a misstep at some point
in the past: once you make one incorrect placement, you are forced to make worse and worse ones over time. Using this, one could imagine a cost
function that evaluates how the average loss has been changing over time, and formulate a line of best fit or some other function to factor in if
the state has been steadily declining in quality. But the assumption of linearity works just fine.

# Results
  All in all, this code mostly acheives what I expected of it at the start. If we are grading this in terms of usability, I would say the results are
poor. The biggest bottleneck by far, is that you need to take a picture of every piece individually. This step took me around 3 hours for my 18x18 puzzle.
not the funnest time of my life. The obvious solution would be to capture many pieces in a single image. This could potentially require quite a change
of the image processing stage. Taking more images at a time means some pieces in the image could be at very different positions relative to the camera,
adding already to the perspective distortions which I think are present. There is already some code in place for correcting these distortions, but I was
unable to see much difference in the results. (I think this is becuase the issue I was trying to solve was somewhere else, I later figured out, so maybe I 
should try again? But it works anyways so its not on the immediate agenda). After loading in all my images, thought, the process was pretty seemless. A
familiarized person could put in images and get solving in under an hour, including manual data correction. The main problem with this is that there are
just so many hand tuned parameters for a dataset. At least 8 in the corner detection alone, and it's still not enough. 5 or so more in the side type
identification, and maybe you couldn't even use this pipeline if you have highly irregular pieces, those without 4 corners, for example. So you can't just
throw anything in here and expect it to work. You could probably get it to that point though if you really wanted to. But computer vision was not the focus
of this project, which is why mine is mediocre. The solving is what drew me in, and it is definitely in a place I am happy with. On my decent setup, an i7,
the 10x6 puzzle takes under 7 seconds to solve, and the 18x18 takes around 18 minutes. I imagine this could go down  considerably with a more sophisticated
search algorithm, or even multithreading (precalculate all match scores with multithreading?).



















