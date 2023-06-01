import numpy, time, cv2, time, numpy as np
from funcs import imscale

vid = cv2.VideoCapture(0)
vid.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
path = f"C:\\Users\\ek\\Desktop\\sdfghj\\puzzle\\testimgs\\dino\\"
c = 0
while 1:
    ret, frame = numpy.array(vid.read())
    s = np.shape(frame)
    marked = cv2.circle(frame, (round(s[1]/2), round(s[0]/2)), radius=10, color=(60, 0, 250), thickness=5)

    cv2.imshow('frame', imscale(marked, .6))
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        cv2.imwrite(f"{path}\\{c}.png", frame)
        c += 1
        time.sleep(1)





















































