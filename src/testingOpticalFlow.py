import os;
import cv2 as cv;
import timeit
import numpy as np



from timeit import default_timer as timer





print(os.getcwd());
filename = "vid2.mp4"

vid = cv.VideoCapture(filename)
x = 0;
runningAverage = 30;
frametime = 5;

def draw_flow(img, flow, step=8):
    h, w = img.shape[:2]
    y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2,-1).astype(int)
    fx, fy = flow[y,x].T
    lines = np.vstack([x, y, x+fx, y+fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)
    vis = img;

    # vis = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
    cv.polylines(vis, lines, 0, (0, 255, 0))
    for (x1, y1), (_x2, _y2) in lines:
        cv.circle(vis, (x1, y1), 1, (0, 255, 0), -1)
    return vis


ret, img = vid.read();
prevgray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)


bs = cv.createBackgroundSubtractorKNN();
mask = bs.apply(img, .5);
cv.imshow("background", mask);

##https://stackoverflow.com/questions/43351950/how-to-extract-velocity-vectors-of-pixels-from-opencv-calcopticalflowfarneback

while True:
    start = timer()
    ret, img = vid.read()
    copy = img;
    if(img is None):
        vid = cv.VideoCapture(filename)
        continue

    mask = bs.apply(copy, .5);
    cv.imshow("background", mask);
    
    end = timer();
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    flow = cv.calcOpticalFlowFarneback(prevgray, gray, None, 0.10, 10, 10, 1, 2, 1.2, 0)
    prevgray = gray

    cv.imshow("flow", draw_flow(img=copy, flow=flow))


    fps = (1/(end-start))


    runningAverage = (fps + runningAverage)/2

    x += 1;
    if cv.waitKey(frametime) & 0xFF == ord('q'):
        break
cv.destroyAllWindows()


