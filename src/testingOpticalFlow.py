import os;
import cv2 as cv;
import timeit
import numpy as np
import matplotlib.pyplot as plt



from timeit import default_timer as timer




def find(name, path):
    for root, dirs, files in os.walk(path):
        if name in files:
            return os.path.join(root, name)



farneback_params={
        'pyr_scale':0.7,
        'levels':10,
        'winsize':20,
        'iterations': 3,
        'poly_n': 2,
        'poly_sigma':1.2,
        'flags':0
    }
x = 0;
frametime = 1;
filename = "vid4.mp4"
path = find(filename, "./")
graphStep = 10

colorType = cv.COLOR_RGB2GRAY
frameSkip = 10

vid = cv.VideoCapture(path)
ret, img = vid.read();

frameSkip = 30
i= 0
fps = vid.get(cv.CAP_PROP_FPS)
print(fps)


def getPeriodicity(anArray):
    count = 0;
    elapsedTime = len(anArray) * (1/fps)
    print("elapsed time", elapsedTime)

    for y in anArray:
        if(y == 1):
            count += 1;
    if(count == 0):
        return -1

    return count/elapsedTime


while(i < frameSkip):
    ret, img = vid.read()
    i += 1

cv.imshow("firstFrame", img)
xarray = []
yarray = []

def draw_flow(img, flow, step=16):
    h, w = img.shape[:2]
    y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2,-1).astype(int)
    fx, fy = flow[y,x].T
    lines = np.vstack([x, y, x+fx, y+fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)
    vis = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
    cv.polylines(vis, lines, 0, (0, 255, 0))
    for (x1, y1), (x2, y2) in lines:
        cv.circle(vis, (x1, y1), 1, (0, 255, 0), -1)
    return vis

prevgray = cv.cvtColor(img, colorType)
flag = False;


##https://stackoverflow.com/questions/43351950/how-to-extract-velocity-vectors-of-pixels-from-opencv-calcopticalflowfarneback


maxMag = 1;
start = timer()


while True:
    ret, img = vid.read()
    end = timer()
    diff = end - start;

    copy = img;
    if(img is None):
        vid = cv.VideoCapture(path)
        continue
    
    gray = cv.cvtColor(img, colorType)
    #flow = cv.calcOpticalFlowFarneback(prevgray, gray,flow=None, **farneback_params)
    flow = cv.calcOpticalFlowFarneback(prevgray, gray, None,  **farneback_params)
    flowImage = draw_flow(gray, flow)


    mag, ang = cv.cartToPolar(flow[..., 0], flow[..., 1])
    #Rough filtering
    currentMag = np.average(mag);
    if(currentMag > maxMag):
        xarray.append(diff)
        yarray.append(1)
    else:
        xarray.append(diff)
        yarray.append(0)
    
    if(x % 10 == 0):
        print(getPeriodicity(yarray))
        print(f'trying flow:', x)

    prevgray = gray
    x += 1;
    
    #Break Structure
    if cv.waitKey(frametime) & 0xFF == ord('g'):
        plt.plot(xarray, yarray)
        plt.pause(.01)
        flag = not flag
    
    if(flag and x%graphStep == 0):
        plt.plot(xarray, yarray)
        print()
        plt.pause(.001)
    
    if cv.waitKey(frametime) & 0xFF == ord('q'):
        break
    
    cv.imshow("FlowImage", flowImage)
    
cv.destroyAllWindows()


