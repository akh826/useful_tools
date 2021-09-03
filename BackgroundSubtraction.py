from __future__ import print_function
import cv2
import argparse
import time
parser = argparse.ArgumentParser(description='This program shows how to use background subtraction methods provided by \
                                              OpenCV. You can process both videos and images.')
parser.add_argument('--input', type=str, help='Path to a video or a sequence of image.', default='vtest.avi')
parser.add_argument('--algo', type=str, help='Background subtraction method (KNN, MOG2).', default='KNN')
args = parser.parse_args()
(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')
if args.algo == 'MOG2':
    backSub = cv2.createBackgroundSubtractorMOG2()
elif args.algo == 'KNN':
    backSub = cv2.createBackgroundSubtractorKNN() # better
else:
    exit
backSub = cv2.createBackgroundSubtractorMOG2()
capture = cv2.VideoCapture(0,cv2.CAP_DSHOW)
start_time = time.time()
# capture.set(cv2.CAP_PROP_BUFFERSIZE, 2)
if not capture.isOpened:
    print('Unable to open: ' + args.input)
    exit(0)
while True:
    ret, frame = capture.read()
    if frame is None:
        break
    
    edges = cv2.Canny(image=frame, threshold1=100, threshold2=200)
    fgMask = backSub.apply(edges)
    
    
    cv2.rectangle(frame, (10, 2), (100,20), (255,255,255), -1)

    # res = cv2.bitwise_and(frame,frame, mask= fgMask)

    fps = 1.0 / (time.time() - start_time)
    cv2.putText(frame, str(float("{:.2f}".format(fps))), (15, 15),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5 , (0,0,0))
    start_time = time.time()
    
    
    cv2.imshow('Frame', frame)
    cv2.imshow('edges', edges)
    cv2.imshow('FG Mask', fgMask)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
            break 
    # keyboard = cv2.waitKey(30)
    # if keyboard == 'q' or keyboard == 27:
    #     break