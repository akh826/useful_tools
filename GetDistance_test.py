import cv2
from GetDistance import GetDistance

def main():
    cam_angle = 60
    cam_width = 7.2 #width between two cameras in cm  closest=7.2

    font                   = cv2.FONT_HERSHEY_SIMPLEX
    locationText1 = (0,20)
    fontScale              = 1
    fontColor              = (0,0,255)
    lineType               = 2

    cap1 = cv2.VideoCapture(0,cv2.CAP_DSHOW)
    cap1.set(cv2.CAP_PROP_BUFFERSIZE, 2)
    cap2 = cv2.VideoCapture(1,cv2.CAP_DSHOW)
    cap2.set(cv2.CAP_PROP_BUFFERSIZE, 2)

    getdistance = GetDistance(cam_width,cam_angle, 1280, 720)
    print("Start of loop")

    while True:
        ret1, img1 = cap1.read()
        ret2, img2 = cap2.read()

        edges = cv2.Canny(image=img1, threshold1=100, threshold2=200)

        if ret1:
            cv2.imshow('frame1', edges)
        if ret2:
            cv2.imshow('frame2', img2)

        
        

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break 

    print("End of loop")

    cap1.release()
    cap2.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
  