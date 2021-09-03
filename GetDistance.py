import cv2
import os
import sys
import math
import numpy as np
import dlib
from imutils import face_utils

class GetDistance():
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades +'haarcascade_frontalface_alt.xml')
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    # face_cascade = cv2.CascadeClassifier('lbpcascadefrontalface.xml')
    
    def __init__(self, cam_width,cam_angle, cam_res_width, cam_res_height):
        self.cam_width = cam_width
        self.cam_angle = cam_angle #diagonal_field_of_view
        self.cam_res_width = cam_res_width  
        self.cam_res_height = cam_res_height 

        self.distance = 0
        self.cam1_angle = 0
        self.cam2_angle = 0

        self.circles1_result = None
        self.circles2_result = None
        self.faces1 = None
        self.faces2 = None
        self.nose1 = None
        self.nose2 = None
        self.shape1 = None
        self.shape2 = None

        self.x_angle_min = 0 #in radian
        self.x_angle_max = 0 #in radian
        self.y_angle_min = 0 #in radian
        self.y_angle_max = 0 #in radian
        self.set_diagonal_field_of_view(cam_angle) #720p 1280x720
        
    def set_diagonal_field_of_view(self,angle):
        diagonal_field_of_view = angle
        Da = math.sqrt(math.pow(self.cam_res_width,2) + math.pow(self.cam_res_height,2))
        hypotenuse = (Da/2) / math.sin(math.radians(diagonal_field_of_view/2))
        hypotenuse_x = math.sqrt(math.pow(hypotenuse,2)-math.pow(self.cam_res_width/2,2))
        hypotenuse_y = math.sqrt(math.pow(hypotenuse,2)-math.pow(self.cam_res_height/2,2))

        self.x_angle_min = math.acos(1 - math.pow(self.cam_res_width,2) / (2 * math.pow(hypotenuse,2)))
        self.x_angle_max = math.acos(1 - math.pow(self.cam_res_width,2) / (2 * math.pow(hypotenuse_x,2)))

        self.y_angle_min = math.acos(1 - math.pow(self.cam_res_height,2) / (2 * math.pow(hypotenuse,2)))
        self.y_angle_max = math.acos(1 - math.pow(self.cam_res_height,2) / (2 * math.pow(hypotenuse_y,2)))

        # print(f"{math.degrees(x_angle_max)} {math.degrees(x_angle_min)}")
        # print(f"{math.degrees(y_angle_max)} {math.degrees(y_angle_min)}")
        

    def process_circle(self):
        gray1 = cv2.cvtColor(self.img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(self.img2, cv2.COLOR_BGR2GRAY)
        circles1 = cv2.HoughCircles(gray1, cv2.HOUGH_GRADIENT, 1.2, 100)
        circles2 = cv2.HoughCircles(gray2, cv2.HOUGH_GRADIENT, 1.2, 100)
        if(circles1 is not None and circles2 is not None):
            self.circles1_result = np.round(circles1[0, :]).astype("int")
            self.circles2_result = np.round(circles2[0, :]).astype("int")
            x1 = self.circles1_result[0][0]  #circle centre x
            # y1 = self.circles1_result[0][1]  #circle centre y
            x2 = self.circles2_result[0][0]  #circle centre x
            # y2 = self.circles2_result[0][1]  #circle centre y
            
            x = (x1 + x2)/2
            cacu_angle = math.degrees(self.x_angle_min + (self.x_angle_max-self.x_angle_min) * abs(2*x/self.cam_res_width -1))

            self.cam2_angle = (180-cacu_angle)/2+x2/self.width2*cacu_angle
            self.cam1_angle = (180-cacu_angle)/2+x1/self.width1*cacu_angle

            if(self.cam1_angle >= 90):
                self.cam1_angle = -(self.cam1_angle - 90)
            else:
                self.cam1_angle = 90 - self.cam1_angle

            if(self.cam2_angle >= 90):
                self.cam2_angle = -(self.cam2_angle - 90)
            else:
                self.cam2_angle = 90 - self.cam2_angle

            try:
                self.distance = self.cam_width / (math.tan(math.radians(self.cam2_angle))-math.tan(math.radians(self.cam1_angle)))
            except ZeroDivisionError:
                print("ZeroDivisionError")
        else:
            self.clear_data()

    def process_face_landmarks(self):
        img1 = self.img1
        img2 = self.img2
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        rects1 = self.detector(gray1, 1)
        rects2 = self.detector(gray2, 1)
        rect1 = None
        rect2 = None

        for (i, rect) in enumerate(rects1):
            rect1 = rect
            break
        for (i, rect) in enumerate(rects2):
            rect2 = rect
            break
        
        if(rect1 is not None and rect2 is not None):
            self.shape1 = self.predictor(gray1, rect1)
            self.shape1 = face_utils.shape_to_np(self.shape1)
            # (x, y, w, h) = face_utils.rect_to_bb(rect1)
            # cv2.rectangle(img1, (x, y), (x + w, y + h), (0, 255, 0), 2)
            self.nose1 =self.shape1[30] # 30 = nose(x,y)
            # cv2.circle(img1, (x1, y1), 1, (0, 0, 255), -1)

            self.shape2 = self.predictor(gray2, rect2)
            self.shape2 = face_utils.shape_to_np(self.shape2)
            self.nose2 =self.shape2[30] # 30 = nose

            if(len(self.nose1) and len(self.nose2)):
                x = (self.nose2[0]+ self.nose1[0])/2
                cacu_angle = math.degrees(self.x_angle_min + (self.x_angle_max-self.x_angle_min) * abs(2*x/self.cam_res_width -1))
                self.cam2_angle = (180-cacu_angle)/2+(self.nose2[0])/self.width2*cacu_angle
                self.cam1_angle = (180-cacu_angle)/2+(self.nose1[0])/self.width1*cacu_angle
                if(self.cam2_angle >= 90):
                    self.cam2_angle = -(self.cam2_angle - 90)
                else:
                    self.cam2_angle = 90 - self.cam2_angle

                if(self.cam1_angle >= 90):
                    self.cam1_angle = -(self.cam1_angle - 90)
                else:
                    self.cam1_angle = 90 - self.cam1_angle

                try:
                    self.distance = self.cam_width / (math.tan(math.radians(self.cam2_angle))-math.tan(math.radians(self.cam1_angle)))
                except ZeroDivisionError:
                    print("ZeroDivisionError")
            else:
                self.clear_data()
        else:
            self.clear_data()



    def process_face(self):
        gray1 = cv2.cvtColor(self.img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(self.img2, cv2.COLOR_BGR2GRAY)

        self.faces1 = self.face_cascade.detectMultiScale(gray1, 1.1, 4)
        self.faces2 = self.face_cascade.detectMultiScale(gray2, 1.1, 4)
        print(self.faces1)

        if(len(self.faces1) and len(self.faces2)):
        # if(self.faces1 is not None and self.faces2 is not None):
            x = (self.faces2[0][0]+self.faces2[0][3]/2 + self.faces1[0][0]+self.faces1[0][3]/2)/2
            cacu_angle = math.degrees(self.x_angle_min + (self.x_angle_max-self.x_angle_min) * abs(2*x/self.cam_res_width -1))
            # print(f"{x} {cacu_angle}")
            
            self.cam2_angle = (180-cacu_angle)/2+(self.faces2[0][0]+self.faces2[0][3]/2)/self.width2*cacu_angle
            self.cam1_angle = (180-cacu_angle)/2+(self.faces1[0][0]+self.faces1[0][3]/2)/self.width1*cacu_angle

            # self.cam2_angle = (180-self.cam_angle)/2+(self.faces2[0][0]+self.faces2[0][3]/2)/self.width2*self.cam_angle
            # self.cam1_angle = (180-self.cam_angle)/2+(self.faces1[0][0]+self.faces1[0][3]/2)/self.width1*self.cam_angle

            if(self.cam2_angle >= 90):
                self.cam2_angle = -(self.cam2_angle - 90)
            else:
                self.cam2_angle = 90 - self.cam2_angle

            if(self.cam1_angle >= 90):
                self.cam1_angle = -(self.cam1_angle - 90)
            else:
                self.cam1_angle = 90 - self.cam1_angle

            try:
                self.distance = self.cam_width / (math.tan(math.radians(self.cam2_angle))-math.tan(math.radians(self.cam1_angle)))
            except ZeroDivisionError:
                print("ZeroDivisionError")
        else:
            self.clear_data()
    
    def clear_data(self):
            self.circles1_result = None
            self.circles2_result = None
            self.nose1 = None
            self.nose2 = None
            self.shape1 = None
            self.shape2 = None
            self.faces1 = None
            self.faces2 = None
            self.distance = 0
            self.cam1_angle = 0
            self.cam2_angle = 0

    def next_frame(self,img1,img2):
        self.img1 = img1
        self.img2 = img2
        # self.img1 = cv2.resize(img1,(460,460))
        # self.img2 = cv2.resize(img2,(460,460))
        self.height1, self.width1 = self.img1.shape[:2]
        self.height2, self.width2 = self.img2.shape[:2]

    def getshape(self): #face landmarks
        if(self.shape1 is not None and self.shape2 is not None):
            return self.shape1,self.shape2
        return None,None

    def getface(self): #face
        if(self.faces1 is not None and self.faces2 is not None):
            return self.faces1,self.faces2
        return None,None

    def getcircle(self): #circle
        if(self.circles1_result is not None and self.circles2_result is not None):
            return self.circles1_result,self.circles2_result
        return None,None
    
    def getnose(self): #face landmarks
        if(self.nose1 is not None and self.nose2 is not None):
            return self.nose1,self.nose2
        return None,None

    def getcamangle(self): #all
        return self.cam1_angle,self.cam2_angle

    def getdistance(self): # all
        return self.distance
    
    def printdata(self): # all
        if(self.cam1_angle is not None and self.cam2_angle is not None and self.distance is not None):
            print("cam1_angle = "+str(self.cam1_angle)+"cam2_angle = "+str(self.cam2_angle))
            print (f"d = {self.distance}")
    


def main():
    font = cv2.FONT_HERSHEY_SIMPLEX
    locationText2 = (0,60)
    fontScale              = 1
    fontColor              = (0,0,255)
    lineType               = 2

    cam_angle = 60
    cam_width = 7.2 #width between two cameras in cm

    directory = os.getcwd()
    image_floder1 =r'Cap_image_cam1'
    image_floder2 =r'Cap_image_cam2'

    Cap_image_dir1 = os.path.join(directory,image_floder1)
    Cap_image_dir2 = os.path.join(directory,image_floder2)

    if not (os.path.exists(image_floder1) and os.path.exists(image_floder2)):
        sys.exit()

    cam1_image = os.listdir(Cap_image_dir1)
    cam2_image = os.listdir(Cap_image_dir2)
    getdistance = GetDistance(cam_width,cam_angle, 1280, 720)

    for no in range(len(cam1_image)):
        img1 = cv2.imread(os.path.join(Cap_image_dir1,cam1_image[no])) 
        img2 = cv2.imread(os.path.join(Cap_image_dir2,cam2_image[no])) 

        getdistance.next_frame(img1,img2)
        getdistance.process_face()
        getdistance.printdata()
        distance = getdistance.getdistance()

        faces1,faces2 = getdistance.getface()
        if(faces1 is not None and faces2 is not None):
            x1,y1,w1,h1=faces1[0]
            cv2.rectangle(img1, (x1, y1), (x1+w1, y1+h1), (0, 255, 0), 2)

            x2,y2,w2,h2=faces2[0]
            cv2.rectangle(img2, (x2, y2), (x2+w2, y2+h2), (255, 0, 0), 2)
            cv2.rectangle(img1, (x2, y2), (x2+w2, y2+h2), (255, 0, 0), 2)

        cv2.putText(img1,"d = "+str(abs(distance)), 
            locationText2, 
            font, 
            fontScale,
            fontColor,
            lineType)


        cv2.imshow("Cam1", img1)
        cv2.imshow("Cam2", img2)
      

        if cv2.waitKey(0) & 0xFF == ord('q'):
            break 

if __name__ == '__main__':
    main()
  
  