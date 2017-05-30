from lib import Rovio
import cv2
import numpy as np
import random
import time
from skimage import filter, img_as_ubyte


class rovioControl(object):
    def __init__(self,url, username, password, port = 80):
        self.rovio = Rovio(url,username=username,password=password, 
                                 port = port)
        self.last = None
        self.key = 0
        
    def night_vision(self,frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = cv2.equalizeHist(frame)
        return frame
        
    def mse(self,imageA, imageB):
        # the 'Mean Squared Error' between the two images is the
        # sum of the squared difference between the two images;
        # NOTE: the two images must have the same dimension
        err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
        err /= float(imageA.shape[0] * imageA.shape[1])
        
        # return the MSE, the lower the error, the more "similar"
        # the two images are
        return err 
        
    def show_battery(self,frame):
        sh = frame.shape
        m,n = sh[0], sh[1]
        battery,charging = self.rovio.battery()
        battery = 100*battery/130.
        bs = "Battery: %0.1f" % battery
        cs = "Status: Roaming"
        if charging == 80:
            cs = "Status: Charging"
        cv2.putText(frame,bs,(20,20),
                    cv2.FONT_HERSHEY_PLAIN,2,(255,0,0))
        
        cv2.putText(frame,cs,(300,20),
                    cv2.FONT_HERSHEY_PLAIN,2,(255,0,0))
        
        return frame
        
    def dance(self):
        x = random.randint(1,5)
        if x == 1 :
            self.rovio.head_down()
            time.sleep(1)
        elif x == 2 :
            self.rovio.rotate_left(angle = 30, speed = 7)
            time.sleep(3)
        elif x == 3 :
            self.rovio.head_up()
            time.sleep(0.5)
        elif x == 4 :
            self.rovio.rotate_right(angle = 30, speed = 7)
            time.sleep(3)      
        elif x == 5 :
            self.rovio.head_middle()
            time.sleep(0.5)
        
 
    def object_detection(self):
        # Stop Rovio so that it can stop down to recognize
        self.rovio.stop()
        #self.rovio.patrol()
        self.rovio.head_middle()
        while True:
        # Input template image and video from rovio
            image = self.rovio.camera.get_frame()
            template = cv2.imread('template3.jpg')
 
       # resize images
            image = cv2.resize(image, (0,0), fx=0.5, fy=0.5) 
            template = cv2.resize(template, (0,0), fx=0.5, fy=0.5) 

       # Convert to grayscale
            imageGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            templateGray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
 
       # Find template and match
            result = cv2.matchTemplate(imageGray,templateGray, cv2.TM_SQDIFF)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
            top_left = min_loc
            h,w = templateGray.shape
            bottom_right = (top_left[0] + w, top_left[1] + h)

            cropped = image[top_left[1]:top_left[1]+h, top_left[0]:top_left[0]+w]
            cropgray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
            
       # edge detection
            edge1 = cv2.Canny(templateGray,100,200)
            edge2 = cv2.Canny(cropgray,100,200)
            
       # calculate Mean Squared Error
            mse_v = self.mse(edge1,edge2)  
            
            #boundingbox
            cv2.rectangle(image,top_left, bottom_right,(0,0,255),4)
            
       # Show result
            cv2.imshow("Template", template) #show template image
            
            image = self.resize(image)
            mseval = "mse: %0.1f" % mse_v
            cv2.putText(image,mseval,(20,20),
                    cv2.FONT_HERSHEY_PLAIN,2,(255,0,0))
            cv2.imshow("Result", image) #show input image with blue bounding box
            
            cv2.imshow("cropped",edge2) #show cropped image within blue bounding box
            
            cv2.moveWindow("Template", 10, 50);
            cv2.moveWindow("Result", 150, 50);
            cv2.moveWindow("cropped", 10, 200);
            
            if mse_v < 2000:
                while True:
                    self.dance()
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("p"):
                        x = 0
                        self.rovio.head_middle()
                        break
                
        # if the `q` key is pressed, break from the lop
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
             
    
    def resize(self,frame, size = (640,480)):
        frame = cv2.resize(frame, size)
        return frame

    def main(self):
        frame = self.rovio.camera.get_frame()
        if not isinstance(frame, np.ndarray):
            return
        frame = self.night_vision(frame)
        #frame = filter.sobel(frame)
        #frame = img_as_ubyte(frame)
        frame = self.resize(frame)
        
        frame = cv2.merge([frame,frame,frame])
        
        frame = self.show_battery(frame)
        
        cv2.imshow("rovio", frame)
        
        self.key = cv2.waitKey(20)
        if self.key > 0:
            #print self.key
            pass
        if self.key == 114: #r
            self.rovio.turn_around()
        elif self.key == 63233 or self.key == 115: #down or s
            self.rovio.backward(speed=1)
        elif self.key == 63232 or self.key == 119: #up or w
            self.rovio.forward(speed=1)
        elif self.key == 63234 or self.key == 113: #left or a
            self.rovio.rotate_left(angle=30,speed=5)
        elif self.key == 63235 or self.key == 101: #right or d
            self.rovio.rotate_right(angle=30,speed=5)
        elif self.key == 97: #left or a
            self.rovio.left(speed=1)
        elif self.key == 100: #right or d
            self.rovio.right(speed=1)
        elif self.key == 44: #comma
            self.rovio.head_down()
        elif self.key == 46: #period
            self.rovio.head_middle()
        elif self.key == 47: #slash
            self.rovio.head_up()
        elif self.key == 32:  # Space Bar
            flag = False
            #self.rovio.stop()
            #while not flag:flag =
            self.object_detection()
            print flag
        
if __name__ == "__main__":
    url = '192.168.137.2'
    user = 'danzig'
    password = "password"
    app = rovioControl(url, user, password)
    while True:
        app.main()
        if app.key == 27:
            break


    