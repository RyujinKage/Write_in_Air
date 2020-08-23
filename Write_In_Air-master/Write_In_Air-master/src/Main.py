import cv2
import numpy as np
import math
from PIL import Image
from predict import prediction

traverse_point = []
rect_start = 75   #dimensions of the r
rect_end = 400

#for math expression evoluation 
expression=''
A=''                    #first operand
B=''                    #second operand    
predicted = ''         
opr=''                  #operation
after=0


#find centroid
def centroid(max_cont):
    moment = cv2.moments(max_cont)
    if moment['m00'] != 0:
        cx = int(moment['m10'] / moment['m00'])
        cy = int(moment['m01'] / moment['m00'])
        return cx, cy
    else:
        return None

#draw trajectory by combining pointes in traverse_point list
def draw_trajectory(frame, pts):
	for i in range(1, len(pts)):
		if pts[i - 1] is None or pts[i] is None:
			continue
		cv2.line(frame, pts[i - 1], pts[i], (0, 0, 0), int(3))


#find highest point of hands which will point to finger
# Note that point at the heightest will have minimum y value
def find_highest_point(points):
    point = min(points, key = lambda t: t[1])
    return point


#mask the frame
def masking(frame):
    #define region of interest
    roi=frame[rect_start:rect_end, rect_start:rect_end]
    kernel = np.ones((5,5),np.uint8)
    cv2.rectangle(frame,(rect_start,rect_start),(rect_end,rect_end),(0,255,0),0)    
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    
    # define range of skin color in HSV
    lower_skin = np.array([0,20,70], dtype=np.uint8)
    upper_skin = np.array([20,255,255], dtype=np.uint8)
    
    #extrect skin colur imagw  
    mask = cv2.inRange(hsv, lower_skin, upper_skin)
    
    #extrapolate the hand to fill dark spots within
    mask = cv2.dilate(mask,kernel,iterations = 4)
    
    #blur the image
    mask = cv2.GaussianBlur(mask,(5,5),100)

    return mask


#function to track the finger and create a trajectory
def track_finger(frame): 
    mask = masking(frame)    
    
    #find contours
    _,contours,hier = cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    #find contour of max area(hand)
    cnt = max(contours, key = lambda x: cv2.contourArea(x))

    #white blank frame to draw the finger trajectory 
    blankFrame = np.zeros(shape=[rect_end,rect_end,3],dtype=np.uint8)
    blankFrame.fill(255) 
    
    #find centroid of hand
    cnt_centroid = centroid(cnt)
    cent = (cnt_centroid[0]+rect_start,cnt_centroid[1]+rect_start) #coordinates of centroid in given rectangle
    cv2.circle(frame, cent, 5, [255, 100, 255], -1) #draw a point on centroid

    #initialize highest points in hand cnt
    # heighest_point1 : heighest point
    # heighest_point2: second heighest point        
    highest_point1 = (1000,1000)
    highest_point2 = (1000,1000)

    if cnt is not None:
        epsilon = 0.0005*cv2.arcLength(cnt,True)
        approx= cv2.approxPolyDP(cnt,epsilon,True)

        areaOfMaxContours = cv2.contourArea(cnt)

        hull = cv2.convexHull(approx, returnPoints=False)
        defects = cv2.convexityDefects(approx, hull)
        
        l=0 #number of defects
        
        for i in range(defects.shape[0]):  #code from cv2 documentation
            s,e,f,d = defects[i,0]
            start = tuple(approx[s][0])
            end = tuple(approx[e][0])
            far = tuple(approx[f][0])
    
            # find length of all sides of triangle
            a = math.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
            b = math.sqrt((far[0] - start[0])**2 + (far[1] - start[1])**2)
            c = math.sqrt((end[0] - far[0])**2 + (end[1] - far[1])**2)

            s = (a+b+c)/2
            ar = math.sqrt(s*(s-a)*(s-b)*(s-c))
            
            #find the point at highest which will be finger because when 
            # writing we need to keep our finger at highest
            highest_point = find_highest_point([start,end,far])
            
            if(highest_point[1]<highest_point1[1]):
                highest_point2=highest_point1
                highest_point1=highest_point
            elif highest_point[1]<highest_point2[1]:
                highest_point2 = highest_point

            d=(2*ar)/a
            
            # apply cosine rule here
            angle = math.acos((b**2 + c**2 - a**2)/(2*b*c)) * 57

            if angle <= 90 and d>30:
                l += 1
        
        l+=1

        font = cv2.FONT_HERSHEY_SIMPLEX
        
        if l==1 and areaOfMaxContours<2000:
            cv2.putText(frame,'No hand in Frame',(0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)
        
        else:
            far_point = ((highest_point1[0]+highest_point2[0])//2,(highest_point1[1]+highest_point2[1])//2)
            
            #due to noise if point is bellow then center then we will add 1 in privous point
            if(far_point[1]>cnt_centroid[1]) and len(traverse_point)>=1:
                ar_point = traverse_point[-1]
                far_point = (ar_point[0]+1,ar_point[1]+1)
            
            ft = (far_point[0]+rect_start,far_point[1]+rect_start)
            cv2.circle(frame, ft, 5, [0, 200, 255], -1)
            
            if len(traverse_point) < 100:
                traverse_point.append(far_point)
            else:
                traverse_point.pop(0)
                traverse_point.append(far_point)
            
            draw_trajectory(blankFrame, traverse_point)
    
    cv2.imshow("Trajectory",blankFrame)
    return frame

def resizeTrajectoryFrame(frame):
    dim = (32,32)
    resized = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)
    return resized


def assign():
    global expression, predicted, opr, after, A, B
    try:
        expression += predicted
        print(expression)
        if predicted=='+' or predicted=='-':
            opr = predicted
            after=1
        elif after:
            B += predicted
            if opr =='+':
                ans = int(A)+int(B)
            else:
                ans = int(A)-int(B)
            print(A,opr,B," = ",ans)
        else:
            A += predicted
    
    except:
        print("Error occurred...")
        print("Reseting the expression")
        expression=''
        A=''
        B=''
        predicted = ''
        opr=''
        after=0

def workOnTrajectory():
    # To store my output 
    blankFrame = np.zeros(shape=[rect_end,rect_end,3],dtype=np.uint8)
    blankFrame.fill(255)

    draw_trajectory(blankFrame, traverse_point)
    
    #dimension of the image to crop
    l = min(traverse_point, key = lambda t: t[0])[0]
    r = max(traverse_point, key = lambda t: t[0])[0]
    u = min(traverse_point, key = lambda t: t[1])[1]   
    d = max(traverse_point, key = lambda t: t[1])[1]
    
    l = l-10 if l-10 >= 0 else 0
    r = r+10 if r+10 <= rect_end else rect_end
    u = u-10 if u-10 >=0 else 0
    d = d+10 if d+10 <= rect_end else rect_end

    blankFrame = blankFrame[u:d,l:r] #crop image
    
    trajectory = resizeTrajectoryFrame(blankFrame)
    
    global predicted
    predicted = prediction(trajectory)
    assign()  #assign values to expression
    
    
def main():
    global expression, predicted, opr, after, A, B
    
    cap = cv2.VideoCapture(0)
    #cap = cv2.VideoCapture('http://192.168.43.167:4747/video') # for the driodcam
    started = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        frame=cv2.flip(frame,1)
        cv2.rectangle(frame,(rect_start,rect_start),(rect_end,rect_end),(0,255,0),0)
        cv2.imshow('frame',frame)
        if started:
            frame = track_finger(frame)
            cv2.imshow("Live Feed", frame)

        k = cv2.waitKey(2) & 0xFF

        #start processing or clear Press s
        if k == ord('s'):
            started = 1
            traverse_point.clear()
    
        #stop processing  Press d
        if k == ord('d'):
            started = 0
            workOnTrajectory()
            traverse_point.clear()

        #reset expression Press r
        if k == ord('r'):
            print("Reset expression")#
            expression=''
            A=''
            B=''
            predicted = ''
            opr=''
            after=0

        if k == 27:  #Press esc
            break

    cv2.destroyAllWindows()
    cap.release()    
    

if __name__ == '__main__':
    main()



