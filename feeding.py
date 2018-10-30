'''
    This file contain the demonstration of using pyautogui with feeding frenzy game
'''
import pyautogui
import cv2


APP_CONFIG = {'RECT_WIDTH':150}
APP_CONFIG['RECT_WIDTH'] = 150
APP_CONFIG['RECT_HEIGHT'] = 150
APP_CONFIG['RUNNING'] = True # default the app is not running


fist_cascade = cv2.CascadeClassifier('HaarCasscades/fist.xml')
# hand_cascade = cv2.CascadeClassifier('HaarCasscades/open_palm.xml')
hand_cascade = cv2.CascadeClassifier('HaarCasscades/smile.xml')
if fist_cascade.empty() or hand_cascade.empty():
  raise IOError('Unable to load the cascades classifier xml files')
cap = cv2.VideoCapture(0)
ds_factor = 0.5

def detectFist(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # detect smile or fist or ...
    mouth_rects = fist_cascade.detectMultiScale(gray, 1.7, 11)
    for (x,y,w,h) in mouth_rects:
        return x and y and w and h
    return False

def shouldRun():
    if (cv2.waitKey(1) == 27):
        APP_CONFIG['RUNNING'] = not APP_CONFIG['RUNNING']
        print(APP_CONFIG['RUNNING'])
    return APP_CONFIG['RUNNING']

def testFedding():
    # while(True):
    #     pyautogui.moveRel(-1*120, 20, duration=0.00001)  # move mouse 10 pixels down
    while(True):
        if (not shouldRun()):
            continue
        frame = getCurrFrame()
        isFist = detectFist(frame)
        if (isFist):
            pyautogui.click()
            continue
        deltaX, deltaY = nextDelta(frame)
        print("deltas {}, {}".format(deltaX,deltaY))
        pyautogui.moveRel(-1*deltaX*3, deltaY*3, duration=0.001)  # move mouse 10 pixels down
        #image = pyautogui.screenshot()


lx = None
ly = None
def nextDelta(frame):
    global lx
    global ly
    hand, x,y = getCurrHandPos(frame)
    if (hand):
        print("hand is true")
        if (lx == None):
            lx = x
            ly = y
        
        res =  x - lx, y - ly
        lx = x
        ly = y
        return res
    return 0,0

def getCurrFrame():
    return cap.read()[1]



def getCurrHandPos(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    hand_rect = hand_cascade.detectMultiScale(gray, 1.7, 11)
    for (x,y,w,h) in hand_rect:
        # y = int(y - 0.15*h)
        # cv2.rectangle(faceFrame, (x,y), (x+w,y+h), (0,255,0), 3)
        return (x and y and w and h), x,y
    return False, None, None



class HandPositionTracker:
    def __init__(self, path):
        self.path = path
        self.lastX = 0
        self.lastY = 0
        self.hand_cascade = cv2.CascadeClassifier(path)

    def nextHandPosition(self, frame):
        self.getCurrHandPos(frame)


    def getCurrHandPos(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hand_rect = self.hand_cascade.detectMultiScale(gray, 1.7, 11)
        for (x,y,w,h) in hand_rect:
            y = int(y - 0.15*h)
            # cv2.rectangle(faceFrame, (x,y), (x+w,y+h), (0,255,0), 3)
            return (x and y and w and h), x,y
        return False
testFedding()