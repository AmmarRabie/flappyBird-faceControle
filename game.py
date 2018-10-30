'''
    This is the flappy bird game played with smile or any cascade that be passed, this game also save random frames and the whole video if save is opened
    To start saving -> press 'n' when flappy bird is focussed
    To pause saving -> press 'f' when flappy bird is focussed
    The default is the game does not save
'''
from __future__ import print_function
from os import makedirs
# import threading
# from math import ceil
import numpy
import cv2
import time
import sys
import pygame
# from smiledetector import SmileDetector
from pygame.locals import K_UP, K_ESCAPE, K_f, K_n
import numpy as np
from datetime import datetime
sys.path.append("game/")
import wrapped_flappy_bird as game

# smileDetector = SmileDetector()

ACTIONS = 2 # number of valid actions

mouth_cascade = cv2.CascadeClassifier('HaarCasscades/smile.xml')
if mouth_cascade.empty():
  raise IOError('Unable to load the mouth cascade classifier xml file')
cap = cv2.VideoCapture(0)


videoSaver = None
ds_factor = 0.5

APP_CONFIG = {'save':False}
APP_CONFIG['RECT_WIDTH'] = 150
APP_CONFIG['RECT_HEIGHT'] = 150
APP_CONFIG['FRAME_WIDTH'] = ds_factor * cap.get(cv2.CAP_PROP_FRAME_WIDTH)
APP_CONFIG['FRAME_HEIGHT'] = ds_factor * cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
APP_CONFIG['save_rate'] = 500 # every 500 ms

frame = None #cv2.resize(cap.read()[1], None, fx=ds_factor, fy=ds_factor, interpolation=cv2.INTER_AREA)

def isMouth(faceFrame):
    mouth = False
    gray = cv2.cvtColor(faceFrame, cv2.COLOR_BGR2GRAY)

    # inc counter
    #global counter
    #counter = counter + 1

    # save if next image and config is on
    # if (counter % 150 and APP_CONFIG['save']):
    #     newImagePath = "{}\\_{}.jpg".format(APP_CONFIG['save_path'],counter)
    #     cv2.imwrite(newImagePath, faceFrame)

    # detect smile or fist or ...
    mouth_rects = mouth_cascade.detectMultiScale(gray, 1.7, 11)
    for (x,y,w,h) in mouth_rects:
        y = int(y - 0.15*h)
        # cv2.rectangle(faceFrame, (x,y), (x+w,y+h), (0,255,0), 3)
        mouth = x and y and w and h
        break
    return mouth


def getGrayFrameFromCamera():
    # get the current photo
    ret, frame = cap.read()
    frame = cv2.resize(frame, None, fx=ds_factor, fy=ds_factor, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return frame

def isMouth2(faceImage):
    return smileDetector.predict(faceImage)
    # image = getGrayFrameFromCamera()
    # if (isMouth):
    #     cv2.circle(image, (50,50), 10, (0,255,0), 5)
    # else:
    #     cv2.circle(image, (50,50), 10, (255,0,0), 5)
    # cv2.imshow('Mouth Detector', image)
    # return isMouth

def playGame():
    # open up a game state to communicate with emulator
    game_state = game.GameState()

    # threading.Timer(0.1, processFrames).start()
    # threading.Timer(1.5, saveFrame).start()
    while True:
        t_start = time.time()
        keys = pygame.key.get_pressed()
        if (keys[K_ESCAPE]):
            cap.release()
            videoSaver.release()
            cv2.destroyAllWindows()
            return
        if (keys[K_f]):
            APP_CONFIG['save'] = False
            print('off')
        if (keys[K_n]):
            APP_CONFIG['save'] = True
            print('on')
        frame = processFrames()
        #drawFitRectangle(frame)
        faceFrame = excludeFace(frame)
        shouldJump = isMouth(faceFrame)
        saveFrame(frame)
        t_end = time.time()
        t_delta = int((t_end - t_start) * 1000)
        print(t_delta)
        while(t_delta >= 0):
            game_state.frame_step(getAction(shouldJump))
            t_delta -= 8
        #game_state.frame_step(getAction(shouldJump))
        drawFrame(frame)



def getAction(jump):
    action = np.zeros(ACTIONS)
    action[0] = 1
    if jump:
        action[0] = 0
        action[1] = 1
    return action

def drawScreenImage():
    # get the current photo
    ret, frame = cap.read()
    frame = cv2.resize(frame, None, fx=ds_factor, fy=ds_factor, interpolation=cv2.INTER_AREA)

    # get points to draw fit rectangle
    #height, width = frame.shape
    #center = (height / 2, width / 2)

    # draw the fit rectangle
    x1, y1, x2, y2 = calcFitRectangle()
    cv2.rectangle(frame,(x1,y1), (x2,y2),(0,0,255), 2)

    # draw a status circle and the whole view
    cv2.circle(frame, (150,0), 20, (255,0,0), 5)

    cv2.imshow('Mouth Detector', frame)

    # crop and return the cropped
    frame = frame[y1:y2, x1:x2]
    return frame

def calcFitRectangle():
    x_start, y_start = 75, 50
    x1, x2, y1, y2 = x_start, x_start + APP_CONFIG['RECT_WIDTH'], y_start, y_start + APP_CONFIG['RECT_HEIGHT']
    return x1, y1, x2, y2

def drawFitRectangle(frame):
    x1, y1, x2, y2 = calcFitRectangle()
    cv2.rectangle(frame,(x1,y1), (x2,y2),(0,0,255), 2)

def excludeFace(frame):
    x1, y1, x2, y2 = calcFitRectangle()
    return frame[y1:y2, x1:x2]

def processFrames():
    # return cap.read()[1]
    return cv2.resize(cap.read()[1], None, fx=ds_factor, fy=ds_factor, interpolation=cv2.INTER_AREA)

def drawFrame(frame):
    cv2.imshow('Mouth Detector', frame)

lastTimeSaveImage = 0
countImages = 0
def saveFrame(frame):
    if (not APP_CONFIG['save']): return
    videoSaver.write(frame)
    global lastTimeSaveImage
    timeNow = time.time() * 1000
    if (timeNow - lastTimeSaveImage > APP_CONFIG['save_rate']):
        lastTimeSaveImage = timeNow
        global countImages
        countImages += 1
        newImagePath = "{}\\_{}.jpg".format(APP_CONFIG['save_path'],countImages)
        cv2.imwrite(newImagePath, frame)
def main():
    now = datetime.now()
    h, minute,y,d,month,s = now.hour, now.minute, now.year, now.day, now.month, now.second
    date = "{}-{}-{}_{}-{}-{}".format(y,month,d,h,minute,s)

    currentImagePath = ".\\pix\\{}".format(date)
    makedirs(currentImagePath)
    randImagesPath = currentImagePath
    APP_CONFIG['save_path'] = randImagesPath

    # width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)   # float
    # height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT) # float
    res = int(APP_CONFIG['FRAME_WIDTH']), int(APP_CONFIG['FRAME_HEIGHT'])
    global videoSaver
    videoSaver = cv2.VideoWriter("{}\\vid.avi".format(APP_CONFIG['save_path']),cv2.VideoWriter_fourcc(*'XVID'), 10.0, res)

    playGame()

if __name__ == "__main__":
    main()
