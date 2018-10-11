#!/usr/bin/env python
from __future__ import print_function
from os import makedirs
# import tensorflow as tf
import cv2
import sys
import pygame
from pygame.locals import K_UP, K_ESCAPE, K_f, K_n
sys.path.append("game/")
import wrapped_flappy_bird as game
import random
import numpy as np
from collections import deque
from datetime import datetime


ACTIONS = 2 # number of valid actions

mouth_cascade = cv2.CascadeClassifier('HaarCasscades/smile.xml')
if mouth_cascade.empty():
  raise IOError('Unable to load the mouth cascade classifier xml file')
cap = cv2.VideoCapture(0)
ds_factor = 0.5

counter = 0
APP_CONFIG = {'save':False}

def isMouth():
    mouth = False
    # get the current photo
    ret, frame = cap.read()
    frame = cv2.resize(frame, None, fx=ds_factor, fy=ds_factor, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # inc counter
    global counter
    counter = counter + 1

    # save if next image and config is on
    if (counter % 150 and APP_CONFIG['save']):
        newImagePath = "{}\\_{}.jpg".format(APP_CONFIG['save_path'],counter)
        cv2.imwrite(newImagePath, frame)

    # detect smile or fist or ...
    mouth_rects = mouth_cascade.detectMultiScale(gray, 1.7, 11)
    for (x,y,w,h) in mouth_rects:
        y = int(y - 0.15*h)
        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 3)
        mouth = x and y and w and h
    cv2.imshow('Mouth Detector', frame)
    return mouth


def playGame():
    # open up a game state to communicate with emulator
    game_state = game.GameState()

    while True:
        keys = pygame.key.get_pressed()
        if (keys[K_ESCAPE]):
            cap.release()
            cv2.destroyAllWindows()
            return
        if (keys[K_f]):
            APP_CONFIG['save'] = False
            print('off')
        if (keys[K_n]):
            APP_CONFIG['save'] = True
            print('on')
        
        action = np.zeros(ACTIONS)
        action[0] = 1
        if isMouth():
            action[0] = 0
            action[1] = 1
        game_state.frame_step(action)

def main():
    now = datetime.now()
    h = now.hour
    minute = now.minute
    y = now.year
    d = now.day
    month = now.month
    s = now.second
    date = "{}-{}-{}_{}-{}-{}".format(y,month,d,h,minute,s)
    currentImagePath = "D:\\imageProcessing game\\(edited) DeepLearningFlappyBird\\pix\\{}".format(date)
    makedirs(currentImagePath)
    randImagesPath = currentImagePath
    APP_CONFIG['save_path'] = randImagesPath
    playGame()

if __name__ == "__main__":
    main()
