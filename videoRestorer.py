'''
    Run this script with arguments shown in 'main' to restore frames in the folder to one video
    This file is related with flappy game, as it is created mainly to restore old runs (when only frames was saved) to the video as game.py do now
'''
import numpy
import cv2
import sys


def main():
    path = sys.argv[1] # path name
    start = int(sys.argv[2]) # start (first file to be read)
    end = int(sys.argv[3]) # end (number of frames that will be included)
    framePerSecond = float(sys.argv[4])

    height, width = cv2.imread("{}\\_{}.jpg".format(path, start)).shape[:2]
    videoRestorer = cv2.VideoWriter("{}\\vid.avi".format(path),cv2.VideoWriter_fourcc(*'XVID'), framePerSecond, (width, height))
    errors = 0
    for count in range(start,end):
        currPath = "{}\\_{}.jpg".format(path, count + errors)
        frame = cv2.imread(currPath)
        while (frame is None):
            errors += 1
            currPath = "{}\\_{}.jpg".format(path, count + errors)
            frame = cv2.imread(currPath)
        print(currPath," ", count)
        videoRestorer.write(frame)

    videoRestorer.release()


if __name__:
    main()