'''
    There is where I try some python using skimage to test code, algorithm, or to implement new required functionality
'''
import numpy as np
import cv2
import imutils
from skimage import morphology as mph, util
from skimage.io import imread, imshow
from skimage.morphology import watershed
from scipy import ndimage as ndi

def grayImage(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image

def readFrame(cap):
    grayed = grayImage(cap.read()[1])
    return grayed
    return cv2.GaussianBlur(grayed,(21, 21), 0)


def imageDifferencing(pre, curr):
    return cv2.absdiff(curr, pre)

def main():
    return simulateMorphologyShape()
    cap = cv2.VideoCapture(0)
    currFrame = prevFrame = readFrame(cap)
    h,w = currFrame.shape
    print(currFrame.dtype)
    print(h,w)
    while(True):
        diffFrame = imageDifferencing(currFrame, prevFrame)
        temp =  np.array(currFrame,dtype=np.uint8)
        # for y in range(h):
        #     for x in range(w):
        #         if (diffFrame[y,x] > 8):
        #             temp[y,x] = 255
        #         else:
        #             temp[y,x] = 0
        # # retval, thresh_gray = cv2.threshold(diffFrame, 12, maxval=255, type=cv2.THRESH_BINARY)
        # # image, contours, hierarchy = cv2.findContours(thresh_gray,cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        # # for c in contours:
        # #     # get the bounding rect
        # #     x, y, w, h = cv2.boundingRect(c)
        # #     # draw a green rectangle to visualize the bounding rect
        # #     cv2.rectangle(temp, (x, y), (x+w, y+h), (0, 255, 0), 2)
        # if len(contours):
        #     cont = contours[0]
        #     x,y,w,h = cv2.boundingRect(cont)
        #     roi=diffFrame[y:y+h,x:x+w]
        #     h_,w_ = roi.shape
        #     cv2.imshow("{}{}".format(h_,w_), roi)
        cv2.imshow("Test Show curr", temp)
        cv2.imshow('Test Show Diff', diffFrame)

        prevFrame = currFrame
        currFrame =  readFrame(cap)
        cv2.waitKey(100)




# thresh = cv2.threshold(diffFrame, 25, 255, cv2.THRESH_BINARY)[1]
# thresh = cv2.dilate(thresh, None, iterations=2)
# cnts = cv2.findContours(thresh[1], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# cnts = cnts[0] if imutils.is_cv2() else cnts[1]

# for c in cnts:
#     # if the contour is too small, ignore it
#     if cv2.contourArea(c) < 1:
#         continue
#     # compute the bounding box for the contour, draw it on the frame,
#     # and update the text
#     (x, y, w, h) = cv2.boundingRect(c)
#     cv2.rectangle(currFrame, (x, y), (x + w, y + h), (0, 255, 0), 2)
def simulateMorphologyShape():
    #cap = cv2.VideoCapture(0)
    sampleImage = imread("sample detected hand.jpg",as_grey=True)
    while(True):
        distance = ndi.distance_transform_edt(sampleImage)
        
        normalizedDistance = (distance - np.min(distance))/np.ptp(distance)
        print(type(distance))
        #print(distance)
        #inverted = np.invert(sampleImage.astype(np.uint8))
        cv2.imshow("Test Show inverted", sampleImage)
        cv2.imshow("Test Show distance", 1 - normalizedDistance)

        
        # # currFrame = readFrame(cap)
        # # cv2.imshow("Test Show curr", currFrame)
        # # x1,y1, w,h = cv2.selectROI("Test Show curr", currFrame)
        # # if(not w and not h):
        # #     continue
        # # print(x1,y1, w,h)
        # # getMorphology(currFrame[y1:y1+h,x1:x1+w])
        # cv2.imshow("loaded", sampleImage)
        # getMorphology(sampleImage)
        cv2.waitKey(30)

def getMorphology(mROI):
    np.argmax(mROI, axis=1)
    for row in mROI:
        if(np.max(row) == 0):
            print('empty row')
        else:
            print('there is a value here in this row')



if __name__ == '__main__':
    main()