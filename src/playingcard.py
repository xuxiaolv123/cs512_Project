import cv2
import numpy as np
from glob import glob
import imutils
import sys


def trainCards():
	print '********************** Start Trainning **********************'

	trainning = {}

	img_mask = '../data/crop_trainset/*.jpg'
	img_names = glob(img_mask)
	for item in img_names:
		image = cv2.imread(item)
		item_modify = item.split('/')[3].split('.')[0]
		#image = cv2.resize(image,(450,450))
   		trainning[item_modify] = preprocessimg(image)
   	print '********************** Trainning complete **********************'
   	print '********************** There are ',len(trainning),' cards are trainned **********************'
   	return trainning

def preprocessimg(img):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray,(5,5),0)
    return blur


def findCard(img):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray,(1,1),1000)
    flag, thresh = cv2.threshold(blur, 120, 255, cv2.THRESH_BINARY) 

    _, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    contours = sorted(contours, key=cv2.contourArea,reverse=True)  


    peri = cv2.arcLength(contours[0],True)
    h = cv2.approxPolyDP(contours[0],0.02*peri,True)
    x,y,w,height = cv2.boundingRect(contours[0])
    h = h.reshape((4,2))
    pts = np.float32(h)
    
    temp_rect = np.zeros((4,2),dtype = np.float32)
    
    s = pts.sum(1)
    
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]

    diff = np.diff(pts, axis = 1)
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]


    if w <= 0.8*height: # If card is vertically oriented
        temp_rect[0] = tl
        temp_rect[1] = tr
        temp_rect[2] = br
        temp_rect[3] = bl

    if w >= 1.2*height: # If card is horizontally oriented
        temp_rect[0] = bl
        temp_rect[1] = tl
        temp_rect[2] = tr
        temp_rect[3] = br

    if w > 0.8*height and w < 1.2*height: #If card is diamond oriented
    # If furthest left point is higher than furthest right point,
    # card is tilted to the left.
        
        if pts[1][1] <= pts[3][1]:
            # If card is titled to the left, approxPolyDP returns points
            # in this order: top right, top left, bottom left, bottom right
            temp_rect[0] = pts[1] # Top left
            temp_rect[1] = pts[0] # Top right
            temp_rect[2] = pts[3] # Bottom right
            temp_rect[3] = pts[2] # Bottom left

        # If furthest left point is lower than furthest right point,
        # card is tilted to the right
        if pts[1][1] > pts[3][1]:
            # If card is titled to the right, approxPolyDP returns points
            # in this order: top left, bottom left, bottom right, top right
            temp_rect[0] = pts[0] # Top left
            temp_rect[1] = pts[3] # Top right
            temp_rect[2] = pts[2] # Bottom right
            temp_rect[3] = pts[1] # Bottom left     

    des = np.array([[0,0],[449,0],[449,449],[0,449]],np.float32)

    transform = cv2.getPerspectiveTransform(temp_rect,des)
    warp = cv2.warpPerspective(img,transform,(450,450))


    return warp


def cardDiff(test_img, training_set):

    found = None
    bestmatch = ''
    # loop over the images to find the template in
    for label, template in training_set.items():
      #print (template)
      template = cv2.Canny(template, 50, 200)
      (tH, tW) = template.shape[:2]
     
      # loop over the scales of the image
      for scale in np.linspace(0.2, 1.0, 20)[::-1]:
        # resize the image according to the scale, and keep track
        # of the ratio of the resizing
        resized = imutils.resize(test_img, width = int(test_img.shape[1] * scale))
        r = test_img.shape[1] / float(resized.shape[1])
     
        # if the resized image is smaller than the template, then break
        # from the loop
        if resized.shape[0] < tH or resized.shape[1] < tW:
            break
        # detect edges in the resized, grayscale image and apply template
        # matching to find the template in the image
        edged = cv2.Canny(resized, 50, 200)
        result = cv2.matchTemplate(edged, template, cv2.TM_CCOEFF)
        (_, maxVal, _, maxLoc) = cv2.minMaxLoc(result)
     
        # if we have found a new maximum correlation value, then ipdate
        # the bookkeeping variable
        if found is None or maxVal > found[0]:
            found = (maxVal, maxLoc, r)
            bestmatch = label
            #print label
            #print found

    return bestmatch

'''
def cardDiff(img1, img2):
    image1 = cv2.GaussianBlur(img1,(5,5),5)
    image2 = cv2.GaussianBlur(img2,(5,5),5)   
    diff = cv2.absdiff(image1,image2)  
    diff = cv2.GaussianBlur(diff,(5,5),5)    
    flag, diff = cv2.threshold(diff, 127, 255, cv2.THRESH_BINARY) 

    return np.sum(diff)

def matchCards(test_img, train_imgs):
    #test_features = preprocessimg(test_img)
    diff_dic = {}
    for label, train_features in train_imgs.items():
        diff_dic[label] = cardDiff(test_img, train_features)
    return sorted(diff_dic.keys(), key=diff_dic.get, reverse=True)
'''

def main():
	if len(sys.argv) < 2:
		print '''
Instruction: 
Use command "python playingcard.py test1.jpg" to detect and recognize card
Program displays original testimage, rotated test image and result image
Note: Image windows can be overlapped, please drag them apart for a better view
			'''
	else:

		img = cv2.imread(sys.argv[1])
		showimg = cv2.resize(img,(300,450))
		test_img = preprocessimg(findCard(img))

		trainset = trainCards()

		print '********************** Start matching cards **********************'

		result = cardDiff(test_img, trainset)
		print result
		#print matchCards(test_img,trainCards())

		cv2.imshow('Original test image',showimg)
		cv2.imshow('Test image after preprocessing and rotation',test_img)
		result_img = cv2.putText(showimg,'%s'%result,(50,225),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255, 0, 0), 2, cv2.LINE_AA)
		cv2.imshow('Result image',result_img)
		#cv2.imshow('test',trainCards()['Diamond of Six'])
		cv2.waitKey(0)
		cv2.destroyAllWindows()


if __name__ == '__main__':
    main()