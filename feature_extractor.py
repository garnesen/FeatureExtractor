import cv2
import os

GRID_SIZE = 8

def getFeaturesOf(img):
	"""
	Returns a list of values representing input for a neural net
	"""
	reduced_img = cv2.resize(cropCharFromImage(img), (GRID_SIZE, GRID_SIZE), fx=0, fy=0, interpolation=cv2.INTER_AREA)
	return reduced_img.ravel()

def showFeaturesOf(img):
	"""
	Shows a visual of the features of an image
	"""
	reduced_img = cv2.resize(cropCharFromImage(img), (GRID_SIZE, GRID_SIZE), fx=0, fy=0, interpolation=cv2.INTER_AREA)
	blown_up = cv2.resize(src=reduced_img, dsize=(500, 500), interpolation=cv2.INTER_NEAREST)
	showImage(blown_up)

def cropCharFromImage(img):
	"""
	Crops an image so that only the char is left
	"""

	# Add a small white border to the image
	img = cv2.copyMakeBorder(src=img, top=1, left=1, bottom=1, right=1, borderType=cv2.BORDER_CONSTANT, value=(255, 255, 255))

	# Find contours
	ret,thresh = cv2.threshold(img, 127, 255, 0)
	img2,contours,hierarchy = cv2.findContours(thresh, 1, 2)

	# Get best fit rectangle of character
	img_size = (len(img[0]), len(img))

	x, y, w, h = getCombinedRect(contours, img_size)
	return img[y:(y + h), x:(x + w)]

def getCombinedRect(contours, img_size):
	"""
	Given contours, finds combined rectangle for bounding
	img_size is a tuple (w, h) of the original image
	"""
	rects = [cv2.boundingRect(c) for c in contours]
	# filter through, getting only the significant rects
	isGoodBoxes = [r for r in rects if isGoodBox(r, img_size)]
	# now, with the good boxes, find the most top-left and bottom-right corners
	lefts = [r[0] for r in isGoodBoxes]
	leftmost = min(lefts)

	rights = [r[0] + r[2] for r in isGoodBoxes]
	rightmost = max(rights)

	tops = [r[1] for r in isGoodBoxes]
	topmost = min(tops)

	bottoms = [r[1] + r[3] for r in isGoodBoxes]
	bottommost = max(bottoms)

	return (leftmost, topmost, rightmost - leftmost,  bottommost - topmost)
	
def isGoodBox(box, img_size):
	"""
	Decides if the given box (x, y, w, h) is a good bounding
	box based on its size
	"""
	x,y,w,h = box
	if w == img_size[0] and h == img_size[1]:
		# the bounding box is of the entire image
		return False
	elif w * h <= img_size[0] * img_size[1] / 100.0:
		return False
	else:
		return True

def loopThroughImages(directory, function):
	"""
	Loops through all images in a directory performing the given function on it
	"""
	for dirs, subdirs, files in os.walk(directory):
		for filename in files:
			if filename.endswith('.png') or filename.endswith('jpg'):
				fullname = os.path.join(dirs, filename)
				print(fullname)
				img = cv2.imread(fullname, cv2.IMREAD_GRAYSCALE)
				function(img)

def showBoundedImage(img, x, y, w, h):
	"""
	Show an image with its bounding box
	"""
	bounded_img = cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
	showImage(bounded_img)

def showImage(img):
	"""
	Show an image
	"""
	cv2.imshow('Test Window', img)
	cv2.waitKey(0)

def readImage(file_loc):
	return cv2.imread(file_loc, cv2.IMREAD_GRAYSCALE)