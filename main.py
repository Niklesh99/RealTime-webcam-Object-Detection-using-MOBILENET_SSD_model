from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import time
import cv2


ap = argparse.ArgumentParser()
ap.add_argument("-c", "--confidence", type=float, default=0.2,help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

#MOBILENET Pre-trained caffe model absolute path
proto = r"C:\Users\Priya\Desktop\Realtime_obj\SSD_MobileNet_prototxt.txt"
caffemodel = r"C:\Users\Priya\Desktop\Realtime_obj\SSD_MobileNet.caffemodel"

phones = 0 			#Initializing the no of phones 
# initialize the classes labels manually
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	"phone", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))   	#assigning color for each object 

print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(proto,caffemodel) 			#reading the caffe model and prototxt file 
print("[INFO] starting video stream...")
vs = VideoStream(0).start()  			# Accessing the web cam and starting to capture the frame
time.sleep(2.0)
fps = FPS().start()						#Initializing the Stream FPS

while True:										#looping over the webcam streaming frames 
	frame = vs.read()
	frame = imutils.resize(frame, width=400)	#resizing the frame to a size of 400 for optimization
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),0.007843, (300, 300), 127.5)			#detecting blob from the frames of the stream and storing it
	print(blob.shape)
	print(net)
	
	net.setInput(blob)					#passing the blob obtained and storing the detection.
	print(net)
	detections = net.forward()

	for i in np.arange(0, detections.shape[2]):				    # looping over the detections obtained from the blob
		confidence = detections[0, 0, i, 2]
		if confidence > args["confidence"]:						#filtering weak confidence by checking the confidence of the detection is above 20 %.

			idx = int(detections[0, 0, i, 1])					#index
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])			#detecting x,y of the bounding box detection
			(startX, startY, endX, endY) = box.astype("int")				
			label = "{}: {:.2f}%".format(CLASSES[idx],confidence * 100)			#Labeling the detection and confidence in frame runtime.
			cv2.rectangle(frame, (startX, startY), (endX, endY),COLORS[idx], 2)	#drawing rectangle over the detected x,y.
			y = startY - 15 if startY - 15 > 15 else startY + 15
			cv2.putText(frame, label, (startX, y),cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)



	cv2.imshow("Frame", frame)				#output window
	key = cv2.waitKey(1) & 0xFF
	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break
	# update the FPS counter
	fps.update()

fps.stop()								#FPS Stop.
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
# do a bit of cleanup
cv2.destroyAllWindows()				
vs.stop()