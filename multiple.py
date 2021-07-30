# USAGE
# python single_image_yolo.py
# Runs on Single image given in image path

# import the necessary packages
import numpy as np
import argparse
import time
import cv2
import RPi.GPIO as GPIO
import os
from glob import glob
import datetime


ok_pin = 19
ng_pin=23 # Board pin 12
in_pin = 15  # Board pin 15
dispH=900 
dispW= 1280
flip=0
camSet='nvarguscamerasrc !  video/x-raw(memory:NVMM), width=3264, height=2464, format=NV12, framerate=21/1 ! nvvidconv flip-method='+str(flip)+' ! video/x-raw, width='+str(dispW)+', height='+str(dispH)+', format=BGRx ! videoconvert ! video/x-raw, format=BGR ! appsink'


GPIO.setmode(GPIO.BOARD)  # BOARD pin-numbering scheme
GPIO.setup(ok_pin, GPIO.OUT)  # LED pin set as output
GPIO.setup(ng_pin, GPIO.OUT)  # LED pin set as output

GPIO.setup(in_pin, GPIO.IN)  # button pin set as 
# Initial state for LEDs:
GPIO.output(ok_pin, GPIO.LOW)
GPIO.output(ng_pin, GPIO.LOW)

print("Starting demo now! Press CTRL+C to exit")
i=1
j=30
while i<30:
	print("Waiting for signal from robot")
	x = GPIO.input(in_pin)
	print(x)
	if x==1:
		print("Signal Received!",i)
		i+=1
		GPIO.output(led_pin, GPIO.LOW)
		camera = cv2.VideoCapture(camSet)
		return_value, image = camera.read()
		#image=image[300:700,470:950]
		dt=str(datetime.datetime.now())
		name = "input/"+"{}.png".format(dt)
		#name=+'.png'
		#out_filename = "input/input" +name		
		
		cv2.imwrite(name, image)
		j+=1
		time.sleep(0.05)
		GPIO.output(led_pin, GPIO.HIGH)
		camera.release()
				
		# construct the argument parse and parse the arguments
		ap = argparse.ArgumentParser()
		# ap.add_argument("-i", "--image", required=True, help="path to input image")
		ap.add_argument(
			"-y", "--yolo", required=False, default="yolo-trained-files", help="base path to YOLO directory"
		)
		ap.add_argument(
			"-c",
			"--confidence",
			type=float,
			default=0.5,
			help="minimum probability to filter weak detections",
		)
		ap.add_argument(
			"-t",
			"--threshold",
			type=float,
			default=0.5,
			help="threshold when applyong non-maxima suppression",
		)
		args = vars(ap.parse_args())

		# load the Digits class labels our YOLO model was trained on
		labelsPath = os.path.sep.join([args["yolo"], "obj.names"])
		LABELS = open(labelsPath).read().strip().split("\n")

		# initialize a list of colors to represent each possible class label
		np.random.seed(24)
		COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")
		black = [0, 0, 0]
		# derive the paths to the YOLO weights and model configuration
		weightsPath = os.path.sep.join([args["yolo"], "custom-yolov4-tiny-detector_best.weights"])
		configPath = os.path.sep.join([args["yolo"], "custom-yolov4-tiny-detector.cfg"])

		# load our YOLO object detector trained on COCO dataset (80 classes)
		print("[INFO] loading Tiny-YOLO V4 from disk...")
		net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

		# load our input image and grab its spatial dimensions
		# image = cv2.imread(args["image"])

		###GIVE IMAGE PATH
		image_path = name


		image = cv2.imread(image_path)

		(H, W) = image.shape[:2]

		# determine only the *output* layer names that we need from YOLO
		ln = net.getLayerNames()
		ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

		# construct a blob from the input image and then perform a forward
		# pass of the YOLO object detector, giving us our bounding boxes and
		# associated probabilities
		blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
		net.setInput(blob)
		start = time.time()
		layerOutputs = net.forward(ln)
		end = time.time()

		# show timing information on YOLO
		print("[INFO] Inference Time: {:.6f} seconds".format(end - start))

		# initialize our lists of detected bounding boxes, confidences, and
		# class IDs, respectively
		boxes = []
		confidences = []
		classIDs = []

		# loop over each of the layer outputs
		for output in layerOutputs:
			# loop over each of the detections
			for detection in output:
				# extract the class ID and confidence (i.e., probability) of
				# the current object detection
				scores = detection[5:]
				classID = np.argmax(scores)
				confidence = scores[classID]

				# filter out weak predictions by ensuring the detected
				# probability is greater than the minimum probability
				if confidence > args["confidence"]:
				    # scale the bounding box coordinates back relative to the
				    # size of the image, keeping in mind that YOLO actually
				    # returns the center (x, y)-coordinates of the bounding
				    # box followed by the boxes' width and height
				    box = detection[0:4] * np.array([W, H, W, H])
				    (centerX, centerY, width, height) = box.astype("int")

				    # use the center (x, y)-coordinates to derive the top and
				    # and left corner of the bounding box
				    x = int(centerX - (width / 2))
				    y = int(centerY - (height / 2))

				    # update our list of bounding box coordinates, confidences,
				    # and class IDs
				    boxes.append([x, y, int(width), int(height)])
				    confidences.append(float(confidence))
				    classIDs.append(classID)

		# apply non-maxima suppression to suppress weak, overlapping bounding
		# boxes
		idxs = cv2.dnn.NMSBoxes(boxes, confidences, args["confidence"], args["threshold"])
		# ensure at least one detection exists
		# print(classIDs)
		# print(idxs.flatten())
		recog_digits = []
		digits_confi = {}
		xcoord_dict = {}
		if len(idxs) > 0:
			# loop over the indexes we are keeping
			for i in idxs.flatten():
				# extract the bounding box coordinates
				(x, y) = (boxes[i][0], boxes[i][1])
				(w, h) = (boxes[i][2], boxes[i][3])

				# draw a bounding box rectangle and label on the image
				color = [int(c) for c in COLORS[classIDs[i]]]
				cv2.rectangle(image, (x, y), (x + w, y + h), color, 1)
				output_text = "{}: {:.1%}".format(LABELS[classIDs[i]], confidences[i])
				# print(output_text)

				digits_confi[classIDs[i]] = confidences[i]

				# xcoord_dict[classIDs[i]] = x
				xcoord_dict[x] = classIDs[i]

				text = "{}".format(LABELS[classIDs[i]])
				cv2.rectangle(image, (x, y - 30), (x + w, y + h - 35), color, thickness=cv2.FILLED)
				cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, black, 2)

		out_filename = "output/" + image_path.split("/")[-1]
		cv2.imwrite(out_filename, image)
		# xcoord_dict = dict(sorted(xcoord_dict.items(), key=lambda item: item[1]))
		xcoord_dict = dict(sorted(xcoord_dict.items()))

		print("\n[OUTPUT]\nRecognized Digit: Confidence")
		recognized_digits = []
		for key, value in xcoord_dict.items():
			print(value, ":", "{:.1%}".format(digits_confi[value]))
			recognized_digits.append(str(value))

		print("\nFinal Output:")
		output_digits = "".join(recognized_digits)
		print(output_digits)

		dest_file = "".join("Output") + ".csv"
		with open(dest_file, "a") as writer:
			writer.write(output_digits+'\t'+dt+"\n")
			

		print("\n[Success]")
		print("Output Image Successfully saved!")
		print("Output Image Path: {}".format(out_filename))
		print("Output Text Path: {}".format(dest_file))
		cv2.imshow(" Output Image", image)
		cv2.waitKey(100)
# show the output image
#
