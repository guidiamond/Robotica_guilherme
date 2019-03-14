import numpy as np
import cv2, argparse, imutils, time
from imutils.video import VideoStream, FPS

# python p3.py -v VIDEO -t TRACKER -o OBJECT -c CONFIDENCE


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", type=str,
	help="path to input video file")
ap.add_argument("-t", "--tracker", type=str, default="kcf",
	help="OpenCV object tracker type")
# ap.add_argument("-p", "--prototxt", required=True,
# 	help="path to Caffe 'deploy' prototxt file")
# ap.add_argument("-m", "--model", required=True,
# 	help="path to Caffe pre-trained model")
ap.add_argument("-c", "--confidence", type=float, default=0.2,
	help="minimum probability to filter weak detections")
ap.add_argument("-o", "--object", type=str, default='dog',
	help="object to be tracked")
args = vars(ap.parse_args())




# initialize the list of class labels MobileNet SSD was trained to detect, then generate a set of bounding box colors for each class
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

net = cv2.dnn.readNetFromCaffe('MobileNetSSD_deploy.prototxt.txt', 'MobileNetSSD_deploy.caffemodel')

def detect(frame):
    image = frame.copy()
    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 0.007843, (300, 300), 127.5)

    # pass the blob through the network and obtain the detections and predictions

    print("[INFO] computing object detections...")
    net.setInput(blob)
    detections = net.forward()

    results = []

    # loop over the detections
    for i in np.arange(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with the prediction
        confidence = detections[0, 0, i, 2]

        # filter out weak detections by ensuring the `confidence` is greater than the minimum confidence
        if confidence > args["confidence"]:
            # extract the index of the class label from the `detections`, then
            # compute the (x, y)-coordinates of the bounding box for the object
            idx = int(detections[0, 0, i, 1])
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            startX, startY, endX, endY = box.astype("int")

            # display the prediction
            label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
            print("[INFO] {}".format(label))
            cv2.rectangle(image, (startX, startY), (endX, endY),
                COLORS[idx], 2)
            y = startY - 15 if startY - 15 > 15 else startY + 15
            cv2.putText(image, label, (startX, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)
            
            # ("CLASS", confidence, (x1, y1, x2, y2))
            results.append((CLASSES[idx], confidence*100, (startX, startY), (endX, endY)))

    return image, results



OPENCV_OBJECT_TRACKERS = {
    "csrt": cv2.TrackerCSRT_create,
    "kcf": cv2.TrackerKCF_create,
    "boosting": cv2.TrackerBoosting_create,
    "mil": cv2.TrackerMIL_create,
    "tld": cv2.TrackerTLD_create,
    "medianflow": cv2.TrackerMedianFlow_create,
    "mosse": cv2.TrackerMOSSE_create}
tracker = OPENCV_OBJECT_TRACKERS[args["tracker"]]()


initBB = None # initialize coords of BB (tracking object)


# cap = cv2.VideoCapture('hall_box_battery_1024.mp4')

if not args.get("video", False):
	print("[INFO] starting video stream...")
	vs = VideoStream(src=0).start()
	time.sleep(1.0)

else: # otherwise, grab a reference to the video file
	vs = cv2.VideoCapture(args["video"])

print("Known classes")
print(CLASSES)

detect_counter = 0

detect_mode = True
fps = FPS().start()
while True:
    frame = vs.read()
    frame = frame[1] if args.get("video", False) else frame
    
    if frame is None: # end of stream
        break

    frame = imutils.resize(frame, width=500) # resize the frame # NOT OK 
    (H, W) = frame.shape[:2] # Get dimensions

    if detect_mode:
        frame, result_tuples = detect(frame)

        if args['object'] in {i[0] for i in result_tuples}: # Check if the object was found
            detect_counter += 1
            if detect_counter >= 5: # Get out of detect mode if the object was detected for 5 continuous frames
                detect_mode = False
                detect_counter = 0 # Reset the counter

        else: # No object was found, reset counter
            detect_counter = 0
        

        # cv2.imshow('frame', result_frame)

        # Prints the structures results:
        # ("CLASS", confidence, (x1, y1, x2, y2))
        for t in result_tuples:
            print(t)

        if not detect_mode:
            # Initialization of tracking mode
            (x1, y1), (x2, y2) = result_tuples[0][2:]
            h = y2 - y1
            l = x2 - x1

            initBB = (x1, y1, l, h)
            tracker.init(frame, initBB)
            
            print("\n\nTracking\n\n")


    if not detect_mode: # Instead of else, so the tracking starts in the same frame

        # Tracking mode
        if initBB is not None: # TODO
            # grab the new bounding box coordinates of the object
            (success, box) = tracker.update(frame)

            if success: # check to see if the tracking was a success
                (x, y, w, h) = [int(v) for v in box]
                cv2.rectangle(frame, (x, y), (x + w, y + h),
                    (0, 255, 0), 2)
            else:
                detect_mode=True

    # update the FPS counter
    fps.update()
    fps.stop()

    # initialize the set of information we'll be displaying on the frame
    info = [
        ("Tracker", args["tracker"]),
        ("Mode", "Detecting" if detect_mode else "Tracking"),
        ("FPS", "{:.2f}".format(fps.fps())),
    ]

    # loop over the info tuples and draw them on our frame
    for (i, (k, v)) in enumerate(info):
        text = "{}: {}".format(k, v)
        cv2.putText(frame, text, (10, H - ((i * 20) + 20)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    
    cv2.imshow("frame", frame)

        


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    

if not args.get("video", False): # if we are using a webcam, release the pointer
	vs.stop()
else: # otherwise, release the file pointer
	vs.release()

cv2.destroyAllWindows()