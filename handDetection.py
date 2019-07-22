from utils import detector_utils as detector_utils
import cv2
import tensorflow as tf
import datetime
import argparse
import time
import pyautogui
import function as fn
from showNotification import showNotif as notification


detection_graph, sess = detector_utils.load_inference_graph()

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-sth',
        '--scorethreshold',
        dest='score_thresh',
        type=float,
        default=0.3,
        help='Score threshold for displaying bounding boxes')
    parser.add_argument(
        '-fps',
        '--fps',
        dest='fps',
        type=int,
        default=1,
        help='Show FPS on detection/display visualization')
    parser.add_argument(
        '-src',
        '--source',
        dest='video_source',
        default=0,
        help='Device index of the camera.')
    parser.add_argument(
        '-wd',
        '--width',
        dest='width',
        type=int,
        default=960,
        help='Width of the frames in the video stream.')
    parser.add_argument(
        '-ht',
        '--height',
        dest='height',
        type=int,
        default=540,
        help='Height of the frames in the video stream.')
    parser.add_argument(
        '-ds',
        '--display',
        dest='display',
        type=int,
        default=1,
        help='Display the detected images using OpenCV. This reduces FPS')
    parser.add_argument(
        '-num-w',
        '--num-workers',
        dest='num_workers',
        type=int,
        default=4,
        help='Number of workers.')
    parser.add_argument(
        '-q-size',
        '--queue-size',
        dest='queue_size',
        type=int,
        default=2,
        help='Size of the queue.')

    notification("Starting Filipa",
                 "Filipa is running in the background to adjust frame open the camera window and ajust",
                 7)

    args = parser.parse_args()

    cap = cv2.VideoCapture(args.video_source)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    start_time = datetime.datetime.now()
    num_frames = 0
    im_width, im_height = (cap.get(3), cap.get(4))
    # max number of hands we want to detect/track
    num_hands_detect = 1

    cv2.namedWindow('Single-Threaded Detection', cv2.WINDOW_AUTOSIZE)

    # List maintaining the with cordinates with time, average distance
    widthList = []  # Array consist of the horizontal dimensions along with timestamp
    heightList = []  # Array consist of the verticle dimentions along with timestamp
    action = True  # Action flag is used to control action making

    desktopFlag = False #desktop flag for monitoring desktop
    
    notification("Filipa started",
                 "Filipa could not detect hands show hands in the camera to continue",
                 7)
    while True:
        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        ret, image_np = cap.read()
        # image_np = cv2.flip(image_np, 1)
        try:
            image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        except:
            print("Error converting to RGB")
        # Actual detection. Variable boxes contains the bounding box cordinates for hands detected,
        # while scores contains the confidence for each of these boxes.
        # Hint: If len(boxes) > 1 , you may assume you have found atleast one hand (within your score threshold)

        boxes, scores = detector_utils.detect_objects(image_np,
                                                      detection_graph, sess)

        (left, right, top, bottom) = (boxes[0][1] * im_width, boxes[0][3] * im_width,
                                      boxes[0][0] * im_height, boxes[0][2] * im_height)

        # Get center of the box
        (width, height) = fn.getCenter(left, right, top, bottom)

        # If the prob of the hand is more than 40%
        if scores[0] > 0.4:

            # Variable for taking decision on the gesture
            left = False
            right = False
            up = False
            down = False
            diagonal = False


            # minimize all windows
            if desktopFlag == False:
                print("Minizing all windows ----->")
                # Minimize all windows and show the desktop
                pyautogui.hotkey('win', 'd')
                desktopFlag = True

            if len(widthList) == 0 or len(heightList) == 0:
                
                widthList.clear()
                heightList.clear()
                widthList.append([width, time.time()])
                heightList.append([height, time.time()])

            else:
                if len(widthList) != 0 and len(heightList) != 0:
                    prev = widthList.pop()
                    tprev = heightList.pop()

                    widthList.append([width, time.time()])
                    heightList.append([height, time.time()])

                    avg = (widthList[0][0] - prev[0])

                    tavg = (heightList[0][0] - tprev[0])

                    if widthList[0][1] - prev[1] > 1.25:
                        widthList.clear()
                    else:
                        if abs(avg) > 0.20*im_width:
                            temp = 0

                            if avg < 0:
                                right = True
                            else:
                                left = True

                            widthList.clear()

                        else:
                            left = False
                            right = False
                            print(f"Nothing detected on width: {abs(avg)}")

                    if heightList[0][1] - tprev[1] > 1.25:
                        heightList.clear()

                    else:
                        if abs(tavg) > 0.3*im_height:

                            # Check the time array for no entry

                            if tavg < 0:
                                up = True

                            else:
                                down = True
                            heightList.clear()

                            # Check for the diagonal swipe
                            if right == True or left == True:
                                diagonal = True

                        else:
                            up = False
                            down = False
                            diagonal = False
                            print(f"Nothing detected on height: {abs(avg)}")

                    # Make decision based on the above conditions
                    if diagonal == True:
                        if action:
                            notification("Actions are halted",
                                         "Actions are turned off and no action will be taken on any gesture until they are turned on again",
                                         5)
                            action = False
                        else:
                            notification("Actions started again",
                                         "Actions are turned on now gesture will be recognised and action will be taken based on the gesture",
                                         5)
                            action = True
                    else:
                        if action:
                            if right == True:
                                print("right ---->")
                                pyautogui.press("right")

                                time.sleep(1.25)
                            elif left == True:
                                print("left <----")
                                pyautogui.press("left")

                                time.sleep(1.25)
                            elif up == True:
                                print("Up <|>")
                                pyautogui.press("up")

                                time.sleep(1.25)
                            elif down == True:
                                print("Down ^|^")
                                pyautogui.press("down")                                

                                time.sleep(1.25)
                        else:
                            print("No action is selected--->")

        # draw bounding boxes on frame
        detector_utils.draw_box_on_image(num_hands_detect, args.score_thresh,
                                         scores, boxes, im_width, im_height,
                                         image_np)

        # Calculate Frames per second (FPS)
        num_frames += 1
        elapsed_time = (datetime.datetime.now() - start_time).total_seconds()
        fps = num_frames / elapsed_time

        # image_np = cv2.flip(image_np, 1)

        if (args.display > 0):
            # Display FPS on frame
            if (args.fps > 0):
                detector_utils.draw_fps_on_image("FPS : " + str(int(fps)),
                                                 image_np)
            cv2.imshow('Single-Threaded Detection',
                       cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))

            if cv2.waitKey(25) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break
        else:
            print("frames processed: ", num_frames, "elapsed time: ",
                  elapsed_time, "fps: ", str(int(fps)))
