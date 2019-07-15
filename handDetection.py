cfrom utils import detector_utils as detector_utils
import cv2
import tensorflow as tf
import datetime
import argparse
import time
import pyautogui

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
        default=860,
        help='Width of the frames in the video stream.')
    parser.add_argument(
        '-ht',
        '--height',
        dest='height',
        type=int,
        default=440,
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

    d = []
    t = []
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

        if scores[0] > 0.5:
            if len(d) == 0:
                d.append([left, right, time.time()])
                t.append([top, bottom, time.time()])
            else:
                prev = d.pop()
                tprev = t.pop()

                d.append([left, right, time.time()])
                t.append([top, bottom, time.time()])

                avg = (d[0][0] - prev[0]) + (d[0][1] - prev[1])
                avg = avg/2

                tavg = (t[0][0] - tprev[0]) + (t[0][1] - tprev[1])
                tavg = tavg/2


                if d[0][2] - prev[2] > 3:
                    d.clear()
                else:
                    if abs(avg) > 0.20*im_width:
                        if avg < 0:
                            print("Its a right swipe")
                            pyautogui.press('right')

                        else:
                            print("Its a left swipe")
                            # pyautogui.press('left')

                    else:
                        print(f"Nothing detected : {abs(avg)}")


                if t[0][2] - tprev[2] > 3:
                    t.clear()
                else:
                    if abs(tavg) > 0.3*im_height:
                        if tavg < 0:
                            print("Up swipe")
                            pyautogui.press('left')
                            #pyautogui.press('win')


                        else:
                            print("Down swipe")
                            # pyautogui.press('down')

                    else:
                        print(f"Nothin detected : {abs(avg)}")
             
            
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