from scipy.spatial import distance as dist
from imutils import face_utils
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2
from typing import List
from define import EAR_THRESH, EAR_CONSEC_FRAMES, EAR_PLOT_PARAMS, COLORS, FRAME_MAX_SIDE, EAR_WINDOW


def eye_aspect_ratio(eye: List) -> float:
    """
    Function to calculate EAR based on eye landmarks.
        :param List eye: List of 6 points of eye contour

        :returns: Calculated EAR
        :rtype: float
    """

    # Euclidean distances used in nominator
    A = dist.euclidean(eye[1], eye[5])  # ||Ul-Dl||
    B = dist.euclidean(eye[2], eye[4])  # ||Ur-Dr||

    # Euclidean distance used in denominator
    C = dist.euclidean(eye[0], eye[3])  # ||L-R||

    # Final formula
    ear = (A + B) / (2.0 * C)

    return ear


def add_new_point(plot_pic: np.ndarray,
                  value: float,
                  color: tuple,
                  radius=1) -> np.ndarray:
    """
    Function used to add new dot to the EAR plot image
        :param numpy.ndarray plot_pic: Plot picture that must be updated
        :param float value: EAR value to add to plot
        :param tuple color: Tuple of (B,G,R) core defining RGB color
        :param int radius: The radius of the point, 1 by default

        :returns: Updated plot picture
        :rtype: numpy.ndarray
    """
    # Get image shape
    height = EAR_PLOT_PARAMS['height']
    width = EAR_PLOT_PARAMS['width']

    # Cutoff to old values
    plot_pic = plot_pic[:, 1:]

    # Add space for new value
    plot_pic = cv2.copyMakeBorder(plot_pic, 0, 0, 0, radius, cv2.BORDER_CONSTANT)

    # Add new point
    y = height - int(height * value * 2)

    plot_pic[y:y + radius, width - radius:width] = color
    return plot_pic


def display_duration(frame: np.ndarray, time: float):
    """
    Display the duration of a blink on the frame
        :param numpy.ndarray frame: Frame on which text will be added
        :param float time: Time to write
    """
    cv2.putText(frame, "Blink Time: {:.2f}".format(time),
                (int(frame.shape[0] * 0.8), int(frame.shape[1] * 0.7)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLORS['text'], 2)


def display_blink_counter(frame: np.ndarray, count: int):
    """
    Display total number of blinks
        :param numpy.ndarray frame: Frame on which text will be added
        :param int count: Number of blinks that will be written
    """
    cv2.putText(frame, "Blinks: {}".format(count), (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLORS['text'], 2)


def display_ear(frame: np.ndarray, ear: float):
    """
    Display current EAR
        :param numpy.ndarray frame: Frame on which text will be added
        :param  float ear: EAR value that will be written
    """
    cv2.putText(frame, "EAR: {:.2f}".format(ear), (int(frame.shape[0] * 0.8), 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLORS['text'], 2)


def display_allert(frame: np.ndarray):
    """
    Display allert
        :param numpy.ndarray frame: Frame on which text will be added
    """
    cv2.putText(frame, "ALERT! TO LONG BLINK!",
                (int(frame.shape[0] * 0.2), int(frame.shape[1] * 0.4)),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 4)


def display_bound(frame: np.ndarray, value: float) -> np.ndarray:
    """
    Display EAR_TRESHOLD in the EAR plot
        :param numpy.ndarray frame: Frame on which line will be added
        :param float value: The level on which line will be (EAR value)

        :returns: Updated picture
        :rtype: numpy.ndarray
    """
    res = frame.copy()
    height = res.shape[0]
    width = res.shape[1]
    y = height - int(height * value * 2)
    cv2.line(res, (0, y), (width - 1, y), (0, 0, 255), 1)
    return res


def is_blink(window: np.ndarray) -> bool:
    """
    Detect quick change in time series
        :param numpy.ndarray window: Last EAR values

        :returns: True is blink is present, False otherwise
        :rtype: bool
    """
    diff = []
    # Calculate differensing with 1 shift
    # No abs is tacken because we want to detect only drops, not rises
    for i in range(1, len(window)):
        diff.append(window[i - 1] - window[i])

    if max(diff) > 0.05:
        return True
    else:
        return False


if __name__ == '__main__':
    # Define arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--shape-predictor", default="models/shape_predictor_68_face_landmarks.dat",
                    help="Path to facial landmark predictor")
    ap.add_argument("-v", "--video", type=str, default="webcam",
                    help="Path to input video file or 'webcam' for webcam stream")
    ap.add_argument("-pl", "--plot-ear", type=bool, default=False,
                    help="True is ear must be ploted in real-time, False otherwise")
    args = vars(ap.parse_args())

    if args['plot_ear']:
        # Create picture that will contain the plot
        EAR_plot = np.zeros((EAR_PLOT_PARAMS['height'], EAR_PLOT_PARAMS['width'], 3), np.uint8)

    print("[INFO] loading facial landmark predictor...")
    # Dlib's face detector (HOG-based)
    detector = dlib.get_frontal_face_detector()
    # Facial landmark predictor
    predictor = dlib.shape_predictor(args['shape_predictor'])

    # Grab interested indexes of the facial landmarks
    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

    BLINK_FR_COUNTER = 0  # Number of subsequent frames classified as blink
    BLINK_COUNTER = 0  # Total number of blinks
    blink_detected = False

    # window for last EAR values to detect drustic change
    ear_window = np.zeros(EAR_WINDOW['blink'])

    # window for last EAR values that ate defined as open eyes
    open_ear_window = np.ones(EAR_WINDOW['open']) * EAR_THRESH

    # Start the video stream from webcam
    print("[INFO] starting video stream ...")
    if args['video'] == 'webcam':
        cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture(args['video'])

    # Get video fps
    fps = cap.get(cv2.CAP_PROP_FPS)

    print("[INFO] Press 'q' to stop stream.")

    while (True):
        # Process each frame
        ok, frame = cap.read()

        if not ok:
            break

        # Resize frame
        if frame.shape[0] > FRAME_MAX_SIDE:
            frac = FRAME_MAX_SIDE / frame.shape[0]
            frame = cv2.resize(frame, (-1, -1), fx=frac, fy=frac)
        elif frame.shape[1] > FRAME_MAX_SIDE:
            frac = FRAME_MAX_SIDE / frame.shape[1]
            frame = cv2.resize(frame, (-1, -1), fx=frac, fy=frac)

        # Convert frame to gray scale image
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect 
        rects = detector(gray, 0)

        for rect in rects:
            # Get facial landmarks for the face region
            shape = predictor(gray, rect)

            # Convert the facial landmark (x, y)-coordinates to a NumPy array
            shape = face_utils.shape_to_np(shape)

            # Extract the left and right eye coordinates
            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]

            # Compure EAR for each eye
            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)

            # Average the eye aspect ratio together for both eyes
            ear = (leftEAR + rightEAR) / 2.0

            ear_window = np.append(ear_window[1:], ear)

            # Draw contours around eyes
            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)
            cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

            # Classify frame as blink or not
            if ear < EAR_THRESH:
                if not blink_detected and is_blink(ear_window):
                    blink_detected = True
            else:
                blink_detected = False

            # CASE 1:  the blink is detected
            if blink_detected:

                # Calculate current blink duration
                cuttent_duration = BLINK_FR_COUNTER / fps
                # Display duration
                display_duration(frame, cuttent_duration)

                BLINK_FR_COUNTER += 1

                # Case when the blink is longer then 2 seconds
                if cuttent_duration > 2:
                    # Display allert message
                    display_allert(frame)
                    if args['plot_ear']:
                        # Add red point to EAR plot
                        EAR_plot = add_new_point(EAR_plot, ear, COLORS['alert'])

                elif args['plot_ear']:
                    # Add orange point to EAR plot
                    EAR_plot = add_new_point(EAR_plot, ear, COLORS['blink'])


            # CASE 2:  the blink is not detected
            else:
                # Reset time of blink start
                if blink_detected:
                    blink_detected = False

                # Define the long enought blink to be true
                if BLINK_FR_COUNTER >= EAR_CONSEC_FRAMES:
                    BLINK_COUNTER += 1

                # Reset Blink Frame counter
                BLINK_FR_COUNTER = 0

                open_ear_window = np.append(open_ear_window[1:], ear)
                # Recalculate the treshold for open eyes
                EAR_THRESH = np.average(open_ear_window) - 0.05

                if args['plot_ear']:
                    # Add green point to EAR plot
                    EAR_plot = add_new_point(EAR_plot, ear, COLORS['ok'])

            # Display number of blinks and EAR calue
            display_blink_counter(frame, BLINK_COUNTER)
            display_ear(frame, ear)

        # Add  EAR plot to frame
        if args['plot_ear']:

            # Make frame and plot be equal in width
            if frame.shape[1] > EAR_plot.shape[1]:
                diff = frame.shape[1] - EAR_PLOT_PARAMS['width']
                EAR_plot = cv2.copyMakeBorder(EAR_plot, 0, 0, diff // 2, diff - diff // 2, cv2.BORDER_CONSTANT)

            elif frame.shape[1] < EAR_plot.shape[1]:
                diff = EAR_PLOT_PARAMS['width'] - frame.shape[1]
                frame = cv2.copyMakeBorder(frame, 0, 0, diff // 2, diff - diff // 2, cv2.BORDER_CONSTANT)

            # Concatinate frame and EAR plot verticaly   
            EAR_plot_with_tresh = display_bound(EAR_plot, EAR_THRESH)
            frame = np.concatenate((frame, EAR_plot_with_tresh), axis=0)

        # Display the resulting frame
        cv2.imshow('frame', frame)

        # Stop stream on key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture and close windows
    print("[INFO] Stopping stream...")
    cap.release()
    cv2.destroyAllWindows()
