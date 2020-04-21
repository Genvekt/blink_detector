# Eye blink counter
## Running program
To run program next parameters may be used:
```pithon
  -p --shape-predictor : Path to facial landmark predictor, default="models/shape_predictor_68_face_landmarks.dat", 
  -v --video           : Path to input video file or 'webcam' for webcam stream,  default="webcam", 
  -pl --plot-ear       : True is ear must be ploted in real-time, False otherwise, default=False               
```

Example of a command :
```sh
git clone https://github.com/Genvekt/blink_detector.git
cd blink_detector
pip install -r requirements.txt
python eye_blink_detection.py -pl=True -v="webcam"
```

**Important!** If you have no dlib installed, it is better to make sure that your system has everything for it to be installed.
Next [guide](https://www.pyimagesearch.com/2017/03/27/how-to-install-dlib/) will help you.

## Theory
The solution was buit using Python, OpenCV and Dlib with next articles as reference:
1. [Eye blink detection](https://www.pyimagesearch.com/2017/04/24/eye-blink-detection-opencv-python-dlib/)
2. [Facial landmark](https://www.pyimagesearch.com/2017/04/10/detect-eyes-nose-lips-jaw-dlib-opencv-python/)
3. [Work with webcam](https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_gui/py_video_display/py_video_display.html)

## Eye blink detection function
In order to detect the eye blink, the facial landmark was used. As a result, each eye is represented with 6 points: Left most L, right most R, two upper points Ul and Ur and two lower points, Dl and Dr

Then the equation called eye aspect ratio (EAR)may be used to detect eye blink: 

EAR = (||Ul-Dl|| + ||Ur-Dr||) / (2*||L-R||)

The blink is detected with two events:
- The value of EAR drows quick (defined by applying  differensing with 1 shift to last 13 values of EAR)
- The theshold of EAR value. It is updated each frame by nexy formula:

TRESHOLD = mean(last 30 EAR witch defined as not blink) - 0.05

## Ploting the EAR
The EAR is ploted in real-time to visualise its change. It is desided to use 
- **green** dots if there is no blink, 
- **orange** dots if there is a short blink
- **red** dots of blink is too long (more then 2 seconds)

Plot is represented by image that is apdated in real-time, wich may be seen on next gif:
