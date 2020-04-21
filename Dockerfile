FROM python:3

ADD eye_blink_detection.py /
ADD define.py /
ADD requirements.txt /
ADD models/shape_predictor_68_face_landmarks.dat /models/



RUN sudo apt-get install build-essential cmake
RUN sudo apt-get install libgtk-3-dev
RUN sudo apt-get install libboost-all-dev

RUN pip install -r requirements.txt

ENV video "webcam"
ENV predictor "models/shape_predictor_68_face_landmarks.dat"
ENV pl "False"

CMD [ "python", "./eye_blink_detection.py", "-v=${video}", "-pl=${pl}", "-p=${predictor}" ]
