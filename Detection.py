import sys
import time
import cv2
import numpy as np
import imutils
import datetime
import sqlite3
import argparse
from facenet.face_contrib import *
from predic_helper import *

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-n", "--num-frames", type=int, default=100,
	help="# of frames to loop over for FPS test")
ap.add_argument("-d", "--display", type=int, default=-1,
	help="Whether or not frames should be displayed")
args = vars(ap.parse_args())

#-------------------------------------------------------------
datetime_object = datetime.datetime.now()

def add_overlays(frame, faces, frame_rate, colors, confidence=0.85):
    #Nhan dien khuon mat
    if faces is not None:
        for idx, face in enumerate(faces):
            face_bb = face.bounding_box.astype(int)
            cv2.rectangle(frame, (face_bb[0], face_bb[1]), (face_bb[2], face_bb[3]), colors[idx], 2)
            if face.name and face.prob:
                if face.prob > confidence:
                    print(face.prob)
                    class_name = face.name
                    #-------------------------------
                    d_age=[]
                    d_gender = []
                    with open("data.txt", "r") as file:
                        for item in file:
                            data = item.split()
                            if data[1] == face.name:
                                d_age = data[2]
                                d_gender = data[3]

                    cv2.putText(frame, class_name, (face_bb[0], face_bb[3] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                    colors[idx], thickness=2, lineType=2)
                    cv2.putText(frame, d_gender + ' , ' + d_age, (face_bb[0] + 10, face_bb[3] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors[idx], thickness=2, lineType=2)
                    #print(class_name + ' ' + d_gender + ' ' + d_age + '  ' + str(datetime_object))
                else:
                    class_name = 'Unknow'
                    cv2.putText(frame, class_name, (face_bb[0], face_bb[3] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            colors[idx], thickness=2, lineType=2)
                    # Du doan tuoi va gioi tinh
                    faces_dt = face_cascade.detectMultiScale(frame, 1.3, 5)
                    for (x, y, w, h) in faces_dt:
                        detected_face = frame[int(y):int(y + h), int(x):int(x + w)]  # crop face
                        try:
                            # Them magin
                            margin = 30
                            margin_x = int((w * margin) / 100)
                            margin_y = int((h * margin) / 100)
                            detected_face = frame[int(y - margin_y):int(y + h + margin_y),
                                            int(x - margin_x):int(x + w + margin_x)]
                        except:
                            print("detected face has no margin")

                        try:
                            # Dua face vao mang predict
                            detected_face = cv2.resize(detected_face, (224, 224))

                            img_pixels = image.img_to_array(detected_face)
                            img_pixels = np.expand_dims(img_pixels, axis=0)
                            img_pixels /= 255

                            # Hien thi thong tin tuoi, gioi tinh
                            age_distributions = age_model.predict(img_pixels)
                            apparent_age = str(int(np.floor(np.sum(age_distributions * output_indexes, axis=1))[0]))

                            gender_distribution = gender_model.predict(img_pixels)[0]
                            gender_index = np.argmax(gender_distribution)
                            if gender_index == 0:
                                cv2.putText(frame, "Female, " + apparent_age, (face_bb[0] + 10, face_bb[3] - 10),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors[idx], thickness=2, lineType=2)
                                print(class_name + ' Female ' + apparent_age + '  ' + str(datetime_object))
                            else:
                                cv2.putText(frame, "Male, " + apparent_age, (face_bb[0] + 10, face_bb[3] - 10),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors[idx], thickness=2, lineType=2)
                                print(class_name + ' Male ' + apparent_age + '  ' + str(datetime_object))


                        except Exception as e:
                            print("exception", str(e))

        cv2.putText(frame, str(frame_rate) + " fps", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0),
                                thickness=2, lineType=2)



def run(model_checkpoint, classifier, video_file=None, output_file=None):
    frame_interval = 5
    fps_display_interval = 2  # seconds
    frame_rate = 0
    frame_count = 0
    if video_file is not None:
        video_capture = cv2.VideoCapture(video_file)
    else:
        video_capture = cv2.VideoCapture(0)
    ret, frame = video_capture.read()
    #frame = imutils.resize(frame, width = 600)
    width = frame.shape[1]
    height = frame.shape[0]
    if output_file is not None:
        video_format = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(output_file, video_format, 20, (width, height))
    face_recognition = Recognition(model_checkpoint, classifier)
    start_time = time.time()
    colors = np.random.uniform(0, 255, size=(1, 3))
    while True:
        # Capture frame-by-frame
        ret, frame = video_capture.read()
        if (frame_count % frame_interval) == 0:
            faces = face_recognition.identify(frame)
            for i in range(len(colors), len(faces)):
                colors = np.append(colors, np.random.uniform(150, 255, size=(1, 3)), axis=0)
            # Check fps
            end_time = time.time()
            if (end_time - start_time) > fps_display_interval:
                frame_rate = int(frame_count / (end_time - start_time))
                start_time = time.time()
                frame_count = 0

        add_overlays(frame, faces, frame_rate, colors)
        frame_count += 1

        cv2.imshow("Frame", frame)
        if output_file is not None:
            out.write(frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


    if output_file is not None:
        out.release()
    video_capture.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    run('models', 'models/your_model.pkl', video_file= "demo.mp4", output_file="demo.avi")
