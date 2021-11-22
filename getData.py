import numpy as np
import sys
import os
import argparse
import time
import imutils
import cv2

num_trainning = 150

def main(id, name, age, gender):
    sampleNum = 0
    count_captures =0
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not os.path.exists('face_data/'+ str(name) ):
            os.makedirs('face_data/'+str(name))

        sampleNum +=1
        face_dir = 'face_data'
        path = os.path.join(face_dir, name)
        if not os.path.isdir(path):
            os.mkdir(path)
        if sampleNum % 3 == 0:
            img_no = sorted([int(fn[:fn.find('.')]) for fn in os.listdir(path) if fn[0] != '.'] + [0])[-1] + 1
            cv2.imwrite('%s/%s.png' % (path, img_no), frame)
            count_captures +=1
            print("Captured image: ", count_captures)
            cv2.imshow('frame', frame)
            cv2.waitKey(1)

        if sampleNum == num_trainning:
            f = open("data.txt", "a")
            f.write(str(id) + ' ' + str(name) + ' ' + str(age) + ' ' + str(gender) + '\n')
            sampleNum += 1

        if sampleNum > num_trainning :
            print("Captured successfully")
            break
    cap.release()
    cv2.destroyAllWindows()

def fId():
    print('Enter person profile ID: ')
    Id = input()


def fname():
    print('Enter person profile name (No spaces): ')
    name = input()
    if name.isalpha():
        return name
    else:
        print('Invalid Input')
        del name
        fname()

def fage():
    print('Enter person profile age: ')
    age = input()
    if age.isdigit():
        if int(age) >100:
            print("Invalid input")
            del age
            fage()
        else:
            return age
    else:
        print("Invalid input")
        fage()

def fgender():
    print('Enter person profile gender: ')
    print('1.Male')
    print('2.Female')
    number = input()
    if number == "1":
        gender = "Male"
    elif number == "2":
        gender = "Female"
    return gender

if __name__ == '__main__':
    id = fId()
    name = fname()
    age = fage()
    gender = fgender()
    main(id, name, age, gender)

