import os
import cv2
import numpy as np
import math as m
import pandas as pd
import mediapipe as mp

def generate(also_distance):
    dim = (640, 480)
    print("here")
    ctime = 0
    ptime = 0
    pt0 = [0, 0, 0]
    width = int(200 * 3)
    height = int(2 * 3)
    mpHands = mp.solutions.hands
    hands = mpHands.Hands()
    mpDrawings = mp.solutions.drawing_utils
    lines = {}
    dim = (width, height)
    gestureName = {'A': {}, 'B': {}, 'C': {}, "D": {}, "E": {}, "F": {}, "G": {}, "H": {}, "I": {}, "J": {}, "K": {},
                   "L": {}, "M": {}, "N": {}, "O": {}, "P": {}, "Q": {}, "R": {}, "S": {}, "T": {},
                   "U": {}, "V": {}, "W": {}, "X": {}, "Y": {}, "Z": {}}

    slope_lines = {}

    index = 4
    Slopes = []
    i = 0
    path = (r"D:\C drive old data\Final year project\pythonProject\ISL Dataset\Frames\*.jpg")
#%%

    counter = 0
    for root, dirs, files in os.walk(r"D:\C drive old data\Final year project\pythonProject\ISL Dataset\Frames"):
        dict = {}
        for f in files:
            a = 2
            if len(f) == 18:
                s = f[12]
                if len(f) == 19:
                    continue
                a = int(s)
            if f.endswith('.jpg') and a < 6 and len(f) < 19:

                currentGesture = f[8]
                image = cv2.imread(root + '\\' + f)
                im_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = hands.process(im_rgb)

                if results.multi_hand_landmarks:

                    for handLms in results.multi_hand_landmarks:
                        for id, ld in enumerate(handLms.landmark):
                            h, w, c = image.shape
                            cx, cy, cz = ld.x * w, ld.y * h, ld.z * c
                            for ID, LD in enumerate(handLms.landmark):
                                scx, scy, scz = LD.x * w, LD.y * h, LD.z * c
                                if ID == id:
                                    continue
                                ptx = cx - scx
                                pty = cy - scy
                                ptz = cz - scz

                                ptx = ptx * ptx
                                pty = pty * pty
                                ptz = ptz * ptz
                                if cx == scx:
                                    cx += cx * 1.2
                                slope = (cy - scy) / (cx - scx)
                                angle = m.atan(slope)
                                distance = m.sqrt(ptx + pty + ptz)
                                if also_distance:
                                #gestureName[currentGesture][str(id) + "to" + str(ID) + "S"] = np.append(gestureName[currentGesture].get(str(id) + "to" + str(ID) + "S"), [slope])
                                    gestureName[currentGesture][str(id) + "to" + str(ID) + "D"] = \
                                        np.append(gestureName[currentGesture].get(str(id) + "to" + str(ID) + "D"), [distance])
                                gestureName[currentGesture][str(id) + "to" + str(ID) + "A"] =\
                                    np.append(gestureName[currentGesture].get(str(id) + "to" + str(ID) + "A"), [angle])
                                if id == 0 and ID == 4:
                                    Slopes.append(slope)
                                if id == 5 and ID == 8:
                                    Slopes.append(slope)
                                if id == 9 and ID == 12:
                                    Slopes.append(slope)
                                if id == 13 and ID == 16:
                                    Slopes.append(slope)
                                if id == 17 and ID == 20:
                                    Slopes.append(slope)

                        counteri = 0
                        for i in Slopes:
                            counterj = 0
                            for j in Slopes:
                                if i ==j:
                                    counterj += 1
                                    continue
                                A = (i - j)/(1+i*j)
                                A = m.fabs(A)
                                A = m.atan(A)
                                gestureName[currentGesture]["finger" + str(counteri) + "to finger" + str(counterj) + "Angle"] = \
                                    np.append(gestureName[currentGesture].get("finger" + str(counteri) + "to finger" + str(counterj) + "Angle"), [A])
                                counterj += 1
                            counteri += 1
                        Slopes.clear()
                        gestureName[currentGesture]["Class"] = np.append(gestureName[currentGesture].get("Class"),[currentGesture])
                        print(f)

                cv2.waitKey(1)
            else:
                continue

            cv2.destroyAllWindows()

    path_Folder = "./ISL Dataset/landmarks_with_lines_2/"
    if also_distance:
        path_Folder = ""
        path_Folder = "./ISL Dataset/landmarks_with_DAL_2/"

    for key in gestureName:
        v = gestureName[key]
        Frame = pd.DataFrame(v)
        path = path_Folder + key + ".csv"
        Frame.to_csv(path, index=False)
        v.clear()















