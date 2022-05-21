import os
import cv2
import time
import numpy as np
import math as m
import pandas as pd
import mediapipe as mp

def generate(also_distance):
    dim = (640, 480)

    ctime = 0
    ptime = 0
    Slopes = []
    pt0 = [0, 0, 0]

    width = int(200 * 3)
    height = int(2 * 3)

    mpHands = mp.solutions.hands
    hands = mpHands.Hands()
    mpDrawings = mp.solutions.drawing_utils

    dim = (width, height)

    for root, dirs, files in os.walk(r"ASL Dataset/Train/asl_alphabet_train"):
        for dir in dirs:
            dict = {}
            folder = ""
            folder = folder + root + "\\" + dir
            for r, d, f in os.walk(folder):
                for image in f:
                    img = cv2.imread(r + "\\" + image)
                    # resized = cv2.resize(image, dim)

                    im_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    results = hands.process(im_rgb)

                    if results.multi_hand_landmarks:
                        for handLms in results.multi_hand_landmarks:
                            for id, ld in enumerate(handLms.landmark):
                                h, w, c = img.shape
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
                                         dict[str(id) + "to" + str(ID) + "D"] = \
                                             np.append(dict.get(str(id) + "to" + str(ID) + "D"), [distance])
                                    dict[str(id) + "to" + str(ID) + "A"] = np.append(
                                        dict.get(str(id) + "to" + str(ID) + "A"), [angle])

                                    if id == 5 and ID == 8:
                                        Slopes.append(slope)
                                    if id == 9 and ID == 12:
                                        Slopes.append(slope)
                                    if id == 13 and ID == 16:
                                        Slopes.append(slope)
                                    if id == 17 and ID == 20:
                                        Slopes.append(slope)
                                    if id == 0 and ID == 4:
                                        Slopes.append(slope)

                            counteri = 0
                            for i in Slopes:
                                counterj = 0
                                for j in Slopes:
                                    if i == j:
                                        counterj += 1
                                        continue
                                    A = (i - j) / (1 + i * j)
                                    A = m.fabs(A)
                                    A = m.atan(A)
                                    dict["finger" + str(counteri) + "to finger" + str(counterj) + "Angle"] =\
                                        np.append(dict.get("finger" + str(counteri) + "to finger" + str(counterj) + "Angle"), [A])
                                    counterj += 1
                                counteri += 1
                            print(len(Slopes))
                            Slopes.clear()
                        print(image)

                        mpDrawings.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)
                    cv2.imshow(dir, img)
                    cv2.waitKey(10)
            cv2.destroyAllWindows()
            Frame = pd.DataFrame(dict)
            Frame = Frame.drop([0])
            Frame["Class"] = dir
            path_folder = "./ASL Dataset/CSV_landmarks_AL_2/"
            if also_distance:
                path_folder = ""
                path_folder = "./ASL Dataset/landmarks_with_extended/"
            Frame.to_csv(path_folder + str(dir) + ".csv", index=False)
            dict.clear()






