from pathlib import Path

import pandas as pd
from mediapipe.python.solutions import hands
import mediapipe as mp
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.model_selection import train_test_split

print("before main")
import Recognition as r
import pickle
import cv2
import os
import math as m
import time
import time
import numpy as np
import uuid

import speech_recognition as sr


def preprocessing():
    Slopes = []
    mpHands = mp.solutions.hands
    hands = mpHands.Hands()
    mpDrawings = mp.solutions.drawing_utils
    also_distance = False

    for root, dirs, files in os.walk(r"images"):
        for D in dirs:
            dict = {}

            for r,d,im in os.walk(root + "\\"+ D):

                for images in im:
                    frame = cv2.imread(r + "\\" + images)
                    # resized = cv2.resize(image, dim)

                    im_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = hands.process(im_rgb)

                    if results.multi_hand_landmarks:
                        for handLms in results.multi_hand_landmarks:
                            for id, ld in enumerate(handLms.landmark):
                                h, w, c = frame.shape
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
                                    dict["finger" + str(counteri) + "to finger" + str(counterj) + "Angle"] = \
                                        np.append(
                                            dict.get("finger" + str(counteri) + "to finger" + str(counterj) + "Angle"),
                                            [A])
                                    counterj += 1
                                counteri += 1
                            Slopes.clear()
                            print(len(dict))

                        mpDrawings.draw_landmarks(frame, handLms, mpHands.HAND_CONNECTIONS)
                    cv2.imshow("video", frame)
                    cv2.waitKey(10)
                cv2.destroyAllWindows()
                print(len(dict))
                Frame = pd.DataFrame(dict)
                Frame = Frame.drop([0])
                Frame["Class"] = D
                path_folder = "./images/"
                if also_distance:
                    path_folder = ""
                    path_folder = "./ASL Dataset/landmarks_with_extended/"
                Frame.to_csv(path_folder + str(D) + ".csv", index=False)
                dict.clear()

def train_model():
    data_combined = pd.DataFrame()
    CSV_data = Path("images/").rglob('*.csv')
    files = [x for x in CSV_data]
    for i in files:
        newData = pd.read_csv(i)
        data_combined = pd.concat([data_combined, newData])
    training_data = data_combined.drop("Class", axis=1)
    print(training_data.shape)

    classes = data_combined["Class"].unique()
    print(classes)

    Xtrain, Xtest, ytrain, ytest = train_test_split(training_data, data_combined["Class"], test_size=0.3)
    model = RandomForestClassifier(n_jobs=-1, verbose=2)
    model.fit(Xtrain, ytrain)


    predicted = model.predict(Xtest)
    print( "Accuracy Score : " + str(round(accuracy_score(ytest,predicted)*100 , 2)))
    print("Recall Score : " + str(recall_score(ytest, predicted , average='macro')))
    print("Precision : " + str(precision_score(ytest, predicted , average='macro')))
    print("f1 Score : " + str(f1_score(ytest, predicted , average='macro')))

    filename = 'pi.sav'
    pickle.dump(model, open("./images/" + filename, 'wb'))


def make_data():
    path = "colectedimages"
    labels = [ "non_gesture"]
    video = cv2.VideoCapture(0)
    import os

    parent_dir = "images\\"

    for i in labels:
        count = 500
        while count < 1200:
            Success, img = video.read()
            cv2.imwrite(parent_dir + "\\" + i + "\\" + str(count) + ".png", img)
            count = count + 1
            cv2.imshow('Video', img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        break


if __name__ == '__main__':
     make_data()
    # preprocessing()
    # train_model()
    #r.recognize_gesture(pickle.load(open("images\\pi.sav", 'rb'),))




