import cv2
import math as m
import mediapipe as mp


def recognize_gesture(loaded_model, lastimag=None):
    print("starting")
    video = cv2.VideoCapture(0)
    Slopes = []

    mpHands = mp.solutions.hands
    hands = mpHands.Hands()
    mpDrawings = mp.solutions.drawing_utils
    lastimag = cv2.cv2
    while True:
        Success,img = video.read()

        #new_image = cv2.imread("white.jpg")
        im_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(im_rgb)

        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks:
                list = []
                for id , ld in enumerate(handLms.landmark):
                    h, w, c = img.shape
                    cx, cy, cz = ld.x * w, ld.y * h, ld.z * c

                    for ID, LD in enumerate(handLms.landmark):
                        scx, scy, scz = LD.x * w, LD.y * h, LD.z * c
                        if ID == id:
                            continue
                        if cx == scx:
                            cx += cx * 1.2

                        slope = (cy - scy) / (cx - scx)
                        angle = m.atan(slope)
                        list.append(angle)

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
                    x = int(cx-70)
                    y = int(cy+35)
                    ###cv2.putText(image, str(id), (x, y), 1, 1, (0, 0, 0), 1)
                for i in Slopes:
                    for j in Slopes:
                        if i == j:
                            continue
                        A = (i - j) / (1 + i * j)
                        A = m.fabs(A)
                        A = m.atan(A)
                        list.append(A)
                Slopes.clear()
                #mpDrawings.draw_landmarks(new_image , handLms , mpHands.HAND_CONNECTIONS)

            Recognize_label = loaded_model.predict([list])


            #output = gTTS(Recognize_label[0])
            #output.save(Recognize_label[0] + '.mp3')
            #playsound("./" + Recognize_label[0] + '.mp3')


            cv2.putText(img, str(Recognize_label[0]) , (10,70) , 2 , 3 ,(255,255,0) , 2)

            #print(Recognize_label[0])


            list.clear()


        cv2.imshow("video", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


    cv2.imwrite("gestureimage3.png",lastimag)

