import cv2
import numpy as np
import dlib
from math import hypot
##import pyglet
import time
import random
import os

import telepot
bot=telepot.Bot("8228440083:AAGd4_rNLiQkLAFEdY5hGhSpklw9Q-mFPuc")
ch_id="6232882173"



cap = cv2.VideoCapture(0)
cv2.ocl.setUseOpenCL(False)
board = np.zeros((300, 700), np.uint8)
board[:] = 255

# detector = dlib.get_frontal_face_detector()
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Keyboard settings
##keyboard = np.zeros((600, 1000, 3), np.uint8)
keyboard = np.zeros((1000, 800, 3), np.uint8)

import subprocess
import platform

def text_to_speech(text):
    """Convert text to speech using system commands"""
    try:
        if platform.system() == "Windows":
            # Windows - using SAPI.SpVoice
            import comtypes.client
            speaker = comtypes.client.CreateObject("SAPI.SpVoice")
            speaker.Speak(text)
        elif platform.system() == "Darwin":  # macOS
            subprocess.call(['say', text])
        else:  # Linux
            subprocess.call(['espeak', text])
    except Exception as e:
        print(f"Text-to-speech error: {e}")


keys_set_1 = {
    0: "IMG\\fan.png",
    1: "IMG\\idea.png",
    2: "IMG\\restroom.png",
    3: "IMG\\medicine.png",
    4: "IMG\\nurse.png",
    5: "IMG\\restaurant.png",
    6: "IMG\\shower.png",
    7: "IMG\\siren.png",
    8: "IMG\\sleep.png",
    9: "IMG\\waterbottle.png",
    10: "IMG\\exit.png",
    11: "IMG\\12.jpg",
    12: "IMG\\13.jpg"
}




def draw_letters(letter_index, text, letter_light):
    x, y = get_coordinates(letter_index)
    width, height = 200, 200
    th = 3

    image_path = keys_set_1[letter_index]
    image = cv2.imread(image_path)
    
##    print(image)
    
    image = cv2.resize(image, (width - 2 * th, height - 2 * th))

    if letter_light is True:
        keyboard[y + th:y + height - th, x + th:x + width - th] = image
    else:
        keyboard[y + th:y + height - th, x + th:x + width - th] = (51, 51, 51)
    cv2.rectangle(keyboard, (x + th, y + th), (x + width - th, y + height - th), (255, 255, 255), th)

def get_coordinates(letter_index):
    coordinates = {
        0: (0, 0),
        1: (200, 0),
        2: (400, 0),
        3: (600, 0),
        4: (0, 200),
        5: (200, 200),
        6: (400, 200),
        7: (600, 200),
        8: (0, 400),
        9: (200, 400),
        10: (400, 400),
        11: (600, 400),
        12: (0, 600),
    }
    return coordinates[letter_index]
    
def draw_menu():
    rows, cols, _ = keyboard.shape
    th_lines = 4 # thickness lines
##    cv2.line(keyboard, (int(cols/2) - int(th_lines/2), 0),(int(cols/2) - int(th_lines/2), rows),
##             (51, 51, 51), th_lines)
    ## cv2.putText(keyboard, "PASSword", (80, 300), font, 6, (255, 255, 255), 5)
##    cv2.putText(keyboard, "RIGHT", (80 + int(cols/2), 300), font, 6, (255, 255, 255), 5)

def midpoint(p1 ,p2):
    return int((p1.x + p2.x)/2), int((p1.y + p2.y)/2)

font = cv2.FONT_HERSHEY_PLAIN

def get_blinking_ratio(eye_points, facial_landmarks):
    left_point = (facial_landmarks.part(eye_points[0]).x, facial_landmarks.part(eye_points[0]).y)
    right_point = (facial_landmarks.part(eye_points[3]).x, facial_landmarks.part(eye_points[3]).y)
    center_top = midpoint(facial_landmarks.part(eye_points[1]), facial_landmarks.part(eye_points[2]))
    center_bottom = midpoint(facial_landmarks.part(eye_points[5]), facial_landmarks.part(eye_points[4]))

##    hor_line = cv2.line(frame, left_point, right_point, (0, 255, 0), 2)
##    ver_line = cv2.line(frame, center_top, center_bottom, (0, 255, 0), 2)

    hor_line_lenght = hypot((left_point[0] - right_point[0]), (left_point[1] - right_point[1]))
    ver_line_lenght = hypot((center_top[0] - center_bottom[0]), (center_top[1] - center_bottom[1]))

    ratio = hor_line_lenght / ver_line_lenght
    return ratio

def eyes_contour_points(facial_landmarks):
    left_eye = []
    right_eye = []
    for n in range(36, 42):
        x = facial_landmarks.part(n).x
        y = facial_landmarks.part(n).y
        left_eye.append([x, y])
    for n in range(42, 48):
        x = facial_landmarks.part(n).x
        y = facial_landmarks.part(n).y
        right_eye.append([x, y])
    left_eye = np.array(left_eye, np.int32)
    right_eye = np.array(right_eye, np.int32)
    return left_eye, right_eye

def get_gaze_ratio(eye_points, facial_landmarks):
    left_eye_region = np.array([(facial_landmarks.part(eye_points[0]).x, facial_landmarks.part(eye_points[0]).y),
                                (facial_landmarks.part(eye_points[1]).x, facial_landmarks.part(eye_points[1]).y),
                                (facial_landmarks.part(eye_points[2]).x, facial_landmarks.part(eye_points[2]).y),
                                (facial_landmarks.part(eye_points[3]).x, facial_landmarks.part(eye_points[3]).y),
                                (facial_landmarks.part(eye_points[4]).x, facial_landmarks.part(eye_points[4]).y),
                                (facial_landmarks.part(eye_points[5]).x, facial_landmarks.part(eye_points[5]).y)], np.int32)
    # cv2.polylines(frame, [left_eye_region], True, (0, 0, 255), 2)

    height, width, _ = frame.shape
    mask = np.zeros((height, width), np.uint8)
    cv2.polylines(mask, [left_eye_region], True, 255, 2)
    cv2.fillPoly(mask, [left_eye_region], 255)
    eye = cv2.bitwise_and(gray, gray, mask=mask)

    min_x = np.min(left_eye_region[:, 0])
    max_x = np.max(left_eye_region[:, 0])
    min_y = np.min(left_eye_region[:, 1])
    max_y = np.max(left_eye_region[:, 1])

    gray_eye = eye[min_y: max_y, min_x: max_x]
    _, threshold_eye = cv2.threshold(gray_eye, 70, 255, cv2.THRESH_BINARY)
    height, width = threshold_eye.shape
    left_side_threshold = threshold_eye[0: height, 0: int(width / 2)]
    left_side_white = cv2.countNonZero(left_side_threshold)

    right_side_threshold = threshold_eye[0: height, int(width / 2): width]
    right_side_white = cv2.countNonZero(right_side_threshold)

    if left_side_white == 0:
        gaze_ratio = 1
    elif right_side_white == 0:
        gaze_ratio = 5
    else:
        gaze_ratio = left_side_white / right_side_white
    return gaze_ratio

# Counters
frames = 0
letter_index = 0
blinking_frames = 0
frames_to_blink = 6
frames_active_letter = 9

# Text and keyboard settings
text = ""
text1=[]
keyboard_selected = "left"
last_keyboard_selected = "left"
select_keyboard_menu = True
keyboard_selection_frames = 0
count=0
pf =['1','2','3','4']

scanned=0

once=1
scanned =1


while True:




     
  
    #time.sleep(1);  
##    print('Scanned the ID Identified in the database ')

    
    
    _, frame = cap.read()
    #frame = cv2.resize(frame, None, fx=0.8, fy=0.8)
    rows, cols, _ = frame.shape
##    keyboard[:] = (26, 26, 26)
    keyboard[:] = (26, 26, 26)
    frames += 1
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # convert to grayscale (ensure proper format)
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = gray.astype('uint8')
    gray = np.ascontiguousarray(gray)
    # gray = gray.astype('uint8')
    # gray = gray.copy()  # Ensures it's a normal contiguous array

    

    # Draw a white space for loading bar
    frame[rows - 50: rows, 0: cols] = (255, 255, 255)

    if select_keyboard_menu is True:
        draw_menu()

    # Keyboard selected
    if keyboard_selected == "left":
        keys_set = keys_set_1
##    else:
##        keys_set = keys_set_2
    active_letter = keys_set[letter_index]
    
    # Face detection
    # faces = detector(gray)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))
    # Face detection using OpenCV Haar Cascade
    # faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))

    for (x, y, w, h) in faces:
        # Convert to dlib rectangle for landmark predictor
        rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))
        landmarks = predictor(gray, rect)

        left_eye, right_eye = eyes_contour_points(landmarks)

        # Detect blinking
        left_eye_ratio = get_blinking_ratio([36, 37, 38, 39, 40, 41], landmarks)
        right_eye_ratio = get_blinking_ratio([42, 43, 44, 45, 46, 47], landmarks)
        blinking_ratio = (left_eye_ratio + right_eye_ratio) / 2

        # Draw eye contours
        cv2.polylines(frame, [left_eye], True, (0, 0, 255), 2)
        cv2.polylines(frame, [right_eye], True, (0, 0, 255), 2)




        if select_keyboard_menu is True:
            # Detecting gaze to select Left or Right keybaord
            gaze_ratio_left_eye = get_gaze_ratio([36, 37, 38, 39, 40, 41], landmarks)
            gaze_ratio_right_eye = get_gaze_ratio([42, 43, 44, 45, 46, 47], landmarks)
            gaze_ratio = (gaze_ratio_right_eye + gaze_ratio_left_eye) / 2
##            print(gaze_ratio)

            if gaze_ratio <= 5:
                keyboard_selected = "right"
                keyboard_selection_frames += 1
                # If Kept gaze on one side more than 15 frames, move to keyboard
                if keyboard_selection_frames == 15:
                    select_keyboard_menu = False
##                    right_sound.play()
                    # Set frames count to 0 when keyboard selected
                    frames = 0
                    keyboard_selection_frames = 0
                if keyboard_selected != last_keyboard_selected:
                    last_keyboard_selected = keyboard_selected
                    keyboard_selection_frames = 0
##            else:
##                keyboard_selected = "left"
##                keyboard_selection_frames += 1
##                # If Kept gaze on one side more than 15 frames, move to keyboard
##                if keyboard_selection_frames == 15:
##                    select_keyboard_menu = False
####                    left_sound.play()
##                    # Set frames count to 0 when keyboard selected
##                    frames = 0
##                if keyboard_selected != last_keyboard_selected:
##                    last_keyboard_selected = keyboard_selected
##                    keyboard_selection_frames = 0

        else:
            # Detect the blinking to select the key that is lighting up
            if blinking_ratio > 5:
                # cv2.putText(frame, "BLINKING", (50, 150), font, 4, (255, 0, 0), thickness=3)
                blinking_frames += 1
                frames -= 1

                # Show green eyes when closed
                cv2.polylines(frame, [left_eye], True, (0, 255, 0), 2)
                cv2.polylines(frame, [right_eye], True, (0, 255, 0), 2)

                # Typing letter
                if blinking_frames == frames_to_blink:
                    if active_letter != "E" and active_letter != "C":
                        count=count+1
                        text += active_letter
                        print(type(letter_index))


                        if letter_index==0:
                            print("FAN.PNG")
                            text_to_speech("NEED FAN")
                        elif letter_index==1:
                            print("IDEA.PNG")
                            text_to_speech("I GOT A IDEA")
                        elif letter_index==2:
                            print("restroom.PNG")
                            text_to_speech("Want to Use Restroom")
                        elif letter_index==3:
                            print("medicine.PNG")
                            text_to_speech("WANT MEDICINE")
                        elif letter_index==4:
                            print("NURSE.PNG")
                            text_to_speech("CALLING NURSE")
                            bot.sendMessage(ch_id,f"Patient is Calling")
                        elif letter_index==5:
                            print("restarunt.PNG")
                            text_to_speech("WANT FOOD")
                        elif letter_index==6:
                            print("BATH.PNG")
                            text_to_speech("WANT SHOWER")
                        elif letter_index==7:
                            print("SIREN.PNG")
                            text_to_speech("EMERGENCY")
                            bot.sendMessage(ch_id,f"Emergency From Patient")
                        elif letter_index==8:
                            print("SLEEP.PNG")
                            text_to_speech("FEELING SLEEPY")
                        elif letter_index==9:
                            print("WATERBOTTLE.PNG")
                            text_to_speech("Need Water")
                        elif letter_index==10:
                            print("EXIT.PNG")
                            # Exit the program when blinking on option 10
                            print("Exiting program...")
                            cap.release()
                            cv2.destroyAllWindows()
                            exit(0)
    
##                    sound.play()
                    select_keyboard_menu = True
                    # time.sleep(1)

            else:
                blinking_frames = 0


    # Display letters on the keyboard
    if select_keyboard_menu is False:
        if frames == frames_active_letter:
            time.sleep(1.5)
            letter_index += 1
            frames = 0
        if letter_index == 11:
            letter_index = 0
        for i in range(11):
            if i == letter_index:
                light = True
            else:
                light = False
            draw_letters(i, keys_set[i], light)

    # Show the text we're writing on the board
    cv2.putText(board, str(text1), (80, 100), font, 9, 0, 3)

    # Blinking loading bar
    percentage_blinking = blinking_frames / frames_to_blink
    loading_x = int(cols * percentage_blinking)
    cv2.rectangle(frame, (0, rows - 50), (loading_x, rows), (51, 51, 51), -1)


    cv2.imshow("Frame", frame)
    cv2.imshow("Virtual keyboard", keyboard)
##    cv2.imshow("Board", board)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

##    key = cv2.waitKey(1)
##    if key == 27:
##        break

cap.release()
cv2.destroyAllWindows()
