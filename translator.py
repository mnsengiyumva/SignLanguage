import pickle
import cv2
import mediapipe as mp
import numpy as np
import pyttsx3
import threading

# 1. Initialize Voice Engine
engine = pyttsx3.init()


def speak(text):
    # We run this in a thread so the video doesn't freeze while talking
    threading.Thread(target=lambda: (engine.say(text), engine.runAndWait()), daemon=True).start()


# 2. Load Model & MediaPipe (Same as Day 4)
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.7)

cap = cv2.VideoCapture(0)

last_spoken = ""
prediction_history = []  # To filter out "flickering" predictions

while True:
    data_aux = []
    success, frame = cap.read()
    if not success: break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        for lm in hand_landmarks.landmark:
            data_aux.extend([lm.x, lm.y, lm.z])

        prediction = model.predict([np.asarray(data_aux)])
        predicted_character = prediction[0]

        # --- LOGIC: Only speak if the sign is stable and new ---
        prediction_history.append(predicted_character)
        if len(prediction_history) > 10:  # Look at last 10 frames
            prediction_history.pop(0)

            # If the most frequent prediction in the last 10 frames is the current one
            most_frequent = max(set(prediction_history), key=prediction_history.count)

            if most_frequent != last_spoken and prediction_history.count(most_frequent) > 8:
                speak(most_frequent.replace("_", " "))  # Say it!
                last_spoken = most_frequent

        cv2.putText(frame, predicted_character.replace("_", " "), (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

    cv2.imshow('Sign Language Voice Translator', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()