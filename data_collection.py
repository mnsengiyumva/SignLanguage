import cv2
import mediapipe as mp
import csv
import os

# 1. Setup MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)

# 2. Prepare CSV File
header = ['label']
for i in range(21):
    header += [f'x{i}', f'y{i}', f'z{i}']

filename = 'hand_data.csv'

# Create file and write header if it doesn't exist
if not os.path.exists(filename):
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)

cap = cv2.VideoCapture(0)

print("Instructions:")
print("Press 'h' to record HELLO")
print("Press 't' to record THANK YOU")
print("Press 'l' to record I LOVE YOU")
print("Press 'q' to quit")

while cap.isOpened():
    success, img = cap.read()
    if not success: break

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    label = None
    key = cv2.waitKey(1)
    if key == ord('h'): label = 'Hello'
    if key == ord('t'): label = 'Thank_You'
    if key == ord('l'): label = 'I_Love_You'
    if key == ord('q'): break

    if results.multi_hand_landmarks and label:
        for handLms in results.multi_hand_landmarks:
            # Flatten the landmarks into a list
            data_row = [label]
            for lm in handLms.landmark:
                data_row.extend([lm.x, lm.y, lm.z])

            # Save to CSV
            with open(filename, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(data_row)
            print(f"Recorded {label}!")

    cv2.imshow("Data Collection", img)

cap.release()
cv2.destroyAllWindows()