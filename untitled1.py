import cv2
import os
import time
import uuid

img_path = "Tensorflow/workspace/images/hindi_alphabets"

labels = ['अ', 'आ', 'इ', 'ई', 'उ', 'ऊ', 'ए', 'ऐ', 'ओ', 'औ', 'अं', 'अः', 'क', 'ख', 'ग']
number_imgs = 15

for label in labels:
    dir_path = os.path.join(img_path, label)
    os.makedirs(dir_path, exist_ok=True)

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print(f"Error: Camera not accessible for label {label}.")
        continue

    print(f"Collecting images for '{label}' in 5 seconds. Press 'q' to quit early.")
    time.sleep(5)

    for imgnum in range(number_imgs):
        ret, frame = cap.read()

        if not ret:
            print(f"Error: Could not read frame for label {label}, image {imgnum}.")
            break

        imgname = os.path.join(dir_path, f"{label}.{str(uuid.uuid1())}.jpg")
        cv2.imwrite(imgname, frame)
        cv2.imshow('frame', frame)

        print(f"Captured image {imgnum+1}/{number_imgs} for {label}")
        time.sleep(2)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Process interrupted by user.")
            break

    cap.release()
    cv2.destroyAllWindows()

print("Data collection complete!")