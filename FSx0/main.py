import cv2
import os
import queue
import keyboard
import threading
from prediction import givepred
from cnnmodel import loadCNN

frame_queue = queue.Queue()

print("hi")
model = loadCNN()

print("model laode")
video_path = 0  # Replace with your video path
sampling_interval = 3  
result = ""

def capture_photo():
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Cannot open video file or webcam.")
        exit()

    # Get the frame rate of the video
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_interval = int(fps * sampling_interval)

    frame_count = 0
    photo_count = 0
    ret = 1

    while True:
        ret, frame = cap.read()
        
        # if not ret:
        #     print("End of video or error reading frame.")
        #     break

        # if keyboard.is_pressed("q"):
        #     break
  
        if frame_count % frame_interval == 0:
            photo_path = os.path.join("FSx0/output_folder", f"photo_{photo_count:03d}.jpg")
            cv2.imwrite(photo_path, frame)
            frame_queue.put(photo_path)
            photo_count+=1

        frame_count += 1

    cap.release()
    print("Processing complete.")

def process_photo():
    while True:
        if not frame_queue.empty():
            photo_path = frame_queue.get()

            label = givepred(photo_path,model)
            print("output:", label)
            os.remove(photo_path)
            # result=result+f"{label},"

def llm_gen():
    while True:
        if result:
            pass

capture_thread = threading.Thread(target=capture_photo, daemon = True)
process_thread = threading.Thread(target=process_photo, daemon = True)
llm_thread = threading.Thread(target=llm_gen, daemon = True)

capture_thread.start()
process_thread.start()
llm_thread.start()

capture_thread.join()
process_thread.join()
llm_thread.join()