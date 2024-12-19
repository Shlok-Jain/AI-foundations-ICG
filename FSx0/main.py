import cv2
import os
import queue
import keyboard
import threading
from prediction import givepred
from cnnmodel import loadCNN

frame_queue = queue.Queue()

model = loadCNN()
print("model loaded!")

end = False
video_path = 0  # Replace with your video path
sampling_interval = 1
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

    while True:
        global end
        ret, frame = cap.read()
        
        height, width, _ = frame.shape
        crop_width = width // 2 
        crop_height = height // 2  
        top_right_frame = frame[0:crop_height, 0:crop_width]

        start_point = (0, 0) 
        end_point = (crop_width, crop_height)      
        color = (0, 255, 0)              
        thickness = 2                         
        cv2.rectangle(frame, start_point, end_point, color, thickness)
        cv2.imshow('Video Feed with Highlighted Region', frame)

        if frame_count % frame_interval == 0:
            photo_path = os.path.join("FSx0/output_folder", f"photo_{photo_count:03d}.jpg")
            cv2.imwrite(photo_path, top_right_frame)
            frame_queue.put(photo_path)
            photo_count+=1

        frame_count += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            end = True
            break

    cap.release()
    print("Processing complete.")

def process_photo():
    global end
    while True:
        if not frame_queue.empty():
            photo_path = frame_queue.get()

            label = givepred(photo_path,model)
            print("output:", label)
            os.remove(photo_path)
            # result=result+f"{label},"
        if end:
            break

def llm_gen():
    global end
    while True:
        if result:
            pass
        if end:
            break

capture_thread = threading.Thread(target=capture_photo, daemon = True)
process_thread = threading.Thread(target=process_photo, daemon = True)
llm_thread = threading.Thread(target=llm_gen, daemon = True)

capture_thread.start()
process_thread.start()
llm_thread.start()

capture_thread.join()
process_thread.join()
llm_thread.join()