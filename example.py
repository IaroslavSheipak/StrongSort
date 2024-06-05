from torch import hub
from torchreid.reid.utils import FeatureExtractor
from strongsort_py import StrongSort as cpp_strongsort
from strongsort import StrongSORT as original_strongsort
import torch
from pathlib import Path
import cv2
import time



state_dict = torch.load("/mnt/c/Users/Lenovo/Desktop/MIPT/DZ_2_SEM/AML/strong-sort-cpp/osnet_x1_0.pth.tar-50",map_location=torch.device('cpu'))
torch.save(state_dict, "/mnt/c/Users/Lenovo/Desktop/MIPT/DZ_2_SEM/AML/strong-sort-cpp/osnet_x1_0.pt")
yolo = hub.load('ultralytics/yolov5', 'yolov5m')
reid = FeatureExtractor('osnet_ain_x1_0', "/mnt/c/Users/Lenovo/Desktop/MIPT/DZ_2_SEM/AML/strong-sort-cpp/osnet_x1_0.pt",device='cpu')
os = original_strongsort(model_weights=Path("osnet_x1_0.pt"),device='cpu',fp16=False)
cs = cpp_strongsort()

# OpenCV video capture
video_path = "/mnt/c/Users/Lenovo/Desktop/MIPT/DZ_2_SEM/AML/strong-sort-cpp/гоооооооол.mp4"
cap = cv2.VideoCapture(video_path)

# Check if the video was opened successfully
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()
python_counter = 0
python_overall = 0
cpp_counter = 0
cpp_overall = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    pred = yolo(frame).pred[0].cpu()

    ltwhs = pred[:, :4]  # left-top width height
    confs = pred[:, 4]   # confidences
    classes = pred[:, 5]  # class ids

    h, w = frame.shape[:2]
    
    start = time.time()
    tracks = os.update(pred, frame)
    end = time.time()
    python_overall += end - start
    python_counter += 1
    if python_counter % 10 == 0:
        print("PYTHON", python_overall / python_counter)
    pred = pred.numpy()

    ltwhs = pred[:, :4]  # left-top width height
    confs = pred[:, 4]   # confidences
    classes = pred[:, 5]  # class ids

    rois = [frame[int(y1):int(y2), int(x1):int(x2)] for x1, y1, x2, y2 in ltwhs.astype(int)]
    start = time.time()
    features = reid(rois).numpy()
    tracks_ss = cs.update(ltwhs, confs, classes, features, (w, h))
    end = time.time()
    cpp_overall += end - start
    cpp_counter += 1
    if cpp_counter % 10 == 0:
        print("CPP", cpp_overall / cpp_counter)
    # # Process and display the tracks
    # for track in tracks:
    #     track_id = track.track_id
    #     l, t, w, h = track.ltwh
    #     cv2.rectangle(frame, (int(l), int(t)), (int(l + w), int(t + h)), (255, 0, 0), 2)
    #     cv2.putText(frame, f'ID: {track_id}', (int(l), int(t) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    # Display the frame
    # cv2.imshow('Frame', frame)
    
    # # Break the loop on 'q' key press
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break

# Release the video capture and close windows
cap.release()
cv2.destroyAllWindows()