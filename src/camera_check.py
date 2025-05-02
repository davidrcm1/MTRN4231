from cv2_enumerate_cameras import enumerate_cameras
import cv2

for camera_info in enumerate_cameras():
    print(f'{camera_info.index}: {camera_info.name}')
    try:
        cap = cv2.VideoCapture(camera_info.index, cv2.CAP_V4L2)  # Use CAP_V4L2 explicitly
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                cv2.imshow(f'Camera {camera_info.index}', frame)
                cv2.waitKey(0)  # Wait for a key press to close the window
            else:
                print(f"Could not read from camera {camera_info.index}")
            cap.release()
        else:
            print(f"Could not open camera {camera_info.index}")
    except Exception as e:
        print(f"Error accessing camera {camera_info.index}: {e}")

cv2.destroyAllWindows()