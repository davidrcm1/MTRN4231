import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from key_pos_msgs.msg import KeyPositionArray, KeyPosition  # Custom message imports

import cv2
import numpy as np
from cv_bridge import CvBridge  # For converting ROS images to OpenCV
from threading import Thread
from ultralytics import YOLO
import logging

logging.getLogger().setLevel(logging.ERROR)  # Suppress non-critical YOLO logs

# Define color ranges in HSV
# Home Colours
# red_lower = np.array([0, 120, 50])
# red_upper = np.array([10, 255, 215])
# orange_lower = np.array([8, 50, 50])
# orange_upper = np.array([20, 255, 255])
# yellow_lower = np.array([20, 100, 100])
# yellow_upper = np.array([30, 255, 255])
# lime_lower = np.array([50, 50, 150])
# lime_upper = np.array([85, 255, 255])
# green_lower = np.array([25, 52, 72])
# green_upper = np.array([85, 200, 150])
# blue_lower = np.array([100, 50, 120])
# blue_upper = np.array([130, 255, 255])
# navy_blue_lower = np.array([120, 50, 50])
# navy_blue_upper = np.array([130, 255, 100])

# Lab colours

red_lower = np.array([153, 103, 224])
red_upper = np.array([173, 219, 255])
orange_lower = np.array([173, 87, 18])
orange_upper = np.array([195, 255, 255])
yellow_lower = np.array([20, 10, 215])
yellow_upper = np.array([42, 200, 255])
lime_lower = np.array([50, 31, 192])
lime_upper = np.array([86, 255, 255])
green_lower = np.array([77, 143, 0])
green_upper = np.array([95, 255, 208])
blue_lower = np.array([95, 100, 228])
blue_upper = np.array([119, 255, 255])
navy_blue_lower = np.array([107, 125, 0])
navy_blue_upper = np.array([121, 255, 140])

# Load YOLOv8 model
model = YOLO('yolov8n.pt')

class Webcam:
    def __init__(self):
        self.video_capture = cv2.VideoCapture(0)
        self.current_frame = self.video_capture.read()[1]

    def start(self):
        Thread(target=self._update_frame, args=()).start()

    def _update_frame(self):
        while True:
            self.current_frame = self.video_capture.read()[1]

    def get_current_frame(self):
        return self.current_frame

# Function to detect people using YOLOv8
def detect_people(frame):
    results = model(frame, verbose=False)  # YOLO detection
    hand_boxes = []
    people_detected = False
    for result in results:
        for box in result.boxes:
            if int(box.cls) == 0:  # Class '0' corresponds to 'person'
                x1, y1, x2, y2 = box.xyxy[0].cpu().int().numpy()  
                hand_boxes.append((x1, y1, x2, y2))
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                people_detected = True
    return frame, hand_boxes, people_detected

def detect_largest_rectangles_with_axes(mask, frame, color):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:2]
    
    centers = []
    axis_lengths = []
    for contour in contours:
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box = np.int0(box)

        # Calculate the lengths of the rectangle's sides (axis lengths)
        side_lengths = np.linalg.norm(box[0] - box[1]), np.linalg.norm(box[1] - box[2])
        longer_side = max(side_lengths)
        shorter_side = min(side_lengths)

        # Check if the rectangle is approximately square (sides are too similar in length)
        if shorter_side / longer_side > 0.8: 
            continue  # Skip this rectangle if it's approximately a square

        # Draw the rectangle and calculate center
        cv2.drawContours(frame, [box], 0, color, 2)
        center = np.mean(box, axis=0).astype(int)
        centers.append(center)
        cv2.circle(frame, tuple(center), 5, color, -1)

        axis_lengths.append(longer_side)
    
    # Determine which rectangle has the longer axis
    if len(axis_lengths) >=2: 
        if axis_lengths[0] > axis_lengths[1]:
            centers = np.array(centers[0]), np.array(centers[1])
        else:
            centers = np.array(centers[1]), np.array(centers[0])

    return centers, axis_lengths

# Function to calculate the gradient between two points
def calculate_gradient(point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    
    if x1 == x2:  # Handle vertical lines
        return None  # Undefined gradient
    return (y2 - y1) / (x2 - x1)

def apply_white_balance(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, _, _ = cv2.split(lab)
    min_val, max_val, _, _ = cv2.minMaxLoc(l_channel)
    gain = 255.0 / max_val
    balanced = cv2.convertScaleAbs(image, alpha=gain, beta=0)
    return balanced

# Function to calculate and return letters and their positions
def draw_letters_on_line(image, p1, p2, letters, extra_letters_left, extra_letters_right):
    # Calculate 9 equidistant points along the line between p1 and p2
    points = [((1 - t) * p1 + t * p2).astype(int) for t in np.linspace(0, 1, len(letters))]
    
    # Calculate the distance between main letters
    if len(points) > 1:
        distance = np.linalg.norm(points[1] - points[0])  # Euclidean distance between the first two points
    else:
        distance = 0  # Default to 0 if not enough points
    
    # Store the letters and their positions in the desired order
    letter_positions = []

    # Draw extra letters to the left of the first letter in reverse order
    for i, letter in enumerate(reversed(extra_letters_left)):
        # Calculate position based on distance to the left
        extra_point = points[0] - (distance * (i + 1) * (p2 - p1) / np.linalg.norm(p2 - p1)).astype(int)
        
        # Store the letter and its position
        letter_positions.append((letter, tuple(extra_point)))

        # Optionally, draw the letters on the image
        text_size = cv2.getTextSize(letter, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        text_x = extra_point[0] - text_size[0] // 2
        text_y = extra_point[1] + text_size[1] // 2
        outline_thickness = 3
        cv2.putText(image, letter, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), outline_thickness)  # Outline
        cv2.putText(image, letter, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)  # Main text

    # Overlay the main letters at each of the equidistant points
    for i, point in enumerate(points):
        letter_positions.append((letters[i], tuple(point)))

        # Optionally, draw the letters on the image
        text_size = cv2.getTextSize(letters[i], cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        text_x = point[0] - text_size[0] // 2
        text_y = point[1] + text_size[1] // 2
        cv2.putText(image, letters[i], (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), outline_thickness)  # Outline
        cv2.putText(image, letters[i], (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)  # Main text

    # Draw extra letters to the right of the last letter
    for i, letter in enumerate(extra_letters_right):
        # Calculate position based on distance to the right
        extra_point = points[-1] + (distance * (i + 1) * (p2 - p1) / np.linalg.norm(p2 - p1)).astype(int)
        
        # Store the letter and its position
        letter_positions.append((letter, tuple(extra_point)))

        # Optionally, draw the letters on the image
        text_size = cv2.getTextSize(letter, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        text_x = extra_point[0] - text_size[0] // 2
        text_y = extra_point[1] + text_size[1] // 2
        cv2.putText(image, letter, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), outline_thickness)  # Outline
        cv2.putText(image, letter, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)  # Main text

    # Return the list of letters with their positions
    return letter_positions

def order_corners(corners):
    # Sort the corners by their x and y positions
    # Top-left will have the smallest sum (x + y), bottom-right will have the largest sum
    # Top-right will have the smallest difference (x - y), bottom-left will have the largest difference
    rect = np.zeros((4, 2), dtype="float32")
    s = corners.sum(axis=1)
    rect[0] = corners[np.argmin(s)]  # Top-left
    rect[2] = corners[np.argmax(s)]  # Bottom-right

    diff = np.diff(corners, axis=1)
    rect[1] = corners[np.argmin(diff)]  # Top-right
    rect[3] = corners[np.argmax(diff)]  # Bottom-left

    return rect

def apply_perspective_transform(frame, corners):
    # Define the destination points (corners of the frame)
    (h, w) = frame.shape[:2]
    h = 575
    w = 575
    dst = np.array([
        [0, 0],        # Top-left
        [w - 1, 0],    # Top-right
        [w - 1, h - 1],# Bottom-right
        [0, h - 1]     # Bottom-left
    ], dtype="float32")
    
    # Order the corners to match the destination order
    ordered_corners = order_corners(np.array(corners, dtype="float32"))
    
    # Compute the perspective transform matrix and apply it
    M = cv2.getPerspectiveTransform(ordered_corners, dst)
    warped = cv2.warpPerspective(frame, M, (w, h))
    
    return warped

def detect_dots(frame):
    # Define the HSV range for dot color
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    dot_lower = np.array([79, 100, 150])
    dot_upper = np.array([119, 255, 255])
    
    # Create a mask for detecting dots
    dot_mask = cv2.inRange(hsv_frame, dot_lower, dot_upper)
    
    # Find contours of the dots
    contours, _ = cv2.findContours(dot_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours that are roughly circular (dots)
    dots = []
    circularity_threshold = 0.7  # Define a circularity threshold
    for contour in contours:
        if len(contour) >= 5:
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            if area > 50:  # Threshold to eliminate noise
                circularity = (4 * np.pi * area) / (perimeter ** 2)  # Circularity formula
                if circularity >= circularity_threshold:  # Check if the circularity meets the threshold
                    center, radius = cv2.minEnclosingCircle(contour)
                    dots.append((center, radius, circularity))

    # Sort dots based on circularity in descending order
    dots = sorted(dots, key=lambda x: x[2], reverse=True)[:4]
    
    # Return nothing if fewer than 4 dots are found
    if len(dots) < 4:
        return []
    
    # Return the 4 most certain dot centers
    return [dot[0] for dot in dots]


class KeyPublisher(Node):
    def __init__(self):
        super().__init__('rectangle_person_publisher')
        self.publisher_ = self.create_publisher(KeyPositionArray, 'key_positions', 10)
        self.timer = self.create_timer(0.5, self.timer_callback)
        self.bridge = CvBridge()
        self.webcam = Webcam()
        self.webcam.start()

        self.processed_frame = self.webcam.get_current_frame()  # Initialize before starting the thread

        # Start a separate thread for displaying the frames
        self.display_thread = Thread(target=self.display_frames)
        self.display_thread.daemon = True
        self.display_thread.start()


    def timer_callback(self):
        frame = self.webcam.get_current_frame()

        # Detect purple dots
        dots = detect_dots(frame)
        dots_detected = False
        # Proceed with transformation only if exactly 4 dots are detected
        if len(dots) == 4:
            # Apply perspective transform
            frame = apply_perspective_transform(frame, dots)
            dots_detected = True

        # frame = apply_white_balance(frame)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Step 2: Create masks for red, green, and yellow
        red_mask = cv2.inRange(hsv, red_lower, red_upper)
        orange_mask = cv2.inRange(hsv, orange_lower, orange_upper)
        yellow_mask = cv2.inRange(hsv, yellow_lower, yellow_upper)
        lime_mask = cv2.inRange(hsv, lime_lower, lime_upper)
        green_mask = cv2.inRange(hsv, green_lower, green_upper)
        blue_mask = cv2.inRange(hsv, blue_lower, blue_upper)
        navy_blue_mask = cv2.inRange(hsv, navy_blue_lower, navy_blue_upper)

         # Define kernel size for the closing operation
        kernel = np.ones((8, 8), np.uint8) 
        
        # Apply closing to each mask
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)
        orange_mask = cv2.morphologyEx(orange_mask, cv2.MORPH_CLOSE, kernel)
        yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_CLOSE, kernel)
        lime_mask = cv2.morphologyEx(lime_mask, cv2.MORPH_CLOSE, kernel)
        lime_mask = cv2.morphologyEx(lime_mask, cv2.MORPH_CLOSE, kernel)
        green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, kernel)
        blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_CLOSE, kernel)
        navy_blue_mask = cv2.morphologyEx(navy_blue_mask, cv2.MORPH_CLOSE, kernel)



        # Step 3: Detect the largest rectangles for each color with axis lengths
        red_centers, red_axes = detect_largest_rectangles_with_axes(red_mask, frame, (0, 0, 255))
        orange_centers, orange_axes = detect_largest_rectangles_with_axes(orange_mask, frame, (0, 165, 255))  # Orange color in BGR
        yellow_centers, yellow_axes = detect_largest_rectangles_with_axes(yellow_mask, frame, (0, 255, 255))
        lime_centers, lime_axes = detect_largest_rectangles_with_axes(lime_mask, frame, (0, 255, 0))
        green_centers, green_axes = detect_largest_rectangles_with_axes(green_mask, frame, (34, 139, 34))  # Dark green in BGR
        blue_centers, blue_axes = detect_largest_rectangles_with_axes(blue_mask, frame, (173, 216, 230))  # Light blue
        navy_blue_centers, navy_blue_axes = detect_largest_rectangles_with_axes(navy_blue_mask, frame, (0, 0, 128))  # Navy blue


        frame, _, people_detected = detect_people(frame)

        # Step 4: Calculate the average gradient between the largest pairs of rectangles
        gradients = []
        
        # if len(red_centers) == 2:
        #     red_gradient = calculate_gradient(red_centers[0], red_centers[1])
        #     if red_gradient is not None:
        #         gradients.append(red_gradient)
        
        # if len(orange_centers) == 2:
        #     orange_gradient = calculate_gradient(orange_centers[0], orange_centers[1])
        #     if orange_gradient is not None:
        #         gradients.append(orange_gradient)
        
        # if len(yellow_centers) == 2:
        #     yellow_gradient = calculate_gradient(yellow_centers[0], yellow_centers[1])
        #     if yellow_gradient is not None:
        #         gradients.append(yellow_gradient)

        if len(lime_centers) == 2:
            lime_gradient = calculate_gradient(lime_centers[0], lime_centers[1])
            if lime_gradient is not None:
                gradients.append(lime_gradient)

        if len(green_centers) == 2:
            green_gradient = calculate_gradient(green_centers[0], green_centers[1])
            if green_gradient is not None:
                gradients.append(green_gradient)

        if len(blue_centers) == 2:
            blue_gradient = calculate_gradient(blue_centers[0], blue_centers[1])
            if blue_gradient is not None:
                gradients.append(blue_gradient)

        if len(navy_blue_centers) == 2:
            navy_blue_gradient = calculate_gradient(navy_blue_centers[0], navy_blue_centers[1])
            if navy_blue_gradient is not None:
                gradients.append(navy_blue_gradient)


        if gradients:
            median_gradient = np.median(gradients)
            filtered_gradients = [g for g in gradients if abs(g - median_gradient) < 0.5]  # Example threshold for deviation
            if filtered_gradients:
                avg_gradient = sum(filtered_gradients) / len(filtered_gradients)
                chosen_colour = "yellow"   

                # Step 5: Draw lines with the average gradient
                if len(red_centers) == 2:
                    x1, y1 = red_centers[0]
                    x2, y2 = red_centers[1]
                    y2 = int(y1 + avg_gradient * (x2 - x1))
                    cv2.line(frame, red_centers[0], (x2, y2), (0, 0, 255), 2)
                
                if len(orange_centers) == 2:
                    x1, y1 = orange_centers[0]
                    x2, y2 = orange_centers[1]
                    y2 = int(y1 + avg_gradient * (x2 - x1))
                    cv2.line(frame, orange_centers[0], (x2, y2), (0, 165, 255), 2)  # Orange color in BGR

                if len(yellow_centers) == 2:
                    x1, y1 = yellow_centers[0]
                    x2, y2 = yellow_centers[1]
                    y2 = int(y1 + avg_gradient * (x2 - x1))
                    cv2.line(frame, yellow_centers[0], (x2, y2), (0, 255, 255), 2)
                
                if len(lime_centers) == 2:
                    x1, y1 = lime_centers[0]
                    x2, y2 = lime_centers[1]
                    y2 = int(y1 + avg_gradient * (x2 - x1))
                    cv2.line(frame, lime_centers[0], (x2, y2), (0, 255, 0), 2)

                if len(green_centers) == 2:
                    x1, y1 = green_centers[0]
                    x2, y2 = green_centers[1]
                    y2 = int(y1 + avg_gradient * (x2 - x1))
                    cv2.line(frame, green_centers[0], (x2, y2), (0, 100, 0), 2)  # Dark green color in BGR

                if len(blue_centers) == 2:
                    x1, y1 = blue_centers[0]
                    x2, y2 = blue_centers[1]
                    y2 = int(y1 + avg_gradient * (x2 - x1))
                    cv2.line(frame, blue_centers[0], (x2, y2), (255, 178, 102), 2)  # Light blue color in BGR

                if len(navy_blue_centers) == 2:
                    x1, y1 = navy_blue_centers[0]
                    x2, y2 = navy_blue_centers[1]
                    y2 = int(y1 + avg_gradient * (x2 - x1))
                    cv2.line(frame, navy_blue_centers[0], (x2, y2), (128, 0, 0), 2)  # Navy blue color in BGR

                match chosen_colour:
                    case "red":
                        # List of letters to display at equidistant points
                        letters = ["R1", "O1", "Y1", "L1", "G1", "B1", "N1", "P1", "R2"]
                        extra_letters_left = [] # Reverse order
                        extra_letters_right = ["O2", "Y2", "L2", "G2", "B2", "N2"]
                        centers = red_centers
                    case "orange":
                        # List of letters to display at equidistant points
                        letters = ["O1", "Y1", "L1", "G1", "B1", "N1", "P1", "R2", "O2"]
                        extra_letters_left = ["R1"] # Reverse order
                        extra_letters_right = ["Y2", "L2", "G2", "B2", "N2"]
                        centers = orange_centers
                    case "yellow":
                        # List of letters to display at equidistant points
                        letters = ["Y1", "L1", "G1", "B1", "N1", "P1", "R2", "O2", "Y2"]
                        extra_letters_left = ["R1", "O1"] # Reverse order
                        extra_letters_right = ["L2", "G2", "B2", "N2"]
                        centers = yellow_centers
                    case "lime":
                        # List of letters to display at equidistant points
                        letters = ["L1", "G1", "B1", "N1", "P1", "R2", "O2", "Y2", "L2"]
                        extra_letters_left = ["R1", "O1", "Y1"] # Reverse order 
                        extra_letters_right = ["G2", "B2", "N2"]
                        centers = green_centers
                    case "green":
                        # List of letters to display at equidistant points
                        letters = ["G1", "B1", "N1", "P1", "R2", "O2", "Y2", "L2", "G2"]
                        extra_letters_left = ["R1", "O1", "Y1", "L1"] # Reverse order 
                        extra_letters_right = ["B2", "N2"]
                        centers = green_centers
                    case "blue":
                        # List of letters to display at equidistant points
                        letters = [ "B1", "N1", "P1", "R2", "O2", "Y2", "L2", "G2", "B2"]
                        extra_letters_left = ["R1", "O1", "Y1", "L1", "G1"]
                        extra_letters_right = [ "N2"]
                        centers = blue_centers
                    case "navy_blue":
                        # List of letters to display at equidistant points
                        letters = [ "N1", "P1", "R2", "O2", "Y2", "L2", "G2", "B2", "N2"]
                        extra_letters_left = ["R1", "O1", "Y1", "L1", "G1", "B1"]
                        extra_letters_right = []
                        centers = navy_blue_centers

                if len(centers) >= 2:
                    # Draw the letters along the line between the centers
                    letter_positions = draw_letters_on_line(frame, centers[0], centers[1], letters, extra_letters_left, extra_letters_right)
                    # Prepare the KeyPositionArray message
                    key_position_array = KeyPositionArray()
                    for letter, position in letter_positions:
                        key_pos = KeyPosition()
                        key_pos.letter = letter
                        key_pos.x = int(position[0]*-1)
                        key_pos.y = int(position[1])
                        key_position_array.key_positions.append(key_pos)

                    key_position_array.people_detected = people_detected
                    key_position_array.dots_detected = dots_detected
                    # Publish the array of key positions
                    self.publisher_.publish(key_position_array)
                    self.get_logger().info(f"Published key positions: {[f'{kp.letter}: ({kp.x},{kp.y})' for kp in key_position_array.key_positions]}, person detected: {people_detected}, dots detected: {dots_detected}")
                
                else:
                    print("Not enough centers detected to draw letters.")
                        
        # Save the processed frame to be displayed
        self.processed_frame = frame

    def display_frames(self):
        while True:
            if self.processed_frame is not None:
                # Display the frame
                cv2.imshow('Color Rectangle Detection', self.processed_frame)
                # Necessary for the frame to update
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        cv2.destroyAllWindows()

            

def main(args=None):
    rclpy.init(args=args)
    node = KeyPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()