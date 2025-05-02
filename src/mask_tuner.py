import cv2
import numpy as np
from threading import Thread
from ultralytics import YOLO
import logging

logging.getLogger().setLevel(logging.ERROR)  # Suppresses all lower-severity messages

# Define initial HSV ranges for colors
color_ranges = {
    "red": [[153, 103, 224], [173, 219, 255]],
    "orange": [[173, 87, 18], [195, 255, 255]],
    "yellow": [[20, 10, 215], [42, 200, 255]],
    "lime": [[50, 31, 192], [86, 255, 255]],
    "green": [[82, 143, 0], [94, 255, 208]],
    "blue": [[95, 100, 228], [119, 255, 255]],
    "navy_blue": [[107, 125, 0], [121, 255, 140]],
    "dot": [[95, 100, 228], [119, 255, 255]],
}

color_index_map = {i: color for i, color in enumerate(color_ranges.keys())}
chosen_colour_index = 0  # Default index for dropdown

# Create windows and trackbars for each color in a grid layout
def create_hsv_windows():
    control_window_name = "Control"
    cv2.namedWindow(control_window_name)
    cv2.createTrackbar("Chosen Colour", control_window_name, chosen_colour_index, len(color_ranges) - 1, lambda x: None)
    cv2.createTrackbar("Print HSV Values", control_window_name, 0, 1, print_hsv_values)
    cv2.moveWindow(control_window_name, 0, 0)

    # HSV sliders for each color, organized in separate windows
    row, col = 0, 0
    for color in color_ranges.keys():
        window_name = f"{color.capitalize()} HSV"
        cv2.namedWindow(window_name)
        cv2.resizeWindow(window_name, 300, 200)
        cv2.moveWindow(window_name, col * 310, row * 220)  # Position each window in a grid

        for i, val in enumerate(['H', 'S', 'V']):
            cv2.createTrackbar(f"Lower {val}", window_name, color_ranges[color][0][i], 360 if val == 'H' else 255, lambda x: None)
            cv2.createTrackbar(f"Upper {val}", window_name, color_ranges[color][1][i], 360 if val == 'H' else 255, lambda x: None)

        # Increment row and column for grid layout (3 columns per row)
        col += 1
        if col >= 3:
            col = 0
            row += 1

def update_hsv_ranges():
    for color in color_ranges.keys():
        window_name = f"{color.capitalize()} HSV"
        for i, val in enumerate(['H', 'S', 'V']):
            color_ranges[color][0][i] = cv2.getTrackbarPos(f"Lower {val}", window_name)
            color_ranges[color][1][i] = cv2.getTrackbarPos(f"Upper {val}", window_name)

def get_chosen_color():
    global chosen_colour_index
    chosen_colour_index = cv2.getTrackbarPos("Chosen Colour", "Control")
    return color_index_map[chosen_colour_index]

def print_hsv_values(dummy_value):
    if dummy_value == 1:
        for color, (lower, upper) in color_ranges.items():
            print(f"{color}_lower = np.array([{lower[0]}, {lower[1]}, {lower[2]}])")
            print(f"{color}_upper = np.array([{upper[0]}, {upper[1]}, {upper[2]}])")
        cv2.setTrackbarPos("Print HSV Values", "Control", 0)

create_hsv_windows()

# Webcam setup
class Webcam:
    def __init__(self):
        self.video_capture = cv2.VideoCapture(3)
        self.current_frame = self.video_capture.read()[1]

    def start(self):
        Thread(target=self._update_frame, args=()).start()

    def _update_frame(self):
        while True:
            self.current_frame = self.video_capture.read()[1]

    def get_current_frame(self):
        return self.current_frame

# Load YOLOv8 model for hand detection
model = YOLO('yolov8n.pt')

# Function to detect the largest rectangles for a given color mask
def detect_largest_rectangles_with_axes(mask, frame, color):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:2]

    centers = []
    for contour in contours:
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box = np.int0(box)

        # Calculate center and draw rectangle
        center = np.mean(box, axis=0).astype(int)
        centers.append(center)
        cv2.drawContours(frame, [box], 0, color, 2)
        cv2.circle(frame, tuple(center), 5, color, -1)

    return centers

# Function to draw letters based on color
def draw_letters_on_line(image, p1, p2, letters, extra_letters_left, extra_letters_right):
    points = [((1 - t) * p1 + t * p2).astype(int) for t in np.linspace(0, 1, len(letters))]
    distance = np.linalg.norm(points[1] - points[0]) if len(points) > 1 else 0
    letter_positions = []

    for i, letter in enumerate(reversed(extra_letters_left)):
        extra_point = points[0] - (distance * (i + 1) * (p2 - p1) / np.linalg.norm(p2 - p1)).astype(int)
        letter_positions.append((letter, tuple(extra_point)))
        text_size = cv2.getTextSize(letter, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        text_x = extra_point[0] - text_size[0] // 2
        text_y = extra_point[1] + text_size[1] // 2
        cv2.putText(image, letter, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    for i, point in enumerate(points):
        letter_positions.append((letters[i], tuple(point)))
        text_size = cv2.getTextSize(letters[i], cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        text_x = point[0] - text_size[0] // 2
        text_y = point[1] + text_size[1] // 2
        cv2.putText(image, letters[i], (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    for i, letter in enumerate(extra_letters_right):
        extra_point = points[-1] + (distance * (i + 1) * (p2 - p1) / np.linalg.norm(p2 - p1)).astype(int)
        letter_positions.append((letter, tuple(extra_point)))
        text_size = cv2.getTextSize(letter, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        text_x = extra_point[0] - text_size[0] // 2
        text_y = extra_point[1] + text_size[1] // 2
        cv2.putText(image, letter, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    return letter_positions

def detect_dots(frame):
    # Define the HSV range for dot color based on sliders
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    dot_lower = np.array(color_ranges["dot"][0], dtype=np.uint8)
    dot_upper = np.array(color_ranges["dot"][1], dtype=np.uint8)

    # Create a mask for detecting dots
    dot_mask = cv2.inRange(hsv_frame, dot_lower, dot_upper)
    dot_mask_color = cv2.cvtColor(dot_mask, cv2.COLOR_GRAY2BGR)  # Convert to BGR for colored circles
    cv2.imshow("Dot Mask", dot_mask_color)  # Display the dot mask

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
                    # Draw circle on the dot mask
                    cv2.circle(dot_mask_color, (int(center[0]), int(center[1])), int(radius), (0, 255, 0), 2)

    # Sort dots based on circularity in descending order
    dots = sorted(dots, key=lambda x: x[2], reverse=True)[:4]
    
    # Update the dot mask with drawn circles and display
    cv2.imshow("Dot Mask", dot_mask_color)

    # Return nothing if fewer than 4 dots are found
    if len(dots) < 4:
        return []
    
    # Return the 4 most certain dot centers
    return [dot[0] for dot in dots]


# Main loop
webcam = Webcam()
webcam.start()

while True:
    frame = webcam.get_current_frame()
    original_frame = frame.copy()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Update HSV ranges and get chosen color
    update_hsv_ranges()
    chosen_colour = get_chosen_color()

    # Define specific letters based on the chosen color
    color_settings = {
        "red": (["R1", "O1", "Y1", "L1", "G1", "B1", "N1", "P1", "R2"], [], ["O2", "Y2", "L2", "G2", "B2", "N2"]),
        "orange": (["O1", "Y1", "L1", "G1", "B1", "N1", "P1", "R2", "O2"], ["R1"], ["Y2", "L2", "G2", "B2", "N2"]),
        "yellow": (["Y1", "L1", "G1", "B1", "N1", "P1", "R2", "O2", "Y2"], ["O1", "R1"], ["L2", "G2", "B2", "N2"]),
        "lime": (["L1", "G1", "B1", "N1", "P1", "R2", "O2", "Y2", "L2"], ["R1", "O1", "Y1"], ["G2", "B2", "N2"]),
        "green": (["G1", "B1", "N1", "P1", "R2", "O2", "Y2", "L2", "G2"], ["R1", "O1", "Y1", "L1"], ["B2", "N2"]),
        "blue": (["B1", "N1", "P1", "R2", "O2", "Y2", "L2", "G2", "B2"], ["R1", "O1", "Y1", "L1", "G1"], ["N2"]),
        "navy_blue": (["N1", "P1", "R2", "O2", "Y2", "L2", "G2", "B2", "N2"], ["R1", "O1", "Y1", "L1", "G1", "B1"], []),
    }
    letters, extra_left, extra_right = color_settings[chosen_colour]
    # Detect and draw dots
    detect_dots(original_frame)
    # Mask creation and rectangle detection
    centers = {}
    for color, (lower, upper) in color_ranges.items():
        if color == "dot":
            continue  # Skip dot
        lower_bound = np.array(lower, dtype=np.uint8)
        upper_bound = np.array(upper, dtype=np.uint8)
        mask = cv2.inRange(frame, lower_bound, upper_bound)
        
        draw_color = {"red": (0, 0, 255), "orange": (0, 165, 255), "yellow": (0, 255, 255),
                      "lime": (0, 255, 0), "green": (34, 139, 34), "blue": (173, 216, 230), "navy_blue": (0, 0, 128)}
        
        detected_centers = detect_largest_rectangles_with_axes(mask, original_frame, draw_color[color])
        if color == chosen_colour and len(detected_centers) == 2:
            draw_letters_on_line(original_frame, detected_centers[0], detected_centers[1], letters, extra_left, extra_right)
        cv2.imshow(f"{color.capitalize()} Mask", mask)
    
    cv2.imshow("Color Rectangle Detection", original_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
