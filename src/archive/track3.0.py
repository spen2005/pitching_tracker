import cv2
import os

def generate_unique_filename(video_dir, video_name):
    output_file_path = os.path.join(video_dir, f'{video_name}-1.txt')
    return output_file_path
    number = 1
    while True:
        output_file_path = os.path.join(video_dir, f'{video_name}-{number}.txt')
        if not os.path.exists(output_file_path):
            return output_file_path
        number += 1

# Input file name
print("Enter the directory name: ")
dir_name = input().strip()

# Load the video
video_path = os.path.join('../video', dir_name, dir_name + '.mp4')
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"Error opening video file: {video_path}")
    exit()

# Get the first frame
ret, frame = cap.read()
if not ret:
    print("Failed to read the first frame from video")
    exit()

# List to store bounding boxes
bbox_list = []

box_size = 12
# Callback function for mouse events
def draw_bbox(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        bbox = (x - box_size, y - box_size, 2 * box_size, 2 * box_size)  # Create a bounding box
        bbox_list.append(bbox)
        cv2.rectangle(frame, (x - box_size, y - box_size), (x + box_size, y + box_size), (255, 0, 0), 2)

# Set the callback function for mouse events
cv2.namedWindow("Select Object")
cv2.setMouseCallback("Select Object", draw_bbox)

while True:
    # Display the frame
    cv2.imshow("Select Object", frame)

    # Wait for the Enter key to be pressed
    if cv2.waitKey(1) & 0xFF == 13:  # ASCII value of Enter is 13
        break

# Initialize each tracker with the bounding boxes
trackers = cv2.legacy.MultiTracker_create()
for bbox in bbox_list:
    tracker = cv2.legacy.TrackerCSRT_create()
    success = tracker.init(frame, bbox)
    if success:
        trackers.add(tracker, frame, bbox)
    else:
        print(f"Failed to initialize tracker with bbox: {bbox}")

# Open file to write tracking results
video_dir = os.path.dirname(video_path)  # Get the directory of the video
video_name = os.path.splitext(os.path.basename(video_path))[0]  # Get the video file name without extension
output_file_path = generate_unique_filename(video_dir, video_name)
final_output_file = open(output_file_path, 'w')

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
Output = []
ct = 0

while True:
    # Read a new frame
    ret, frame = cap.read()
    if not ret:
        break

    # Update all trackers
    success, boxes = trackers.update(frame)
    final_output = [0.0] * 13  # Initialize the result for each frame
    if success:
        for i, bbox in enumerate(boxes):
            x, y, w, h = [int(v) for v in bbox]
            center_x = x + w // 2
            center_y = y + h // 2
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.circle(frame, (center_x, center_y), 5, (0, 255, 0), -1)
            if i < 5:
                final_output[i * 2] = -center_y
                final_output[i * 2 + 1] = center_x
            else:
                final_output[11] = -center_y
                final_output[12] = center_x
    else:
        print(f"Failed to update trackers on frame {ct}")

    Output.append(final_output)

    # Display the result
    cv2.imshow("Tracking", frame)

    # Press 'q' to end tracking
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    ct += 1

# Write Output to final_output_file
for i in range(ct):
    for j in range(10):
        final_output_file.write(f"{Output[0][j]} ")
    final_output_file.write(f"{(i + 1) / ct * 9} ")
    final_output_file.write(f"{Output[i][11]} {Output[i][12]}\n")

# Release resources
cap.release()
cv2.destroyAllWindows()

# Close the file
final_output_file.close()
