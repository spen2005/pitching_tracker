import cv2
import os

def generate_unique_filename(video_dir, video_name):
    number = 1
    while True:
        output_file_path = os.path.join(video_dir, f'{video_name}-{number}.txt')
        if not os.path.exists(output_file_path):
            return output_file_path
        number += 1

# Load the video
video_path = '../video/fastball_7/fastball_7.mp4'
cap = cv2.VideoCapture(video_path)

# Initialize the tracker
trackers = cv2.legacy.MultiTracker_create()

# Get the first frame
ret, frame = cap.read()

# List to store bounding boxes
bbox_list = []

box_size = 12
# Callback function for mouse events
def draw_bbox(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        bbox = (x-1*box_size, y-1*box_size, 2*box_size, 2*box_size)  # Create a 100x100 square bounding box
        bbox_list.append(bbox)
        cv2.rectangle(frame, (x-1*box_size, y-1*box_size), (x+1*box_size, y+1*box_size), (255, 0, 0), 2)

# Set the callback function for mouse events
cv2.namedWindow("Select Object")
cv2.setMouseCallback("Select Object", draw_bbox)

while True:
    # Display the frame
    cv2.imshow("Select Object", frame)

    # Wait for the Enter key to be pressed
    if cv2.waitKey(1) & 0xFF == 13:  # ASCII value of Enter is 13
        break

# Initialize the tracker with the bounding boxes
for bbox in bbox_list:
    tracker = cv2.legacy.TrackerCSRT_create()
    trackers.add(tracker, frame, bbox)

# Rest of the code remains the same
# 打開文件準備寫入追蹤結果
video_dir = os.path.dirname(video_path)  # 獲取影片所在目錄
video_name = os.path.splitext(os.path.basename(video_path))[0]  # 獲取影片文件名，不包括擴展名
output_file_path = generate_unique_filename(video_dir, video_name)
final_output_file = open(output_file_path, 'w')

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
Output = []
ct = 0
while True:
    # 讀取新的一幀影像
    ret, frame = cap.read()
    if not ret:
        break
    
    # 更新追蹤器
    success, boxes = trackers.update(frame)
    final_output = [0.0] * 13  # 初始化每一幀的結果
    for i, bbox in enumerate(boxes):
        # 繪製追蹤框
        x, y, w, h = [int(pos) for pos in bbox]
        # 繪製追蹤框的中心點
        center_x = x + w // 2
        center_y = y + h // 2
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.circle(frame, (center_x, center_y), 5, (0, 255, 0), -1)
        # 將中心點坐標寫入文件
        if i < 5:
            final_output[i*2] = -center_y
            final_output[i*2+1] = center_x
        else:
            final_output[11] = -center_y
            final_output[12] = center_x
    # 將每一幀的結果添加到Output列表中
    Output.append(final_output)

    # 顯示結果
    cv2.imshow("Tracking", frame)
    
    # 按下 'q' 鍵結束追蹤
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    ct += 1

#write Output to final_output_file
for i in range(ct):
    for j in range(10):
        final_output_file.write(f"{Output[0][j]} ")
    final_output_file.write(f"{(i+1)/ct*9} ")
    final_output_file.write(f"{Output[i][11]} {Output[i][12]}\n")

# 釋放資源
cap.release()
cv2.destroyAllWindows()

# 關閉文件
final_output_file.close()
