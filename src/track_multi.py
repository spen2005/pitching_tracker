import cv2
import os

# 載入影片
video_path = '../video/slider_1/slider_1.mp4'  # 根據新的結構更新影片路徑
cap = cv2.VideoCapture(video_path)

# 初始化追蹤器列表
trackers = cv2.legacy.MultiTracker_create()

# 獲取第一幀的影像
ret, frame = cap.read()

# 選擇要追蹤的物體
bbox_list = []
for i in range(6):
    bbox = cv2.selectROI("Select Object", frame, False)
    bbox_list.append(bbox)

# 初始化追蹤器
for bbox in bbox_list:
    tracker = cv2.legacy.TrackerCSRT_create()
    trackers.add(tracker, frame, bbox)

# 打開文件準備寫入追蹤結果
video_dir = os.path.dirname(video_path)  # 獲取影片所在目錄
output_file_path = os.path.join(video_dir, 'slider_1_final.txt')  # 將結果文件保存在影片目錄下
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

# 將Output寫入結果文件
for i in range(ct):
    for j in range(13):
        final_output_file.write(f"{Output[i][j]} ")
    final_output_file.write("\n")

# 釋放資源
cap.release()
cv2.destroyAllWindows()

# 關閉文件
final_output_file.close()
