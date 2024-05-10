import cv2

# 載入影片
video_path = 'video/splitter_1.mp4'
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
output_files = [open(f"{video_path.split('.')[0]}_output_{i}.txt", 'w') for i in range(len(bbox_list))]

while True:
    # 讀取新的一幀影像
    ret, frame = cap.read()
    if not ret:
        break
    
    # 更新追蹤器
    success, boxes = trackers.update(frame)
    
    for i, bbox in enumerate(boxes):
        # 繪製追蹤框
        x, y, w, h = [int(pos) for pos in bbox]
        # 繪製追蹤框的中心點
        center_x = x + w // 2
        center_y = y + h // 2
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.circle(frame, (center_x, center_y), 5, (0, 255, 0), -1)
        # 將中心點坐標寫入文件
        output_files[i].write(f"{center_x},{center_y}\n")
    
    # 顯示結果
    cv2.imshow("Tracking", frame)
    
    # 按下 'q' 鍵結束追蹤
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 釋放資源
cap.release()
cv2.destroyAllWindows()

# 關閉文件
for file in output_files:
    file.close()