import cv2

# 載入影片
video_path = 'video/splitter_1.mp4'
cap = cv2.VideoCapture(video_path)

# 初始化追蹤器
tracker = cv2.TrackerCSRT_create()

# 獲取第一幀的影像
ret, frame = cap.read()

# 選擇要追蹤的物體
bbox = cv2.selectROI("Select Object", frame, False)

# 初始化追蹤器
tracker.init(frame, bbox)

# 打開文件準備寫入追蹤結果
output_file = open(video_path.split('.')[0] + '_output.txt', 'w')

while True:
    # 讀取新的一幀影像
    ret, frame = cap.read()
    if not ret:
        break
    
    # 更新追蹤器
    success, bbox = tracker.update(frame)
    
    # 繪製追蹤框
    if success:
        x, y, w, h = [int(i) for i in bbox]
        # 繪製追蹤框的中心點
        center_x = x + w // 2
        center_y = y + h // 2
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.circle(frame, (center_x, center_y), 5, (0, 255, 0), -1)
        # 將中心點坐標寫入文件
        output_file.write(f"{center_x},{center_y}\n")
    else:
        cv2.putText(frame, "Lost", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)
    
    # 顯示結果
    cv2.imshow("Tracking", frame)
    
    # 按下 'q' 鍵結束追蹤
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 釋放資源
cap.release()
cv2.destroyAllWindows()

# 關閉文件
output_file.close()
