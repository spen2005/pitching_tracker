import os
import cv2
import torch
import torch.nn as nn
import math
import bpy

# Video processing part
def generate_unique_filename(video_dir, video_name):
    output_file_path = os.path.join(video_dir, f'{video_name}-1.txt')
    return output_file_path
    number = 1
    while True:
        output_file_path = os.path.join(video_dir, f'{video_name}-{number}.txt')
        if not os.path.exists(output_file_path):
            return output_file_path
        number += 1

def process_video(dir_name):
    video_path = os.path.join('../video', dir_name, dir_name + '.mp4')
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error opening video file: {video_path}")
        return None

    ret, frame = cap.read()
    if not ret:
        print("Failed to read the first frame from video")
        return None

    bbox_list = []
    box_size = 12

    def draw_bbox(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            bbox = (x - box_size, y - box_size, 2 * box_size, 2 * box_size)
            bbox_list.append(bbox)
            cv2.rectangle(frame, (x - box_size, y - box_size), (x + box_size, y + box_size), (255, 0, 0), 2)

    cv2.namedWindow("Select Object")
    cv2.setMouseCallback("Select Object", draw_bbox)

    while True:
        cv2.imshow("Select Object", frame)
        if cv2.waitKey(1) & 0xFF == 13:
            break

    trackers = cv2.legacy.MultiTracker_create()
    for bbox in bbox_list:
        tracker = cv2.legacy.TrackerCSRT_create()
        success = tracker.init(frame, bbox)
        if success:
            trackers.add(tracker, frame, bbox)
        else:
            print(f"Failed to initialize tracker with bbox: {bbox}")

    video_dir = os.path.dirname(video_path)
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    output_file_path = generate_unique_filename(video_dir, video_name)
    final_output_file = open(output_file_path, 'w')

    Output = []
    ct = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        success, boxes = trackers.update(frame)
        final_output = [0.0] * 13
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
        cv2.imshow("Tracking", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        ct += 1

    for i in range(ct):
        for j in range(10):
            final_output_file.write(f"{Output[0][j]} ")
        final_output_file.write(f"{(i + 1) / ct * 9} ")
        final_output_file.write(f"{Output[i][11]} {Output[i][12]}\n")

    cap.release()
    cv2.destroyAllWindows()
    final_output_file.close()
    return output_file_path

# Neural network prediction part
class ComplexNN(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        super(ComplexNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, hidden_size2)
        self.fc4 = nn.Linear(hidden_size2, hidden_size1)
        self.fc5 = nn.Linear(hidden_size1, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = self.fc5(x)
        return x

def load_model():
    model = ComplexNN(input_size=11, hidden_size1=128, hidden_size2=64, output_size=3)
    model.load_state_dict(torch.load('../models/model.pth', map_location=torch.device('cpu')))
    model.eval()
    return model

def user_input(input_data):
    input_data = [float(num) for num in input_data]
    normalized_input_data = []

    for i in range(1, 5):
        input_data[i * 2] -= input_data[0]
        input_data[i * 2 + 1] -= input_data[1]

    regularization_value = math.sqrt(input_data[2] ** 2 + input_data[3] ** 2)
    for i in range(1, 5):
        vec_x = input_data[i * 2]
        vec_y = input_data[i * 2 + 1]

        if regularization_value != 0:
            vec_x /= regularization_value
            vec_y /= regularization_value
        normalized_input_data.extend([vec_x, vec_y])
    timestamp = input_data[10]
    x_prime = input_data[11]
    y_prime = input_data[12]
    x_prime = x_prime - input_data[0]
    y_prime = y_prime - input_data[1]

    if regularization_value != 0:
        x_prime /= regularization_value
        y_prime /= regularization_value
    normalized_input_data.extend([timestamp, x_prime, y_prime])
    return normalized_input_data

def predict(model, input_data):
    with torch.no_grad():
        output = model(torch.tensor(input_data, dtype=torch.float32).unsqueeze(0))
    return output.squeeze().cpu().numpy()

def make_predictions(model, dir_name, file_name):
    with open(f'../video/{dir_name}/{dir_name}-{file_name}.txt', 'r') as file:
        input_lines = file.readlines()

    predictions = []
    for line in input_lines:
        input_data = line.split()
        input_data = user_input(input_data)
        prediction = predict(model, input_data)
        predictions.append(prediction)

    formatted_predictions = []
    for prediction in predictions:
        formatted_prediction = [f"{x/100.0:.3f}" for x in prediction]
        formatted_predictions.append("(" + ", ".join(formatted_prediction) + ")")

    output_string = "\n".join(formatted_predictions)
    output_file_path = f'../video/{dir_name}/{dir_name}-{file_name}_location.txt'
    with open(output_file_path, 'w') as file:
        file.write(output_string)
    return output_file_path

# Blender animation part
def create_sphere(location, diameter=0.07):
    bpy.ops.mesh.primitive_uv_sphere_add(radius=diameter / 2, location=location)
    return bpy.context.object

def create_animation(dir_name, file_name):
    model_path = os.path.abspath('../model.blend')
    predictions_path = os.path.abspath(f'../video/{dir_name}/{dir_name}-{file_name}_location.txt')
    output_blend_path = os.path.abspath(f'../video/{dir_name}/animated_scene.blend')
    output_mp4_path = os.path.abspath(f'../video/{dir_name}/animated_scene.mp4')

    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Cannot find file: {model_path}")

    bpy.ops.wm.open_mainfile(filepath=model_path)

    predictions = []
    with open(predictions_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            line = line.strip('()\n')
            values = line.split(', ')
            if len(values) == 3:
                predictions.append([float(values[0]), float(values[1]), float(values[2])])
            else:
                print(f"Skipping invalid line: {line}")

    frame_number = 0
    frame_step = 5
    main_sphere = create_sphere(location=predictions[0])

    for i, loc in enumerate(predictions):
        frame = frame_number + i * frame_step
        bpy.context.scene.frame_set(frame)
        main_sphere.location = (loc[0], loc[1], loc[2])
        main_sphere.keyframe_insert(data_path="location", frame=frame)

    for i, loc in enumerate(predictions):
        frame = frame_number + i * frame_step
        sphere = create_sphere(location=(loc[0], loc[1], loc[2]))

        sphere.hide_viewport = True
        sphere.hide_render = True
        sphere.keyframe_insert(data_path="hide_viewport", frame=frame - 1)
        sphere.keyframe_insert(data_path="hide_render", frame=frame - 1)

        bpy.context.scene.frame_set(frame)
        sphere.hide_viewport = False
        sphere.hide_render = False
        sphere.keyframe_insert(data_path="hide_viewport", frame=frame)
        sphere.keyframe_insert(data_path="hide_render", frame=frame)

        sphere.keyframe_insert(data_path="location", frame=frame)

    bpy.context.scene.frame_end = frame_number + len(predictions) * frame_step
    bpy.ops.wm.save_as_mainfile(filepath=output_blend_path)
    bpy.context.scene.render.filepath = output_mp4_path
    bpy.context.scene.render.image_settings.file_format = 'FFMPEG'
    bpy.context.scene.render.ffmpeg.format = 'MPEG4'
    bpy.context.scene.render.ffmpeg.codec = 'H264'
    bpy.ops.render.render(animation=True)

if __name__ == "__main__":
    print("Enter the directory name: ")
    dir_name = input().strip()

    print("Processing video...")
    txt_path = process_video(dir_name)
    if txt_path is None:
        print("Video processing failed.")
        exit()

    model = load_model()

    print("Enter the file number: ")
    #file_name = input().strip()

    print("Making predictions...")
    predictions_path = make_predictions(model, dir_name, 1)

    print("Creating animation...")
    create_animation(dir_name, 1)

    print("Done!")
