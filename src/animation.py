import bpy
import os

# Ensure the file path is correct
model_path = os.path.abspath('../model.blend')
predictions_path = os.path.abspath('../video/fastball_7/fastball_7-1_location.txt')
output_blend_path = os.path.abspath('../video/fastball_7/animated_scene.blend')
output_mp4_path = os.path.abspath('../video/fastball_7/animated_scene.mp4')

# Check if model.blend exists
if not os.path.isfile(model_path):
    raise FileNotFoundError(f"Cannot find file: {model_path}")

# Load your base scene
bpy.ops.wm.open_mainfile(filepath=model_path)

# Function to create a sphere at a specific location
def create_sphere(location, diameter=0.07):
    bpy.ops.mesh.primitive_uv_sphere_add(radius=diameter/2, location=location)
    return bpy.context.object

# Read predictions from file
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

# Create spheres and animate
frame_number = 0
frame_step = 5

for i, loc in enumerate(predictions):
    if i % 3 == 0:
        sphere = create_sphere(location=(loc[0], loc[1], loc[2]))

    frame = frame_number + (i % 3) * frame_step
    sphere.location = (loc[0], loc[1], loc[2])
    sphere.keyframe_insert(data_path="location", frame=frame)

    if i % 3 == 2:
        frame_number += 3 * frame_step

# Save the animation
bpy.context.scene.frame_end = frame_number
bpy.ops.wm.save_as_mainfile(filepath=output_blend_path)

# Render the animation to an MP4 file
bpy.context.scene.render.filepath = output_mp4_path
bpy.context.scene.render.image_settings.file_format = 'FFMPEG'
bpy.context.scene.render.ffmpeg.format = 'MPEG4'
bpy.context.scene.render.ffmpeg.codec = 'H264'
bpy.ops.render.render(animation=True)

