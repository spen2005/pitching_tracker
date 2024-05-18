import bpy
import os

# Ensure the file path is correct
model_path = os.path.abspath('../model.blend')

print("Enter the directory:")
dir_name = input()
print("Enter the file:")
file_name = input()

predictions_path = os.path.abspath('../video/'+dir_name+'/'+dir_name+'-'+file_name+'_location.txt')
output_blend_path = os.path.abspath('../video/'+dir_name+'/animated_scene.blend')
output_mp4_path = os.path.abspath('../video/'+dir_name+'/animated_scene.mp4')

# Check if model.blend exists
if not os.path.isfile(model_path):
    raise FileNotFoundError(f"Cannot find file: {model_path}")

# Load your base scene
bpy.ops.wm.open_mainfile(filepath=model_path)

# Function to create a sphere at a specific location
def create_sphere(location, diameter=0.07):
    bpy.ops.mesh.primitive_uv_sphere_add(radius=diameter / 2, location=location)
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

# Create and animate the main sphere
frame_number = 0
frame_step = 5

# Create the initial sphere at the first location for smooth movement
main_sphere = create_sphere(location=predictions[0])

for i, loc in enumerate(predictions):
    frame = frame_number + i * frame_step

    # Set the current frame
    bpy.context.scene.frame_set(frame)

    # Move the main sphere to the new location and insert keyframe
    main_sphere.location = (loc[0], loc[1], loc[2])
    main_sphere.keyframe_insert(data_path="location", frame=frame)

# Create and animate spheres that appear at each location
for i, loc in enumerate(predictions):
    frame = frame_number + i * frame_step
    sphere = create_sphere(location=(loc[0], loc[1], loc[2]))

    # Hide the sphere before the current frame
    sphere.hide_viewport = True
    sphere.hide_render = True
    sphere.keyframe_insert(data_path="hide_viewport", frame=frame - 1)
    sphere.keyframe_insert(data_path="hide_render", frame=frame - 1)

    # Show the sphere at the current frame
    bpy.context.scene.frame_set(frame)
    sphere.hide_viewport = False
    sphere.hide_render = False
    sphere.keyframe_insert(data_path="hide_viewport", frame=frame)
    sphere.keyframe_insert(data_path="hide_render", frame=frame)

    # Insert location keyframe
    sphere.keyframe_insert(data_path="location", frame=frame)

# Set animation end frame
bpy.context.scene.frame_end = frame_number + len(predictions) * frame_step

# Save the animation
bpy.ops.wm.save_as_mainfile(filepath=output_blend_path)

# Render the animation to an MP4 file
bpy.context.scene.render.filepath = output_mp4_path
bpy.context.scene.render.image_settings.file_format = 'FFMPEG'
bpy.context.scene.render.ffmpeg.format = 'MPEG4'
bpy.context.scene.render.ffmpeg.codec = 'H264'
bpy.ops.render.render(animation=True)
