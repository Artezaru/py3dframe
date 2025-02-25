print("# ======== py3dframe: Using FrameTree and Frame ========")


#####
print("\n\nInstantiating a FrameTree")
print("------------------------")

from py3dframe import FrameTree, Frame
import numpy as np

# Create a FrameTree instance
frame_tree = FrameTree()
print("A new FrameTree has been created.")

# Expected Output:
# ----------------
# A new FrameTree has been created.


#####
print("\n\nCreating Frames and Adding Them to the Tree")
print("------------------------")

# Create frames with specific origins and identity rotation matrices

# Frame1: Translation [1, 0, 0]
frame1 = Frame(origin=np.array([1, 0, 0]).reshape((3, 1)),
               rotation_matrix=np.eye(3))
frame_tree.add_frame(frame1, "frame1")
print("Added 'frame1' with origin [1, 0, 0].")

# Frame2: Translation [0, 2, 0] relative to frame1
frame2 = Frame(origin=np.array([0, 2, 0]).reshape((3, 1)),
               rotation_matrix=np.eye(3))
frame_tree.add_frame(frame2, "frame2", parent="frame1")
print("Added 'frame2' with origin [0, 2, 0], as a child of 'frame1'.")

# Frame3: Translation [0, 0, 3] in world coordinates
frame3 = Frame(origin=np.array([0, 0, 3]).reshape((3, 1)),
               rotation_matrix=np.eye(3))
frame_tree.add_frame(frame3, "frame3")
print("Added 'frame3' with origin [0, 0, 3] in the world frame.")

# Print the tree structure
print("\nCurrent FrameTree structure:")
print(frame_tree)

# Expected Output:
# ----------------
# world
# ├── frame1
# │   └── frame2
# └── frame3


#####
print("\n\nQuerying Frames from the Tree")
print("------------------------")

# List all frames
print(f"Frames in the tree: {frame_tree.names}")
# Expected Output:
# Frames in the tree: ['frame1', 'frame2', 'frame3']

# Get a frame directly
f1 = frame_tree.get_frame("frame1")
print(f"Retrieved 'frame1': Origin = {f1.origin.flatten()}")
# Expected Output:
# Retrieved 'frame1': Origin = [1. 0. 0.]


#####
print("\n\nTransforming Points Between Frames")
print("------------------------")

# Define a point in world coordinates
world_point = np.array([[3], [3], [0]])
print(f"World point: {world_point.flatten()}")
# Expected Output:
# World point: [3 3 0]

# Convert world point to frame2's local coordinates
point_in_frame2 = frame_tree.from_frame_to_frame(input_name=None, output_name="frame2", point=world_point)
print(f"Point in 'frame2' coordinates: {point_in_frame2.flatten()}")
# Expected Output (depends on transforms):
# Point in 'frame2' coordinates: [2. 1. 0.]

# Convert it back to world coordinates
point_back_to_world = frame_tree.from_frame_to_frame(input_name="frame2", output_name=None, point=point_in_frame2)
print(f"Converted back to world coordinates: {point_back_to_world.flatten()}")
# Expected Output:
# Converted back to world coordinates: [3. 3. 0.]


#####
print("\n\nModifying the FrameTree")
print("------------------------")

# Rename frame1 to 'base_frame'
frame_tree.set_name("frame1", "base_frame")
print("Renamed 'frame1' to 'base_frame'.")

# Print updated tree
print("\nFrameTree after renaming 'frame1' to 'base_frame':")
print(frame_tree)
# Expected Output:
# world
# ├── base_frame
# │   └── frame2
# └── frame3

# Change parent of frame3 to 'base_frame'
frame_tree.set_parent("frame3", "frame2")
print("Set 'base_frame' as the parent of 'frame3'.")

# Print updated tree
print("\nFrameTree after setting 'frame2' as parent of 'frame3':")
print(frame_tree)
# Expected Output:
# world
# └── base_frame
#     └── frame2
#         └── frame3

# Remove frame2 from the tree
frame_tree.remove_frame("frame2")
print("'frame2' has been removed from the tree.")

# Print updated tree
print("\nFrameTree after removing 'frame2':")
print(frame_tree)
# Expected Output:
# world
# ├── base_frame
# └── frame3

# Display remaining frames
print(f"Remaining frames: {frame_tree.names}")
# Expected Output:
# Remaining frames: ['base_frame', 'frame3']


#####
print("\n\nClearing the FrameTree")
print("------------------------")

# Clear the tree
frame_tree.clear()
print("FrameTree has been cleared.")

# Print cleared tree
print("\nFrameTree after clearing:")
print(frame_tree)
# Expected Output:
# world

# Display frames after clearing
print(f"Frames after clearing: {frame_tree.names}")
# Expected Output:
# Frames after clearing: []
