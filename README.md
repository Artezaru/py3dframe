# py3dframe

## Description

3D Frame of Reference managing tools. Only orthogonal and right-handed frames are supported.

# Usage

This section will guide you through the basic usage of the `py3dframe` package.

## Construct a right-handed frame

To create a frame, you can provide the origin and the axes of the frame as follows:

```python
import numpy as np
from py3dframe import Frame

origin = np.array([1, 2, 3])
x_axis = np.array([1, 0, 0])
y_axis = np.array([0, 1, 0])
z_axis = np.array([0, 0, 1])

frame = Frame.from_axes(origin=origin, x_axis=x_axis, y_axis=y_axis, z_axis=z_axis)
```

You can also construct a frame from a rotation and a translation using one of the 8 possible conventions:

| Index | Formula |
|-------|---------|
| 0 | **X_E** = **R** ⋅ **X_F** + **T** |
| 1 | **X_E** = **R** ⋅ **X_F** − **T** |
| 2 | **X_E** = **R** (**X_F** + **T**) |
| 3 | **X_E** = **R** (**X_F** − **T**) |
| 4 | **X_F** = **R** ⋅ **X_E** + **T** |
| 5 | **X_F** = **R** ⋅ **X_E** − **T** |
| 6 | **X_F** = **R** (**X_E** + **T**) |
| 7 | **X_F** = **R** (**X_E** − **T**) |

Where **X_E** is the point expressed in the parent (or global) frame E, **X_F** is the point expressed in the child (or local) frame F, **R** is the rotation matrix and **T** is the translation vector.

```python
from py3dframe import Frame, Rotation

rotation = Rotation.from_euler('xyz', [0, 0, 0], degrees=True)
translation = np.array([1, 2, 3]).reshape(3, 1)

frame = Frame.from_rotation(translation=translation, rotation=rotation, convention=0)
```

You can check the coordinates of the frame by plotting it into a 3D plot:

```python
import matplotlib.pyplot as plt

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

frame.plot(ax)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

plt.show()
```

## Construct a system of frames

Let’s consider a person walking in a train.
The person is represented by a frame F and the train is represented by a frame E.
It is possible to represent the position of the person in the train by defining the frame F in the frame E coordinates.

```python
from py3dframe import Frame, FrameTransform

rotation = Rotation.from_euler('xyz', [0, 0, 0], degrees=True)
translation = np.array([1, 2, 3]).reshape(3, 1)

frame_E = Frame.from_rotation(translation=translation, rotation=rotation, convention=0)

rotation = Rotation.from_euler('xyz', [0, 0, 0], degrees=True)
translation = np.array([0, 0, 0]).reshape(3, 1)

frame_F = Frame.from_rotation(translation=translation, rotation=rotation, convention=0, parent=frame_E)
```

In this case, when the frame E moves, the frame F moves with it.

## Transformation between frames

The transformation between two frames can be computed using the `FrameTransform` class:

```python
from py3dframe import Frame, FrameTransform

rotation = Rotation.from_euler('xyz', [0, 0, 0], degrees=True)
translation = np.array([1, 2, 3]).reshape(3, 1)

frame_E = Frame.from_rotation(translation=translation, rotation=rotation, convention=0)

rotation = Rotation.from_euler('xyz', [0, 0, 0], degrees=True)
translation = np.array([0, 0, 0]).reshape(3, 1)

frame_F = Frame.from_rotation(translation=translation, rotation=rotation, convention=0, parent=frame_E)

transform = FrameTransform(input_frame=frame_E, output_frame=frame_F)

print(transform.translation)
print(transform.rotation.as_euler('xyz', degrees=True))
```

This object can be used to transform points or vectors from one frame to another:

```python
point_E = np.array([1, 2, 3]).reshape(3, 1)
point_F = transform.transform(point=point_E) # In convention 0: pE = R * pF + T
point_E = transform.inverse_transform(point=point_F)

vector_E = np.array([1, 2, 3]).reshape(3, 1)
vector_F = transform.transform(vector=vector_E) # In convention 0: vE = R * vF
vector_E = transform.inverse_transform(vector=vector_F)
```

When the frame E moves, the `transform` object will automatically update the transformation between the two frames.

## Authors

- Artezaru <artezaru.github@proton.me>

- **Git Plateform**: https://github.com/Artezaru/py3dframe.git
- **Online Documentation**: https://Artezaru.github.io/py3dframe

## Installation

Install with pip

```
pip install py3dframe
```

```
pip install git+https://github.com/Artezaru/py3dframe.git
```

Clone with git

```
git clone https://github.com/Artezaru/py3dframe.git
```

## License

Copyright 2025 Artezaru

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
