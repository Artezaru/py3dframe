# Copyright 2025 Artezaru
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from .frame import Frame
from .frame_transform import FrameTransform

from typing import Optional, List
import numpy
import json

class FrameTree(object):
    r"""
    A class to manage a tree/system of Frame objects better than using individual Frame objects with parent/child relationships.
    
    .. warning::
    
        All the connected frames are stored in a tree structure must have ``parent=None`` because the :class:`FrameTree` manages the relationships.
        Avoid modifying the ``parent`` attribute of the Frame objects connected to a FrameTree to prevent inconsistencies.
        
    By default, the root frame is considered the canonical frame of :math:`\mathcal{R}^3` and is associated with the name 'root'.
    
    .. code-block:: console
    
        Root_Frame (Canonical Frame)
        ├── Child_Frame_1
        │   ├── Grandchild_Frame_1
        │   └── Grandchild_Frame_2
        └── Child_Frame_2
            └── Grandchild_Frame_3
            
    parameters
    ----------
    root_frame : Optional[Frame], optional
        The root Frame of the FrameTree. If None, the canonical Frame of :math:`\mathcal{R}^3` is used. Default is None.
            
    """
    def __init__(self, root_frame: Optional[Frame] = None) -> None:
        if root_frame is not None and not isinstance(root_frame, Frame):
            raise TypeError("root_frame must be a Frame object or None.")
        if root_frame is None:
            root_frame = Frame.canonical()
        
        self._names = {'root'}
        self._frames = {'root': root_frame}
        self._parents = {'root': None}
        self._children = {'root': []}
        
        
    def set_root_frame(self, frame: Frame) -> None:
        r"""
        Set the root Frame of the FrameTree.
        
        Parameters
        ----------
        frame : Frame
            The Frame object to set as the root frame.
        
        Returns
        -------
        None
        
        
        Examples
        --------
        
        .. code-block:: python
        
            from py3dframe import Frame, FrameTree
            
            # Create a FrameTree
            frame_tree = FrameTree()
            
            # Create a new root frame
            new_root_frame = Frame(name="New_Root_Frame")
            
            # Set the new root frame
            frame_tree.set_root_frame(new_root_frame)
        
        """
        if not isinstance(frame, Frame):
            raise TypeError("frame must be a Frame object.")
        
        self._frames['root'] = frame
        
        
    def replace_frame(self, name: str, frame: Frame) -> None:
        r"""
        Replace a Frame in the FrameTree with another Frame.
        
        The new Frame will inherit the parent and children of the replaced Frame.
        
        Parameters
        ----------
        name : str
            The name of the frame to replace.
        frame : Frame
            The new Frame object to replace the existing frame.
        
        Returns
        -------
        None
        
        
        Examples
        --------
        
        .. code-block:: python
        
            from py3dframe import Frame, FrameTree
            
            # Create some frames
            root_frame = Frame(name="Root_Frame")
            child_frame = Frame(name="Child_Frame")
            
            # Create a FrameTree and add frames
            frame_tree = FrameTree()
            frame_tree.add_frame(name="Root_Frame", frame=root_frame)
            frame_tree.add_frame(name="Child_Frame", frame=child_frame, parent_name="Root_Frame")
            
            # Create a new frame to replace the existing one
            new_child_frame = Frame(name="New_Child_Frame")
            
            # Replace the existing frame
            frame_tree.replace_frame(name="Child_Frame", frame=new_child_frame)
        
        """
        if not isinstance(name, str):
            raise TypeError("name must be a string.")
        if name not in self._names:
            raise ValueError(f"The frame '{name}' does not exist in the FrameTree.")
        
        if not isinstance(frame, Frame):
            raise TypeError("frame must be a Frame object.")
        if frame.parent is not None:
            raise ValueError("The new frame must have parent=None. The FrameTree manages the parent/child relationships to avoid inconsistencies.")
        
        # remove the current relationships
        current_frame = self._frames[name]
        current_frame.parent = None
        
        # Get the parent and children of the existing frame
        parent_name = self._parents[name]
        child_names = self._children[name]
        
        # Set the parent of the new frame
        parent_frame = self._frames[parent_name]
        frame.parent = parent_frame
        
        # Replace the frame in the tree
        self._frames[name] = frame
        
        # Update the children to have the new frame as their parent
        for child_name in child_names:
            child_frame = self._frames[child_name]
            child_frame.parent = frame
        
        
    def connect_frame(self, name: str, frame: Frame, parent_name: Optional[str] = None) -> None:
        r"""
        Add a Frame to the FrameTree.
        
        If the parent_name is None (equivalent to 'root'), the frame is added as the root frame (Canonical frame of :math:`\mathcal{R}^3`). Otherwise, it is added as a child of the specified parent frame.
        
        .. seealso::
        
            :meth:`FrameTree.disconnect_frame` : Remove a frame from the FrameTree.
        
        Parameters
        ----------
        name : str
            The name of the frame to add.
        frame : Frame
            The Frame object to add.
        parent_name : Optional[str], optional
            The name of the parent frame. If None, the frame is added as the root frame. Default is None.
        
        Returns
        -------
        None
        
        
        Examples
        --------
        
        .. code-block:: python
        
            from py3dframe import Frame, FrameTree
            
            # Create some frames
            root_frame = Frame(name="Root_Frame")
            child_frame_1 = Frame(name="Child_Frame_1")
            child_frame_2 = Frame(name="Child_Frame_2")
            grandchild_frame_1 = Frame(name="Grandchild_Frame_1")
            
            # Create a FrameTree and add frames
            frame_tree = FrameTree()
            frame_tree.add_frame(name="Root_Frame", frame=root_frame)
            frame_tree.add_frame(name="Child_Frame_1", frame=child_frame_1, parent_name="Root_Frame")
            frame_tree.add_frame(name="Child_Frame_2", frame=child_frame_2, parent_name="Root_Frame")
            frame_tree.add_frame(name="Grandchild_Frame_1", frame=grandchild_frame_1, parent_name="Child_Frame_1")
        
        """
        # Input validation
        if not isinstance(frame, Frame):
            raise TypeError("frame must be a Frame object.")
        if frame.parent is not None:
            raise ValueError("The frame to be added must have parent=None. The FrameTree manages the parent/child relationships to avoid inconsistencies.")
        
        if not isinstance(name, str):
            raise TypeError("name must be a string.")
        if name in self._names:
            raise ValueError(f"A frame with the name '{name}' already exists in the FrameTree.")
        
        if parent_name is None:
            parent_name = 'root'
        if not isinstance(parent_name, str):
            raise TypeError("parent_name must be a string or None.")
        if parent_name not in self._names:
            raise ValueError(f"The parent frame '{parent_name}' does not exist in the FrameTree.")
        
        # Set the parent frame
        parent_frame = self._frames[parent_name]
        frame.parent = parent_frame
        
        # Add the frame to the tree
        self._frames[name] = frame
        self._parents[name] = parent_name
        self._children[name] = []
        self._names.add(name)
        self._children[parent_name].append(name)
        
        
    def disconnect_frame(self, name: str, recursive: bool = True) -> None:
        r"""
        Remove a Frame from the FrameTree.
        
        The frame is removed from the FrameTree and its parent is set to None.
        
        .. seealso::
        
            :meth:`FrameTree.connect_frame` : Add a frame to the FrameTree.
            
        .. warning::
        
            The root frame (Canonical frame of :math:`\mathcal{R}^3`) cannot be removed from the FrameTree.
        
        Parameters
        ----------
        name : str
            The name of the frame to remove.
            
        recursive : bool, optional
            If True, all child frames of the specified frame are also removed recursively. If False, the child frames are not removed and will have their parent set to the parent of the removed frame. Default is True.
        
        Returns
        -------
        None
        
        
        Examples
        --------
        
        .. code-block:: python
        
            from py3dframe import Frame, FrameTree
            
            # Create some frames
            root_frame = Frame(name="Root_Frame")
            child_frame_1 = Frame(name="Child_Frame_1")
            
            # Create a FrameTree and add frames
            frame_tree = FrameTree()
            frame_tree.add_frame(name="Root_Frame", frame=root_frame)
            frame_tree.add_frame(name="Child_Frame_1", frame=child_frame_1, parent_name="Root_Frame")
            
            # Disconnect a frame
            frame_tree.disconnect_frame(name="Child_Frame_1", recursive=True)
        
        """
        # Input validation
        if not isinstance(name, str):
            raise TypeError("name must be a string.")
        if name not in self._names:
            raise ValueError(f"The frame '{name}' does not exist in the FrameTree.")
        if name == 'root':
            raise ValueError("The root frame cannot be removed from the FrameTree.")
        
        if not isinstance(recursive, bool):
            raise TypeError("recursive must be a boolean.")
        
        # Remove child frames recursively if specified
        child_names = self.get_child_names(name)
        
        if recursive:
            for child_name in child_names:
                self.disconnect_frame(child_name, recursive=True)
        else:
            for child_name in child_names:
                # Set the parent of the child frame to the parent of the removed frame
                child_frame = self._frames[child_name]
                parent_name = self._parents[name]
                parent_frame = self._frames[parent_name]
                child_frame.parent = parent_frame
                
                # Update the tree structure
                self._parents[child_name] = parent_name
                self._children[parent_name].append(child_name)
                
        # Set the parent of the frame to None
        frame = self._frames[name]
        frame.parent = None
        
        # Remove the frame from the tree
        parent_name = self._parents[name]
        self._children[parent_name].remove(name)
        del self._frames[name]
        del self._parents[name]
        del self._children[name]
        self._names.remove(name)
        
        
    def move_frame(self, name: str, new_parent_name: Optional[str] = None) -> None:
        r"""
        Move a Frame to a new parent frame in the FrameTree.
        
        The frame's parent is updated to the new parent frame.
        
        Parameters
        ----------
        name : str
            The name of the frame to move.
            
        new_parent_name : Optional[str], optional
            The name of the new parent frame. If None, the frame is moved to be a child of the root frame. Default is None.
        
        Returns
        -------
        None
        
        
        Examples
        --------
        
        .. code-block:: python
        
            from py3dframe import Frame, FrameTree
            
            # Create some frames
            root_frame = Frame(name="Root_Frame")
            child_frame_1 = Frame(name="Child_Frame_1")
            child_frame_2 = Frame(name="Child_Frame_2")
            
            # Create a FrameTree and add frames
            frame_tree = FrameTree()
            frame_tree.add_frame(name="Root_Frame", frame=root_frame)
            frame_tree.add_frame(name="Child_Frame_1", frame=child_frame_1, parent_name="Root_Frame")
            frame_tree.add_frame(name="Child_Frame_2", frame=child_frame_2, parent_name="Root_Frame")
            
            # Move a frame to a new parent
            frame_tree.move_frame(name="Child_Frame_1", new_parent_name="Child_Frame_2")
        
        """
        # Input validation
        if not isinstance(name, str):
            raise TypeError("name must be a string.")
        if name not in self._names:
            raise ValueError(f"The frame '{name}' does not exist in the FrameTree.")
        if name == 'root':
            raise ValueError("The root frame cannot be moved in the FrameTree.")
        
        if new_parent_name is None:
            new_parent_name = 'root'
        if not isinstance(new_parent_name, str):
            raise TypeError("new_parent_name must be a string or None.")
        if new_parent_name not in self._names:
            raise ValueError(f"The new parent frame '{new_parent_name}' does not exist in the FrameTree.")
        
        # Update the parent frame
        frame = self._frames[name]
        old_parent_name = self._parents[name]
        new_parent_frame = self._frames[new_parent_name]
        frame.parent = new_parent_frame
        
        # Update the tree structure
        self._parents[name] = new_parent_name
        
        # Remove from old parent's children
        self._children[old_parent_name].remove(name)
        
        # Add to new parent's children
        self._children[new_parent_name].append(name)
        
    
    def get_frame(self, name: str) -> Frame:
        r"""
        Get a Frame from the FrameTree by its name.
        
        Parameters
        ----------
        name : str
            The name of the frame to retrieve.
        
        Returns
        -------
        Frame
            The Frame object with the specified name.
        
        
        Examples
        --------
        
        .. code-block:: python
        
            from py3dframe import Frame, FrameTree
            
            # Create some frames
            root_frame = Frame(name="Root_Frame")
            
            # Create a FrameTree and add frames
            frame_tree = FrameTree()
            frame_tree.add_frame(name="Root_Frame", frame=root_frame)
            
            # Get a frame by name
            retrieved_frame = frame_tree.get_frame(name="Root_Frame")
        
        """
        # Input validation
        if not isinstance(name, str):
            raise TypeError("name must be a string.")
        if name not in self._names:
            raise ValueError(f"The frame '{name}' does not exist in the FrameTree.")
        
        return self._frames[name]
    
    
    def get_child_names(self, name: str) -> List[str]:
        r"""
        Get the names of the child frames of a specified frame in the FrameTree.
        
        Parameters
        ----------
        name : str
            The name of the frame whose child names to retrieve.
        
        Returns
        -------
        List[str]
            A list of names of the child frames.
        
        
        Examples
        --------
        
        .. code-block:: python
        
            from py3dframe import Frame, FrameTree
            
            # Create some frames
            root_frame = Frame(name="Root_Frame")
            child_frame_1 = Frame(name="Child_Frame_1")
            
            # Create a FrameTree and add frames
            frame_tree = FrameTree()
            frame_tree.add_frame(name="Root_Frame", frame=root_frame)
            frame_tree.add_frame(name="Child_Frame_1", frame=child_frame_1, parent_name="Root_Frame")
            
            # Get child frame names
            child_names = frame_tree.get_child_names(name="Root_Frame")
        
        """
        # Input validation
        if not isinstance(name, str):
            raise TypeError("name must be a string.")
        if name not in self._names:
            raise ValueError(f"The frame '{name}' does not exist in the FrameTree.")
        
        return self._children[name].copy()
    
    
    def get_parent_name(self, name: str) -> Optional[str]:
        r"""
        Get the name of the parent frame of a specified frame in the FrameTree.
        
        Parameters
        ----------
        name : str
            The name of the frame whose parent name to retrieve.
        
        Returns
        -------
        Optional[str]
            The name of the parent frame, or None if the frame is the root frame.
        
        
        Examples
        --------
        
        .. code-block:: python
        
            from py3dframe import Frame, FrameTree
            
            # Create some frames
            root_frame = Frame(name="Root_Frame")
            child_frame_1 = Frame(name="Child_Frame_1")
            
            # Create a FrameTree and add frames
            frame_tree = FrameTree()
            frame_tree.add_frame(name="Root_Frame", frame=root_frame)
            frame_tree.add_frame(name="Child_Frame_1", frame=child_frame_1, parent_name="Root_Frame")
            
            # Get parent frame name
            parent_name = frame_tree.get_parent_name(name="Child_Frame_1")
        
        """
        # Input validation
        if not isinstance(name, str):
            raise TypeError("name must be a string.")
        if name not in self._names:
            raise ValueError(f"The frame '{name}' does not exist in the FrameTree.")
        
        return self._parents[name]
    
    
    def list_frames(self) -> List[str]:
        r"""
        List all frame names in the FrameTree.
        
        Returns
        -------
        List[str]
            A list of all frame names in the FrameTree.
        
        
        Examples
        --------
        
        .. code-block:: python
        
            from py3dframe import Frame, FrameTree
            
            # Create some frames
            root_frame = Frame(name="Root_Frame")
            child_frame_1 = Frame(name="Child_Frame_1")
            
            # Create a FrameTree and add frames
            frame_tree = FrameTree()
            frame_tree.add_frame(name="Root_Frame", frame=root_frame)
            frame_tree.add_frame(name="Child_Frame_1", frame=child_frame_1, parent_name="Root_Frame")
            
            # List all frame names
            frame_names = frame_tree.list_frames()
        
        """
        return list(self._names)
    
    
    def get_transform(self, input_frame: Optional[str] = None, output_frame: Optional[str] = None) -> FrameTransform:
        r"""
        Get the FrameTransform between two frames in the FrameTree.
        
        Parameters
        ----------
        input_frame : Optional[str], optional
            The name of the input frame. If None, the root frame is used. Default is None.
        output_frame : Optional[str], optional
            The name of the output frame. If None, the root frame is used. Default is None.
        
        Returns
        -------
        FrameTransform
            The FrameTransform object representing the transformation from the input frame to the output frame.
        
        
        Examples
        --------
        
        .. code-block:: python
        
            from py3dframe import Frame, FrameTree
            
            # Create some frames
            root_frame = Frame(name="Root_Frame")
            child_frame_1 = Frame(name="Child_Frame_1")
            
            # Create a FrameTree and add frames
            frame_tree = FrameTree()
            frame_tree.add_frame(name="Root_Frame", frame=root_frame)
            frame_tree.add_frame(name="Child_Frame_1", frame=child_frame_1, parent_name="Root_Frame")
            
            # Get the FrameTransform between two frames
            transform = frame_tree.get_transform(input_frame="Child_Frame_1", output_frame="Root_Frame")
        
        """
        if input_frame is None:
            input_frame = 'root'
        if output_frame is None:
            output_frame = 'root'
        
        input_f = self.get_frame(input_frame)
        output_f = self.get_frame(output_frame)
        
        return FrameTransform(input_frame=input_f, output_frame=output_f)
    
    
    def transform(self, input_frame: Optional[str] = None, output_frame: Optional[str] = None, *, point: Optional[numpy.ndarray] = None, vector: Optional[numpy.ndarray] = None) -> numpy.ndarray:
        r"""
        Transform a point or a vector from the input frame to the output frame.

        If the point is provided, the method will return the coordinates of the point in the output frame.
        If the vector is provided, the method will return the coordinates of the vector in the output frame.

        Several points / vectors can be transformed at the same time by providing a 2D numpy array with shape (3, N).

        If both the point and the vector are provided, the method will raise a ValueError.
        If neither the point nor the vector is provided, the method will return None.

        In the convention 0:

        .. math::

            X_{\text{output_frame}} = R^{-1} * (X_{\text{input_frame}} - T)

        .. math::

            V_{\text{output_frame}} = R^{-1} * V_{\text{input_frame}}
            
        .. seealso::
        
            :class:`FrameTransform` : Class to represent transformations between two frames and apply them to points and vectors.
            
        Parameters
        ----------
        point : Optional[array_like], optional
            The coordinates of the point in the input frame with shape (3, N). Default is None.
        
        vector : Optional[array_like], optional
            The coordinates of the vector in the input frame with shape (3, N). Default is None.

        Returns
        -------
        numpy.ndarray
            The coordinates of the point or the vector in the output frame with shape (3, N).

        Raises
        ------
        ValueError
            If the point or the vector is not provided.
            If point and vector are both provided.

        Examples
        --------
        
        Lets create a FrameTree object with the global frame as input frame and a local frame as output frame.

        .. code-block:: python

            import numpy as np
            from py3dframe import Frame, FrameTransform

            frame_E = Frame.canonical() # Input frame - Global frame
            frame_F = Frame.from_axes(origin=[1, 2, 3], x_axis=[1, 0, 0], y_axis=[0, 1, 0], z_axis=[0, 0, 1]) # Output frame - Local frame

            frame_tree = FrameTree()
            frame_tree.connect_frame(name="E", frame=frame_E)
            frame_tree.connect_frame(name="F", frame=frame_F, parent_name="E")

        The FrameTree object can be used to transform points or vectors from the input frame to the output frame (same as FrameTransform).

        .. code-block:: python

            X_i = np.array([1, 2, 3]).reshape((3, 1)) # Point in the input frame coordinates
            X_o = frame_tree.transform(input_frame="E", output_frame="F", point=X_i) # Transform the point to the output frame coordinates
            print(X_o)
            # Output: [[0.] [0.] [0.]]

            V_i = np.array([1, 0, 0]).reshape((3, 1)) # Vector in the input frame coordinates
            V_o = frame_tree.transform(input_frame="E", output_frame="F", vector=V_i) # Transform the vector to the output frame coordinates
            print(V_o)
            # Output: [[1.] [0.] [0.]]
            
        """
        transform = self.get_tranform(input_frame=input_frame, output_frame=output_frame)
        return transform.transform(point=point, vector=vector)
    

    def print_tree(self) -> None:
        r"""
        Print the FrameTree structure.
        
        Returns
        -------
        None
        
        
        Examples
        --------
        
        .. code-block:: python
        
            from py3dframe import Frame, FrameTree
            
            # Create some frames
            root_frame = Frame(name="Root_Frame")
            child_frame_1 = Frame(name="Child_Frame_1")
            grandchild_frame_1 = Frame(name="Grandchild_Frame_1")
            
            # Create a FrameTree and add frames
            frame_tree = FrameTree()
            frame_tree.add_frame(name="Root_Frame", frame=root_frame)
            frame_tree.add_frame(name="Child_Frame_1", frame=child_frame_1, parent_name="Root_Frame")
            frame_tree.add_frame(name="Grandchild_Frame_1", frame=grandchild_frame_1, parent_name="Child_Frame_1")
            
            # Print the FrameTree structure
            frame_tree.print_tree()
        
        """
        def _print_subtree(name: str, prefix: Optional[str] = None) -> None:
            if prefix is None:
                prefix = ""
                print(prefix + name)
            else:
                print(prefix[:-4] + "├── " + name)
            children = self.get_child_names(name)
            for i, child_name in enumerate(children):
                _print_subtree(child_name, prefix + "│   ")
        
        _print_subtree('root')
        
        
    def __len__(self) -> int:
        return len(self._names)
    
    def __contains__(self, name: str) -> bool:
        return name in self._names
    
    def __repr__(self) -> str:
        return f"FrameTree(num_frames={len(self)})"
    
    def __str__(self) -> str:
        return f"FrameTree with {len(self)} frames."
    
    def __getitem__(self, name: str) -> Frame:
        return self.get_frame(name)
    
    def __setitem__(self, name: str, frame: Frame) -> None:
        self.connect_frame(name, frame)
        
    def __delitem__(self, name: str) -> None:
        self.disconnect_frame(name)
    
    def __bool__(self) -> bool:
        return len(self) > 0
    
    
    def to_dict(self) -> dict:
        r"""
        Serialize the FrameTree to a dictionary.
        
        The dictionary contains the serialized Frame objects and their relationships.
        
        .. seealso::
        
            - :meth:`Frame.to_dict` : Serialize a Frame object to a dictionary.
            - :meth:`FrameTree.from_dict` : Deserialize a FrameTree from a dictionary.
            
        The serialization have the following structure:
        
        .. code-block:: python
        
            {
                "frames": {
                    "frame_name_1": { ... Frame serialized dictionary ... },
                    "frame_name_2": { ... Frame serialized dictionary ... },
                    ...
                },
                "parents": {
                    "frame_name_1": "parent_frame_name",
                    "frame_name_2": "parent_frame_name",
                    ...
                }
            }
            
        Returns
        -------
        dict
            The serialized FrameTree as a dictionary.
            
        """
        data = {
            "frames": {},
            "parents": {},
        }
        for name in self._names:
            frame = self._frames[name]
            data["frames"][name] = frame.to_dict()
            parent_name = self._parents[name]
            data["parents"][name] = parent_name
        
        return data
    
    @classmethod
    def from_dict(cls, data: dict) ->  FrameTree:
        r"""
        Deserialize a FrameTree from a dictionary.
        
        The dictionary must contain the serialized Frame objects and their relationships.
        
        .. seealso::
        
            - :meth:`Frame.from_dict` : Deserialize a Frame object from a dictionary.
            - :meth:`FrameTree.to_dict` : Serialize a FrameTree to a dictionary.
        
        Parameters
        ----------
        data : dict
            The serialized FrameTree as a dictionary.
        
        Returns
        -------
        FrameTree
            The deserialized FrameTree object.
            
        """
        frame_tree = cls()
        
        frames_data = data["frames"]
        parents_data = data["parents"]
        
        # First, create all frames without parents
        for name, frame_data in frames_data.items():           
            if name == 'root':
                frame = Frame.from_dict(frame_data)
                frame_tree.set_root_frame(frame)
            else:
                frame = Frame.from_dict(frame_data)
                frame_tree._frames[name] = frame
                frame_tree._names.add(name)
                frame_tree._children[name] = []
                frame_tree._parents[name] = 'root'
                frame_tree._children['root'].append(name)
        
        # Then, connect frames to their parents
        for name, parent_name in parents_data.items():
            if name != 'root':
                frame_tree.move_frame(name, new_parent_name=parent_name)
            
        return frame_tree
    
    
    def to_json(self, filename: str) -> None:
        """
        Save the FrameTree object to a JSON file.

        .. seealso::

            - method :meth:`to_dict` to save the FrameTree object to a dictionary.
            
        Parameters
        ----------
        filename : str
            The name of the JSON file.
        
        Returns
        -------
        None
        """
        data = self.to_dict()
        with open(filename, 'w') as f:
            json.dump(data, f, indent=4)
            
            
    @classmethod
    def from_json(cls, filename: str) -> FrameTree:
        """
        Load a FrameTree object from a JSON file.

        .. seealso::

            - method :meth:`from_dict` to load the FrameTree object from a dictionary.
            
        Parameters
        ----------
        filename : str
            The name of the JSON file.
            
        Returns
        -------
        FrameTree
            The loaded FrameTree object.
            
        """
        with open(filename, 'r') as f:
            data = json.load(f)
        frame_tree = cls.from_dict(data)
        return frame_tree