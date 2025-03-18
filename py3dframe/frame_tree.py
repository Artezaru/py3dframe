from __future__ import annotations
from typing import Optional, List, Dict
import numpy
import uuid
import json
from copy import copy as copycopy
from .frame import Frame
from .inverse_frame import inverse_frame


class FrameTree:
    r"""
    The FrameTree class is a container for Frame instances.

    Each frame is stored with a name and can be linked to another frame.
    If a frame is linked to another frame, the global reference of the child frame became the parent frame.
    By default, the global frame is the default frame of the FrameTree.

    .. warning::

        If you use the FrameTree to store the frames, please do not set the parent of the frame manually.
        Otherwise, the FrameTree will not work correctly and errors will occurs.
        If you prefer manage the parent of the frame manually, use the :class:`py3dframe.Frame` class directly without the FrameTree.

    Examples
    --------
    >>> from py3dframe import FrameTree, Frame
    >>> # Create a FrameTree
    >>> frame_tree = FrameTree()
    >>> # Create several frames
    >>> frame1, frame2, frame3 = Frame(), Frame(), Frame()
    >>> # Add the frames to the FrameTree
    >>> frame_tree.add_frame(frame, "frame1")
    >>> frame_tree.add_frame(frame, "frame2", parent="frame1")
    >>> frame_tree.add_frame(frame, "frame3")

    The output of the code above is a FrameTree with three frames: frame1, frame2, and frame3 with the following structure:

    .. code-block:: console

        global
        ├── frame1
        │   └── frame2
        └── frame3
    """

    def __init__(self) -> None:
        self.uuid_to_name = {} # key: uuid, value: name
        self.name_to_uuid = {} # key: name, value: uuid
        self._frames = {} # key: uuid, value: Frame 
        # Warning self._frames (uuid, Frame) differs from self.frames (name, Frame)



    # Property getters and setters
    @property
    def names(self) -> List[str]:
        """Get a list of the names of the frames."""
        return list(self.name_to_uuid.keys())
    

    @property
    def frames(self) -> Dict[str, Frame]:
        """Get a dictionary of the names of the frames and the frames."""
        return {name: self.get_frame(name) for name in self.names}
    


    @property
    def num_frames(self) -> int:
        """Get the number of frames in the FrameTree."""
        return self.__len__()
    


    # Private methods
    def _exist_name(self, name: str) -> bool:
        """Check if a frame with the given name exists."""
        if not isinstance(name, str):
            raise TypeError("name must be a string.")
        frame_uuid = self.name_to_uuid.get(name, None)
        return frame_uuid is not None



    def _identity_frame(self) -> Frame:
        """Return the default identity frame"""
        return Frame()


    
    def __len__(self) -> int:
        """Get the number of frames in the FrameTree."""
        return len(self._frames)

    
    def __getitem__(self, name: str) -> Frame:
        """Get a frame from the FrameTree."""
        return self.get_frame(name, copy=False)

    

    def __delitem__(self, name: str) -> None:
        """Remove a frame from the FrameTree."""
        self.remove_frame(name)



    def __setitem__(self, name: str, frame: Frame) -> None:
        """Add a frame to the FrameTree."""
        self.add_frame(frame, name, copy=False)


    
    def __repr__(self) -> str:
        """
        Return a string representation of the FrameTree in a tree structure.
        """
        def build_tree(name, prefix="", is_last=True):
            """Recursively build the tree structure."""
            connector = "└── " if is_last else "├── "
            lines = [f"{prefix}{connector}{name}"]
            children = self.get_children_names(name)
            for index, child in enumerate(sorted(children)):
                is_child_last = index == (len(children) - 1)
                child_prefix = prefix + ("    " if is_last else "│   ")
                lines.extend(build_tree(child, child_prefix, is_child_last))
            return lines

        # Create the parents dictionary
        parents = {name: self.get_parent_name(name) for name in self.names}

        # Build the tree structure
        lines = ["global"]
        roots = [name for name, parent in parents.items() if parent is None]
        for index, root in enumerate(sorted(roots)):
            is_last_root = index == (len(roots) - 1)
            lines.extend(build_tree(root, "", is_last_root))

        return "\n".join(lines)



    # Get and set methods 
    def get_frame(self, name: str, copy: bool = False) -> Frame:
        r"""
        Get a frame from the FrameTree.

        If the ``copy`` parameter is False, the method is equivalent to the following code:

        >>> frame = frame_binder[name]

        Parameters
        ----------
        name : str
            The name of the frame to get.
        
        copy : bool, optional
            Get a copy of the frame. Defaults to False.
        
        Returns
        -------
        Frame
            The frame with the given name.
        
        Raises
        -------
        TypeError
            If an argument is not the correct type.
        ValueError
            If the frame name does not exist.
        """
        # Check the types of the arguments
        if not isinstance(copy, bool):
            raise TypeError("copy must be a boolean.")
        
        # Get the uuid of the frame
        if not self._exist_name(name):
            raise ValueError(f"Frame with name '{name}' does not exist.")

        # Get the frame from the FrameTree
        frame_uuid = self.name_to_uuid[name]
        if copy:
            return copycopy(self._frames[frame_uuid])
        return self._frames[frame_uuid]
    


    def set_name(self, old_name: str, new_name: str) -> None:
        """
        Set a new name for a frame in the FrameTree.

        Parameters
        ----------
        old_name : str
            The old name of the frame.
        
        new_name : str
            The new name of the frame.
        
        Raises
        ------
        ValueError
            If the frame name does not exist.
            If the new frame name already exists.
        """
        # Check if the frame name already exists
        if not self._exist_name(old_name):
            raise ValueError(f"Frame with name '{old_name}' does not exist.")
        if self._exist_name(new_name):
            raise ValueError(f"Frame with name '{new_name}' already exists.")
        
        # Set the new name for the frame
        frame_uuid = self.name_to_uuid.pop(old_name)
        self.name_to_uuid[new_name] = frame_uuid
        self.uuid_to_name[frame_uuid] = new_name


    
    def get_parent(self, name: str, copy: bool = False) -> Optional[Frame]:
        """
        Get the parent of a frame in the FrameTree.

        Parameters
        ----------
        name : str
            The name of the frame.

        copy : bool, optional
            Get a copy of the parent frame. Defaults to False
        
        Returns
        -------
        Frame
            The parent frame of the frame with the given name.
        
        Raises
        ------
        ValueError
            If the frame name does not exist.
        """       
        # Get the frame from the FrameTree
        frame = self.get_frame(name)
        parent = frame.parent

        # Return the parent frame
        if copy and parent is not None:
            return copycopy(parent)
        return parent
    


    def set_parent(self, name: str, parent: Optional[str] = None) -> None:
        """
        Set a parent for a frame in the FrameTree.

        If the ``parent`` parameter is None, the frame will be unlinked from the parent frame.
        That means the global reference of the frame will be the default global frame.

        Parameters
        ----------
        name : str
            The name of the frame.
        
        parent : str, optional
            The name of the frame to set as the parent. Defaults to None.

        Raises
        ------
        ValueError
            If the frame name does not exist.
            If the parent name does not exist.

        Examples
        --------
        If the FrameTree has the following structure:

        .. code-block:: console

            global
            ├── frame1
            │   └── frame2
            └── frame3

        The code below will set the parent of frame3 to frame1:

        >>> frame_tree.set_parent("frame3", "frame1")

        The new structure of the FrameTree will be:

        .. code-block:: console

            global
            ├── frame1
            │   ├── frame2
            │   └── frame3
            
        """
        # Check if the frame name already exists
        if not self._exist_name(name):
            raise ValueError(f"Frame with name '{name}' does not exist.")

        # Check if the parent name already exists
        if parent is not None and not self._exist_name(parent):
            raise ValueError(f"Link '{parent}' does not exist.")
        
        # Set the parent for the frame
        frame_uuid = self.name_to_uuid[name]
        parent_uuid = self.name_to_uuid[parent] if parent is not None else None
        parent = self._frames[parent_uuid] if parent is not None else None

        self._frames[frame_uuid].parent = parent



    def get_parent_name(self, name: str) -> Optional[str]:
        """
        Get the name of the parent of a frame in the FrameTree.

        Parameters
        ----------
        name : str
            The name of the frame.
        
        Returns
        -------
        str
            The name of the parent frame of the frame with the given name.
        
        Raises
        ------
        ValueError
            If the frame name does not exist.
        """
        parent = self.get_parent(name)
        if parent is None:
            return None
        return self.uuid_to_name[parent._uuid]
    


    def get_children_names(self, name: Optional[str] = None) -> List[str]:
        """
        Get the children names of a frame in the FrameTree.

        Parameters
        ----------
        name : str, optional
            The name of the frame. Defaults to None.
        
        Returns
        -------
        List[str]
            The names of the children frames of the frame with the given name.
        
        Raises
        ------
        ValueError
            If the frame name does not exist.
        """
        # Check if the frame name already exists
        if name is not None and not self._exist_name(name):
            raise ValueError(f"Frame with name '{name}' does not exist.")

        # Get the children of the frame
        if name is None:
            children = [child_name for child_name in self.names if self.get_parent_name(child_name) is None]
        else:
            children = [child_name for child_name in self.names if self.get_parent_name(child_name) == name]
        return children

    

    # Public methods
    def add_frame(
        self, 
        frame: Frame,
        name: str,
        parent: Optional[str] = None,
        copy: bool = False
        ) -> None:
        r"""
        Add a frame to the FrameTree.

        The frame can be linked to another frame with the ``parent`` parameter.

        If the ``copy`` parameter is False and the ``parent`` parameter is None, the method is equivalent to the following code:

        >>> frame_tree[name] = frame

        .. warning::

            The parent attribute of the frame is overwritten by the FrameTree !!! 
            Please do not set the parent attribute of the frame manually.

        Parameters
        ----------
        frame : Frame
            The frame to be added.
        
        name : str
            The name of the frame.
        
        parent : str, optional
            The name of the frame to set as the parent. Defaults to None.
        
        copy : bool, optional
            Add a copy of the frame. Defaults to False.
        
        Raises
        ------
        TypeError
            If the frame is not a Frame instance.
            If an argument is not the correct type.
        ValueError
            If the frame name already exists.
            If the parent name does not exist.

        Examples
        --------
        >>> from py3dframe import FrameTree, Frame
        >>> # Create a FrameTree
        >>> frame_tree = FrameTree()
        >>> # Create several frames
        >>> frame1, frame2, frame3 = Frame(), Frame(), Frame()
        >>> # Add the frames to the FrameTree
        >>> frame_tree.add_frame(frame, "frame1")
        >>> frame_tree.add_frame(frame, "frame2", parent="frame1")
        >>> frame_tree.add_frame(frame, "frame3")

        The output of the code above is a FrameTree with three frames: frame1, frame2, and frame3 with the following structure:

        .. code-block:: console

            global
            ├── frame1
            │   └── frame2
            └── frame3
        """
        # Check the types of the arguments
        if not isinstance(frame, Frame):
            raise TypeError("frame is not Frame instance.")
        if parent is not None and not isinstance(parent, str):
            raise TypeError("parent must be a string.")
        if not isinstance(copy, bool):
            raise TypeError("copy must be a boolean.")

        # Check if the frame name already exists
        if self._exist_name(name):
            raise ValueError(f"Frame with name '{name}' already exists.")
        if parent is not None and not self._exist_name(parent):
            raise ValueError(f"Link '{parent}' does not exist.")

        # Copy the frame
        if copy:
            frame = copycopy(frame)

        # Create the uuid of the frame
        frame_uuid = uuid.uuid4()
        frame._uuid = frame_uuid
                
        # Setting the parent attribute of the frame
        parent = self.get_frame(parent) if parent is not None else None
        frame.parent = parent

        self._frames[frame_uuid] = frame
        self.name_to_uuid[name] = frame_uuid
        self.uuid_to_name[frame_uuid] = name



    def remove_frame(self, name: str) -> None:
        r"""
        Remove a frame from the FrameTree.

        The frames linked to the removed frame will be linked to the global frame.

        This method is equivalent to the following code:

        >>> del frame_binder[name]

        .. seealso:: 
        
            :meth:`recursive_remove_frame`

        Parameters
        ----------
        name : str
            The name of the frame to be removed.
        
        Raises
        ------
        TypeError
            If an argument is not the correct type.
        ValueError
            If the frame name does not exist.

        Examples
        --------
        If the FrameTree has the following structure:

        .. code-block:: console

            global
            ├── frame1
            │   └── frame2
            |       └── frame3
            └── frame4

        The code below will remove the frame2:

        >>> frame_tree.remove_frame("frame2")

        The new structure of the FrameTree will be:

        .. code-block:: console

            global
            ├── frame1
            ├── frame3
            └── frame4
        """
        # Check if the frame name already exists
        if not self._exist_name(name):
            raise ValueError(f"Frame with name '{name}' does not exist.")

        # Remove the frame from the FrameTree
        frame_uuid = self.name_to_uuid[name]
        children = self.get_children_names(name)

        # Setting the parent attribute of the children frames
        for child_name in children:
            self.set_parent(child_name, None)

        _ = self._frames.pop(frame_uuid)
        _ = self.name_to_uuid.pop(name)
        _ = self.uuid_to_name.pop(frame_uuid)



    def recursive_remove_frame(self, name: str) -> None:
        r"""
        Recursively remove a frame and all its children frames from the FrameTree.

        Parameters
        ----------
        name : str
            The name of the frame to be removed.
        
        Raises
        ------
        TypeError
            If an argument is not the correct type.
        ValueError
            If the frame name does not exist.

        Examples
        --------
        If the FrameTree has the following structure:

        .. code-block:: console

            global
            ├── frame1
            │   └── frame2
            |       └── frame3
            └── frame4

        The code below will remove the frame2 and all its linked frames:

        >>> frame_tree.recursive_remove_frame("frame2")

        The new structure of the FrameTree will be:

        .. code-block:: console

            global
            ├── frame1
            └── frame4
        """
        # Check if the frame name already exists
        if not self._exist_name(name):
            raise ValueError(f"Frame with name '{name}' does not exist.")

        # Recursively remove the frame and its linked frames
        to_remove = [name]
        while len(to_remove) > 0:
            name = to_remove.pop()
            frame_uuid = self.name_to_uuid[name]
            children_names = self.get_children_names(name)

            # Set to None the parent attribute of the children frames to avoid KeyError
            for child_name in children_names:
                self.set_parent(child_name, None)
            
            # Add the children frames to the list of frames to remove
            to_remove.extend(children_names)
            
            # Removing the frame
            _ = self._frames.pop(frame_uuid)
            _ = self.name_to_uuid.pop(name)
            _ = self.uuid_to_name.pop(frame_uuid)



    

    def get_global_frame(self, name: Optional[str] = None) -> Frame:
        r"""
        The return frame is the transformation between the global frame and the frame with the given name.

        If the name is None, the function will return the global frame.

        .. seealso::

            :meth:`py3dframe.Frame.get_global_frame`

        Parameters
        ----------
        name : str, optional
            The name of the frame. Defaults to None.

        Returns
        -------
        Frame
            The composed frame with the given name.
        
        Raises
        -------
        ValueError
            If the frame name does not exist.
        """
        # Case 0. If name is None, return the identity frame
        if name is None:
            return self._identity_frame()
        
        # Check if the frame name already exists 
        if not self._exist_name(name):
            raise ValueError(f"Frame with name '{name}' does not exist.")
        
        # Get the frame
        frame = self.get_frame(name)
        return frame.get_global_frame()



    def get_composed_frame(
        self, 
        input_name: Optional[str] = None,
        output_name: Optional[str] = None,
        ) -> Frame:
        """
        The return frame is the transformation between the input frame and the output frame.

        if input_name is None, the function will return the transformation between the global frame and the output frame.
        if output_name is None, the function will return the transformation between the input frame and the global frame.

        Parameters
        ----------
        input_name : str, optional
            The name of the input frame. Defaults to None.
        
        output_name : str, optional
            The name of the output frame. Defaults to None.
        
        Returns
        -------
        Frame
            The composed frame between the input and output frames.
        
        Raises
        -------
        ValueError
            If a frame name does not exist.

        Examples
        --------
        If the FrameTree has the following structure:

        .. code-block:: console

            global
            ├── frame1
            │   └── frame2
            └── frame3

        The code below will return the composed frame between frame2 and frame3:

        >>> frame_2_3 = frame_tree.get_composed_frame("frame2", "frame3")
        >>> point_in_3 = frame_2_3.from_global_to_frame(point=point_in_2) # Convert point from frame2 to frame3 

        .. seealso::

            :meth:`from_frame_to_frame`
        """
        # Check if the frame name already exists
        if input_name is not None and not self._exist_name(input_name):
            raise ValueError(f"Frame with name '{input_name}' does not exist.")
        if output_name is not None and not self._exist_name(output_name):
            raise ValueError(f"Frame with name '{output_name}' does not exist.")

        # Case 0. If input_name is None and output_name is None, return the identity frame
        if (input_name is None and output_name is None) or (input_name == output_name):
            return self._identity_frame()

        # Case 1. Construct the composed frame
        input_frame = self.get_global_frame(input_name) # global -> Input
        inverse_input_frame = inverse_frame(input_frame) # Input -> global
        output_frame = self.get_global_frame(output_name) # global -> Output

        composed_frame = inverse_input_frame * output_frame # Input -> Output
        return composed_frame


    
    def from_frame_to_frame(
        self,
        input_name: Optional[str] = None,
        output_name: Optional[str] = None,
        *,
        point: Optional[numpy.ndarray] = None,
        vector: Optional[numpy.ndarray] = None,
        ) -> Optional[numpy.ndarray]:
        r"""
        Convert a point or vector from one frame to another frame.

        if point is not None and vector is not None, the function will raise a ValueError.
        if point is None and vector is None, the function will return None.

        if name is None, the function will return the point or vector in global coordinates.

        Parameters
        ----------
        input_name : str, optional
            The name of the input frame. Defaults to None.
        
        output_name : str, optional
            The name of the output frame. Defaults to None.
        
        point : numpy.ndarray, optional
            The point to convert with shape (3, N). Defaults to None.
        
        vector : numpy.ndarray, optional
            The vector to convert with shape (3, N). Defaults to None.
        
        Returns
        -------
        numpy.ndarray
            The converted point or vector with shape (3, N).
        
        Raises
        ------
        ValueError
            If point and vector are not None.
            If a frame name does not exist.
        """
        compose_frame = self.get_composed_frame(input_name=input_name, output_name=output_name)
        return compose_frame.from_parent_to_frame(point=point, vector=vector)



    def clear(self) -> None:
        """Clear all frames and parents in the FrameTree."""
        self._frames.clear()
        self.name_to_uuid.clear()
        self.uuid_to_name.clear()



    def save_to_dict(self, description: str = "") -> Dict:
        """
        Export the Frame's data to a dictionary.

        The structure of the dictionary is as follows:

        .. code-block:: python

            {
                "type": "FrameTree [py3dframe]",
                "description": "Description of the frame tree",
                "frames": {
                    "frame1": {
                        "origin": [1.0, 2.0, 3.0],
                        "quaternion": [0.5, 0.5, 0.5, 0.5],
                        "direct": True,
                        "parent": "frame2"
                    },
                    "frame2": {
                        "origin": [1.0, 2.0, 4.0],
                        "quaternion": [0.5, 0.2, 0.3, 0.0],
                        "direct": False,
                        "parent": null
                    }
                }
            }

        Parameters
        ----------
        description : str, optional
            A description of the frame, by default "".

        Returns
        -------
        dict
            A dictionary containing the frames.

        Raises
        ------
        ValueError
            If the description is not a string.
        """
        # Check the description
        if not isinstance(description, str):
            raise ValueError("Description must be a string.")

        # Create the dictionary
        data = {
            "type": "FrameTree [py3dframe]",
            "frames": {}
        }

        # Add the description
        if len(description) > 0:
            data["description"] = description

        # Add the frames to the dictionary
        for name in self.names:
            frame = self.get_frame(name)
            parent_name = self.get_parent_name(name)
            data["frames"][name] = frame.save_to_dict(parent=False)
            data["frames"][name]["parent"] = parent_name
        
        return data



    @classmethod
    def load_from_dict(cls, data: Dict) -> FrameTree:
        """
        Create a FrameTree instance from a dictionary.

        The structure of the dictionary should be as provided by the :meth:`py3dframe.FrameTree.save_to_dict` method.

        The other keys in the dictionary are ignored.

        Parameters
        ----------
        data : dict
            A dictionary containing the frames.
        
        Returns
        -------
        FrameTree
            A FrameTree instance.
        
        Raises
        ------
        ValueError
            If the data is not a dictionary.
        """
        # Check for the input type
        if not isinstance(data, dict):
            raise ValueError("data must be a dictionary.")
        
        # Create the Frame instance
        frame_tree = cls()
        
        # Add the frames to the FrameTree
        for name, frame_data in data["frames"].items():
            frame = Frame.load_from_dict(frame_data, parent=False)
            parent = frame_data["parent"]
            frame_tree.add_frame(frame, name, parent)

        return frame_tree



    def save_to_json(self, filepath: str, description: str = "") -> None:
        """
        Export the Frame's data to a JSON file.

        The structure of the JSON file follows the :meth:`py3dframe.FrameTree.save_to_dict` method.

        Parameters
        ----------
        filepath : str
            The path to the JSON file.
        
        description : str, optional
            A description of the frame, by default "".
        
        Raises
        ------
        FileNotFoundError
            If the filepath is not a valid path.
        """
        # Create the dictionary
        data = self.save_to_dict(description=description)

        # Save the dictionary to a JSON file
        with open(filepath, "w") as file:
            json.dump(data, file, indent=4)


    
    @classmethod
    def load_from_json(cls, filepath: str) -> FrameTree:
        """
        Create a FrameTree instance from a JSON file.

        The structure of the JSON file follows the :meth:`py3dframe.FrameTree.save_to_dict` method.

        Parameters
        ----------
        filepath : str
            The path to the JSON file.
        
        Returns
        -------
        FrameTree
            A FrameTree instance.
        
        Raises
        ------
        FileNotFoundError
            If the filepath is not a valid path.
        """
        # Load the dictionary from the JSON file
        with open(filepath, "r") as file:
            data = json.load(file)
        
        # Create the Frame instance
        return cls.load_from_dict(data)
