from __future__ import annotations
from typing import Optional, List, Dict
import numpy
import json
from copy import copy as copycopy
from .frame import Frame


class FrameTree:
    r"""
    The FrameTree class is a container for Frame instances.

    Each frame is stored with a name and can be linked to another frame.
    If a frame is linked to another frame, the global reference of the child frame became the parent frame.
    By default, the global frame is the default frame of the FrameTree.

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
        self._frames = {}
        self._parents = {}



    # Property getters and setters
    @property
    def names(self) -> List[str]:
        """Get a list of the names of the frames."""
        return list(self._frames.keys())
    


    @property
    def frames(self) -> Dict[str, Frame]:
        """Get a dictionary of the frames."""
        return self._frames



    @property
    def parents(self) -> Dict[str, str]:
        """Get a dictionary of the parents of the frames."""
        return self._parents
    


    @property
    def num_frames(self) -> int:
        """Get the number of frames in the FrameTree."""
        return self.__len__()
    


    # Private methods
    def _exist_name(self, name: str) -> bool:
        """Check if a frame with the given name exists."""
        if not isinstance(name, str):
            raise TypeError("name must be a string.")
        return name in self._frames



    def _global_frame(self) -> Frame:
        """Return the default global frame"""
        return Frame()


    
    def __len__(self) -> int:
        """Get the number of frames in the FrameTree."""
        return len(self._frames.keys())


    
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
        """Return a string representation of the FrameTree in a tree structure."""
        def build_tree(name, prefix="", is_last=True):
            """Recursively build the tree structure."""
            connector = "└── " if is_last else "├── "
            lines = [f"{prefix}{connector}{name}"]
            children = [child for child, parent in self._parents.items() if parent == name]
            for index, child in enumerate(sorted(children)):
                is_child_last = index == (len(children) - 1)
                child_prefix = prefix + ("    " if is_last else "│   ")
                lines.extend(build_tree(child, child_prefix, is_child_last))
            return lines

        # Start building the tree from the global frame
        lines = ["global"]
        roots = [name for name, parent in self._parents.items() if parent is None]
        for index, root in enumerate(sorted(roots)):
            is_last_root = index == (len(roots) - 1)
            lines.extend(build_tree(root, "", is_last_root))

        return "\n".join(lines)



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

        >>> frame_binder[name] = frame

        Parameters
        ----------
        frame : Frame
            The frame to be added.
        
        name : str
            The name of the frame.
        
        parent : str, optional
            The name of the frame to link to. Defaults to None.
        
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
        if parent is not None and parent not in self._frames:
            raise ValueError(f"Link '{parent}' does not exist.")

        # Add the frame to the FrameTree
        if copy:
            frame = copycopy(frame)
        self._frames[name] = frame
        self._parents[name] = parent



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
        for key, value in list(self._parents.items()):
            if value == name:
                self._parents[key] = None
                
        del self._frames[name]
        del self._parents[name]


    

        


    def recursive_remove_frame(self, name: str) -> None:
        r"""
        Recursively remove a frame and all its linked frames from the FrameTree.

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

            # Adding the linked frames to the list of frames to remove
            for key, value in self._parents.items():
                if value == name:
                    to_remove.append(key)

            # Removing the frame
            del self._frames[name]
            del self._parents[name]
    


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

        # Check if the frame name already exists
        if not self._exist_name(name):
            raise ValueError(f"Frame with name '{name}' does not exist.")

        # Get the frame from the FrameTree
        if copy:
            return copycopy(self._frames[name])
        return self._frames[name]

    

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
        self._frames[new_name] = self._frames.pop(old_name)
        self._parents[new_name] = self._parents.pop(old_name)

        # Update the parents
        for key, value in self._parents.items():
            if value == old_name:
                self._parents[key] = new_name
    


    def set_parent(self, name: str, parent: Optional[str] = None) -> None:
        """
        Set a parent for a frame in the FrameTree.

        If the ``parent`` parameter is None, the frame will be unlinked from the parent frame.
        That means the global reference of the frame will be the default global frame.

        Parameters
        ----------
        name : str
            The name of the frame to set the link.
        
        parent : str, optional
            The name of the frame to link to. Defaults to None.

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
        self._parents[name] = parent



    def get_globalcompose_frame(self, name: Optional[str] = None) -> Frame:
        r"""
        The return frame is the transformation between the global frame and the frame with the given name.

        If the name is None, the function will return the global frame.

        .. seealso::

            :meth:`get_compose_frame`

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
        # Case 0. If name is None, return the global frame
        if name is None:
            return self._global_frame()

        # Check if the frame name already exists
        if not self._exist_name(name):
            raise ValueError(f"Frame with name '{name}' does not exist.")
        
        # Case 1. Construct the composed frame
        compose_frame = self._frames[name]
        parent = self._parents[name]
        while parent is not None:
            parent_frame = self._frames[parent]
            compose_frame = parent_frame * compose_frame
            parent = self._parents[parent]
        return compose_frame



    def get_compose_frame(
        self, 
        input_name: Optional[str] = None,
        output_name: Optional[str] = None,
        ) -> Frame:
        """
        The return frame is the transformation between the input frame and the output frame.

        if input_name is None, the function will return the globalcompose frame of the output frame.
        if output_name is None, the function will return the inverse globalcompose frame of the input frame.

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

        >>> frame_2_3 = frame_tree.get_compose_frame("frame2", "frame3")
        >>> point_in_3 = frame_2_3.from_global_to_frame(point=point_in_2) # Convert point from frame2 to frame3 

        .. seealso::

            :meth:`from_frame_to_frame`
        """
        # Check if the frame name already exists
        if input_name is not None and not self._exist_name(input_name):
            raise ValueError(f"Frame with name '{input_name}' does not exist.")
        if output_name is not None and not self._exist_name(output_name):
            raise ValueError(f"Frame with name '{output_name}' does not exist.")

        # Case 0. If input_name is None and output_name is None, return the global frame
        if (input_name is None and output_name is None) or (input_name == output_name):
            return self._global_frame()

        # Case 1. Construct the composed frame
        input_frame = self.get_globalcompose_frame(input_name) # global -> Input
        inverse_input_frame = input_frame.get_inverse_frame() # Input -> global
        output_frame = self.get_globalcompose_frame(output_name) # global -> Output
        return inverse_input_frame * output_frame # Input -> Output


    
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
        compose_frame = self.get_compose_frame(input_name=input_name, output_name=output_name)
        return compose_frame.from_global_to_frame(point=point, vector=vector)



    def clear(self) -> None:
        """Clear all frames and parents in the FrameTree."""
        self._frames.clear()
        self._parents.clear()



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
        for name, frame in self._frames.items():
            data["frames"][name] = frame.save_to_dict()
            data["frames"][name]["parent"] = self._parents[name]
        
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
            frame = Frame.load_from_dict(frame_data)
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
