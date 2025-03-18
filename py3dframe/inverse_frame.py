
import numpy

from .frame import Frame

def inverse_frame(frame: Frame, glob: bool = True) -> Frame:
    r"""
    Compute the inverse frame of a frame.

    By convention, the frame is defined from the global to the frame.

    .. math::

        \mathbf{p}_{\text{global}} = \mathbf{R} \mathbf{p}_{\text{frame}} + \mathbf{O}

    .. math::

        \mathbf{p}_{\text{frame}} = \mathbf{R}^T (\mathbf{p}_{\text{global}} - \mathbf{O})
    
    The inverse frame is defined from the frame to the global frame.
    So the inverse frame is defined by:

    .. math::

        \mathbf{R}_{\text{inverse}} = \mathbf{R}^T

    .. math::

        \mathbf{p}_{\text{inverse}} = -\mathbf{R}^T \mathbf{O}

    If the frame has a parent such as global -> parent -> frame, the inverse frame is defined from the frame to the global frame.
    If glob is False, the inverse frame is defined from the frame to the parent frame.

    .. note::

        The parent frame of the inverse frame is the frame itself.

    Parameters
    ----------
    frame : Frame
        The frame to compute the inverse frame.

    glob : bool
        If True, the inverse frame is defined from the frame to the global frame.
        If False, the inverse frame is defined from the frame to the parent frame.
    
    Returns
    -------
    Frame
        The inverse frame.

    Raises
    ------
    TypeError
        If the frame is not a Frame
    """
    if not isinstance(frame, Frame):
        raise TypeError("The frame must be a Frame.")
    
    # Get the global frame
    if glob:
        frame = frame.get_global_frame()
    
    # Compute the inverse rotation matrix and origin
    inverse_rotation_matrix = frame.rotation_matrix.T
    inverse_origin = -numpy.dot(inverse_rotation_matrix, frame.origin)

    # Create the inverse frame
    return Frame(origin=inverse_origin, rotation_matrix=inverse_rotation_matrix, parent=frame)

