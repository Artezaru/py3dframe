import numpy as np
import pytest
from py3dframe import Rotation, switch_RT_convention

def test_identity():
    """ Verifies that if input_convention == output_convention, the values remain identical. """
    R = Rotation.from_euler('xyz', [45, 30, 60], degrees=True)
    T = np.array([[1], [2], [3]])

    for i in range(8):
        R_out, T_out = switch_RT_convention(R, T, i, i)
        assert np.allclose(R_out.as_matrix(), R.as_matrix()), f"Rotation error for convention {i}"
        assert np.allclose(T_out, T), f"Translation error for convention {i}"

def test_known_transformation():
    """ Tests a known case where we know the expected result. """
    R = Rotation.from_euler('xyz', [0, 0, 90], degrees=True)  # 90° rotation around Z
    T = np.array([[1], [0], [0]])

    R_out, T_out = switch_RT_convention(R, T, 0, 1)
    assert np.allclose(R_out.as_matrix(), R.as_matrix()), "The rotation should not change"
    assert np.allclose(T_out, -T), "The translation should be inverted"

def test_inverse_transformation():
    """ Verifies that switch_RT_convention is bijective. """
    R = Rotation.from_euler('xyz', [10, 20, 30], degrees=True)
    T = np.array([[3], [-2], [5]])

    for i in range(8):
        for j in range(8):
            R_out, T_out = switch_RT_convention(R, T, i, j)
            R_recovered, T_recovered = switch_RT_convention(R_out, T_out, j, i)
            
            assert np.allclose(R_recovered.as_matrix(), R.as_matrix()), f"Inversion error for {i} -> {j} -> {i}"
            assert np.allclose(T_recovered, T), f"Translation error for {i} -> {j} -> {i}"

def test_random_cyclic_transformations(N=100):
    """ Verifies that passing through random cycles always returns to the original state. """
    for _ in range(N):
        R_init = Rotation.from_euler('xyz', np.random.uniform(-180, 180, size=3), degrees=True)
        T_init = np.random.uniform(-10, 10, size=(3, 1))

        # Generate a random cycle
        cycle = np.random.choice(range(8), size=np.random.randint(3, 10), replace=True).tolist()
        cycle.append(cycle[0])  # Ensure a return to the initial convention

        R_trans, T_trans = R_init, T_init
        for i in range(len(cycle) - 1):
            R_trans, T_trans = switch_RT_convention(R_trans, T_trans, cycle[i], cycle[i + 1])

        # Verify that we return to the initial state
        assert np.allclose(R_trans.as_matrix(), R_init.as_matrix(), atol=1e-10), \
            f"Rotation error after a complete cycle: {cycle}"
        assert np.allclose(T_trans, T_init, atol=1e-10), \
            f"Translation error after a complete cycle: {cycle}"

def test_invalid_inputs():
    """ Verifies that errors are correctly raised. """
    R = Rotation.from_euler('xyz', [10, 20, 30], degrees=True)
    T = np.array([[1], [2], [3]])

    with pytest.raises(TypeError):
        switch_RT_convention(R, T, "invalid", 0)  # Wrong type for input_convention

    with pytest.raises(TypeError):
        switch_RT_convention(R, T, 0, "invalid")  # Wrong type for output_convention

    with pytest.raises(ValueError):
        switch_RT_convention(R, T, -1, 0)  # input_convention out of bounds

    with pytest.raises(ValueError):
        switch_RT_convention(R, T, 0, 8)  # output_convention out of bounds

    with pytest.raises(ValueError):
        switch_RT_convention(R, np.array([1, 2, 3, 4]), 0, 1)  # Incorrect translation size

    with pytest.raises(TypeError):
        switch_RT_convention("not a rotation", T, 0, 1)  # Wrong type for rotation
