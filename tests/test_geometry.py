from safety_vision.geometry import point_in_polygon


def test_point_in_polygon():
    square = [(0, 0), (100, 0), (100, 100), (0, 100)]
    assert point_in_polygon((50, 50), square)
    assert not point_in_polygon((150, 50), square)

