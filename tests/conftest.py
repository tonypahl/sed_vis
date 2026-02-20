import pytest

class DummyAxis:
    class DummyTransform:
        def __init__(self, y):
            self.y = y
        def transform(self, xy):
            return [0, self.y]

    def __init__(self, y):
        self.transLimits = self.DummyTransform(y)

@pytest.fixture
def axis_top():
    return DummyAxis(0.95)

@pytest.fixture
def axis_middle():
    return DummyAxis(0.5)

@pytest.fixture
def axis_bottom():
    return DummyAxis(0.05)
