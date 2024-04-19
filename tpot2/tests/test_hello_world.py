"""
Test hello world.
Notes:
    parameterizing the test_input and expected values allows tests continue running even if one fails.
    xfail marks a test as expected to fail.  This is useful for tests that are not yet implemented.
    fixtures are used to setup and teardown tests.  They are useful for tests that require a lot of setup.
        We can implement fixtures if we need them.
"""

import pytest


@pytest.mark.parametrize("test_input,expected", [
    ("Hello World", "Hello World"),
])
def test_hello_world(test_input, expected):
    assert test_input is expected



def test_print(capture_stdout):
    print("Hello World")
    assert capture_stdout["stdout"] == "Hello World\n"
