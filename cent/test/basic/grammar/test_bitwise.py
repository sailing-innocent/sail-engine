# the bitwise operation in python
import pytest 

@pytest.mark.func
def test_16bit():
    a = 0x0000
    b = 0x0001
    assert a == 0
    assert b == 1

@pytest.mark.func 
def test_2bit():
    a = 0b0000
    b = 0b0001
    assert a == 0
    assert b == 1

@pytest.mark.func
def test_bit_and():
    a = 0b1100
    b = 0b1010
    assert a & b == 0b1000

@pytest.mark.func
def test_bit_or():
    a = 0b1100
    b = 0b1010
    assert a | b == 0b1110

@pytest.mark.func
def test_bit_xor():
    a = 0b1100
    b = 0b1010
    assert a ^ b == 0b0110

@pytest.mark.func
def test_bit_not():
    a = 0b1100
    assert ~a == -0b1101

@pytest.mark.func
def test_bit_shift_left():
    a = 0b1100
    assert a << 1 == 0b11000

@pytest.mark.func
def test_bit_shift_right():
    a = 0b1100
    assert a >> 1 == 0b110
    