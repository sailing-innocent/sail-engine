# test tensorflow gpu operation
import tensorflow as tf 

def test_gpu():
    assert tf.test.is_gpu_available()