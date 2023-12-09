import pytest 
import os 
import cv2 
import matplotlib.pyplot as plt
import numpy as np 

@pytest.mark.current 
def test_hdr():
    workspace_dir = "D:/workspace/hdr/"
    explosure_times = [0.033, 0.25, 2.5, 15]
    times = np.array(explosure_times, dtype=np.float32)
    filenames = [workspace_dir + "img_" + str(time_stamp) + ".jpg" for time_stamp in explosure_times]
    assert os.path.exists(filenames[0])
    images = []

    for filename in filenames:
        images.append(cv2.imread(filename))

    # align images
    alignMTB = cv2.createAlignMTB()
    alignMTB.process(images, images)

    # recorver CRF
    calibrateDebevec = cv2.createCalibrateDebevec()
    responseDebevec = calibrateDebevec.process(images, times)

    # merge images
    mergeDebevec = cv2.createMergeDebevec()
    hdrDebevec = mergeDebevec.process(images, times, responseDebevec)

    # save imagas
    # cv2.imwrite("hdrDebevec.hdr", hdrDebevec)

    # tonemap
    tonemapDrago = cv2.createTonemapDrago(1.0, 0.7)
    ldrDrago = tonemapDrago.process(hdrDebevec)
    ldrDrago = 3 * ldrDrago
    ldrDrago = np.clip(ldrDrago, 0, 1).astype(np.float32)

    plt.imshow(ldrDrago)
    plt.show()
