"""How face_models/yunet_320_opencv41.onnx was made (run on a PC, not the Jetson).

JetPack's OpenCV 4.1.1 cannot load the original YuNet (opencv_zoo,
face_detection_yunet_2023mar.onnx, MIT licence): dynamic shapes, an empty
Resize 'roi' tensor and opset-11 Resize. This fixes the input to 1x3x320x320,
folds the shapes and replaces Resize with a plain 2x nearest Upsample.

    pip install onnx onnxsim
    python tools/convert_yunet.py face_detection_yunet_2023mar.onnx face_models/yunet_320_opencv41.onnx
"""
import sys

import onnx
import onnxsim
from onnx import helper


def convert(source, target):
    model, ok = onnxsim.simplify(onnx.load(source), overwrite_input_shapes={'input': [1, 3, 320, 320]})
    assert ok
    for node in model.graph.node:
        if node.op_type == 'Resize':
            node.op_type = 'Upsample'
            del node.input[1:]
            del node.attribute[:]
            node.attribute.extend([helper.make_attribute('mode', 'nearest'),
                                   helper.make_attribute('scales', [1.0, 1.0, 2.0, 2.0])])
    used = set(name for node in model.graph.node for name in node.input)
    keep = [t for t in model.graph.initializer if t.name in used]
    del model.graph.initializer[:]
    model.graph.initializer.extend(keep)
    onnx.save(model, target)


if __name__ == '__main__':
    convert(sys.argv[1], sys.argv[2])
