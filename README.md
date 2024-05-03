# TensorRT_YOLOv8_CPP

trtexec --onnx=workspace/yolov8n.transd.onnx \
    --minShapes=images:1x3x640x640 \
    --maxShapes=images:16x3x640x640 \
    --optShapes=images:1x3x640x640 \
    --saveEngine=workspace/yolov8n.transd.engine