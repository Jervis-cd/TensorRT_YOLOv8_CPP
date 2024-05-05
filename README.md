# TensorRT_YOLOv8_CPP
## 简介
1.用于YOLOv8的TensorRT推理的CPP代码，包含详细注释          \
2.不支持最新TensorRT 10版本（API变化较大）                \
3.本仓库支持多批次推理，也支持一张图像推理
## 使用流程
### step1: YOLOv8.pt $\Longrightarrow$ YOLOv8.onnx
1.YOLOv8 export()函数设置动态维度时，会同时将batch,height,width都设置成动态维度（后面两者无必要，且会造成错误）             \
2.可以用<https://github.com/Jervis-cd/Parse-YOLOv8>仓库中```export.py```进行模型导出（只将batch维度设置为动态）

### step2: YOLOv8.onnx $\Longrightarrow$ YOLOv8.transd.onnx
使用workspace下v8trans.py脚本交换onnx模型输出
```
cd workspace
python v8trans.py yolov8n.onnx
```
### step3: YOLOv8.transd.onnx $\Longrightarrow$ YOLOv8n.transd.engine
1.使用TensorRT自带工具trtexec工具进行转换               \
2.直接在终端中使用trtexec工具需要将TensorRT路径添加到环境变量中
``` 
trtexec --onnx=workspace/yolov8n.transd.onnx        \
    --minShapes=images:1x3x640x640                  \
    --maxShapes=images:16x3x640x640                 \
    --optShapes=images:1x3x640x640                  \
    --saveEngine=workspace/yolov8n.transd.engine
```