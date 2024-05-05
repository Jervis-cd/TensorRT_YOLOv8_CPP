#include<opencv2/opencv.hpp>

#include "yolo.hpp"

static const char *cocolabels[]={"person",        "bicycle",      "car",
                                 "motorcycle",    "airplane",     "bus",
                                 "train",         "truck",        "boat",
                                 "traffic light", "fire hydrant", "stop sign",
                                 "parking meter", "bench",        "bird",
                                 "cat",           "dog",          "horse",
                                 "sheep",         "cow",          "elephant",
                                 "bear",          "zebra",        "giraffe",
                                 "backpack",      "umbrella",     "handbag",
                                 "tie",           "suitcase",     "frisbee",
                                 "skis",          "snowboard",    "sports ball",
                                 "kite",          "baseball bat", "baseball glove",
                                 "skateboard",    "surfboard",    "tennis racket",
                                 "bottle",        "wine glass",   "cup",
                                 "fork",          "knife",        "spoon",
                                 "bowl",          "banana",       "apple",
                                 "sandwich",      "orange",       "broccoli",
                                 "carrot",        "hot dog",      "pizza",
                                 "donut",         "cake",         "chair",
                                 "couch",         "potted plant", "bed",
                                 "dining table",  "toilet",       "tv",
                                 "laptop",        "mouse",        "remote",
                                 "keyboard",      "cell phone",   "microwave",
                                 "oven",          "toaster",      "sink",
                                 "refrigerator",  "book",         "clock",
                                 "vase",          "scissors",     "teddy bear",
                                 "hair drier",    "toothbrush"};

yolo::Image cvimg(const cv::Mat &image){
  return yolo::Image(image.data,image.cols,image.rows);
}

void batch_inference(){
  // 推理原始图像
  std::vector<cv::Mat> images{cv::imread("images/car.jpg"),cv::imread("images/zand.jpg")};
  // 加载engine文件，加载失败直接返回
  auto yolo=yolo::load("yolov8n.transd.engine");
  if(yolo==nullptr) return;

  // 将images中元素，逐一执行cvimg函数后，存储在yoloimages中
  std::vector<yolo::Image> yoloimages(images.size());
  std::transform(images.begin(),images.end(),yoloimages.begin(),cvimg);

  auto batched_result=yolo->forwards(yoloimages);         // 执行inference
  // 解码模型输出
  for(int ib=0;ib<(int)batched_result.size();++ib){
    auto &objs=batched_result[ib];
    auto &image=images[ib];
    for(auto &obj:objs){
      uint8_t b,g,r;
      std::tie(b,g,r)=yolo::random_color(obj.class_label);
      cv::rectangle(image,cv::Point(obj.left,obj.top),cv::Point(obj.right,obj.bottom),
                    cv::Scalar(b,g,r),5);

      auto name=cocolabels[obj.class_label];
      auto caption=cv::format("%s %.2f",name,obj.confidence);
      int width=cv::getTextSize(caption,0,1,2,nullptr).width+10;
      cv::rectangle(image,cv::Point(obj.left-3,obj.top-33),
                    cv::Point(obj.left+width,obj.top),cv::Scalar(b,g,r),-1);
      cv::putText(image,caption,cv::Point(obj.left,obj.top-5),0,1,cv::Scalar::all(0),2,16);
    }
    printf("Save result to Result.jpg, %d objects\n",(int)objs.size());
    cv::imwrite(cv::format("Result%d.jpg",ib),image);
  }
}

int main(int argc,char *argv[]){
  batch_inference();
  return 0;
}