#include<memory>

#include "yolo.hpp"
#include "infer.hpp"

namespace yolo{

class InferImpl:public Infer{
 public:
  std::shared_ptr<trt::Infer> trt_;

  virtual ~InferImpl() = default;

  virtual BoxArray forward(const Image &image,void *stream=nullptr) override{
    auto output=forwards({image},stream);
    return output[0];
  }

  virtual std::vector<BoxArray> forwards(const std::vector<Image> &images,void *stream=nullptr) override{
    int num_image=images.size();
    if(num_image==0) return {};
    auto input_dims=trt_->static_dims(0);
  }
};

Infer *loadraw(const std::string &engine_file,float confidence_threshold,float nms_threshold){
  InferImpl *impl = new InferImpl();
  if (!impl->load(engine_file,confidence_threshold, nms_threshold)) {
    delete impl;
    impl = nullptr;
  }
  return impl;
}

std::shared_ptr<Infer> load(const std::string &engine_file,float confidence_threshold,float nms_threshold){
  return std::shared_ptr<InferImpl>((InferImpl *)loadraw(engine_file,confidence_threshold,nms_threshold));
}
} // namespace yolo

