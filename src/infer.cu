#include<NvInfer.h>
#include<cuda_runtime.h>
#include<stdarg.h>

#include<fstream>
#include<numeric>
#include<sstream>
#include<unordered_map>

#include "infer.hpp"

namespace trt{

#define checkRuntime(call)                                                                 \
  do{                                                                                      \
    auto ___call__ret_code__=(call);                                                       \
    if(___call__ret_code__!=cudaSuccess){                                                  \
      INFO("CUDA Runtime error %s # %s, code = %s [ %d ]",#call,                           \
           cudaGetErrorString(___call__ret_code__),cudaGetErrorName(___call__ret_code__),  \
           ___call__ret_code__);                                                           \
      abort();                                                                             \
    }                                                                                      \
  }while(0)

#define checkKernel(...)                 \
  do{                                    \
    {(__VA_ARGS__);}                     \
    checkRuntime(cudaPeekAtLastError()); \
  }while(0)

#define Assert(op)                 \
  do{                              \
    bool cond=!(!(op));            \
    if(!cond){                     \
      INFO("Assert failed, " #op); \
      abort();                     \
    }                              \
  }while(0)

#define Assertf(op,...)                             \
  do{                                                \
    bool cond=!(!(op));                              \
    if(!cond){                                       \
      INFO("Assert failed, " #op " : " __VA_ARGS__); \
      abort();                                       \
    }                                                \
  }while(0)


class __nativate_nvinfer_logger:public nvinfer1::ILogger{
 public:
  virtual void log(Severity severity,const char *msg) noexcept override{
    if (severity==Severity::kINTERNAL_ERROR) {
      INFO("NVInfer INTERNAL_ERROR: %s",msg);
      abort();
    } else if (severity==Severity::kERROR) {
      INFO("NVInfer: %s",msg);
    }
  }
};

static __nativate_nvinfer_logger gLogger;

template<typename _T>
static void destroy_nvidia_pointer(_T *ptr){
  if(ptr) ptr->destroy();
}

static std::vector<uint8_t> load_file(const std::string &file){
  std::ifstream in(file,std::ios::in | std::ios::binary);
  if(!in.is_open()) return {};

  in.seekg(0,std::ios::end);
  size_t length = in.tellg();

  std::vector<uint8_t> data;
  if (length>0){
    in.seekg(0,std::ios::beg);
    data.resize(length);

    in.read((char *)&data[0],length);
  }
  in.close();
  return data;
}

class __nativate_engine_context{
 public:
  virtual ~__nativate_engine_context(){destroy();}
  std::shared_ptr<nvinfer1::IExecutionContext> context_;
  std::shared_ptr<nvinfer1::ICudaEngine> engine_;
  std::shared_ptr<nvinfer1::IRuntime> runtime_=nullptr;

  bool construct(const void *pdata,size_t size){
    destroy();  
    if(pdata==nullptr||size==0) return false;

    runtime_=std::shared_ptr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(gLogger),
                                                 destroy_nvidia_pointer<nvinfer1::IRuntime>);
    if(runtime_==nullptr) return false;

    engine_=std::shared_ptr<nvinfer1::ICudaEngine>(runtime_->deserializeCudaEngine(pdata,size,nullptr),
                                                   destroy_nvidia_pointer<nvinfer1::ICudaEngine>);
    if(engine_==nullptr) return false;

    context_=std::shared_ptr<nvinfer1::IExecutionContext>(engine_->createExecutionContext(),
                                                          destroy_nvidia_pointer<nvinfer1::IExecutionContext>);
    return context_!=nullptr;
  }

 private:
  void destroy(){
    context_.reset();
    engine_.reset();
    runtime_.reset();
  }
};

class InferImpl:public Infer{
 public:
  std::shared_ptr<__nativate_engine_context> context_;
  std::unordered_map<std::string,int> binding_name_to_index_;

  virtual ~InferImpl()=default;

  void setup(){
    auto engine=this->context_->engine_;
    int nbBindings=engine->getNbBindings();

    binding_name_to_index_.clear();

    for(int i=0;i<nbBindings;++i){
      const char *bindingName=engine->getBindingName(i);
      binding_name_to_index_[bindingName]=i;
    }
  }

  bool construct(const void *data,size_t size){
    context_=std::make_shared<__nativate_engine_context>();

    if(!context_->construct(data,size)) return false;

    setup();
    return true;
  }

  bool load(const std::string &file){
    auto data=load_file(file);
    if(data.empty()){
      INFO("An empty file has been loaded. Please confirm your file path: %s",file.c_str());
      return false;
    }

    return this->construct(data.data(),data.size());
  }

  virtual int index(const std::string &name) override{
    auto iter=binding_name_to_index_.find(name);
    Assertf(iter!=binding_name_to_index_.end(),"Can not found the binding name: %s",name.c_str());

    return iter->second;
  }

  virtual bool forward(const std::vector<void *> &bindings,void *stream,void *input_consum_event) override{
    return this->context_->context_->enqueueV2((void**)bindings.data(),
                                               (cudaStream_t)stream,
                                               (cudaEvent_t *)input_consum_event);
  }

  virtual std::vector<int> run_dims(const std::string &name) override{
    return run_dims(index(name));
  }

  virtual std::vector<int> run_dims(int ibinding) override{
    auto dim=this->context_->context_->getBindingDimensions(ibinding);
    return std::vector<int>(dim.d,dim.d+dim.nbDims);
  }
 };


} // namespace trt