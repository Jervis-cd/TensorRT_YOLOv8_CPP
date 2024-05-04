#include "infer.hpp"
#include "yolo.hpp"

namespace yolo{

#define GPU_BLOCK_THREADS 512

/* runtime API检查 */
#define checkRuntime(call)                                                                 \
  do {                                                                                     \
    auto ___call__ret_code__=(call);                                                       \
    if(___call__ret_code__!=cudaSuccess){                                                  \
      INFO("CUDA Runtime error %s # %s, code = %s [ %d ]",#call,                           \
           cudaGetErrorString(___call__ret_code__),cudaGetErrorName(___call__ret_code__),  \
           ___call__ret_code__);                                                           \
      abort();                                                                             \
    }                                                                                      \
  }while(0)

/* 核函数检查 */
#define checkKernel(...)                 \
  do {                                   \
    { (__VA_ARGS__); }                   \
    checkRuntime(cudaPeekAtLastError()); \
  }while(0)

/* 归一化类型 */
enum class NormType:int { None=0,MeanStd=1,AlphaBeta=2 };

/* 通道类型bgr/rgb */
enum class ChannelType:int { None=0,SwapRB=1 };

/* 归一化操作，可以支持均值标准差，alpha beta，和swap RB */
struct Norm {
  float mean[3];
  float std[3];
  float alpha,beta;
  NormType type=NormType::None;
  ChannelType channel_type=ChannelType::None;

  // out=(x*alpha-mean)/std
  static Norm mean_std(const float mean[3],const float std[3],float alpha=1/255.0f,
                       ChannelType channel_type=ChannelType::None);
  // out=x*alpha+beta
  static Norm alpha_beta(float alpha,float beta=0,ChannelType channel_type=ChannelType::None);
  // None
  static Norm None();
};

Norm Norm::mean_std(const float mean[3],const float std[3],float alpha,ChannelType channel_type){
  Norm out;
  out.type=NormType::MeanStd;
  out.alpha=alpha;
  out.channel_type=channel_type;
  memcpy(out.mean,mean,sizeof(out.mean));
  memcpy(out.std,std,sizeof(out.std));
  return out;
}

Norm Norm::alpha_beta(float alpha,float beta,ChannelType channel_type) {
  Norm out;
  out.type=NormType::AlphaBeta;
  out.alpha=alpha;
  out.beta=beta;
  out.channel_type=channel_type;
  return out;
}

Norm Norm::None(){ return Norm(); }

const int NUM_BOX_ELEMENT=8;  // left, top, right, bottom, confidence, class,
                                // keepflag, row_index(output)
const int MAX_IMAGE_BOXES=1024;

/* 内存对齐计算 */
inline int upbound(int n,int align=32) { return (n+align-1)/align*align; }

/* 单点仿射变换 */
static __host__ __device__ void affine_project(float *matrix,float x,float y,float *ox,float *oy){
  *ox=matrix[0]*x+matrix[1]*y+matrix[2];
  *oy=matrix[3]*x+matrix[4]*y+matrix[5];
}

/* 核函数（由host函数调用），网络v8模型推理结果解码 */
static __global__ void decode_kernel_v8(float *predict,int num_bboxes,int num_classes,
                                        int output_cdim,float confidence_threshold,
                                        float *invert_affine_matrix,float *parray,
                                        int MAX_IMAGE_BOXES){
  int position=blockDim.x*blockIdx.x+threadIdx.x;         // 当前线程id以及执行的位置
  if(position>=num_bboxes) return;        // 位置大于anchor point数量，直接返回

  float *pitem=predict+output_cdim*position;    // 从预测结果中获取当前执行的起始指针位置
  float *class_confidence=pitem+4;          // 获取类别的起始位置

  // 从类别起始位置开始循环获取其中最大值作为置信度以及获取对应的label
  float confidence=*class_confidence++;   // 解引用赋值之后，指针位置+1
  int label=0;
  for(int i=1;i<num_classes;++i,++class_confidence){
    if(*class_confidence>confidence){
      confidence=*class_confidence;
      label=i;
    }
  }
  // 如果confidence小于阈值，直接返回
  if (confidence<confidence_threshold) return;

  int index=atomicAdd(parray,1);              // 将parray指针指向的值+1后，写到parray中
                                              // 用于记录检测图像的总数
  if (index>=MAX_IMAGE_BOXES) return;     // 如果大于设置的最大检测数量，直接返回

  // 解码模型获取box center x,center y,w,h
  float cx=*pitem++;
  float cy=*pitem++;
  float width=*pitem++;
  float height=*pitem++;

  // center x,center y,w,h ---> left-top,right-bottom
  float left=cx-width*0.5f;
  float top=cy-height*0.5f;
  float right=cx+width*0.5f;
  float bottom=cy+height*0.5f;

  // 逆仿射变换，转换到原图坐标
  affine_project(invert_affine_matrix,left,top,&left,&top);
  affine_project(invert_affine_matrix,right,bottom,&right,&bottom);

  //todo 输出量为什么直接加载parray之后，parray分配了多少内存
  float *pout_item=parray+1+index*NUM_BOX_ELEMENT;
  *pout_item++=left;            // 将left指针赋值给pout_item解引用之后的值，然后指针+1
  *pout_item++=top;
  *pout_item++=right;
  *pout_item++=bottom;
  *pout_item++=confidence;
  *pout_item++=label;
  *pout_item++=1;               // 1=keep, 0=ignore, 记录该anchor point是否保留
  *pout_item++=position;
}

/* 设备函数（在设备上执行，由核函数或其他设备函数调用），计算IOU */
static __device__ float box_iou(float aleft,float atop,float aright,float abottom,float bleft,
                                float btop,float bright,float bbottom){
  // 计算交域的left-top/right-bottom                                
  float cleft=max(aleft,bleft);
  float ctop=max(atop,btop);
  float cright=min(aright,bright);
  float cbottom=min(abottom,bbottom);

  float c_area=max(cright-cleft,0.0f)*max(cbottom-ctop,0.0f);

  if(c_area==0.0f) return 0.0f;

  float a_area=max(0.0f,aright-aleft)*max(0.0f,abottom-atop);
  float b_area=max(0.0f,bright-bleft)*max(0.0f,bbottom-btop);
  return c_area/(a_area+b_area-c_area);
}

/* nms核函数，快速执行nms */
static __global__ void fast_nms_kernel(float *bboxes,int MAX_IMAGE_BOXES,float threshold){
  int position=(blockDim.x*blockIdx.x+threadIdx.x);
  int count=min((int)*bboxes,MAX_IMAGE_BOXES);        // 统计经过置信度阈值筛选之后的bboxes
  if(position>=count) return;

  // left,top,right,bottom,confidence,class,keepflag
  float *pcurrent=bboxes+1+position*NUM_BOX_ELEMENT;    // 当前线程计算的box

  for(int i=0;i<count;++i){
    float *pitem=bboxes+1+i*NUM_BOX_ELEMENT;
    if(i==position || pcurrent[5]!=pitem[5]) continue;    // 如果类别不相同/同一box跳过

    // todo
    if(pitem[4]>=pcurrent[4]){            // pitem置信度大于等于pcurrent
      if(pitem[4]==pcurrent[4] && i<position) continue;   // 置信度相同，并且比较位置小于当前位置直接跳过

      // 计算IOU，将大于阈值并将keepflag
      float iou=box_iou(pcurrent[0],pcurrent[1],pcurrent[2],pcurrent[3],pitem[0],pitem[1],
                        pitem[2],pitem[3]);
      if(iou>threshold){
        pcurrent[6]=0;  // 1=keep, 0=ignore
        return;
      }
    }
  }
}

/* 根据任务数量, 获取核函数执行grid维度 */
static dim3 grid_dims(int numJobs){
  int numBlockThreads=numJobs<GPU_BLOCK_THREADS?numJobs:GPU_BLOCK_THREADS;
  return dim3(((numJobs+numBlockThreads-1)/(float)numBlockThreads));
}

/* 获取核函数执行block维度 */
static dim3 block_dims(int numJobs){
  return numJobs<GPU_BLOCK_THREADS?numJobs:GPU_BLOCK_THREADS;
}

/* 模型输出解码主程序 */
static void decode_kernel_invoker(float *predict,int num_bboxes,int num_classes,int output_cdim,
                                  float confidence_threshold,float nms_threshold,
                                  float *invert_affine_matrix,float *parray,int MAX_IMAGE_BOXES,
                                  cudaStream_t stream){
  auto grid=grid_dims(num_bboxes);
  auto block=block_dims(num_bboxes);

  checkKernel(decode_kernel_v8<<<grid,block,0,stream>>>(
      predict,num_bboxes,num_classes,output_cdim,confidence_threshold,invert_affine_matrix,parray,
      MAX_IMAGE_BOXES));
  
  grid=grid_dims(MAX_IMAGE_BOXES);
  block=block_dims(MAX_IMAGE_BOXES);
  checkKernel(fast_nms_kernel<<<grid,block,0,stream>>>(parray,MAX_IMAGE_BOXES,nms_threshold));
}

/* 双线性差值以及对数据进行进行对应的Norm */
static __global__ void warp_affine_bilinear_and_normalize_plane_kernel(
    uint8_t *src,int src_line_size,int src_width,int src_height,float *dst,int dst_width,
    int dst_height,uint8_t const_value_st,float *warp_affine_matrix_2_3,Norm norm) {
  int dx=blockDim.x*blockIdx.x+threadIdx.x;
  int dy=blockDim.y*blockIdx.y+threadIdx.y;
  if(dx>=dst_width || dy>=dst_height) return;

  float m_x1=warp_affine_matrix_2_3[0];
  float m_y1=warp_affine_matrix_2_3[1];
  float m_z1=warp_affine_matrix_2_3[2];
  float m_x2=warp_affine_matrix_2_3[3];
  float m_y2=warp_affine_matrix_2_3[4];
  float m_z2=warp_affine_matrix_2_3[5];

  float src_x=m_x1*dx+m_y1*dy+m_z1;
  float src_y=m_x2*dx+m_y2*dy+m_z2;
  float c0,c1,c2;

  if (src_x<=-1 || src_x>=src_width || src_y<=-1 || src_y>=src_height){
    // out of range
    c0=const_value_st;
    c1=const_value_st;
    c2=const_value_st;
  }else{
    int y_low=floorf(src_y);
    int x_low=floorf(src_x);
    int y_high=y_low+1;
    int x_high=x_low+1;

    uint8_t const_value[]={const_value_st,const_value_st,const_value_st};
    float ly=src_y-y_low;
    float lx=src_x - x_low;
    float hy=1-ly;
    float hx=1-lx;
    float w1=hy*hx,w2=hy*lx,w3=ly*hx,w4=ly*lx;
    uint8_t *v1=const_value;
    uint8_t *v2=const_value;
    uint8_t *v3=const_value;
    uint8_t *v4=const_value;
    if(y_low>=0){
      if(x_low>=0) v1=src+y_low*src_line_size+x_low*3;
      if(x_high<src_width) v2=src+y_low*src_line_size+x_high*3;
    }

    if(y_high<src_height){
      if (x_low>=0) v3=src+y_high*src_line_size+x_low*3;
      if (x_high<src_width) v4=src+y_high*src_line_size+x_high*3;
    }

    c0=floorf(w1*v1[0]+w2*v2[0]+w3*v3[0]+w4*v4[0]+0.5f);
    c1=floorf(w1*v1[1]+w2*v2[1]+w3*v3[1]+w4*v4[1]+0.5f);
    c2=floorf(w1*v1[2]+w2*v2[2]+w3*v3[2]+w4*v4[2]+0.5f);
  }

  if(norm.channel_type==ChannelType::SwapRB){
    float t=c2;
    c2=c0;
    c0=t;
  }

  if(norm.type==NormType::MeanStd){
    c0=(c0*norm.alpha-norm.mean[0])/norm.std[0];
    c1=(c1*norm.alpha-norm.mean[1])/norm.std[1];
    c2=(c2*norm.alpha-norm.mean[2])/norm.std[2];
  }else if(norm.type == NormType::AlphaBeta){
    c0=c0*norm.alpha+norm.beta;
    c1=c1*norm.alpha+norm.beta;
    c2=c2*norm.alpha+norm.beta;
  }

  int area=dst_width*dst_height;
  float *pdst_c0=dst+dy*dst_width+dx;
  float *pdst_c1=pdst_c0+area;
  float *pdst_c2=pdst_c1+area;
  *pdst_c0=c0;
  *pdst_c1=c1;
  *pdst_c2=c2;
}

/* 双线性插值归一化和函数调用 */
static void warp_affine_bilinear_and_normalize_plane(uint8_t *src,int src_line_size,int src_width,
                                                     int src_height,float *dst,int dst_width,
                                                     int dst_height,float *matrix_2_3,
                                                     uint8_t const_value,const Norm &norm,
                                                     cudaStream_t stream){
  dim3 grid((dst_width+31)/32,(dst_height+31)/32);
  dim3 block(32,32);

  checkKernel(warp_affine_bilinear_and_normalize_plane_kernel<<<grid,block,0,stream>>>(
      src,src_line_size,src_width,src_height,dst,dst_width,dst_height,const_value,
      matrix_2_3,norm));
}

/* 构建仿射矩阵 */
struct AffineMatrix{
  float i2d[6];  // image to dst(network), 2x3 matrix
  float d2i[6];  // dst to image, 2x3 matrix

  // 根据输入图像和网络输入计算仿射变换矩阵
  void compute(const std::tuple<int,int> &from,const std::tuple<int,int> &to){
    // 获取x，y的缩放比例
    float scale_x=std::get<0>(to)/(float)std::get<0>(from);
    float scale_y=std::get<1>(to)/(float)std::get<1>(from);
    float scale=std::min(scale_x,scale_y);

    i2d[0]=scale;
    i2d[1]=0;
    i2d[2]=-scale*std::get<0>(from)*0.5+std::get<0>(to)*0.5+scale*0.5-0.5;
    i2d[3]=0;
    i2d[4]=scale;
    i2d[5]=-scale*std::get<1>(from)*0.5+std::get<1>(to)*0.5+scale*0.5-0.5;

    // 求解逆矩阵
    double D=i2d[0]*i2d[4]-i2d[1]*i2d[3];
    D=D!=0.?double(1.)/D:double(0.);
    double A11=i2d[4]*D,A22=i2d[0]*D,A12=-i2d[1]*D,A21=-i2d[3]*D;
    double b1=-A11*i2d[2]-A12*i2d[5];
    double b2=-A21*i2d[2]-A22*i2d[5];

    d2i[0]=A11;
    d2i[1]=A12;
    d2i[2]=b1;
    d2i[3]=A21;
    d2i[4]=A22;
    d2i[5]=b2;
  }
};

/* yolo模型推理 */
class InferImpl:public Infer{
public:
  std::shared_ptr<trt::Infer> trt_;          // 创建trt::Infer对象
  std::string engine_file_;                  // 模型序列化文件地址
  float confidence_threshold_;          // 置信度阈值
  float nms_threshold_;                 // nms阈值

  std::vector<std::shared_ptr<trt::Memory<unsigned char>>> preprocess_buffers_; // 创建预处理需要的buffer
  trt::Memory<float> input_buffer_,bbox_predict_,output_boxarray_;

  int network_input_width_,network_input_height_;    // 网络输入的h,w
  Norm normalize_;              // 数据归一化类型
  std::vector<int> bbox_head_dims_;     // 网络输出头的维度
  int num_classes_=0;
  bool isdynamic_model_=false;          // 模型是否为动态batch

  virtual ~InferImpl()=default;

  // 根据批次大小调整共享内存大小
  void adjust_memory(int batch_size){
    size_t input_numel=network_input_width_*network_input_height_*3;
    input_buffer_.gpu(batch_size*input_numel);              // 分配gpu内存
    bbox_predict_.gpu(batch_size*bbox_head_dims_[1]*bbox_head_dims_[2]);      // 分配网络输出gpu内存
    output_boxarray_.gpu(batch_size*(32+MAX_IMAGE_BOXES*NUM_BOX_ELEMENT));    // 分配输出box的
    output_boxarray_.cpu(batch_size*(32+MAX_IMAGE_BOXES*NUM_BOX_ELEMENT));

    // 如果preprocess_buffers_的size小于batch_size，进行扩展
    if((int)preprocess_buffers_.size()<batch_size) {
      for(int i=preprocess_buffers_.size();i<batch_size;++i)
        preprocess_buffers_.push_back(std::make_shared<trt::Memory<unsigned char>>());
    }
  }

  /* 对输入图像进行预处理 */
  void preprocess(int ibatch,const Image &image,
                  std::shared_ptr<trt::Memory<unsigned char>> preprocess_buffer,
                  AffineMatrix &affine,void *stream=nullptr){
    // 根据输入图像以及网络输入计算转换仿射矩阵
    affine.compute(std::make_tuple(image.width,image.height),
                   std::make_tuple(network_input_width_,network_input_height_));

    // 网络输入number element
    size_t input_numel=network_input_width_*network_input_height_*3;

    // todo
    float *input_device=input_buffer_.gpu()+ibatch*input_numel;

    size_t size_image=image.width*image.height*3;       // 原始图像元素个数
    size_t size_matrix=upbound(sizeof(affine.d2i),32);  // 内存对齐

    uint8_t *gpu_workspace=preprocess_buffer->gpu(size_matrix+size_image);
    float *affine_matrix_device=(float *)gpu_workspace;
    uint8_t *image_device=gpu_workspace+size_matrix;

    uint8_t *cpu_workspace=preprocess_buffer->cpu(size_matrix+size_image);
    float *affine_matrix_host=(float *)cpu_workspace;
    uint8_t *image_host=cpu_workspace+size_matrix;

    cudaStream_t stream_=(cudaStream_t)stream;
    memcpy(image_host,image.bgrptr,size_image);
    memcpy(affine_matrix_host,affine.d2i,sizeof(affine.d2i));
    // 将image数据从host复制到device
    checkRuntime(
        cudaMemcpyAsync(image_device,image_host,size_image,cudaMemcpyHostToDevice,stream_));
    // 将反射矩阵从host复制到device
    checkRuntime(cudaMemcpyAsync(affine_matrix_device,affine_matrix_host,sizeof(affine.d2i),
                                 cudaMemcpyHostToDevice,stream_));
    // 反射变化、双线性插值、归一化
    warp_affine_bilinear_and_normalize_plane(image_device,image.width*3,image.width,
                                             image.height,input_device,network_input_width_,
                                             network_input_height_,affine_matrix_device,114,
                                             normalize_,stream_);
  }

  // 加载trtexec生成的engine模型
  bool load(const std::string &engine_file,float confidence_threshold,float nms_threshold){
    trt_=trt::load(engine_file);      // 加载engine文件，返回trt::infer派生类智能指针
    if(trt_==nullptr) return false;   // 加载失败直接返回

    trt_->print();      // 打印网络输入输出信息
    this->confidence_threshold_=confidence_threshold;
    this->nms_threshold_=nms_threshold;

    auto input_dim=trt_->static_dims(0);      // 获取输入维度
    bbox_head_dims_=trt_->static_dims(1);     // 获取输出维度

    network_input_width_=input_dim[3];       // input_dim={-1,c,h,w}
    network_input_height_=input_dim[2];
    isdynamic_model_=trt_->has_dynamic_dim();   // 判断模型是否用动态的维度

    // yolov8归一化方式以及class维度
    normalize_=Norm::alpha_beta(1/255.0f,0.0f,ChannelType::SwapRB);
    num_classes_=bbox_head_dims_[2]-4;
    return true;
  }

  // 模型推理，单张图像
  virtual BoxArray forward(const Image &image,void *stream=nullptr) override {
    auto output=forwards({image},stream);
    if(output.empty()) return {};        // 返回返回值的初始化值
    return output[0];
  }

  // 推理多张图像
  virtual std::vector<BoxArray> forwards(const std::vector<Image> &images,void *stream=nullptr) override {
    int num_image=images.size();            // 获取输入图像数量
    if(num_image==0) return {};            // 如果输入数量为0，直接返回
    auto input_dims=trt_->static_dims(0);   // 获取模型输入维度
    // 推理时的batch大小
    int infer_batch_size=input_dims[0];
    // 如果模型输入batch size和输入图像数量不同
    if(infer_batch_size!=num_image){
      // 如果是动态输入模型，设置模型infer时输入维度
      if(isdynamic_model_){
        infer_batch_size=num_image;
        input_dims[0]=num_image;
        if(!trt_->set_run_dims(0,input_dims)) return {};        // 设置输入维度失败返回空的vector<BoxArray>
      }else{
        // 如果不是动态batch模型，并且infer batch size小于输入图像数量，直接返回空vector<BoxArray>
        if(infer_batch_size<num_image){
          INFO(
              "When using static shape model, number of images[%d] must be "
              "less than or equal to the maximum batch[%d].",
              num_image, infer_batch_size);
          return {};
        }
      }
    }
    // 根据batch size调整memory
    adjust_memory(infer_batch_size);

    std::vector<AffineMatrix> affine_matrixs(num_image);
    cudaStream_t stream_=(cudaStream_t)stream;        // 定义cuda stream
    // 逐张预处理输入图像
    for (int i=0;i<num_image;++i)
      preprocess(i,images[i],preprocess_buffers_[i],affine_matrixs[i],stream);

    float *bbox_output_device=bbox_predict_.gpu();            // 返回创建gpu内存对应的指针
    std::vector<void *> bindings{input_buffer_.gpu(),bbox_output_device};

    if(!trt_->forward(bindings,stream)){
      INFO("Failed to tensorRT forward.");
      return {};
    }

    for (int ib=0;ib<num_image;++ib){
      float *boxarray_device=output_boxarray_.gpu()+ib*(32+MAX_IMAGE_BOXES*NUM_BOX_ELEMENT);
      float *affine_matrix_device=(float *)preprocess_buffers_[ib]->gpu();
      float *image_based_bbox_output=bbox_output_device+ib*(bbox_head_dims_[1]*bbox_head_dims_[2]);
      // 将内存设置为0
      checkRuntime(cudaMemsetAsync(boxarray_device,0,sizeof(int),stream_));
      // 解码网络输出
      decode_kernel_invoker(image_based_bbox_output,bbox_head_dims_[1],num_classes_,
                            bbox_head_dims_[2],confidence_threshold_,nms_threshold_,
                            affine_matrix_device,boxarray_device,MAX_IMAGE_BOXES,stream_);
    }

    // 将数据从gpu复制到cpu
    checkRuntime(cudaMemcpyAsync(output_boxarray_.cpu(),output_boxarray_.gpu(),
                                 output_boxarray_.gpu_bytes(),cudaMemcpyDeviceToHost,stream_));
    checkRuntime(cudaStreamSynchronize(stream_));         // 同步流

    std::vector<BoxArray> arrout(num_image);
    int imemory=0;
    for (int ib=0;ib<num_image;++ib) {
      float *parray=output_boxarray_.cpu()+ib*(32+MAX_IMAGE_BOXES*NUM_BOX_ELEMENT);
      int count=min(MAX_IMAGE_BOXES,(int)*parray);
      BoxArray &output=arrout[ib];
      output.reserve(count);
      for(int i=0;i<count;++i){
        float *pbox=parray+1+i*NUM_BOX_ELEMENT;
        int label=pbox[5];
        int keepflag=pbox[6];
        if (keepflag==1){
          Box result_object_box(pbox[0],pbox[1],pbox[2],pbox[3],pbox[4],label);
          output.emplace_back(result_object_box);
        }
      }
    }
    return arrout;
  }
};

// 加载原始数据
Infer *loadraw(const std::string &engine_file,float confidence_threshold,float nms_threshold){
  InferImpl *impl=new InferImpl();
  if(!impl->load(engine_file,confidence_threshold,nms_threshold)){
    delete impl;
    impl=nullptr;
  }
  return impl;
}

// 加载模型
std::shared_ptr<Infer> load(const std::string &engine_file,float confidence_threshold,float nms_threshold){
  return std::shared_ptr<InferImpl>((InferImpl *)loadraw(engine_file,confidence_threshold,nms_threshold));
}

// hsv转换成bgr
std::tuple<uint8_t,uint8_t,uint8_t>hsv2bgr(float h,float s,float v){
  const int h_i=static_cast<int>(h*6);
  const float f=h * 6 - h_i;
  const float p=v * (1 - s);
  const float q=v * (1 - f * s);
  const float t=v * (1 - (1 - f) * s);
  float r, g, b;
  switch (h_i) {
    case 0:
      r = v, g = t, b = p;
      break;
    case 1:
      r = q, g = v, b = p;
      break;
    case 2:
      r = p, g = v, b = t;
      break;
    case 3:
      r = p, g = q, b = v;
      break;
    case 4:
      r = t, g = p, b = v;
      break;
    case 5:
      r = v, g = p, b = q;
      break;
    default:
      r = 1, g = 1, b = 1;
      break;
  }
  return std::make_tuple(static_cast<uint8_t>(b * 255), static_cast<uint8_t>(g * 255),
                    static_cast<uint8_t>(r * 255));
}
// 随机颜色画图
std::tuple<uint8_t, uint8_t, uint8_t> random_color(int id) {
  float h_plane = ((((unsigned int)id << 2) ^ 0x937151) % 100) / 100.0f;
  float s_plane = ((((unsigned int)id << 3) ^ 0x315793) % 100) / 100.0f;
  return hsv2bgr(h_plane, s_plane, 1);
}
};  // namespace yolo