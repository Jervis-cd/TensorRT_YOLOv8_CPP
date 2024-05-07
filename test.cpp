#include<iostream>
#include<string>
#include<vector>
#include<tuple>

#include "src/infer.hpp"

BaseMemory::BaseMemory(void *cpu,size_t cpu_bytes,void *gpu,size_t gpu_bytes){
  // 构造函数指定cpu/gpu的指针以及容量
  reference(cpu,cpu_bytes,gpu,gpu_bytes);
}

void BaseMemory::reference(void *cpu,size_t cpu_bytes,void *gpu,size_t gpu_bytes){
  release();

  if(cpu==nullptr || cpu_bytes_==0){
    cpu=nullptr;
    cpu_bytes=0;
  }

  if(gpu==nullptr || gpu_bytes_==0){
    gpu=nullptr;
    gpu_bytes=0;
  }

  this->cpu_=cpu;
  this->cpu_capacity_=cpu_bytes;
  this->cpu_bytes_=cpu_bytes;
  this->gpu_=gpu;
  this->gpu_capacity_=gpu_bytes;
  this->gpu_bytes_=gpu_bytes;

  this->owner_cpu_=!(cpu && cpu_bytes>0);
  this->owner_gpu_=!(gpu && gpu_bytes>0);
}

BaseMemory::~BaseMemory(){release();}       // 析构时释放cpu/gpu指向的内存

void *BaseMemory::gpu_realloc(size_t bytes){
  if(gpu_capacity_<bytes){
    release_gpu();

    gpu_capacity_=bytes;
    checkRuntime(cudaMalloc(&gpu_,bytes));
  }
  gpu_bytes_=bytes;
  return gpu_;
}

void *BaseMemory::cpu_realloc(size_t bytes){
  if(cpu_capacity_<bytes){
    release_cpu();

    cpu_capacity_=bytes;
    checkRuntime(cudaMallocHost(&cpu_,bytes));
    Assert(cpu_!=nullptr);        // cpu内存分配失败会返回空指针，直接终止程序
  }
  cpu_bytes_=bytes;
  return cpu_;
}

// 同时释放gpu和cpu内存
void BaseMemory::release(){
  release_cpu();
  release_gpu();
}

void BaseMemory::release_cpu(){
  if(cpu_){
    if(owner_cpu_){
      checkRuntime(cudaFreeHost(cpu_));
    }
    cpu_=nullptr;
  }
  cpu_capacity_=0;
  cpu_bytes_=0;
}

void BaseMemory::release_gpu(){
  if(gpu_){
    if(owner_gpu_){
      checkRuntime(cudaFree(gpu_));
    }
    gpu_=nullptr;
  }
  gpu_capacity_=0;
  gpu_bytes_=0;
}


int main(){

    trt::Memory<float> input_buffer_,bbox_predict_,output_boxarray_;
    std::cout<<sizeof(input_buffer_)<<std::endl;
    std::cout<<sizeof(int *)<<std::endl;
}




