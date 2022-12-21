#ifndef CAFFE_MACROS_HPP_
#define CAFFE_MACROS_HPP_

#if __CUDACC_VER_MAJOR__ >= 9
#undef __CUDACC_VER__
#define __CUDACC_VER__ \
  ((__CUDACC_VER_MAJOR__ * 10000) + (__CUDACC_VER_MINOR__ * 100))
#endif

#if BOOST_VERSION >= 106100
// error: class "boost::common_type<long, long>" has no member "type"
#define BOOST_NO_CXX11_VARIADIC_TEMPLATES
#if defined(__CUDACC_VER_MAJOR__) && defined(__CUDACC_VER_MINOR__) && defined(__CUDACC_VER_BUILD__)
#define BOOST_CUDA_VERSION \
  __CUDACC_VER_MAJOR__ * 1000000 + __CUDACC_VER_MINOR__ * 10000 + __CUDACC_VER_BUILD__
#else
#define BOOST_CUDA_VERSION 8000000
#endif
#endif

// Ubuntu 18.04 fails to build boost::property_tree by NVCC 10.1
#if BOOST_VERSION == 106501 && CUDA_VERSION == 10010
#define CAFFE_NO_BOOST_PROPERTY_TREE 1
#endif

#if CUDA_VERSION >= 8000
#  define CAFFE_DATA_HALF CUDA_R_16F
#else
#  define CAFFE_DATA_HALF CUBLAS_DATA_HALF
#endif
// Convert macro to string
#define STRINGIFY(m) #m
#define AS_STRING(m) STRINGIFY(m)

// gflags 2.1 issue: namespace google was changed to gflags without warning.
// Luckily we will be able to use GFLAGS_GFLAGS_H_ to detect if it is version
// 2.1. If yes, we will add a temporary solution to redirect the namespace.
// TODO(Yangqing): Once gflags solves the problem in a more elegant way, let's
// remove the following hack.
#ifndef GFLAGS_GFLAGS_H_
namespace gflags = google;
#endif  // GFLAGS_GFLAGS_H_

// Disable the copy and assignment operator for a class.
#define DISABLE_COPY_MOVE_AND_ASSIGN(classname) \
  classname(const classname&) = delete;\
  classname(classname&&) = delete;\
  classname& operator=(const classname&) = delete; \
  classname& operator=(classname&&) = delete

#define INSTANTIATE_CLASS_CPU(classname) \
  char gInstantiationGuard##classname; \
  template class classname<float>; \
  template class classname<double>

#define INSTANTIATE_CLASS_CPU_FB(classname) \
  char gInstantiationGuard##classname; \
  template class classname<float, float>; \
  template class classname<float, double>; \
  template class classname<double, float>; \
  template class classname<double, double>

// Instantiate a class with float and double specifications.
# define INSTANTIATE_CLASS(classname) \
    INSTANTIATE_CLASS_CPU(classname); \
    template class classname<float16>

# define INSTANTIATE_CLASS_FB(classname) \
    INSTANTIATE_CLASS_CPU_FB(classname); \
    template class classname<float16, float>; \
    template class classname<float, float16>; \
    template class classname<float16, double>; \
    template class classname<double, float16>; \
    template class classname<float16, float16>

# define INSTANTIATE_LAYER_GPU_FORWARD(classname) \
  template void classname<float>::Forward_gpu( \
      const std::vector<Blob*>& bottom, \
      const std::vector<Blob*>& top); \
  template void classname<double>::Forward_gpu( \
      const std::vector<Blob*>& bottom, \
      const std::vector<Blob*>& top)

# define INSTANTIATE_LAYER_GPU_FORWARD_F16_FB(classname, member) \
  template void classname<float16, float>::member( \
      const std::vector<Blob*>& bottom, \
      const std::vector<Blob*>& top); \
  template void classname<float, float16>::member( \
      const std::vector<Blob*>& bottom, \
      const std::vector<Blob*>& top); \
  template void classname<float16, double>::member( \
      const std::vector<Blob*>& bottom, \
      const std::vector<Blob*>& top); \
  template void classname<double, float16>::member( \
      const std::vector<Blob*>& bottom, \
      const std::vector<Blob*>& top); \
  template void classname<float16, float16>::member( \
      const std::vector<Blob*>& bottom, \
      const std::vector<Blob*>& top)

# define INSTANTIATE_LAYER_GPU_BACKWARD_F16_FB(classname, member) \
  template void classname<float16, float>::member( \
      const std::vector<Blob*>& top, \
      const std::vector<bool>& propagate_down, \
      const std::vector<Blob*>& bottom); \
  template void classname<float, float16>::member( \
      const std::vector<Blob*>& top, \
      const std::vector<bool>& propagate_down, \
      const std::vector<Blob*>& bottom); \
  template void classname<float16, double>::member( \
      const std::vector<Blob*>& top, \
      const std::vector<bool>& propagate_down, \
      const std::vector<Blob*>& bottom); \
  template void classname<double, float16>::member( \
      const std::vector<Blob*>& top, \
      const std::vector<bool>& propagate_down, \
      const std::vector<Blob*>& bottom); \
  template void classname<float16, float16>::member( \
      const std::vector<Blob*>& top, \
      const std::vector<bool>& propagate_down, \
      const std::vector<Blob*>& bottom)

# define INSTANTIATE_LAYER_GPU_FORWARD_FB(classname, member) \
  template void classname<float, float>::member( \
      const std::vector<Blob*>& bottom, \
      const std::vector<Blob*>& top); \
  template void classname<float, double>::member( \
      const std::vector<Blob*>& bottom, \
      const std::vector<Blob*>& top); \
  template void classname<double, float>::member( \
      const std::vector<Blob*>& bottom, \
      const std::vector<Blob*>& top); \
  template void classname<double, double>::member( \
      const std::vector<Blob*>& bottom, \
      const std::vector<Blob*>& top);

# define INSTANTIATE_LAYER_GPU_BACKWARD_FB(classname, member) \
  template void classname<float, float>::member( \
      const std::vector<Blob*>& top, \
      const std::vector<bool>& propagate_down, \
      const std::vector<Blob*>& bottom); \
  template void classname<float, double>::member( \
      const std::vector<Blob*>& top, \
      const std::vector<bool>& propagate_down, \
      const std::vector<Blob*>& bottom); \
  template void classname<double, float>::member( \
      const std::vector<Blob*>& top, \
      const std::vector<bool>& propagate_down, \
      const std::vector<Blob*>& bottom); \
  template void classname<double, double>::member( \
      const std::vector<Blob*>& top, \
      const std::vector<bool>& propagate_down, \
      const std::vector<Blob*>& bottom)

#  define INSTANTIATE_LAYER_GPU_FORWARD_ONLY_FB(classname) \
    INSTANTIATE_LAYER_GPU_FORWARD_FB(classname, Forward_gpu); \
    INSTANTIATE_LAYER_GPU_FORWARD_F16_FB(classname, Forward_gpu)

#  define INSTANTIATE_LAYER_GPU_BACKWARD_ONLY_FB(classname) \
    INSTANTIATE_LAYER_GPU_BACKWARD_FB(classname, Backward_gpu); \
    INSTANTIATE_LAYER_GPU_BACKWARD_F16_FB(classname, Backward_gpu)

#  define INSTANTIATE_LAYER_GPU_FUNCS_FB(classname) \
    INSTANTIATE_LAYER_GPU_FORWARD_FB(classname, Forward_gpu); \
    INSTANTIATE_LAYER_GPU_FORWARD_F16_FB(classname, Forward_gpu); \
    INSTANTIATE_LAYER_GPU_BACKWARD_FB(classname, Backward_gpu); \
    INSTANTIATE_LAYER_GPU_BACKWARD_F16_FB(classname, Backward_gpu)

#  define INSTANTIATE_LAYER_GPU_FW_MEMBER_FB(classname, member) \
    INSTANTIATE_LAYER_GPU_FORWARD_FB(classname, member); \
    INSTANTIATE_LAYER_GPU_FORWARD_F16_FB(classname, member)

#  define INSTANTIATE_LAYER_GPU_BW_MEMBER_FB(classname, member) \
    INSTANTIATE_LAYER_GPU_BACKWARD_FB(classname, member); \
    INSTANTIATE_LAYER_GPU_BACKWARD_F16_FB(classname, member)

// A simple macro to mark codes that are not implemented, so that when the code
// is executed we will see a fatal log.
#define NOT_IMPLEMENTED LOG(FATAL) << "Not Implemented Yet"

// Common functions and classes from std that caffe often uses.
using std::fstream;
using std::ios;
using std::iterator;
using std::make_pair;
using std::map;
using std::unordered_map;
using std::ostringstream;
using std::pair;
using std::set;
using std::string;
using std::stringstream;
using std::vector;
using std::unique_ptr;
using std::mutex;
using std::lock_guard;
using std::ostringstream;
// std::shared_ptr would be better but pycaffe breaks
using boost::shared_ptr;
using boost::weak_ptr;
using boost::make_shared;
using boost::shared_mutex;
using boost::shared_lock;
using boost::upgrade_lock;
using boost::unique_lock;
using boost::upgrade_to_unique_lock;

#define CAFFE_WS_CONV 0
#define CAFFE_WS_CONV_WEIGHTS 1
#define CAFFE_WS_TOTAL 2

#endif  // CAFFE_MACROS_HPP_
