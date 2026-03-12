// SPDX-FileCopyrightText: Copyright contributors to the kvcached project
// SPDX-License-Identifier: Apache-2.0

#pragma once

#ifdef USE_ROCM

#include <hip/hip_runtime.h>

// Type aliases
using gpu_device_t = hipDevice_t;
using gpu_devptr_t = hipDeviceptr_t;
using gpu_mem_handle_t = hipMemGenericAllocationHandle_t;
using gpu_result_t = hipError_t;
using gpu_context_t = hipCtx_t;
using gpu_mem_alloc_prop_t = hipMemAllocationProp;
using gpu_mem_access_desc_t = hipMemAccessDesc;
using gpu_rt_error_t = hipError_t;

// Success codes
#define GPU_SUCCESS hipSuccess
#define GPU_RT_SUCCESS hipSuccess

// Memory allocation type/location constants
#define GPU_MEM_ALLOCATION_TYPE_PINNED hipMemAllocationTypePinned
#define GPU_MEM_LOCATION_TYPE_DEVICE hipMemLocationTypeDevice
#define GPU_MEM_ACCESS_FLAGS_PROT_READWRITE hipMemAccessFlagsProtReadWrite
#define GPU_MEM_ALLOC_GRANULARITY_MINIMUM hipMemAllocationGranularityMinimum

// VMM API functions
#define gpuMemCreate hipMemCreate
#define gpuMemRelease hipMemRelease
#define gpuMemMap hipMemMap
#define gpuMemUnmap hipMemUnmap
#define gpuMemSetAccess hipMemSetAccess
#define gpuMemAddressReserve hipMemAddressReserve
#define gpuMemAddressFree hipMemAddressFree
#define gpuMemGetAllocationGranularity hipMemGetAllocationGranularity

// Context API functions
#define gpuCtxGetDevice hipCtxGetDevice
#define gpuCtxGetCurrent hipCtxGetCurrent

// Runtime API functions
#define gpuFree hipFree
#define gpuGetErrorString hipGetErrorString

// Error string helper: HIP returns const char* directly
inline const char *gpu_get_drv_error_string(gpu_result_t result) {
  return hipGetErrorString(result);
}

inline const char *gpu_get_rt_error_string(gpu_rt_error_t result) {
  return hipGetErrorString(result);
}

#else // CUDA

#include <cuda.h>
#include <cuda_runtime.h>

// Type aliases
using gpu_device_t = CUdevice;
using gpu_devptr_t = CUdeviceptr;
using gpu_mem_handle_t = CUmemGenericAllocationHandle;
using gpu_result_t = CUresult;
using gpu_context_t = CUcontext;
using gpu_mem_alloc_prop_t = CUmemAllocationProp;
using gpu_mem_access_desc_t = CUmemAccessDesc;
using gpu_rt_error_t = cudaError_t;

// Success codes
#define GPU_SUCCESS CUDA_SUCCESS
#define GPU_RT_SUCCESS cudaSuccess

// Memory allocation type/location constants
#define GPU_MEM_ALLOCATION_TYPE_PINNED CU_MEM_ALLOCATION_TYPE_PINNED
#define GPU_MEM_LOCATION_TYPE_DEVICE CU_MEM_LOCATION_TYPE_DEVICE
#define GPU_MEM_ACCESS_FLAGS_PROT_READWRITE CU_MEM_ACCESS_FLAGS_PROT_READWRITE
#define GPU_MEM_ALLOC_GRANULARITY_MINIMUM CU_MEM_ALLOC_GRANULARITY_MINIMUM

// VMM API functions
#define gpuMemCreate cuMemCreate
#define gpuMemRelease cuMemRelease
#define gpuMemMap cuMemMap
#define gpuMemUnmap cuMemUnmap
#define gpuMemSetAccess cuMemSetAccess
#define gpuMemAddressReserve cuMemAddressReserve
#define gpuMemAddressFree cuMemAddressFree
#define gpuMemGetAllocationGranularity cuMemGetAllocationGranularity

// Context API functions
#define gpuCtxGetDevice cuCtxGetDevice
#define gpuCtxGetCurrent cuCtxGetCurrent

// Runtime API functions
#define gpuFree cudaFree
#define gpuGetErrorString cudaGetErrorString

// Error string helper: CUDA uses out-parameter
inline const char *gpu_get_drv_error_string(gpu_result_t result) {
  const char *errMsg = nullptr;
  cuGetErrorString(result, &errMsg);
  return errMsg ? errMsg : "unknown error";
}

inline const char *gpu_get_rt_error_string(gpu_rt_error_t result) {
  return cudaGetErrorString(result);
}

#endif // USE_ROCM
