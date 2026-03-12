// SPDX-FileCopyrightText: Copyright contributors to the kvcached project
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "gpu_compat.hpp"

#include <cassert>
#include <iostream>

#define LOGE(format, ...)                                                      \
  fprintf(stderr, "ERROR: %s:%d: " format "\n", __FILE__, __LINE__,            \
          ##__VA_ARGS__);                                                      \
  fflush(stderr);

#define LOGW(format, ...)                                                      \
  fprintf(stderr, "WARNING: %s:%d: " format "\n", __FILE__, __LINE__,          \
          ##__VA_ARGS__);                                                      \
  fflush(stderr);

#define ASSERT(cond, ...)                                                      \
  {                                                                            \
    if (!(cond)) {                                                             \
      LOGE(__VA_ARGS__);                                                       \
      assert(0);                                                               \
    }                                                                          \
  }

#define WARN(cond, ...)                                                        \
  {                                                                            \
    if (!(cond)) {                                                             \
      LOGE(__VA_ARGS__);                                                       \
    }                                                                          \
  }

#define DRV_CALL(call)                                                         \
  {                                                                            \
    gpu_result_t result = (call);                                              \
    if (GPU_SUCCESS != result) {                                               \
      const char *errMsg = gpu_get_drv_error_string(result);                   \
      ASSERT(0, "Error when exec " #call " %s-%d code:%d err:%s",             \
             __FUNCTION__, __LINE__, (int)result, errMsg);                     \
    }                                                                          \
  }

#define DRV_CALL_RET(call, status_val)                                         \
  {                                                                            \
    gpu_result_t result = (call);                                              \
    if (GPU_SUCCESS != result) {                                               \
      const char *errMsg = gpu_get_drv_error_string(result);                   \
      WARN(0, "Error when exec " #call " %s-%d code:%d err:%s", __FUNCTION__, \
           __LINE__, (int)result, errMsg);                                     \
    }                                                                          \
    status_val = result;                                                       \
  }

static inline void checkRtError(gpu_rt_error_t res, const char *tok,
                                const char *file, unsigned line) {
  if (res != GPU_RT_SUCCESS) {
    std::cerr << file << ':' << line << ' ' << tok
              << " failed in GPU runtime (" << (unsigned)res
              << "): " << gpu_get_rt_error_string(res) << std::endl;
    abort();
  }
}

#define CHECK_RT(x) checkRtError(x, #x, __FILE__, __LINE__)

static inline void checkDrvError(gpu_result_t res, const char *tok,
                                 const char *file, unsigned line) {
  if (res != GPU_SUCCESS) {
    const char *errStr = gpu_get_drv_error_string(res);
    std::cerr << file << ':' << line << ' ' << tok
              << " failed in GPU driver (" << (unsigned)res << "): " << errStr
              << std::endl;
    abort();
  }
}

#define CHECK_DRV(x) checkDrvError(x, #x, __FILE__, __LINE__)
