// SPDX-FileCopyrightText: Copyright contributors to the kvcached project
// SPDX-License-Identifier: Apache-2.0

#include "page.hpp"
#include "constants.hpp"
#include "gpu_utils.hpp"

namespace kvcached {

GPUPage::GPUPage(page_id_t page_id, int dev_idx, size_t page_size)
    : page_id_(page_id), dev_(dev_idx),
      page_size_(page_size > 0 ? page_size : kPageSize), handle_(0) {
  // CHECK_DRV(gpuCtxGetDevice(&dev_));

  gpu_mem_alloc_prop_t prop = {
      .type = GPU_MEM_ALLOCATION_TYPE_PINNED,
      .location =
          {
              .type = GPU_MEM_LOCATION_TYPE_DEVICE,
              .id = dev_,
          },
  };
  CHECK_DRV(gpuMemCreate(&handle_, page_size_, &prop, 0));
}

GPUPage::~GPUPage() { CHECK_DRV(gpuMemRelease(handle_)); }

bool GPUPage::map(generic_ptr_t vaddr, bool set_access) {
  gpu_mem_access_desc_t accessDesc_{
      .location =
          {
              .type = GPU_MEM_LOCATION_TYPE_DEVICE,
              .id = dev_,
          },
      .flags = GPU_MEM_ACCESS_FLAGS_PROT_READWRITE,
  };
  CHECK_DRV(gpuMemMap(reinterpret_cast<gpu_devptr_t>(vaddr), page_size_, 0,
                      handle_, 0));
  if (set_access)
    CHECK_DRV(gpuMemSetAccess(reinterpret_cast<gpu_devptr_t>(vaddr), page_size_,
                              &accessDesc_, 1));
  return true;
}

// TODO: finish CPUPage impl.
CPUPage::CPUPage(page_id_t page_id, size_t page_size)
    : page_id_(page_id), page_size_(page_size > 0 ? page_size : kPageSize),
      mapped_addr_(nullptr) {}

CPUPage::~CPUPage() {}

bool CPUPage::map(void *vaddr, bool set_access) {
  mapped_addr_ = vaddr;
  return true;
}

} // namespace kvcached
