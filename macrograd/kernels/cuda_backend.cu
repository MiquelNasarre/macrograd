#include "macrograd_error.h"
#include "macrograd.h"
#include "cuda_backend.h"

#include <math.h>

#include <cuda_runtime.h>
#include <cublasLt.h>
#include <curand_kernel.h>
#include <math_constants.h>
#include <device_launch_parameters.h>

/*
--------------------------------------------------------------------------------------------------------------------------
 CUDA Macros
--------------------------------------------------------------------------------------------------------------------------
*/

// Macro for intellisense not to bother me with launch calls.
#ifdef __INTELLISENSE__
#define CUDA_LAUNCH(kernel, grid, block, ...) kernel(__VA_ARGS__)
#else
#define CUDA_LAUNCH(kernel, grid, block, ...) kernel<<<(grid), (block), 0, stream::get()>>>(__VA_ARGS__)
#endif

#ifdef __INTELLISENSE__
#define CUDART_INF_F 0.0f
inline float __expf(float x) { return expf(x); }
inline float __logf(float x) { return logf(x); }
inline void __syncthreads() {}
#endif

// Checks CUDA expression, if 'cudaError != cudaSuccess' raises a TENSOR_ERROR. 
#define CUDA_CHECK(x)       do { cudaError_t err = (x); if(err != cudaSuccess) TENSOR_ERROR("(CUDA ERROR)\n%s\n", cudaGetErrorString(err)); } while(0)
#define CUBLAS_CHECK(x)     do { cublasStatus_t _s = (x); if (_s != CUBLAS_STATUS_SUCCESS) TENSOR_ERROR("(CUBLAS ERROR)\nError Code: 0x%08X", _s); } while(0)

/*
--------------------------------------------------------------------------------------------------------------------------
 Global Stream Class
--------------------------------------------------------------------------------------------------------------------------
*/

// Global Stream class.
class stream
{
private:
    static inline cudaStream_t create() { cudaStream_t s; CUDA_CHECK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking)); return s; }
public:
    // Global stream used by all ops.
    static inline cudaStream_t get() { static thread_local cudaStream_t s = create(); return s; }
};

class device
{
private:
    static inline int count()
    {
        int device;
        CUDA_CHECK(cudaGetDevice(&device));

        int sm;
        CUDA_CHECK(cudaDeviceGetAttribute(
            &sm,
            cudaDevAttrMultiProcessorCount,
            device
        ));
        return sm;
    }
public:
    static int sm_count()
    {
        static int value = count();
        return value;
    }
};

/*
--------------------------------------------------------------------------------------------------------------------------
 Memory Pool functions
--------------------------------------------------------------------------------------------------------------------------
*/

// Get your free GPU memory here! Freshly zeroed!

void* MemPool::allocate(size_t byte_size)
{
    if (!byte_size) return nullptr;

    void* data_ptr = nullptr;
    CUDA_CHECK(cudaMallocAsync(&data_ptr, byte_size, stream::get()));
    CUDA_CHECK(cudaMemsetAsync(data_ptr, 0, byte_size, stream::get()));
    return data_ptr;
}

// Recicle your GPU memory here!

void MemPool::free(void* data_ptr)
{
    if (!data_ptr) return;
    CUDA_CHECK(cudaFreeAsync(data_ptr, stream::get()));
}

/*
--------------------------------------------------------------------------------------------------------------------------
 Other functions
--------------------------------------------------------------------------------------------------------------------------
*/

void cuda::zero_data(void* data_ptr, size_t byte_size)
{
    CUDA_CHECK(cudaMemsetAsync(data_ptr, 0, byte_size, stream::get()));
}

// Copies the data from the CPU pointer to the GPU pointer.

void cuda::copy_cpu_to_gpu(void* gpu_data_ptr, const void* cpu_data_ptr, size_t byte_size)
{
    CUDA_CHECK(cudaMemcpyAsync(gpu_data_ptr, cpu_data_ptr, byte_size, cudaMemcpyHostToDevice, stream::get()));
    CUDA_CHECK(cudaStreamSynchronize(stream::get()));
}

// Copies the data from the GPU pointer to the CPU pointer.

void cuda::copy_gpu_to_cpu(void* cpu_data_ptr, const void* gpu_data_ptr, size_t byte_size)
{
    CUDA_CHECK(cudaMemcpyAsync(cpu_data_ptr, gpu_data_ptr, byte_size, cudaMemcpyDeviceToHost, stream::get()));
    CUDA_CHECK(cudaStreamSynchronize(stream::get()));
}

// Copies data from one GPU poiner to another GPU pointer.

void cuda::copy_gpu_to_gpu(void* dst_ptr, const void* src_ptr, size_t byte_size)
{
    CUDA_CHECK(cudaMemcpyAsync(dst_ptr, src_ptr, byte_size, cudaMemcpyDeviceToDevice, stream::get()));
}

// Waits until global stream is done.

void cuda::synchronize()
{
    CUDA_CHECK(cudaStreamSynchronize(stream::get()));
}

// Set gradient element value to one for backprop.

void cuda::set_to_one(void* data_ptr)
{
    static float one = 1.f;
    CUDA_CHECK(cudaMemcpyAsync(data_ptr, &one, sizeof(float), cudaMemcpyHostToDevice, stream::get()));
}

/*
--------------------------------------------------------------------------------------------------------------------------
 Kernel functions
--------------------------------------------------------------------------------------------------------------------------
*/

__global__ void set_scalar_kernel_vec(float4* out, float val, size_t num_vec)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    for (size_t i = idx; i < num_vec; i += stride)
    {
        float4 ov;

        ov.x = val;
        ov.y = val;
        ov.z = val;
        ov.w = val;

        out[i] = ov;
    }
}
__global__ void set_scalar_kernel(float* out, float val, size_t num_elements)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    for (size_t i = idx; i < num_elements; i += stride)
        out[i] = val;
}
void kernel_ops::set_scalar(void* out_data, float val, size_t num_elements)
{
    const int BLOCK_SIZE = 256;
    const int grid_size = device::sm_count() * 4;

    // Vectorized addition
    size_t num_vec = num_elements / 4;
    if (num_vec)
    {
        CUDA_LAUNCH(set_scalar_kernel_vec, grid_size, BLOCK_SIZE,
            (float4*)out_data,
            val,
            num_vec
        );
        CUDA_CHECK(cudaGetLastError());
    }

    // Remainder addition
    size_t remainder = num_elements % 4;
    if (remainder)
    {
        size_t offset = num_vec * 4;
        CUDA_LAUNCH(set_scalar_kernel, grid_size, BLOCK_SIZE,
            (float*)out_data + offset,
            val,
            remainder
        );
        CUDA_CHECK(cudaGetLastError());
    }
}

__global__ void add_scalar_kernel_vec(float4* out, float val, size_t num_vec)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    for (size_t i = idx; i < num_vec; i += stride)
    {
        float4 ov = out[i];

        ov.x += val;
        ov.y += val;
        ov.z += val;
        ov.w += val;

        out[i] = ov;
    }
}
__global__ void add_scalar_kernel(float* out, float val, size_t num_elements)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    for (size_t i = idx; i < num_elements; i += stride)
        out[i] += val;
}
void kernel_ops::add_scalar(void* out_data, float val, size_t num_elements)
{
    const int BLOCK_SIZE = 256;
    const int grid_size = device::sm_count() * 4;

    // Vectorized addition
    size_t num_vec = num_elements / 4;
    if (num_vec)
    {
        CUDA_LAUNCH(add_scalar_kernel_vec, grid_size, BLOCK_SIZE,
            (float4*)out_data,
            val,
            num_vec
        );
        CUDA_CHECK(cudaGetLastError());
    }

    // Remainder addition
    size_t remainder = num_elements % 4;
    if (remainder)
    {
        size_t offset = num_vec * 4;
        CUDA_LAUNCH(add_scalar_kernel, grid_size, BLOCK_SIZE,
            (float*)out_data + offset,
            val,
            remainder
        );
        CUDA_CHECK(cudaGetLastError());
    }
}

__global__ void add_tensor_kernel_vec(float4* out, const float4* sum, size_t num_vec)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    for (size_t i = idx; i < num_vec; i += stride)
    {
        float4 ov = out[i];
        float4 sv = sum[i];

        ov.x += sv.x;
        ov.y += sv.y;
        ov.z += sv.z;
        ov.w += sv.w;

        out[i] = ov;
    }
}
__global__ void add_tensor_kernel(float* out, const float* sum, size_t num_elements)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    for (size_t i = idx; i < num_elements; i += stride)
        out[i] += sum[i];
}
void kernel_ops::add_tensor(void* out_data, const void* sum_data, size_t num_elements)
{
    const int BLOCK_SIZE = 256;
    const int grid_size = device::sm_count() * 4;

    // Vectorized addition
    size_t num_vec = num_elements / 4;
    if (num_vec)
    {
        CUDA_LAUNCH(add_tensor_kernel_vec, grid_size, BLOCK_SIZE,
            (float4*)out_data,
            (const float4*)sum_data,
            num_vec
        );
        CUDA_CHECK(cudaGetLastError());
    }

    // Remainder addition
    size_t remainder = num_elements % 4;
    if (remainder)
    {
        size_t offset = num_vec * 4;
        CUDA_LAUNCH(add_tensor_kernel, grid_size, BLOCK_SIZE,
            (float*)out_data + offset,
            (const float*)sum_data + offset,
            remainder
        );
        CUDA_CHECK(cudaGetLastError());
    }
}

__global__ void add_tensor_tensor_kernel_vec(float4* out, const float4* sum0, const float4* sum1, size_t num_vec)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    for (size_t i = idx; i < num_vec; i += stride)
    {
        float4 v0 = sum0[i];
        float4 v1 = sum1[i];

        float4 ov;
        ov.x = v0.x + v1.x;
        ov.y = v0.y + v1.y;
        ov.z = v0.z + v1.z;
        ov.w = v0.w + v1.w;

        out[i] = ov;
    }
}
__global__ void add_tensor_tensor_kernel(float* out, const float* sum0, const float* sum1, size_t num_elements)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    for (size_t i = idx; i < num_elements; i += stride)
        out[i] = sum0[i] + sum1[i];
}
void kernel_ops::add_tensor_tensor(void* out_data, const void* sum0_data, const void* sum1_data, size_t num_elements)
{
    const int BLOCK_SIZE = 256;
    const int grid_size = device::sm_count() * 4;

    // Vectorized addition
    size_t num_vec = num_elements / 4;
    if (num_vec)
    {
        CUDA_LAUNCH(add_tensor_tensor_kernel_vec, grid_size, BLOCK_SIZE,
            (float4*)out_data, 
            (const float4*)sum0_data, 
            (const float4*)sum1_data, 
            num_vec
        );
        CUDA_CHECK(cudaGetLastError());
    }

    // Remainder addition
    size_t remainder = num_elements % 4;
    if (remainder)
    {
        size_t offset = num_vec * 4;
        CUDA_LAUNCH(add_tensor_tensor_kernel, grid_size, BLOCK_SIZE,
            (float*)out_data + offset, 
            (const float*)sum0_data + offset, 
            (const float*)sum1_data + offset, 
            remainder
        );
        CUDA_CHECK(cudaGetLastError());
    }
}

__global__ void multiply_scalar_kernel_vec(float4* out, float val, size_t num_vec)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    for (size_t i = idx; i < num_vec; i += stride)
    {
        float4 ov = out[i];

        ov.x *= val;
        ov.y *= val;
        ov.z *= val;
        ov.w *= val;

        out[i] = ov;
    }
}
__global__ void multiply_scalar_kernel(float* out, float val, size_t num_elements)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    for (size_t i = idx; i < num_elements; i += stride)
        out[i] *= val;
}
void kernel_ops::multiply_scalar(void* out_data, float val, size_t num_elements)
{
    const int BLOCK_SIZE = 256;
    const int grid_size = device::sm_count() * 4;

    // Vectorized addition
    size_t num_vec = num_elements / 4;
    if (num_vec)
    {
        CUDA_LAUNCH(multiply_scalar_kernel_vec, grid_size, BLOCK_SIZE,
            (float4*)out_data,
            val,
            num_vec
        );
        CUDA_CHECK(cudaGetLastError());
    }

    // Remainder addition
    size_t remainder = num_elements % 4;
    if (remainder)
    {
        size_t offset = num_vec * 4;
        CUDA_LAUNCH(multiply_scalar_kernel, grid_size, BLOCK_SIZE,
            (float*)out_data + offset,
            val,
            remainder
        );
        CUDA_CHECK(cudaGetLastError());
    }
}

__global__ void multiply_tensor_kernel_vec(float4* out, const float4* fac, size_t num_vec)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    for (size_t i = idx; i < num_vec; i += stride)
    {
        float4 ov = out[i];
        float4 sv = fac[i];

        ov.x *= sv.x;
        ov.y *= sv.y;
        ov.z *= sv.z;
        ov.w *= sv.w;

        out[i] = ov;
    }
}
__global__ void multiply_tensor_kernel(float* out, const float* fac, size_t num_elements)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    for (size_t i = idx; i < num_elements; i += stride)
        out[i] *= fac[i];
}
void kernel_ops::multiply_tensor(void* out_data, const void* fac_data, size_t num_elements)
{
    const int BLOCK_SIZE = 256;
    const int grid_size = device::sm_count() * 4;

    // Vectorized addition
    size_t num_vec = num_elements / 4;
    if (num_vec)
    {
        CUDA_LAUNCH(multiply_tensor_kernel_vec, grid_size, BLOCK_SIZE,
            (float4*)out_data,
            (const float4*)fac_data,
            num_vec
        );
        CUDA_CHECK(cudaGetLastError());
    }

    // Remainder addition
    size_t remainder = num_elements % 4;
    if (remainder)
    {
        size_t offset = num_vec * 4;
        CUDA_LAUNCH(multiply_tensor_kernel, grid_size, BLOCK_SIZE,
            (float*)out_data + offset,
            (const float*)fac_data + offset,
            remainder
        );
        CUDA_CHECK(cudaGetLastError());
    }
}

__global__ void multiply_tensor_tensor_kernel_vec(float4* out, const float4* fac0, const float4* fac1, size_t num_vec)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    for (size_t i = idx; i < num_vec; i += stride)
    {
        float4 v0 = fac0[i];
        float4 v1 = fac1[i];

        float4 ov;
        ov.x = v0.x * v1.x;
        ov.y = v0.y * v1.y;
        ov.z = v0.z * v1.z;
        ov.w = v0.w * v1.w;

        out[i] = ov;
    }
}
__global__ void multiply_tensor_tensor_kernel(float* out, const float* fac0, const float* fac1, size_t num_elements)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    for (size_t i = idx; i < num_elements; i += stride)
        out[i] = fac0[i] * fac1[i];
}
void kernel_ops::multiply_tensor_tensor(void* out_data, const void* fac0_data, const void* fac1_data, size_t num_elements)
{
    const int BLOCK_SIZE = 256;
    const int grid_size = device::sm_count() * 4;

    // Vectorized addition
    size_t num_vec = num_elements / 4;
    if (num_vec)
    {
        CUDA_LAUNCH(multiply_tensor_tensor_kernel_vec, grid_size, BLOCK_SIZE,
            (float4*)out_data, 
            (const float4*)fac0_data, 
            (const float4*)fac1_data, 
            num_vec
        );
        CUDA_CHECK(cudaGetLastError());
    }

    // Remainder addition
    size_t remainder = num_elements % 4;
    if (remainder)
    {
        size_t offset = num_vec * 4;
        CUDA_LAUNCH(multiply_tensor_tensor_kernel, grid_size, BLOCK_SIZE,
            (float*)out_data + offset, 
            (const float*)fac0_data + offset, 
            (const float*)fac1_data + offset, 
            remainder
        );
        CUDA_CHECK(cudaGetLastError());
    }
}

__global__ void add_multiply_scalar_tensor_kernel_vec(float4* out, float val, const float4* fac, size_t num_vec)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    for (size_t i = idx; i < num_vec; i += stride)
    {
        float4 v = fac[i];
        float4 ov = out[i];

        ov.x += val * v.x;
        ov.y += val * v.y;
        ov.z += val * v.z;
        ov.w += val * v.w;

        out[i] = ov;
    }
}
__global__ void add_multiply_scalar_tensor_kernel(float* out, float val, const float* fac, size_t num_elements)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    for (size_t i = idx; i < num_elements; i += stride)
        out[i] += val * fac[i];
}
void kernel_ops::add_multiply_scalar_tensor(void* out_data, float val, const void* fac_data, size_t num_elements)
{
    const int BLOCK_SIZE = 256;
    const int grid_size = device::sm_count() * 4;

    // Vectorized addition
    size_t num_vec = num_elements / 4;
    if (num_vec)
    {
        CUDA_LAUNCH(add_multiply_scalar_tensor_kernel_vec, grid_size, BLOCK_SIZE,
            (float4*)out_data, val, 
            (const float4*)fac_data, 
            num_vec
        );
        CUDA_CHECK(cudaGetLastError());
    }

    // Remainder addition
    size_t remainder = num_elements % 4;
    if (remainder)
    {
        size_t offset = num_vec * 4;
        CUDA_LAUNCH(add_multiply_scalar_tensor_kernel, grid_size, BLOCK_SIZE,
            (float*)out_data + offset, val,
            (const float*)fac_data + offset, 
            remainder
        );
        CUDA_CHECK(cudaGetLastError());
    }
}

__global__ void subtract_tensor_kernel_vec(float4* out, const float4* sub, size_t num_vec)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    for (size_t i = idx; i < num_vec; i += stride)
    {
        float4 ov = out[i];
        float4 sv = sub[i];

        ov.x -= sv.x;
        ov.y -= sv.y;
        ov.z -= sv.z;
        ov.w -= sv.w;

        out[i] = ov;
    }
}
__global__ void subtract_tensor_kernel(float* out, const float* sub, size_t num_elements)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    for (size_t i = idx; i < num_elements; i += stride)
        out[i] -= sub[i];
}
void kernel_ops::subtract_tensor(void* out_data, const void* sub_data, size_t num_elements)
{
    const int BLOCK_SIZE = 256;
    const int grid_size = device::sm_count() * 4;

    // Vectorized addition
    size_t num_vec = num_elements / 4;
    if (num_vec)
    {
        CUDA_LAUNCH(subtract_tensor_kernel_vec, grid_size, BLOCK_SIZE,
            (float4*)out_data,
            (const float4*)sub_data,
            num_vec
        );
        CUDA_CHECK(cudaGetLastError());
    }

    // Remainder addition
    size_t remainder = num_elements % 4;
    if (remainder)
    {
        size_t offset = num_vec * 4;
        CUDA_LAUNCH(subtract_tensor_kernel, grid_size, BLOCK_SIZE,
            (float*)out_data + offset,
            (const float*)sub_data + offset,
            remainder
        );
        CUDA_CHECK(cudaGetLastError());
    }
}

__global__ void subtract_tensor_tensor_kernel_vec(float4* out, const float4* sum, const float4* sub, size_t num_vec)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    for (size_t i = idx; i < num_vec; i += stride)
    {
        float4 v0 = sum[i];
        float4 v1 = sub[i];

        float4 ov;
        ov.x = v0.x - v1.x;
        ov.y = v0.y - v1.y;
        ov.z = v0.z - v1.z;
        ov.w = v0.w - v1.w;

        out[i] = ov;
    }
}
__global__ void subtract_tensor_tensor_kernel(float* out, const float* sum, const float* sub, size_t num_elements)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    for (size_t i = idx; i < num_elements; i += stride)
        out[i] = sum[i] - sub[i];
}
void kernel_ops::subtract_tensor_tensor(void* out_data, const void* sum_data, const void* sub_data, size_t num_elements)
{
    const int BLOCK_SIZE = 256;
    const int grid_size = device::sm_count() * 4;

    // Vectorized addition
    size_t num_vec = num_elements / 4;
    if (num_vec)
    {
        CUDA_LAUNCH(subtract_tensor_tensor_kernel_vec, grid_size, BLOCK_SIZE,
            (float4*)out_data, 
            (const float4*)sum_data, 
            (const float4*)sub_data, 
            num_vec
        );
        CUDA_CHECK(cudaGetLastError());
    }

    // Remainder addition
    size_t remainder = num_elements % 4;
    if (remainder)
    {
        size_t offset = num_vec * 4;
        CUDA_LAUNCH(subtract_tensor_tensor_kernel, grid_size, BLOCK_SIZE,
            (float*)out_data + offset, 
            (const float*)sum_data + offset, 
            (const float*)sub_data + offset, 
            remainder
        );
        CUDA_CHECK(cudaGetLastError());
    }
}

__global__ void divide_tensor_kernel_vec(float4* out, const float4* den, size_t num_vec)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    for (size_t i = idx; i < num_vec; i += stride)
    {
        float4 ov = out[i];
        float4 sv = den[i];

        ov.x /= sv.x;
        ov.y /= sv.y;
        ov.z /= sv.z;
        ov.w /= sv.w;

        out[i] = ov;
    }
}
__global__ void divide_tensor_kernel(float* out, const float* den, size_t num_elements)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    for (size_t i = idx; i < num_elements; i += stride)
        out[i] /= den[i];
}
void kernel_ops::divide_tensor(void* out_data, const void* den_data, size_t num_elements)
{
    const int BLOCK_SIZE = 256;
    const int grid_size = device::sm_count() * 4;

    // Vectorized addition
    size_t num_vec = num_elements / 4;
    if (num_vec)
    {
        CUDA_LAUNCH(divide_tensor_kernel_vec, grid_size, BLOCK_SIZE,
            (float4*)out_data,
            (const float4*)den_data,
            num_vec
        );
        CUDA_CHECK(cudaGetLastError());
    }

    // Remainder addition
    size_t remainder = num_elements % 4;
    if (remainder)
    {
        size_t offset = num_vec * 4;
        CUDA_LAUNCH(divide_tensor_kernel, grid_size, BLOCK_SIZE,
            (float*)out_data + offset,
            (const float*)den_data + offset,
            remainder
        );
        CUDA_CHECK(cudaGetLastError());
    }
}

__global__ void divide_tensor_tensor_kernel_vec(float4* out, const float4* num, const float4* den, size_t num_vec)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    for (size_t i = idx; i < num_vec; i += stride)
    {
        float4 v0 = num[i];
        float4 v1 = den[i];

        float4 ov;
        ov.x = v0.x / v1.x;
        ov.y = v0.y / v1.y;
        ov.z = v0.z / v1.z;
        ov.w = v0.w / v1.w;

        out[i] = ov;
    }
}
__global__ void divide_tensor_tensor_kernel(float* out, const float* num, const float* den, size_t num_elements)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    for (size_t i = idx; i < num_elements; i += stride)
        out[i] = num[i] / den[i];
}
void kernel_ops::divide_tensor_tensor(void* out_data, const void* num_data, const void* den_data, size_t num_elements)
{
    const int BLOCK_SIZE = 256;
    const int grid_size = device::sm_count() * 4;

    // Vectorized addition
    size_t num_vec = num_elements / 4;
    if (num_vec)
    {
        CUDA_LAUNCH(divide_tensor_tensor_kernel_vec, grid_size, BLOCK_SIZE,
            (float4*)out_data, 
            (const float4*)num_data, 
            (const float4*)den_data, 
            num_vec
        );
        CUDA_CHECK(cudaGetLastError());
    }

    // Remainder addition
    size_t remainder = num_elements % 4;
    if (remainder)
    {
        size_t offset = num_vec * 4;
        CUDA_LAUNCH(divide_tensor_tensor_kernel, grid_size, BLOCK_SIZE,
            (float*)out_data + offset, 
            (const float*)num_data + offset, 
            (const float*)den_data + offset, 
            remainder
        );
        CUDA_CHECK(cudaGetLastError());
    }
}

/*
--------------------------------------------------------------------------------------------------------------------------
 Non Linearities
--------------------------------------------------------------------------------------------------------------------------
*/

__global__ void sign_kernel_vec(float4* __restrict__ out, const float4* __restrict__ in, size_t num_vec)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    for (size_t i = idx; i < num_vec; i += stride)
    {
        float4 v = in[i];

        v.x = (v.x > 0.0f);
        v.y = (v.y > 0.0f);
        v.z = (v.z > 0.0f);
        v.w = (v.w > 0.0f);

        out[i] = v;
    }
}
__global__ void sign_kernel(float* __restrict__ out, const float* __restrict__ in, size_t num_elements)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    for (size_t i = idx; i < num_elements; i += stride)
        out[i] = (in[i] > 0.0f);
}
void kernel_ops::sign(void* out_data, const void* in_data, size_t num_elements)
{
    const int BLOCK_SIZE = 256;
    const int grid_size = device::sm_count() * 4;

    // Vectorized addition
    size_t num_vec = num_elements / 4;
    if (num_vec)
    {
        CUDA_LAUNCH(sign_kernel_vec, grid_size, BLOCK_SIZE,
            (float4*)out_data,
            (const float4*)in_data,
            num_vec
        );
        CUDA_CHECK(cudaGetLastError());
    }

    // Remainder addition
    size_t remainder = num_elements % 4;
    if (remainder)
    {
        size_t offset = num_vec * 4;
        CUDA_LAUNCH(sign_kernel, grid_size, BLOCK_SIZE,
            (float*)out_data + offset,
            (const float*)in_data + offset,
            remainder
        );
        CUDA_CHECK(cudaGetLastError());
    }
}

__global__ void exp_kernel_vec(float4* __restrict__ out, const float4* __restrict__ in, size_t num_vec)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    for (size_t i = idx; i < num_vec; i += stride)
    {
        float4 v = in[i];

        v.x = __expf(v.x);
        v.y = __expf(v.y);
        v.z = __expf(v.z);
        v.w = __expf(v.w);

        out[i] = v;
    }
}
__global__ void exp_kernel(float* __restrict__ out, const float* __restrict__ in, size_t num_elements)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    for (size_t i = idx; i < num_elements; i += stride)
        out[i] = __expf(in[i]);
}
void kernel_ops::exp(void* out_data, const void* in_data, size_t num_elements)
{
    const int BLOCK_SIZE = 256;
    const int grid_size = device::sm_count() * 4;

    // Vectorized addition
    size_t num_vec = num_elements / 4;
    if (num_vec)
    {
        CUDA_LAUNCH(exp_kernel_vec, grid_size, BLOCK_SIZE,
            (float4*)out_data,
            (const float4*)in_data,
            num_vec
        );
        CUDA_CHECK(cudaGetLastError());
    }

    // Remainder addition
    size_t remainder = num_elements % 4;
    if (remainder)
    {
        size_t offset = num_vec * 4;
        CUDA_LAUNCH(exp_kernel, grid_size, BLOCK_SIZE,
            (float*)out_data + offset,
            (const float*)in_data + offset,
            remainder
        );
        CUDA_CHECK(cudaGetLastError());
    }
}

__global__ void log_kernel_vec(float4* __restrict__ out, const float4* __restrict__ in, size_t num_vec)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    for (size_t i = idx; i < num_vec; i += stride)
    {
        float4 v = in[i];

        v.x = __logf(v.x);
        v.y = __logf(v.y);
        v.z = __logf(v.z);
        v.w = __logf(v.w);

        out[i] = v;
    }
}
__global__ void log_kernel(float* __restrict__ out, const float* __restrict__ in, size_t num_elements)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    for (size_t i = idx; i < num_elements; i += stride)
        out[i] = __logf(in[i]);
}
void kernel_ops::log(void* out_data, const void* in_data, size_t num_elements)
{
    const int BLOCK_SIZE = 256;
    const int grid_size = device::sm_count() * 4;

    // Vectorized addition
    size_t num_vec = num_elements / 4;
    if (num_vec)
    {
        CUDA_LAUNCH(log_kernel_vec, grid_size, BLOCK_SIZE,
            (float4*)out_data,
            (const float4*)in_data,
            num_vec
        );
        CUDA_CHECK(cudaGetLastError());
    }

    // Remainder addition
    size_t remainder = num_elements % 4;
    if (remainder)
    {
        size_t offset = num_vec * 4;
        CUDA_LAUNCH(log_kernel, grid_size, BLOCK_SIZE,
            (float*)out_data + offset,
            (const float*)in_data + offset,
            remainder
        );
        CUDA_CHECK(cudaGetLastError());
    }
}

__global__ void relu_kernel_vec(float4* __restrict__ out, const float4* __restrict__ in, size_t num_vec)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    for (size_t i = idx; i < num_vec; i += stride)
    {
        float4 v = in[i];

        v.x = fmaxf(v.x, 0.0f);
        v.y = fmaxf(v.y, 0.0f);
        v.z = fmaxf(v.z, 0.0f);
        v.w = fmaxf(v.w, 0.0f);

        out[i] = v;
    }
}
__global__ void relu_kernel(float* __restrict__ out, const float* __restrict__ in, size_t num_elements)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    for (size_t i = idx; i < num_elements; i += stride)
        out[i] = fmaxf(in[i], 0.0f);;
}
void kernel_ops::relu(void* out_data, const void* in_data, size_t num_elements)
{
    const int BLOCK_SIZE = 256;
    const int grid_size = device::sm_count() * 4;

    // Vectorized addition
    size_t num_vec = num_elements / 4;
    if (num_vec)
    {
        CUDA_LAUNCH(relu_kernel_vec, grid_size, BLOCK_SIZE,
            (float4*)out_data,
            (const float4*)in_data,
            num_vec
        );
        CUDA_CHECK(cudaGetLastError());
    }

    // Remainder addition
    size_t remainder = num_elements % 4;
    if (remainder)
    {
        size_t offset = num_vec * 4;
        CUDA_LAUNCH(relu_kernel, grid_size, BLOCK_SIZE,
            (float*)out_data + offset,
            (const float*)in_data + offset,
            remainder
        );
        CUDA_CHECK(cudaGetLastError());
    }
}

__global__ void silu_kernel_vec(float4* __restrict__ out, const float4* __restrict__ in, size_t num_vec)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    for (size_t i = idx; i < num_vec; i += stride)
    {
        float4 v = in[i];

        v.x = v.x / (1.0f + __expf(-v.x));
        v.y = v.y / (1.0f + __expf(-v.y));
        v.z = v.z / (1.0f + __expf(-v.z));
        v.w = v.w / (1.0f + __expf(-v.w));

        out[i] = v;
    }
}
__global__ void silu_kernel(float* __restrict__ out, const float* __restrict__ in, size_t num_elements)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    for (size_t i = idx; i < num_elements; i += stride)
        out[i] = in[i] / (1.0f + __expf(-in[i]));
}
void kernel_ops::silu(void* out_data, const void* in_data, size_t num_elements)
{
    const int BLOCK_SIZE = 256;
    const int grid_size = device::sm_count() * 4;

    // Vectorized addition
    size_t num_vec = num_elements / 4;
    if (num_vec)
    {
        CUDA_LAUNCH(silu_kernel_vec, grid_size, BLOCK_SIZE,
            (float4*)out_data,
            (const float4*)in_data,
            num_vec
        );
        CUDA_CHECK(cudaGetLastError());
    }

    // Remainder addition
    size_t remainder = num_elements % 4;
    if (remainder)
    {
        size_t offset = num_vec * 4;
        CUDA_LAUNCH(silu_kernel, grid_size, BLOCK_SIZE,
            (float*)out_data + offset,
            (const float*)in_data + offset,
            remainder
        );
        CUDA_CHECK(cudaGetLastError());
    }
}

__global__ void gelu_kernel_vec(float4* __restrict__ out, const float4* __restrict__ in, size_t num_vec)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    for (size_t i = idx; i < num_vec; i += stride)
    {
        float4 v = in[i];

        v.x = 0.5f * v.x * (1.0f + erff(v.x * 0.70710678118f));
        v.y = 0.5f * v.y * (1.0f + erff(v.y * 0.70710678118f));
        v.z = 0.5f * v.z * (1.0f + erff(v.z * 0.70710678118f));
        v.w = 0.5f * v.w * (1.0f + erff(v.w * 0.70710678118f));

        out[i] = v;
    }
}
__global__ void gelu_kernel(float* __restrict__ out, const float* __restrict__ in, size_t num_elements)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    for (size_t i = idx; i < num_elements; i += stride)
        out[i] = 0.5f * in[i] * (1.0f + erff(in[i] * 0.70710678118f));
}
void kernel_ops::gelu(void* out_data, const void* in_data, size_t num_elements)
{
    const int BLOCK_SIZE = 256;
    const int grid_size = device::sm_count() * 4;

    // Vectorized addition
    size_t num_vec = num_elements / 4;
    if (num_vec)
    {
        CUDA_LAUNCH(gelu_kernel_vec, grid_size, BLOCK_SIZE,
            (float4*)out_data,
            (const float4*)in_data,
            num_vec
        );
        CUDA_CHECK(cudaGetLastError());
    }

    // Remainder addition
    size_t remainder = num_elements % 4;
    if (remainder)
    {
        size_t offset = num_vec * 4;
        CUDA_LAUNCH(gelu_kernel, grid_size, BLOCK_SIZE,
            (float*)out_data + offset,
            (const float*)in_data + offset,
            remainder
        );
        CUDA_CHECK(cudaGetLastError());
    }
}

__global__ void sigmoid_kernel_vec(float4* __restrict__ out, const float4* __restrict__ in, size_t num_vec)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    for (size_t i = idx; i < num_vec; i += stride)
    {
        float4 v = in[i];

        v.x = 1.0f / (1.0f + __expf(-v.x));
        v.y = 1.0f / (1.0f + __expf(-v.y));
        v.z = 1.0f / (1.0f + __expf(-v.z));
        v.w = 1.0f / (1.0f + __expf(-v.w));

        out[i] = v;
    }
}
__global__ void sigmoid_kernel(float* __restrict__ out, const float* __restrict__ in, size_t num_elements)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    for (size_t i = idx; i < num_elements; i += stride)
        out[i] = 1.0f / (1.0f + __expf(-in[i]));
}
void kernel_ops::sigmoid(void* out_data, const void* in_data, size_t num_elements)
{
    const int BLOCK_SIZE = 256;
    const int grid_size = device::sm_count() * 4;

    // Vectorized addition
    size_t num_vec = num_elements / 4;
    if (num_vec)
    {
        CUDA_LAUNCH(sigmoid_kernel_vec, grid_size, BLOCK_SIZE,
            (float4*)out_data,
            (const float4*)in_data,
            num_vec
        );
        CUDA_CHECK(cudaGetLastError());
    }

    // Remainder addition
    size_t remainder = num_elements % 4;
    if (remainder)
    {
        size_t offset = num_vec * 4;
        CUDA_LAUNCH(sigmoid_kernel, grid_size, BLOCK_SIZE,
            (float*)out_data + offset,
            (const float*)in_data + offset,
            remainder
        );
        CUDA_CHECK(cudaGetLastError());
    }
}

__global__ void tanh_kernel_vec(float4* __restrict__ out, const float4* __restrict__ in, size_t num_vec)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    for (size_t i = idx; i < num_vec; i += stride)
    {
        float4 v = in[i];

        v.x = tanhf(v.x);
        v.y = tanhf(v.y);
        v.z = tanhf(v.z);
        v.w = tanhf(v.w);

        out[i] = v;
    }
}
__global__ void tanh_kernel(float* __restrict__ out, const float* __restrict__ in, size_t num_elements)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    for (size_t i = idx; i < num_elements; i += stride)
        out[i] = tanhf(in[i]);
}
void kernel_ops::tanh(void* out_data, const void* in_data, size_t num_elements)
{
    const int BLOCK_SIZE = 256;
    const int grid_size = device::sm_count() * 4;

    // Vectorized addition
    size_t num_vec = num_elements / 4;
    if (num_vec)
    {
        CUDA_LAUNCH(tanh_kernel_vec, grid_size, BLOCK_SIZE,
            (float4*)out_data,
            (const float4*)in_data,
            num_vec
        );
        CUDA_CHECK(cudaGetLastError());
    }

    // Remainder addition
    size_t remainder = num_elements % 4;
    if (remainder)
    {
        size_t offset = num_vec * 4;
        CUDA_LAUNCH(tanh_kernel, grid_size, BLOCK_SIZE,
            (float*)out_data + offset,
            (const float*)in_data + offset,
            remainder
        );
        CUDA_CHECK(cudaGetLastError());
    }
}

__global__ void sqrt_kernel_vec(float4* __restrict__ out, const float4* __restrict__ in, size_t num_vec)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    for (size_t i = idx; i < num_vec; i += stride)
    {
        float4 v = in[i];

        v.x = sqrtf(v.x);
        v.y = sqrtf(v.y);
        v.z = sqrtf(v.z);
        v.w = sqrtf(v.w);

        out[i] = v;
    }
}
__global__ void sqrt_kernel(float* __restrict__ out, const float* __restrict__ in, size_t num_elements)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    for (size_t i = idx; i < num_elements; i += stride)
        out[i] = sqrtf(in[i]);
}
void kernel_ops::sqrt(void* out_data, const void* in_data, size_t num_elements)
{
    const int BLOCK_SIZE = 256;
    const int grid_size = device::sm_count() * 4;

    // Vectorized addition
    size_t num_vec = num_elements / 4;
    if (num_vec)
    {
        CUDA_LAUNCH(sqrt_kernel_vec, grid_size, BLOCK_SIZE,
            (float4*)out_data,
            (const float4*)in_data,
            num_vec
        );
        CUDA_CHECK(cudaGetLastError());
    }

    // Remainder addition
    size_t remainder = num_elements % 4;
    if (remainder)
    {
        size_t offset = num_vec * 4;
        CUDA_LAUNCH(sqrt_kernel, grid_size, BLOCK_SIZE,
            (float*)out_data + offset,
            (const float*)in_data + offset,
            remainder
        );
        CUDA_CHECK(cudaGetLastError());
    }
}

__global__ void square_kernel_vec(float4* __restrict__ out, const float4* __restrict__ in, size_t num_vec)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    for (size_t i = idx; i < num_vec; i += stride)
    {
        float4 v = in[i];

        v.x = v.x * v.x;
        v.y = v.y * v.y;
        v.z = v.z * v.z;
        v.w = v.w * v.w;

        out[i] = v;
    }
}
__global__ void square_kernel(float* __restrict__ out, const float* __restrict__ in, size_t num_elements)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    for (size_t i = idx; i < num_elements; i += stride)
        out[i] = in[i] * in[i];
}
void kernel_ops::square(void* out_data, const void* in_data, size_t num_elements)
{
    const int BLOCK_SIZE = 256;
    const int grid_size = device::sm_count() * 4;

    // Vectorized addition
    size_t num_vec = num_elements / 4;
    if (num_vec)
    {
        CUDA_LAUNCH(square_kernel_vec, grid_size, BLOCK_SIZE,
            (float4*)out_data,
            (const float4*)in_data,
            num_vec
        );
        CUDA_CHECK(cudaGetLastError());
    }

    // Remainder addition
    size_t remainder = num_elements % 4;
    if (remainder)
    {
        size_t offset = num_vec * 4;
        CUDA_LAUNCH(square_kernel, grid_size, BLOCK_SIZE,
            (float*)out_data + offset,
            (const float*)in_data + offset,
            remainder
        );
        CUDA_CHECK(cudaGetLastError());
    }
}

__global__ void pow_kernel_vec(float4* __restrict__ out, const float4* __restrict__ in, float exp, size_t num_vec)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    for (size_t i = idx; i < num_vec; i += stride)
    {
        float4 v = in[i];

        v.x = powf(v.x, exp);
        v.y = powf(v.y, exp);
        v.z = powf(v.z, exp);
        v.w = powf(v.w, exp);

        out[i] = v;
    }
}
__global__ void pow_kernel(float* __restrict__ out, const float* __restrict__ in, float exp, size_t num_elements)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    for (size_t i = idx; i < num_elements; i += stride)
        out[i] = powf(in[i], exp);
}
void kernel_ops::pow(void* out_data, const void* in_data, float exp, size_t num_elements)
{
    const int BLOCK_SIZE = 256;
    const int grid_size = device::sm_count() * 4;

    // Vectorized addition
    size_t num_vec = num_elements / 4;
    if (num_vec)
    {
        CUDA_LAUNCH(pow_kernel_vec, grid_size, BLOCK_SIZE,
            (float4*)out_data,
            (const float4*)in_data, exp,
            num_vec
        );
        CUDA_CHECK(cudaGetLastError());
    }

    // Remainder addition
    size_t remainder = num_elements % 4;
    if (remainder)
    {
        size_t offset = num_vec * 4;
        CUDA_LAUNCH(pow_kernel, grid_size, BLOCK_SIZE,
            (float*)out_data + offset,
            (const float*)in_data + offset, exp,
            remainder
        );
        CUDA_CHECK(cudaGetLastError());
    }
}

/*
--------------------------------------------------------------------------------------------------------------------------
 RNG initialization
--------------------------------------------------------------------------------------------------------------------------
*/

#define CURAND_CHECK(x) do { curandStatus_t st = (x); if (st != CURAND_STATUS_SUCCESS) TENSOR_ERROR("cuRAND error"); } while(0)

class rng
{
private:
    static curandGenerator_t gen;

    static void ensure_created()
    {
        if (!gen)
        {
            CURAND_CHECK(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_PHILOX4_32_10));
            CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(gen, 42ULL));
            CURAND_CHECK(curandSetStream(gen, stream::get()));
        }
    }
public:
    static void set_seed(unsigned long long seed)
    {
        ensure_created();
        CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(gen, seed));
    }

    static curandGenerator_t generator()
    {
        ensure_created();
        return gen;
    }
};
curandGenerator_t rng::gen = nullptr;

void kernel_ops::set_seed(size_t seed)
{
    rng::set_seed(seed);
}

void kernel_ops::normal(void* data_ptr, float mean, float std, size_t num_elements)
{
    curandGenerateNormal(rng::generator(), (float*)data_ptr, num_elements, mean, std);
}

void kernel_ops::uniform(void* data_ptr, float min, float max, size_t num_elements)
{
    curandGenerateUniform(rng::generator(), (float*)data_ptr, num_elements);
    multiply_scalar(data_ptr, max - min, num_elements);
    add_scalar(data_ptr, min, num_elements);
}

/*
--------------------------------------------------------------------------------------------------------------------------
 Row-Wise Operators
--------------------------------------------------------------------------------------------------------------------------
*/

template<int BLOCK>
__inline__ __device__ float block_max(float x)
{
    __shared__ float shared[32];
    int lane = threadIdx.x % 32;
    int warp = threadIdx.x / 32;

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        x = fmaxf(x, __shfl_down_sync(0xffffffff, x, offset));

    if (lane == 0) shared[warp] = x;
    __syncthreads();

    x = (threadIdx.x < (BLOCK / 32)) ? shared[lane] : -CUDART_INF_F;

    if (warp == 0)
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            x = fmaxf(x, __shfl_down_sync(0xffffffff, x, offset));

    return x;
}

template<int BLOCK>
__inline__ __device__ float block_sum(float x) 
{
    __shared__ float shared[32]; // up to 32 warps
    int lane = threadIdx.x % 32;
    int warp = threadIdx.x / 32;

    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2)
        x += __shfl_down_sync(0xffffffff, x, offset);
    if (lane == 0) shared[warp] = x;
    __syncthreads();

    x = (threadIdx.x < (BLOCK / 32)) ? shared[lane] : 0.0f;
    if (warp == 0)
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2)
            x += __shfl_down_sync(0xffffffff, x, offset);

    return x;
}

struct Welford
{
    float mean, M2;
    int n;
};
__inline__ __device__ Welford empty_welford()
{
    Welford w;
    w.mean = 0.f;
    w.M2 = 0.f;
    w.n = 0;
    return w;
}
__inline__ __device__ void welford_add(Welford& w, float val)
{
    w.n += 1;
    const float old_mean = w.mean;
    w.mean += (val - w.mean) / w.n;
    w.M2 += (val - old_mean) * (val - w.mean);
}
__inline__ __device__ void welford_add4(Welford& w, float4 v)
{
    // mean of 4
    float sum = v.x + v.y + v.z + v.w;
    float mean = 0.25f * sum;

    // M2 of 4: Σ (xi - mean)^2
    float dx = v.x - mean;
    float dy = v.y - mean;
    float dz = v.z - mean;
    float dw = v.w - mean;
    float M2 = dx * dx + dy * dy + dz * dz + dw * dw;

    // merge this 4-sample batch
    float delta = mean - w.mean;
    int new_n = w.n + 4;

    w.mean += delta * 4.f / new_n;
    w.M2 += M2 + delta * delta * (float)w.n * 4.f / new_n;
    w.n = new_n;
}
__inline__ __device__ void welford_add_welford(Welford& w, float mean, float M2, int n)
{
    if (!n) return;
    if (!w.n) 
    {
        w.mean = mean;
        w.M2 = M2;
        w.n = n;
        return;
    }

    float delta = mean - w.mean;
    int new_n = w.n + n;

    w.mean += delta * (float)n / new_n;
    w.M2 += M2 + delta * delta * (float)w.n * (float)n / new_n;
    w.n = new_n;
}
template<int BLOCK>
__inline__ __device__ Welford welford_merge(Welford w)
{
    __shared__ Welford shared[32]; // up to 32 warps
    int lane = threadIdx.x % 32;
    int warp = threadIdx.x / 32;
    
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2)
        welford_add_welford(w,
            __shfl_down_sync(0xffffffff, w.mean, offset),
            __shfl_down_sync(0xffffffff, w.M2  , offset),
            __shfl_down_sync(0xffffffff, w.n   , offset)
        );

    if (lane == 0) shared[warp] = w;
    __syncthreads();

    w = (threadIdx.x < (BLOCK / 32)) ? shared[lane] : empty_welford();
    if (warp == 0)
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2)
            if (lane + offset < 32)
                welford_add_welford(w,
                    __shfl_down_sync(0xffffffff, w.mean, offset),
                    __shfl_down_sync(0xffffffff, w.M2  , offset),
                    __shfl_down_sync(0xffffffff, w.n   , offset)
                );

    return w;
}

template<int BLOCK>
__global__ void sum_lastdim_vec(float* __restrict__ out, const float* __restrict__ in, int element_count, int rows)
{
    int row = blockIdx.x;
    if (row >= rows) return;

    const float* base = in + row * element_count;
    float sum = 0.0f;

    if (element_count % 4 == 0)
    {
        // Can vectorize because everything is 16byte aligned.
        int vecN = element_count / 4;
        const float4* vptr = (const float4*)base;

        for (int i = threadIdx.x; i < vecN; i += BLOCK)
        {
            float4 v = vptr[i];
            sum += v.x + v.y + v.z + v.w;
        }
    }
    else
    {
        // Sum individually to avoid misalignement.
        for (int i = threadIdx.x; i < element_count; i += BLOCK)
            sum += base[i];
    }

    sum = block_sum<BLOCK>(sum);
    if (threadIdx.x == 0)
        out[row] = sum;
}
template<int BLOCK>
__global__ void sum_dim_strided(float* __restrict__ out, const float* __restrict__ in, int element_stride, int element_count, int rows)
{
    int row = blockIdx.x;
    if (row >= rows) return;

    // Compute pointer position based on row number.
    int outer_stride = element_count * element_stride;
    int outer_idx = row / element_stride;
    int inner_idx = row % element_stride;

    const float* base = in + outer_idx * outer_stride + inner_idx;

    float sum = 0.0f;
    // stride along the reduced dimension is 'inner'
    for (int k = threadIdx.x; k < element_count; k += BLOCK)
        sum += base[k * element_stride];

    sum = block_sum<BLOCK>(sum);
    if (threadIdx.x == 0)
        out[row] = sum;
}
void kernel_ops::sum(void* out_data, const void* in_data, int element_stride, int element_count, int rows)
{
    constexpr int BLOCK = 256;

    // Don't run empty tensors.
    if (!element_count) return;

    // If can use the fast kernel use it.
    if (element_stride == 1)
    {
        CUDA_LAUNCH(sum_lastdim_vec<BLOCK>, rows, BLOCK,
            (float*)out_data,
            (const float*)in_data,
            element_count, rows
        );
        CUDA_CHECK(cudaGetLastError());
    }
    // Else use the strided kernel.
    else
    {
        CUDA_LAUNCH(sum_dim_strided<BLOCK>, rows, BLOCK,
            (float*)out_data,
            (const float*)in_data,
            element_stride,
            element_count, rows
        );
        CUDA_CHECK(cudaGetLastError());
    }
}

template<int BLOCK>
__global__ void mean_lastdim_vec(float* __restrict__ out, const float* __restrict__ in, int element_count, int rows)
{
    int row = blockIdx.x;
    if (row >= rows) return;

    const float* base = in + row * element_count;
    float sum = 0.0f;

    if (element_count % 4 == 0)
    {
        // Can vectorize because everything is 16byte aligned.
        int vecN = element_count / 4;
        const float4* vptr = (const float4*)base;

        for (int i = threadIdx.x; i < vecN; i += BLOCK) 
        {
            float4 v = vptr[i];
            sum += v.x + v.y + v.z + v.w;
        }
    }
    else
    {
        // Sum individually to avoid misalignement.
        for (int i = threadIdx.x; i < element_count; i += BLOCK)
            sum += base[i];
    }

    sum = block_sum<BLOCK>(sum);
    if (threadIdx.x == 0)
        out[row] = sum / element_count;
}
template<int BLOCK>
__global__ void mean_dim_strided(float* __restrict__ out, const float* __restrict__ in, int element_stride, int element_count, int rows)
{
    int row = blockIdx.x;
    if (row >= rows) return;

    // Compute pointer position based on row number.
    int outer_stride = element_count * element_stride;
    int outer_idx = row / element_stride;
    int inner_idx = row % element_stride;

    const float* base = in + outer_idx * outer_stride + inner_idx;

    float sum = 0.0f;
    // stride along the reduced dimension is 'inner'
    for (int k = threadIdx.x; k < element_count; k += BLOCK)
        sum += base[k * element_stride];

    sum = block_sum<BLOCK>(sum);
    if (threadIdx.x == 0)
        out[row] = sum / element_count;
}
void kernel_ops::mean(void* out_data, const void* in_data, int element_stride, int element_count, int rows)
{
    constexpr int BLOCK = 256;

    // Don't run empty tensors.
    if (!element_count) return;

    // If can use the fast kernel use it.
    if (element_stride == 1)
    {
        CUDA_LAUNCH(mean_lastdim_vec<BLOCK>, rows, BLOCK,
            (float*)out_data,
            (const float*)in_data,
            element_count, rows
        );
        CUDA_CHECK(cudaGetLastError());
    }
    // Else use the strided kernel.
    else
    {
        CUDA_LAUNCH(mean_dim_strided<BLOCK>, rows, BLOCK,
            (float*)out_data,
            (const float*)in_data,
            element_stride,
            element_count, rows
        );
        CUDA_CHECK(cudaGetLastError());
    }
}

template<int BLOCK>
__global__ void var_lastdim_vec(float* __restrict__ out, const float* __restrict__ in, int element_count, int rows)
{
    int row = blockIdx.x;
    if (row >= rows) return;

    const float* base = in + row * element_count;
    Welford w = empty_welford();

    if (element_count % 4 == 0)
    {
        // Can vectorize because everything is 16byte aligned.
        int vecN = element_count / 4;
        const float4* vptr = (const float4*)base;

        for (int i = threadIdx.x; i < vecN; i += BLOCK)
            welford_add4(w, vptr[i]);
    }
    else
    {
        // Sum individually to avoid misalignement.
        for (int i = threadIdx.x; i < element_count; i += BLOCK)
            welford_add(w, base[i]);
    }

    w = welford_merge<BLOCK>(w);
    if (threadIdx.x == 0)
        out[row] = w.M2 / element_count;
}
template<int BLOCK>
__global__ void var_dim_strided(float* __restrict__ out, const float* __restrict__ in, int element_stride, int element_count, int rows)
{
    int row = blockIdx.x;
    if (row >= rows) return;

    // Compute pointer position based on row number.
    int outer_stride = element_count * element_stride;
    int outer_idx = row / element_stride;
    int inner_idx = row % element_stride;

    const float* base = in + outer_idx * outer_stride + inner_idx;

    Welford w = empty_welford();
    // stride along the reduced dimension is 'inner'
    for (int k = threadIdx.x; k < element_count; k += BLOCK)
        welford_add(w, base[k * element_stride]);

    w = welford_merge<BLOCK>(w);
    if (threadIdx.x == 0)
        out[row] = w.M2 / element_count;
}
void kernel_ops::var(void* out_data, const void* in_data, int element_stride, int element_count, int rows)
{
    constexpr int BLOCK = 256;

    // Don't run empty tensors.
    if (!element_count) return;

    // If can use the fast kernel use it.
    if (element_stride == 1)
    {
        CUDA_LAUNCH(var_lastdim_vec<BLOCK>, rows, BLOCK,
            (float*)out_data,
            (const float*)in_data,
            element_count, rows
        );
        CUDA_CHECK(cudaGetLastError());
    }
    // Else use the strided kernel.
    else
    {
        CUDA_LAUNCH(var_dim_strided<BLOCK>, rows, BLOCK,
            (float*)out_data,
            (const float*)in_data,
            element_stride,
            element_count, rows
        );
        CUDA_CHECK(cudaGetLastError());
    }
}

template<int BLOCK>
__global__ void std_lastdim_vec(float* __restrict__ out, const float* __restrict__ in, int element_count, int rows)
{
    int row = blockIdx.x;
    if (row >= rows) return;

    const float* base = in + row * element_count;
    Welford w = empty_welford();

    if (element_count % 4 == 0)
    {
        // Can vectorize because everything is 16byte aligned.
        int vecN = element_count / 4;
        const float4* vptr = (const float4*)base;

        for (int i = threadIdx.x; i < vecN; i += BLOCK)
            welford_add4(w, vptr[i]);
    }
    else
    {
        // Sum individually to avoid misalignement.
        for (int i = threadIdx.x; i < element_count; i += BLOCK)
            welford_add(w, base[i]);
    }

    w = welford_merge<BLOCK>(w);
    if (threadIdx.x == 0)
        out[row] = sqrtf(w.M2 / element_count);
}
template<int BLOCK>
__global__ void std_dim_strided(float* __restrict__ out, const float* __restrict__ in, int element_stride, int element_count, int rows)
{
    int row = blockIdx.x;
    if (row >= rows) return;

    // Compute pointer position based on row number.
    int outer_stride = element_count * element_stride;
    int outer_idx = row / element_stride;
    int inner_idx = row % element_stride;

    const float* base = in + outer_idx * outer_stride + inner_idx;

    Welford w = empty_welford();
    // stride along the reduced dimension is 'inner'
    for (int k = threadIdx.x; k < element_count; k += BLOCK)
        welford_add(w, base[k * element_stride]);

    w = welford_merge<BLOCK>(w);
    if (threadIdx.x == 0)
        out[row] = sqrtf(w.M2 / element_count);
}
void kernel_ops::std(void* out_data, const void* in_data, int element_stride, int element_count, int rows)
{
    constexpr int BLOCK = 256;

    // Don't run empty tensors.
    if (!element_count) return;

    // If can use the fast kernel use it.
    if (element_stride == 1)
    {
        CUDA_LAUNCH(std_lastdim_vec<BLOCK>, rows, BLOCK,
            (float*)out_data,
            (const float*)in_data,
            element_count, rows
        );
        CUDA_CHECK(cudaGetLastError());
    }
    // Else use the strided kernel.
    else
    {
        CUDA_LAUNCH(std_dim_strided<BLOCK>, rows, BLOCK,
            (float*)out_data,
            (const float*)in_data,
            element_stride,
            element_count, rows
        );
        CUDA_CHECK(cudaGetLastError());
    }
}

template<int BLOCK>
__global__ void softmax_lastdim(float* __restrict__ out, const float* __restrict__ in, int element_count, int rows)
{
    int row = blockIdx.x;
    if (row >= rows) return;

    const float* base_in = in + row * element_count;
    float* base_out = out + row * element_count;

    // Shared maximum and sum.
    __shared__ float sh_max;
    __shared__ float sh_sum;

    // First pass compute max.
    float max = -CUDART_INF_F;
    for (int i = threadIdx.x; i < element_count; i += BLOCK)
        max = fmaxf(max, base_in[i]);

    // synch max.
    max = block_max<BLOCK>(max);
    if (threadIdx.x == 0) sh_max = max;
    __syncthreads();
    max = sh_max;

    // Second pass compute exponential and accumulate sum.
    float sum = 0.0f;
    for (int i = threadIdx.x; i < element_count; i += BLOCK)
    {
        float exp = __expf(base_in[i] - max);
        base_out[i] = exp;
        sum += exp;
    }

    // synch sum.
    sum = block_sum<BLOCK>(sum);
    if (threadIdx.x == 0) sh_sum = sum;
    __syncthreads();
    float inv_s = 1.f / sh_sum;

    // Third pass divides by sum.
    for (int i = threadIdx.x; i < element_count; i += BLOCK)
        base_out[i] *= inv_s;
}
template<int BLOCK>
__global__ void softmax_dim_strided(float* __restrict__ out, const float* __restrict__ in, int element_stride, int element_count, int rows)
{
    int row = blockIdx.x;
    if (row >= rows) return;

    // Compute pointer position based on row number.
    int outer_stride = element_count * element_stride;
    int outer_idx = row / element_stride;
    int inner_idx = row % element_stride;

    const float* base_in = in + outer_idx * outer_stride + inner_idx;
    float* base_out = out + outer_idx * outer_stride + inner_idx;

    // Shared maximum and sum.
    __shared__ float sh_max;
    __shared__ float sh_sum;

    // First pass compute max.
    float max = -CUDART_INF_F;
    for (int i = threadIdx.x; i < element_count; i += BLOCK)
        max = fmaxf(max, base_in[i * element_stride]);

    // synch max.
    max = block_max<BLOCK>(max);
    if (threadIdx.x == 0) sh_max = max;
    __syncthreads();
    max = sh_max;

    // Second pass compute exponential and accumulate sum.
    float sum = 0.0f;
    for (int i = threadIdx.x; i < element_count; i += BLOCK)
    {
        float exp = __expf(base_in[i * element_stride] - max);
        base_out[i * element_stride] = exp;
        sum += exp;
    }

    // synch sum.
    sum = block_sum<BLOCK>(sum);
    if (threadIdx.x == 0) sh_sum = sum;
    __syncthreads();
    float inv_s = 1.f / sh_sum;

    // Third pass divides by sum.
    for (int i = threadIdx.x; i < element_count; i += BLOCK)
        base_out[i * element_stride] *= inv_s;
}
void kernel_ops::softmax(void* out_data, const void* in_data, int element_stride, int element_count, int rows)
{
    constexpr int BLOCK = 256;

    // Don't run empty tensors.
    if (!element_count) return;

    // If can use the fast kernel use it.
    if (element_stride == 1)
    {
        CUDA_LAUNCH(softmax_lastdim<BLOCK>, rows, BLOCK,
            (float*)out_data,
            (const float*)in_data,
            element_count, rows
        );
        CUDA_CHECK(cudaGetLastError());
    }
    // Else use the strided kernel.
    else
    {
        CUDA_LAUNCH(softmax_dim_strided<BLOCK>, rows, BLOCK,
            (float*)out_data,
            (const float*)in_data,
            element_stride,
            element_count, rows
        );
        CUDA_CHECK(cudaGetLastError());
    }
}

/*
--------------------------------------------------------------------------------------------------------------------------
 Shape Modifiers Operators
--------------------------------------------------------------------------------------------------------------------------
*/

template<int TILE_DIM, int BLOCK_ROWS>
__global__ void transpose_last2_kernel_vec(float* __restrict__ out, const float* __restrict__ in, int A, int B, int num_matrices)
{
    // Each matrix is A x B, contiguous.
    const int m = (int)blockIdx.z;
    if (m >= num_matrices) return;

    const int tx = (int)threadIdx.x;  // 0..TILE_DIM-1
    const int ty = (int)threadIdx.y;  // 0..BLOCK_ROWS-1

    // Tile origin in input matrix:
    // x0 spans columns (B), y0 spans rows (A)
    const int x0 = (int)blockIdx.x * TILE_DIM;
    const int y0 = (int)blockIdx.y * TILE_DIM;

    // Shared tile (+1 pad avoids bank conflicts in transpose)
    __shared__ float tile[TILE_DIM][TILE_DIM + 1];

    // Base pointers for this matrix
    const size_t base = (size_t)m * (size_t)A * (size_t)B;

    // ---- Load tile from input (coalesced). Use float4 loads when possible.
    #pragma unroll
    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
    {
        const int y = y0 + ty + j;  // input row
        const int x = x0 + tx;      // input col

        if (y < A && x < B)
        {
            // Vectorized load for groups of 4 columns.
            if (((tx & 3) == 0) && (x + 3 < B))
            {
                const float4 v4 = *(const float4*)(in + base + y * B + x);
                tile[ty + j][tx + 0] = v4.x;
                tile[ty + j][tx + 1] = v4.y;
                tile[ty + j][tx + 2] = v4.z;
                tile[ty + j][tx + 3] = v4.w;
            }
            // Scalar path for the other lanes.
            else
                tile[ty + j][tx] = in[base + y * B + x];
        }
        // Out of bounds: only need to write the locations we might read later.
        // We can just guard scalar stores; vector lanes will be skipped by the y/x checks.
        else if (tx / 4 != 0 || x + 3 >= B) tile[ty + j][tx] = 0.0f;
    }

    __syncthreads();

    // Store transposed tile to output.
    // Output matrix shape is B x A.
    #pragma unroll
    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
    {
        const int yT = x0 + ty + j; // output row index (0..B-1)
        const int xT = y0 + tx;     // output col index (0..A-1)

        if (yT < B && xT < A)
            out[base + yT * A + xT] = tile[tx][ty + j];
    }
}
template<int TILE_DIM, int BLOCK_ROWS>
__global__ void transpose_kernel_vec(float* __restrict__ out, const float* __restrict__ in, int A, int B, 
    int outer_size, int middle_size, int inner_size, int in_stride, int out_stride)
{
    // Flatten (outer,middle) into a single batch index.
    const int batch = (int)blockIdx.z;
    if (batch >= outer_size * middle_size) return;

    const int outter = batch / middle_size;
    const int middle = batch % middle_size;

    // Base offsets for this (outer,middle) batch
    const int in_base = outter * in_stride * A + middle * inner_size * B;
    const int out_base = outter * out_stride * B + middle * inner_size * A;

    // 2D tile indices in the (dim0, dim1) plane
    const int x0 = (int)blockIdx.x * TILE_DIM; // along dim1 (B)
    const int y0 = (int)blockIdx.y * TILE_DIM; // along dim0 (A)

    const int tx = (int)threadIdx.x;  // [0..31]
    const int ty = (int)threadIdx.y;  // [0..BLOCK_ROWS-1]

    // Vectorized: copy float4 packets along the inner dimension.
    // We still tile across (dim0, dim1) using shared memory transpose.

    // Using float4 shared tile; +1 column padding to reduce bank conflicts.
    __shared__ float4 tile4[TILE_DIM][TILE_DIM + 1];

    const int inner4 = inner_size / 4; // number of float4 packets

    // For each float4 packet along the inner tail:
    for (int p4 = 0; p4 < inner4; p4++)
    {
        // Load tile: coalesced reads
        #pragma unroll
        for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
        {
            const int y = y0 + ty + j; // dim0 index
            const int x = x0 + tx;     // dim1 index

            float4 v;
            v.x = v.y = v.z = v.w = 0.0f;

            if (y < A && x < B)
                v = *(const float4*)(in + in_base + y * in_stride + x * inner_size + p4 * 4);

            tile4[ty + j][tx] = v;
        }

        __syncthreads();

        // Store transposed tile: coalesced writes
        #pragma unroll
        for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
        {
            const int yT = x0 + ty + j; // becomes dim0 in output (was dim1)
            const int xT = y0 + tx;     // becomes dim1 in output (was dim0)

            if (xT < A && yT < B)
                *(float4*)(out + out_base + yT * out_stride + xT * inner_size + p4 * 4) = tile4[tx][ty + j];
        }

        __syncthreads();
    }
}
template<int TILE_DIM, int BLOCK_ROWS>
__global__ void transpose_kernel(float* __restrict__ out, const float* __restrict__ in, int A, int B, 
    int outer_size, int middle_size, int inner_size, int in_stride, int out_stride)
{
    // Flatten (outer,middle) into a single batch index.
    const int batch = (int)blockIdx.z;
    if (batch >= outer_size * middle_size) return;

    const int outter = batch / middle_size;
    const int middle = batch % middle_size;

    // Base offsets for this (outer,middle) batch
    const int in_base = outter * in_stride * A + middle * inner_size * B;
    const int out_base = outter * out_stride * B + middle * inner_size * A;

    // 2D tile indices in the (dim0, dim1) plane
    const int x0 = (int)blockIdx.x * TILE_DIM; // along dim1 (B)
    const int y0 = (int)blockIdx.y * TILE_DIM; // along dim0 (A)

    const int tx = (int)threadIdx.x;  // [0..31]
    const int ty = (int)threadIdx.y;  // [0..BLOCK_ROWS-1]

    // Scalar fallback: classic shared transpose of floats
    __shared__ float tile[TILE_DIM][TILE_DIM + 1];

    for (int p = 0; p < inner_size; ++p)
    {
        // Load tile
        #pragma unroll
        for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
        {
            const int y = y0 + ty + j;
            const int x = x0 + tx;

            float v = 0.0f;
            if (y < A && x < B)
                v = in[in_base + y * in_stride + x * inner_size + p];

            tile[ty + j][tx] = v;
        }

        __syncthreads();

        // Store transposed tile
        #pragma unroll
        for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
        {
            const int yT = x0 + ty + j;
            const int xT = y0 + tx;

            if (xT < A && yT < B)
                out[out_base + yT * out_stride + xT * inner_size + p] = tile[tx][ty + j];
        }

        __syncthreads();
    }
}
void kernel_ops::transpose(void* out_data, const void* in_data, int A, int B, int outter_size, int middle_size, int inner_size, int in_stride, int out_stride)
{
    constexpr int TILE_DIM = 32;
    constexpr int BLOCK_ROWS = 8;

    constexpr dim3 BLOCK(TILE_DIM, BLOCK_ROWS, 1);
    dim3 GRID(
        (unsigned)((B + TILE_DIM - 1) / TILE_DIM),
        (unsigned)((A + TILE_DIM - 1) / TILE_DIM),
        (unsigned)(outter_size * middle_size)
    );

    // Don't run empty tensors.
    if (!(outter_size * middle_size * inner_size)) return;

    // If can use the fast kernel use it.
    if (inner_size % 4 == 0)
    {
        CUDA_LAUNCH((transpose_kernel_vec<TILE_DIM, BLOCK_ROWS>), GRID, BLOCK,
            (float*)out_data,
            (const float*)in_data,
            A, B, outter_size, middle_size, inner_size,
            in_stride, out_stride
        );
        CUDA_CHECK(cudaGetLastError());
    }
    // If its the last two dimensions and we have alignement use their kernel.
    else if (inner_size == 1 && middle_size == 1 && B % 4 == 0)
    {
        CUDA_LAUNCH((transpose_last2_kernel_vec<TILE_DIM, BLOCK_ROWS>), GRID, BLOCK,
            (float*)out_data,
            (const float*)in_data,
            A, B, outter_size
        );
        CUDA_CHECK(cudaGetLastError());
    }
    // Else use the element wise one.
    else
    {
        CUDA_LAUNCH((transpose_kernel<TILE_DIM, BLOCK_ROWS>), GRID, BLOCK,
            (float*)out_data,
            (const float*)in_data,
            A, B, outter_size, middle_size, inner_size,
            in_stride, out_stride
        );
        CUDA_CHECK(cudaGetLastError());
    }
}

static inline void single_reduction_memcpy2d(void* out_data, const void* in_data, size_t prev_dim, size_t new_dim, size_t start_dim, size_t inner_size, size_t outer_size)
{
    const size_t widthBytes = new_dim * inner_size * sizeof(float);
    const size_t srcPitchBytes = prev_dim * inner_size * sizeof(float);
    const size_t dstPitchBytes = new_dim * inner_size * sizeof(float);

    const char* src = (const char*)in_data + start_dim * inner_size * sizeof(float);
    char* dst = (char*)out_data;

    if (outer_size == 1)
        CUDA_CHECK(cudaMemcpyAsync(dst, src, widthBytes, cudaMemcpyDeviceToDevice, stream::get()));

    else 
        CUDA_CHECK(cudaMemcpy2DAsync(
            dst, dstPitchBytes,
            src, srcPitchBytes,
            widthBytes, outer_size,
            cudaMemcpyDeviceToDevice,
            stream::get()
        ));
}
void kernel_ops::subset(const Shape& out_shape, const Shape& in_shape, const Shape& idxs, void* out_data, const void* in_data)
{
    Shape running_shape = in_shape;
    void* temp_data = nullptr;

    while (true)
    {
        float max_reduction = 1.f;
        size_t stride = 1u;
        unsigned reduction_dim = 0u;
        unsigned reduction_count = 0u;
        size_t reduction_stride = 0u;

        for (int d = running_shape.dim() - 1; d >= 0; d--)
        {
            if (running_shape[d] > out_shape[d])
            {
                reduction_count++;
                float reduc = float(out_shape[d]) / running_shape[d];

                if (reduc < max_reduction)
                {
                    max_reduction = reduc;
                    reduction_dim = d;
                    reduction_stride = stride;
                }
            }
            stride *= running_shape[d];
        }
        size_t outer = stride / (reduction_stride * running_shape[reduction_dim]);

        if (reduction_count == 1)
        {
            single_reduction_memcpy2d(out_data, temp_data ? temp_data : in_data,
                running_shape[reduction_dim], out_shape[reduction_dim], idxs[reduction_dim], reduction_stride, outer
            );

            if (temp_data) MemPool::free(temp_data);
            return;
        }
        else
        {
            void* new_temp = MemPool::allocate(sizeof(float) * stride * out_shape[reduction_dim] / running_shape[reduction_dim]);

            single_reduction_memcpy2d(new_temp, temp_data ? temp_data : in_data,
                running_shape[reduction_dim], out_shape[reduction_dim], idxs[reduction_dim], reduction_stride, outer
            );

            if (temp_data) MemPool::free(temp_data);
            temp_data = new_temp;

            running_shape[reduction_dim] = out_shape[reduction_dim];
        }
    }
}

struct InjectDesc
{
    static constexpr int MAX_DIMS = 32;

    int   ndim;
    size_t inner;                  // in_sizes[last]
    size_t total;                  // product(in_sizes[0..last-1])
    size_t in_sizes[MAX_DIMS];     // shape of input patch
    size_t out_strides[MAX_DIMS];  // strides of output
    size_t offsets[MAX_DIMS];      // output offsets
};
__device__ __forceinline__ size_t compute_out_base_for_row(size_t row, const InjectDesc& d)
{
    // Computes the output base offset for a given row.
    size_t out_base = 0;
    for (int dim = d.ndim - 2; dim >= 0; --dim)
    {
        const size_t size = d.in_sizes[dim];
        const size_t idx = row % size;
        row /= size;

        out_base += (idx + d.offsets[dim]) * d.out_strides[dim];
    }
    return out_base;
}
__global__ void inject_patch_kernel(float* __restrict__ out, const float* __restrict__ in, InjectDesc desc)
{
    const size_t tid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= desc.total) return;

    const size_t row = tid / desc.inner;
    const size_t col = tid - row * desc.inner;

    const size_t out_base = compute_out_base_for_row(row, desc);
    const size_t out_off = out_base + (desc.offsets[desc.ndim - 1] + col);
    out[out_off] = in[tid];
}
__global__ void inject_patch_kernel_vec(float4* __restrict__ out, const float4* __restrict__ in, InjectDesc desc)
{
    // Vectorized across last dim: inner must be multiple of 4 and last-offset multiple of 4.
    const size_t inner4 = desc.inner / 4;
    const size_t total4 = desc.total / 4;

    const size_t tid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total4) return;

    const size_t row = tid / inner4;
    const size_t col4 = tid - row * inner4;

    // Base in elements excluding last dim:
    const size_t out_base = compute_out_base_for_row(row, desc);

    // Convert element offset to float4 offset.
    const size_t offsets_last = desc.offsets[desc.ndim - 1];
    const size_t out_elem = out_base + offsets_last + col4 * 4;
    const size_t out4_idx = out_elem / 4;

    out[out4_idx] = in[tid];
}
void kernel_ops::modify(const Shape& out_shape, const Shape& in_shape, const Shape& out_strides, const Shape& idxs, void* out_data, const void* in_data)
{
    size_t stride = 1u;
    unsigned injection_dim = 0u;
    unsigned injection_count = 0u;
    size_t injection_stride = 0u;

    for (int d = in_shape.dim() - 1; d >= 0; d--)
    {
        if (out_shape[d] > in_shape[d])
        {
            injection_count++;
            injection_dim = d;
            injection_stride = stride;
        }
        stride *= in_shape[d];
    }
    
    // If just copying memory just copy memory.
    if (!injection_count)
        cuda::copy_gpu_to_gpu(out_data, in_data, stride * sizeof(float));

    // If only expanding once use a 2D memcpy.
    else if (injection_count == 1)
    {
        const size_t widthBytes = in_shape[injection_dim] * injection_stride * sizeof(float);
        const size_t srcPitchBytes = in_shape[injection_dim] * injection_stride * sizeof(float);
        const size_t dstPitchBytes = out_shape[injection_dim] * injection_stride * sizeof(float);
        const size_t outer = stride / (injection_stride * in_shape[injection_dim]);

        // Move out pointer to starting dimension.
        void* dst = (float*)out_data + idxs[injection_dim] * injection_stride;

        if (outer == 1)
            CUDA_CHECK(cudaMemcpyAsync(dst, in_data, widthBytes, cudaMemcpyDeviceToDevice, stream::get()));

        else
            CUDA_CHECK(cudaMemcpy2DAsync(
                dst, dstPitchBytes,
                in_data, srcPitchBytes,
                widthBytes, outer,
                cudaMemcpyDeviceToDevice,
                stream::get()
            ));
    }
    // Else use a kernel to copy everything.
    else
    {
        InjectDesc desc = {};
        desc.ndim = in_shape.dim();
        desc.inner = in_shape[-1];
        desc.total = stride;

        for (int d = 0; d < desc.ndim; ++d)
        {
            desc.in_sizes[d] = (size_t)in_shape[d];
            desc.offsets[d] = (size_t)idxs[d];
            desc.out_strides[d] = (size_t)out_strides[d];
        }

        constexpr int BLOCK = 256;
        // If possible use vectorized kernel.
        if ((desc.inner % 4u == 0u) && (idxs[-1] % 4u == 0u))
        {
            CUDA_LAUNCH(inject_patch_kernel_vec, unsigned(desc.total / 4u + BLOCK - 1) / BLOCK, BLOCK,
                (float4*)out_data,
                (const float4*)in_data,
                desc
            );
            CUDA_CHECK(cudaGetLastError());
        }
        // Else fallback to scalar.
        else
        {
            CUDA_LAUNCH(inject_patch_kernel, unsigned(desc.total + BLOCK - 1) / BLOCK, BLOCK,
                (float*)out_data,
                (const float*)in_data,
                desc
            );
            CUDA_CHECK(cudaGetLastError());
        }
    }
}

__global__ void repeat_lastdim_kernel(float* __restrict__ out, const float* __restrict__ in, size_t num_elements)
{
    size_t in_stride = blockDim.x * gridDim.x;
    size_t out_stride = blockDim.x * gridDim.x * gridDim.y;

    size_t in_idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t out_idx = blockIdx.y + in_idx * gridDim.y;

    for (size_t i = in_idx, j = out_idx; i < num_elements; i += in_stride, j += out_stride)
        out[j] = in[i];
}
template<int BLOCK>
__global__ void repeat_kernel_vec(float4* __restrict__ out, const float4* __restrict__ in, size_t inner_vec)
{
    size_t in_idx = blockIdx.x * inner_vec + threadIdx.x;
    size_t out_idx = blockIdx.x * (inner_vec * gridDim.y) + blockIdx.y * inner_vec + threadIdx.x;

    size_t in_end = (blockIdx.x + 1) * inner_vec;

    for (size_t i = in_idx, j = out_idx; i < in_end; i += BLOCK, j += BLOCK)
        out[j] = in[i];
}
template<int BLOCK>
__global__ void repeat_kernel(float* __restrict__ out, const float* __restrict__ in, size_t inner_elements)
{
    size_t in_idx = blockIdx.x * inner_elements + threadIdx.x;
    size_t out_idx = blockIdx.x * (inner_elements * gridDim.y) + blockIdx.y * inner_elements + threadIdx.x;

    size_t in_end = (blockIdx.x + 1) * inner_elements;

    for (size_t i = in_idx, j = out_idx; i < in_end; i += BLOCK, j += BLOCK)
        out[j] = in[i];
}
void kernel_ops::repeat(void* out_data, const void* in_data, int outer_size, int inner_size, int repetitions)
{
    constexpr int BLOCK = 256;

    if (outer_size == 1 && repetitions < 16)
        for (int k = 0; k < repetitions; k++)
            CUDA_CHECK(cudaMemcpyAsync(((float*)out_data) + k * inner_size, in_data, inner_size * sizeof(float), cudaMemcpyDeviceToDevice, stream::get()));

    else if (inner_size == 1)
    {
        dim3 GRID(
            (unsigned)(device::sm_count() * 4 + repetitions - 1) / repetitions,
            (unsigned)repetitions
        );
        CUDA_LAUNCH(repeat_lastdim_kernel, GRID, BLOCK,
            (float*)out_data,
            (const float*)in_data,
            outer_size
        );
        CUDA_CHECK(cudaGetLastError());
    }
    else if (inner_size % 4 == 0)
    {
        dim3 GRID(
            (unsigned)outer_size,
            (unsigned)repetitions
        );
        CUDA_LAUNCH(repeat_kernel_vec<BLOCK>, GRID, BLOCK,
            (float4*)out_data,
            (const float4*)in_data,
            inner_size / 4
        );
        CUDA_CHECK(cudaGetLastError());
    }
    else
    {
        dim3 GRID(
            (unsigned)outer_size,
            (unsigned)repetitions
        );
        CUDA_LAUNCH(repeat_kernel<BLOCK>, GRID, BLOCK,
            (float*)out_data,
            (const float*)in_data,
            inner_size
        );
        CUDA_CHECK(cudaGetLastError());
    }
}

/*
--------------------------------------------------------------------------------------------------------------------------
 Functional Namespace Operators
--------------------------------------------------------------------------------------------------------------------------
*/

class cublas
{
private:
    static inline cublasLtHandle_t create_handle()
    {
        cublasLtHandle_t lt;
        CUBLAS_CHECK(cublasLtCreate(&lt));
        return lt;
    }
    static inline cublasLtMatmulPreference_t init_preference()
    {
        cublasLtMatmulPreference_t pref = nullptr;
        CUBLAS_CHECK(cublasLtMatmulPreferenceCreate(&pref));
        CUBLAS_CHECK(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &cublas::workspace_size, sizeof(cublas::workspace_size)));
        return pref;
    }
    
public:
    static constexpr size_t workspace_size = 0x01000000ULL;

    static inline void* workspace()
    {
        static thread_local void* workspace = MemPool::allocate(cublas::workspace_size);
        return workspace;
    }
    static inline cublasLtHandle_t handle()
    {
        static cublasLtHandle_t lt = create_handle();
        return lt;
    }
    static inline cublasLtMatmulPreference_t preference()
    {
        static cublasLtMatmulPreference_t pref = init_preference();
        return pref;
    }
};

void kernel_ops::matmul(void* out_data, const void* A_data, const void* B_data, const Shape& A_shape, const Shape& B_shape)
{
    constexpr size_t storage_size = 0x1000ull;
    constexpr size_t mask = storage_size - 1;

    constexpr float alpha = 1.0f;
    constexpr float beta  = 0.0f;

    struct GEMM_data
    {
        cublasLtMatmulAlgo_t algorithm = {};
        cublasLtMatmulDesc_t opDesc = nullptr;
        cublasLtMatrixLayout_t aLayout = nullptr;
        cublasLtMatrixLayout_t bLayout = nullptr;
        cublasLtMatrixLayout_t cLayout = nullptr;
        size_t key = 0u;
    }
    static thread_local storage[storage_size] = {};

    class gemm_data_bank
    {
    private:
        static inline size_t hash(const Shape& A_shape, const Shape& B_shape)
        {
            auto splitmix = [](size_t _seed)
            {
                _seed += 0x9E3779B97F4A7C15ull;
                _seed = (_seed ^ (_seed >> 30)) * 0xBF58476D1CE4E5B9ull;
                _seed = (_seed ^ (_seed >> 27)) * 0x94D049BB133111EBull;
                _seed ^= (_seed >> 31);
                return _seed;
            };

            size_t key = 123ull;
            for (unsigned i = 0; i < A_shape.dim(); i++)
                key ^= splitmix(i) * A_shape[i] + splitmix(splitmix(i)) * B_shape[i];

            return key;
        }
    public:
        static inline GEMM_data& get_data(const Shape& A_shape, const Shape& B_shape)
        {
            // Find your data slot and retrieve the entry.
            size_t key = hash(A_shape, B_shape);

            // Do 2-slot, if any matches we are done here.
            GEMM_data* slot0 = &storage[key & mask];
            if (slot0->key == key) return *slot0;

            GEMM_data* slot1 = &storage[(key & mask) ^ 0x1];
            if (slot1->key == key) return *slot1;

            // Pick a data slot based on if they are occupied.
            GEMM_data& data = slot0->key ? *slot1 : *slot0;

            // If there is something here too we'll have to clean it.
            if (data.key)
            {
                CUBLAS_CHECK(cublasLtMatmulDescDestroy(data.opDesc));
                CUBLAS_CHECK(cublasLtMatrixLayoutDestroy(data.aLayout));
                CUBLAS_CHECK(cublasLtMatrixLayoutDestroy(data.bLayout));
                CUBLAS_CHECK(cublasLtMatrixLayoutDestroy(data.cLayout));
            }

            // --- Else time to generate the data ---
            
            // Different possible layout cases to consider.
            bool b_batched = true;
            bool arbitrary = false;

            // Integer to store the batch size.
            int batch_count = 1u;

            // Find out which case we are dealing with.
            for (unsigned i = 0; i < A_shape.dim() - 2; i++)
            {
                int a = A_shape[i], b = B_shape[i];
                batch_count *= (a > b) ? a : b;

                if (a > b) b_batched = false;
                if (b > a || (!b_batched && b != 1)) arbitrary = true;
            }

            // Extract matrix dimensions.
            int64_t M = A_shape[-2];
            int64_t N = B_shape[-1];
            int64_t K = A_shape[-1];

            // Create a matmul descriptor.
            CUBLAS_CHECK(cublasLtMatmulDescCreate(&data.opDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F));

            // Create col-major layouts for A^T(K, M), B^T(N, K), C^T(N,M)
            CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&data.aLayout, CUDA_R_32F, K, M, K));
            CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&data.bLayout, CUDA_R_32F, N, K, N));
            CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&data.cLayout, CUDA_R_32F, N, M, N));

            // If it is batched set the batch mode.
            if (batch_count > 1)
            {
                int64_t strideB = (b_batched) ? K * N : 0ull;
                int64_t strideA = M * K;
                int64_t strideC = M * N;

                cublasLtBatchMode_t batch_mode = CUBLASLT_BATCH_MODE_STRIDED;

                CUBLAS_CHECK(cublasLtMatrixLayoutSetAttribute(data.aLayout, CUBLASLT_MATRIX_LAYOUT_BATCH_MODE, &batch_mode, sizeof(batch_mode)));
                CUBLAS_CHECK(cublasLtMatrixLayoutSetAttribute(data.aLayout, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch_count, sizeof(batch_count)));
                CUBLAS_CHECK(cublasLtMatrixLayoutSetAttribute(data.aLayout, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &strideA, sizeof(strideA)));

                CUBLAS_CHECK(cublasLtMatrixLayoutSetAttribute(data.bLayout, CUBLASLT_MATRIX_LAYOUT_BATCH_MODE, &batch_mode, sizeof(batch_mode)));
                CUBLAS_CHECK(cublasLtMatrixLayoutSetAttribute(data.bLayout, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch_count, sizeof(batch_count)));
                CUBLAS_CHECK(cublasLtMatrixLayoutSetAttribute(data.bLayout, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &strideB, sizeof(strideB)));

                CUBLAS_CHECK(cublasLtMatrixLayoutSetAttribute(data.cLayout, CUBLASLT_MATRIX_LAYOUT_BATCH_MODE, &batch_mode, sizeof(batch_mode)));
                CUBLAS_CHECK(cublasLtMatrixLayoutSetAttribute(data.cLayout, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch_count, sizeof(batch_count)));
                CUBLAS_CHECK(cublasLtMatrixLayoutSetAttribute(data.cLayout, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &strideC, sizeof(strideC)));
            }
            if (arbitrary)
            {
                TENSOR_ERROR("Arbitrarily sized batch GEMMs not yet implemented.");
            }

            // Heuristic algo selection
            cublasLtMatmulHeuristicResult_t heur = {};
            int out = 0;
            CUBLAS_CHECK(cublasLtMatmulAlgoGetHeuristic(cublas::handle(), data.opDesc, 
                data.bLayout, data.aLayout, data.cLayout, data.cLayout, cublas::preference(), 1, &heur, &out));
            TENSOR_CHECK(out != 0, "No cuBLASLt heuristic algorithm found for this GEMM.");

            // Store the algorithm and return the data.
            data.algorithm = heur.algo; data.key = key;
            return data;
        }
    };

    // Load data from bank.
    GEMM_data& data = gemm_data_bank::get_data(A_shape, B_shape);

    // Execute cuBLAS Lt matmul.
    CUBLAS_CHECK(cublasLtMatmul(
        cublas::handle(),
        data.opDesc,
        &alpha,
        B_data, data.bLayout,
        A_data, data.aLayout,
        &beta,
        out_data, data.cLayout,
        out_data, data.cLayout,
        &data.algorithm,
        cublas::workspace(), 
        cublas::workspace_size,
        stream::get()
    ));
}

void kernel_ops::matmul_bias(void* out_data, const void* A_data, const void* B_data, const void* bias, const Shape& A_shape, const Shape& B_shape, const Shape& bias_shape)
{
    constexpr size_t storage_size = 0x1000ull;
    constexpr size_t mask = storage_size - 1;

    constexpr float alpha = 1.0f;
    constexpr float beta  = 0.0f;

    struct GEMM_B_data
    {
        cublasLtMatmulAlgo_t algorithm = {};
        cublasLtMatmulDesc_t opDesc = nullptr;
        cublasLtMatrixLayout_t aLayout = nullptr;
        cublasLtMatrixLayout_t bLayout = nullptr;
        cublasLtMatrixLayout_t cLayout = nullptr;
        size_t key = 0u;
        bool bias_fused = false;
    }
    static thread_local storage[storage_size] = {};
    
    class gemm_b_data_bank
    {
    private:
        static inline size_t hash(const Shape& A_shape, const Shape& B_shape, const Shape& bias_shape)
        {
            auto splitmix = [](size_t _seed)
            {
                _seed += 0x9E3779B97F4A7C15ull;
                _seed = (_seed ^ (_seed >> 30)) * 0xBF58476D1CE4E5B9ull;
                _seed = (_seed ^ (_seed >> 27)) * 0x94D049BB133111EBull;
                _seed ^= (_seed >> 31);
                return _seed;
            };
    
            // Remena nena.
            size_t key = 123ull;
            for (unsigned i = 0; i < A_shape.dim(); i++)
                key ^= splitmix(i) * A_shape[i] + 
                splitmix(splitmix(i)) * B_shape[i] + 
                splitmix(splitmix(splitmix(i))) * bias_shape[i];
    
            return key;
        }
    public:
        static inline GEMM_B_data& get_data(const Shape& A_shape, const Shape& B_shape, const Shape& bias_shape)
        {
            // Find your data slot and retrieve the entry.
            size_t key = hash(A_shape, B_shape, bias_shape);
    
            // Do 2-slot, if any matches we are done here.
            GEMM_B_data* slot0 = &storage[key & mask];
            if (slot0->key == key) return *slot0;
    
            GEMM_B_data* slot1 = &storage[(key & mask) ^ 0x1];
            if (slot1->key == key) return *slot1;
    
            // Pick a data slot based on if they are occupied.
            GEMM_B_data& data = slot0->key ? *slot1 : *slot0;
    
            // If there is something here too we'll have to clean it.
            if (data.key)
            {
                CUBLAS_CHECK(cublasLtMatmulDescDestroy(data.opDesc));
                CUBLAS_CHECK(cublasLtMatrixLayoutDestroy(data.aLayout));
                CUBLAS_CHECK(cublasLtMatrixLayoutDestroy(data.bLayout));
                CUBLAS_CHECK(cublasLtMatrixLayoutDestroy(data.cLayout));
            }
    
            // --- Else time to generate the data ---
            
            // Different possible layout cases to consider.
            bool B_batched = true;
            bool b_batched = true;
            bool arbitrary = false;
            data.bias_fused = true;

            // Integer to store the batch size.
            int batch_count = 1u;

            // Find out which case we are dealing with.
            for (unsigned i = 0; i < A_shape.dim() - 2; i++)
            {
                int A = A_shape[i];
                int B = B_shape[i];
                int b = bias_shape[i];
                batch_count *= (A >= B) ? A : B;

                if (A > B) B_batched = false;
                if (A > b) b_batched = false;
                if (B > A || (!B_batched && B != 1)) arbitrary = true;
                if (!b_batched && b != 1) data.bias_fused = false;
            }

            // Last check for fused bias. make sure is (1,N) case.
            if (bias_shape[-1] == 1 || bias_shape[-2] != 1)
                data.bias_fused = false;

            // Extract matrix dimensions.
            int64_t M = A_shape[-2];
            int64_t N = B_shape[-1];
            int64_t K = A_shape[-1];

            // Create a matmul descriptor.
            CUBLAS_CHECK(cublasLtMatmulDescCreate(&data.opDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F));

            // Create col-major layouts for A^T(K, M), B^T(N, K), C^T(N,M)
            CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&data.aLayout, CUDA_R_32F, K, M, K));
            CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&data.bLayout, CUDA_R_32F, N, K, N));
            CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&data.cLayout, CUDA_R_32F, N, M, N));

            if (arbitrary)
            {
                TENSOR_ERROR("Arbitrarily sized batch GEMMs not yet implemented.");
            }
            // If batched add the batches.
            else if (batch_count > 1)
            {
                int64_t strideB = (B_batched) ? K * N : 0;
                int64_t strideA = M * K;
                int64_t strideC = M * N;

                cublasLtBatchMode_t batch_mode = CUBLASLT_BATCH_MODE_STRIDED;

                CUBLAS_CHECK(cublasLtMatrixLayoutSetAttribute(data.aLayout, CUBLASLT_MATRIX_LAYOUT_BATCH_MODE, &batch_mode, sizeof(batch_mode)));
                CUBLAS_CHECK(cublasLtMatrixLayoutSetAttribute(data.aLayout, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch_count, sizeof(batch_count)));
                CUBLAS_CHECK(cublasLtMatrixLayoutSetAttribute(data.aLayout, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &strideA, sizeof(strideA)));

                CUBLAS_CHECK(cublasLtMatrixLayoutSetAttribute(data.bLayout, CUBLASLT_MATRIX_LAYOUT_BATCH_MODE, &batch_mode, sizeof(batch_mode)));
                CUBLAS_CHECK(cublasLtMatrixLayoutSetAttribute(data.bLayout, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch_count, sizeof(batch_count)));
                CUBLAS_CHECK(cublasLtMatrixLayoutSetAttribute(data.bLayout, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &strideB, sizeof(strideB)));

                CUBLAS_CHECK(cublasLtMatrixLayoutSetAttribute(data.cLayout, CUBLASLT_MATRIX_LAYOUT_BATCH_MODE, &batch_mode, sizeof(batch_mode)));
                CUBLAS_CHECK(cublasLtMatrixLayoutSetAttribute(data.cLayout, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch_count, sizeof(batch_count)));
                CUBLAS_CHECK(cublasLtMatrixLayoutSetAttribute(data.cLayout, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &strideC, sizeof(strideC)));
            }

            if (data.bias_fused)
            {
                cublasLtEpilogue_t epi = CUBLASLT_EPILOGUE_BIAS;
                CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(data.opDesc, CUBLASLT_MATMUL_DESC_EPILOGUE, &epi, sizeof(epi)));

                if (b_batched)
                    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(data.opDesc, CUBLASLT_MATMUL_DESC_BIAS_BATCH_STRIDE, &N, sizeof(N)));
            }

            // Heuristic algo selection
            cublasLtMatmulHeuristicResult_t heur = {};
            int out = 0;
            CUBLAS_CHECK(cublasLtMatmulAlgoGetHeuristic(cublas::handle(), data.opDesc,
                data.bLayout, data.aLayout, data.cLayout, data.cLayout, cublas::preference(), 1, &heur, &out));
            TENSOR_CHECK(out != 0, "No cuBLASLt heuristic algorithm found for this GEMM.");

            // Store the algorithm and return the data.
            data.algorithm = heur.algo; data.key = key;
            return data;
        }
    };

    // Load data from bank.
    GEMM_B_data& data = gemm_b_data_bank::get_data(A_shape, B_shape, bias_shape);

    if (data.bias_fused)
        CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(data.opDesc, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias, sizeof(bias)));

    // Execute cuBLAS Lt matmul.
    CUBLAS_CHECK(cublasLtMatmul(
        cublas::handle(),
        data.opDesc,
        &alpha,
        B_data, data.bLayout,
        A_data, data.aLayout,
        &beta,
        out_data, data.cLayout,
        out_data, data.cLayout,
        &data.algorithm,
        cublas::workspace(), 
        cublas::workspace_size,
        stream::get()
    ));

    if (!data.bias_fused)
    {
        TENSOR_ERROR("Matmul fallback without fused bias not implemented yet.");
    }
}

void kernel_ops::cat(void* out_data, const void* in0_data, const void* in1_data, size_t inner_size, size_t outer_size, int size0, int size1)
{
    const size_t widthBytes0 = size0 * inner_size * sizeof(float);
    const size_t widthBytes1 = size1 * inner_size * sizeof(float);
    const size_t dstPitchBytes = (size0 + size1) * inner_size * sizeof(float);

    // If last dimension just copy twice.
    if (outer_size == 1)
    {
        cuda::copy_gpu_to_gpu(out_data                      , in0_data, widthBytes0);
        cuda::copy_gpu_to_gpu((char*)out_data + widthBytes0 , in1_data, widthBytes1);
    }
    // Else do 2D memcpy instead.
    else
    {
        CUDA_CHECK(cudaMemcpy2DAsync(
            out_data, dstPitchBytes,
            in0_data, widthBytes0,
            widthBytes0, outer_size,
            cudaMemcpyDeviceToDevice,
            stream::get()
        ));
        CUDA_CHECK(cudaMemcpy2DAsync(
            (char*)out_data + widthBytes0, dstPitchBytes,
            in1_data, widthBytes1,
            widthBytes1, outer_size,
            cudaMemcpyDeviceToDevice,
            stream::get()
        ));
    }
}

template<int BLOCK>
__global__ void mse_kernel_vec(float* __restrict__ out, const float4* x, const float4* y, size_t num_vec)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    float sum = 0.0f;
    for (size_t i = idx; i < num_vec; i += stride)
    {

        float4 v = x[i];
        float4 u = y[i];

        v.x = (v.x - u.x) * (v.x - u.x);
        v.y = (v.y - u.y) * (v.y - u.y);
        v.z = (v.z - u.z) * (v.z - u.z);
        v.w = (v.w - u.w) * (v.w - u.w);

        sum += v.x + v.y + v.z + v.w;
    }

    sum = block_sum<BLOCK>(sum);
    if (threadIdx.x == 0) *out = sum / (num_vec * 4);
}
template<int BLOCK>
__global__ void mse_kernel(float* __restrict__ out, const float* x, const float* y, size_t num_elements)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    float sum = 0.0f;
    for (size_t i = idx; i < num_elements; i += stride)
        sum += (x[i] - y[i]) * (x[i] - y[i]);

    sum = block_sum<BLOCK>(sum);
    if (threadIdx.x == 0) *out = sum / num_elements;
}
void kernel_ops::mse(void* out_data, const void* x_data, const void* y_data, size_t num_elements)
{
    constexpr int BLOCK = 1024;

    // If possible use the fast kernel
    if (num_elements % 4 == 0)
    {
        CUDA_LAUNCH(mse_kernel_vec<BLOCK>, 1, BLOCK,
            (float*)out_data,
            (const float4*)x_data,
            (const float4*)y_data,
            num_elements / 4
        );
        CUDA_CHECK(cudaGetLastError());
    }
    // Else fallback to element-wise
    else
    {
        CUDA_LAUNCH(mse_kernel<BLOCK>, 1, BLOCK,
            (float*)out_data,
            (const float*)x_data,
            (const float*)y_data,
            num_elements
        );
        CUDA_CHECK(cudaGetLastError());
    }
}

__global__ void causal_mask_kernel(float* __restrict__ out, int L)
{
    const int row = blockIdx.x;
    float* out_row = out + row * L;

    for (int col = row + threadIdx.x + 1; col < L; col += blockDim.x)
        out_row[col] = -CUDART_INF_F;
}
void kernel_ops::causal_mast(void* out_data, int L)
{
    constexpr int BLOCK = 256;

    CUDA_LAUNCH(causal_mask_kernel, L, BLOCK, (float*)out_data, L);
    CUDA_CHECK(cudaGetLastError());
}
