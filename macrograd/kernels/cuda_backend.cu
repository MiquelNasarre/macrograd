#include "macrograd_error.h"
#include "macrograd.h"
#include "cuda_backend.h"

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <device_launch_parameters.h>
#include <math.h>

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

// Checks CUDA expression, if 'cudaError != cudaSuccess' raises a TENSOR_ERROR. 
#define CUDA_CHECK(x)   do { cudaError_t err = (x); if(err != cudaSuccess) TENSOR_ERROR("\nCUDA error string:\n", cudaGetErrorString(err)); } while(0)

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

#ifdef __INTELLISENSE__
inline float __expf(float x) { return expf(x); }
inline float __logf(float x) { return logf(x); }
#endif

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