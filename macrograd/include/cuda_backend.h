#pragma once
#include "macrograd.h"
#include <stdint.h>

class MemPool
{
	MemPool() = delete;
public:
	// Get your free GPU memory here! Freshly zeroed!
	static void* allocate(size_t byte_size);

	// Recicle your GPU memory here!
	static void free(void* data_ptr);
};

namespace cuda
{
	// Sets the data to zero for the specified byte size.
	void zero_data(void* data_ptr, size_t byte_size);

	// Copies the data from the CPU pointer to the GPU pointer.
	void copy_cpu_to_gpu(void* gpu_data_ptr, const void* cpu_data_ptr, size_t byte_size);

	// Copies the data from the GPU pointer to the CPU pointer.
	void copy_gpu_to_cpu(void* cpu_data_ptr, const void* gpu_data_ptr, size_t byte_size);

	// Copies data from one GPU poiner to another GPU pointer.
	void copy_gpu_to_gpu(void* dst_ptr, const void* src_ptr, size_t byte_size);

	// Waits until global stream is done.
	void synchronize();

	// Set gradient element value to one for backprop.
	void set_to_one(void* data_ptr);
}

namespace kernel_ops
{
	void set_scalar(void* out_data, float val, size_t num_elements);

	void add_scalar(void* out_data, float val, size_t num_elements);

	void add_tensor(void* out_data, const void* sum_data, size_t num_elements);

	void add_tensor_tensor(void* out_data, const void* sum0_data, const void* sum1_data, size_t num_elements);

	void multiply_scalar(void* out_data, float val, size_t num_elements);

	void multiply_tensor(void* out_data, const void* fac_data, size_t num_elements);

	void multiply_tensor_tensor(void* out_data, const void* fac0_data, const void* fac1_data, size_t num_elements);

	void add_multiply_scalar_tensor(void* out_data, float val, const void* fac_data, size_t num_elements);

	void subtract_tensor(void* out_data, const void* sub_data, size_t num_elements);

	void subtract_tensor_tensor(void* out_data, const void* sum_data, const void* sub_data, size_t num_elements);

	void divide_tensor(void* out_data, const void* den_data, size_t num_elements);

	void divide_tensor_tensor(void* out_data, const void* num_data, const void* den_data, size_t num_elements);

	// --- Element-Wise Operators ---

	void sign	(void* out_data, const void* in_data, size_t num_elements);
	void exp	(void* out_data, const void* in_data, size_t num_elements);
	void log    (void* out_data, const void* in_data, size_t num_elements);
	void relu   (void* out_data, const void* in_data, size_t num_elements);
	void silu   (void* out_data, const void* in_data, size_t num_elements);
	void gelu   (void* out_data, const void* in_data, size_t num_elements);
	void sigmoid(void* out_data, const void* in_data, size_t num_elements);
	void tanh	(void* out_data, const void* in_data, size_t num_elements);
	void sqrt	(void* out_data, const void* in_data, size_t num_elements);
	void square	(void* out_data, const void* in_data, size_t num_elements);
	void pow	(void* out_data, const void* in_data, float exp, size_t num_elements);

	// --- RNG Initialization ---

	void set_seed(size_t seed);
	void normal(void* data_ptr, float mean, float std, size_t num_elements);
	void uniform(void* data_ptr, float min, float max, size_t num_elements);
}
