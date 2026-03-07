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
	void add_scalar(void* out_data, const void* ten_data, float val, size_t num_elements);
	void add_tensor(void* out_data, const void* sum0_data, const void* sum1_data, size_t num_elements);
	void multiply_scalar(void* out_data, const void* fac_data, float val, size_t num_elements);
	void multiply_tensor(void* out_data, const void* fac0_data, const void* fac1_data, size_t num_elements);
	void subtract_from_scalar(void* out_data, const void* sub_data, float val, size_t num_elements);
	void divide_from_scalar(void* out_data, const void* den_data, float val, size_t num_elements);
	void add_multiply_scalar_tensor(void* out_data, float val, const void* fac_data, size_t num_elements);
	void subtract_tensor(void* out_data, const void* sum_data, const void* sub_data, size_t num_elements);
	void divide_tensor(void* out_data, const void* num_data, const void* den_data, size_t num_elements);

	void tensor_bracket_op(void* out_data, const void* ten_data, const void* indices_data, size_t num_indices, size_t stride, size_t range);
	void vector_bracket_op(void* out_data, const void* vec_data, const void* indices_data, size_t num_indices, size_t range);

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
	void shuffle(void* values, size_t length);
	void arange(void* data_ptr, int a, int stride, size_t count);

	// --- Row-Wise Operators ---

	void sum(void* out_data, const void* in_data, int element_stride, int element_count, int rows);
	void mean(void* out_data, const void* in_data, int element_stride, int element_count, int rows);
	void var(void* out_data, const void* in_data, int element_stride, int element_count, int rows);
	void std(void* out_data, const void* in_data, int element_stride, int element_count, int rows);
	void softmax(void* out_data, const void* in_data, int element_stride, int element_count, int rows);

	// --- Shape modifiers ---

	void transpose(void* out_data, const void* in_data, int A, int B, int outter_size, int middle_size, int inner_size, int in_stride, int out_stride);
	void subset(const Shape& out_shape, const Shape& in_shape, const Shape& idxs, void* out_data, const void* in_data);
	void modify(const Shape& out_shape, const Shape& in_shape, const Shape& out_strides, const Shape& idxs, void* out_data, const void* in_data);
	void repeat(void* out_data, const void* in_data, int outer_size, int inner_size, int repetitions);

	// --- Functional Namespace ---

	void matmul(void* out_data, const void* A_data, const void* B_data, const Shape& out_shape, const Shape& A_shape, const Shape& B_shape);
	void matmul_bias(void* out_data, const void* A_data, const void* B_data, const void* bias, const Shape& out_shape, const Shape& A_shape, const Shape& B_shape, const Shape& bias_shape);
	void cat(void* out_data, const void* in0_data, const void* in1_data, size_t inner_size, size_t outer_size, int size0, int size1);
	void mse(void* out_data, const void* x_data, const void* y_data, size_t num_elements);
	void cross_entropy_loss(void* out_data, void* probs_data, const void* logits_data, const void* labels_data, size_t num_cases, size_t num_classes);
	void negative_log_likelihood(void* out_data, const void* probs_data, const void* labels_data, size_t num_cases, size_t num_classes);
	void one_hot(void* out_data, const void* labels_data, size_t num_cases, size_t num_classes);
	void causal_mast(void* out_data, int L);

	// --- Regular Operators ---

	void shaped_add(void* out_data, const void* in0_data, const void* in1_data, const Shape& out_shape, const Shape& in_shape);
	void shaped_subtract(void* out_data, const void* in0_data, const void* in1_data, const Shape& out_shape, const Shape& in_shape);
	void shaped_multiply(void* out_data, const void* in0_data, const void* in1_data, const Shape& out_shape, const Shape& in_shape);
	void shaped_divide(void* out_data, const void* in0_data, const void* in1_data, const Shape& out_shape, const Shape& in_shape);
}
