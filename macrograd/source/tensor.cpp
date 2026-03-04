#include "macrograd.h"
#include "macrograd_error.h"
#include "cuda_backend.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <new>

// Macro to help with data acess on CPU.
#define __data ((float*)_internals->_data)

/*
--------------------------------------------------------------------------------------------------------------------------
 Shape functions
--------------------------------------------------------------------------------------------------------------------------
*/

Shape::Shape(unsigned dim, int* sizes) : _dim{ dim }
{
	if (!dim)
		return;

	_sizes = new int[dim];

	if (sizes)
		for (unsigned i = 0; i < dim; i++)
			_sizes[i] = sizes[i];
	else
		for (unsigned i = 0; i < dim; i++)
			_sizes[i] = 0;
}

Shape& Shape::operator=(const Shape& other)
{
	if (_dim == other._dim)
	{
		for (unsigned i = 0; i < _dim; i++)
			_sizes[i] = other._sizes[i];
		return *this;
	}

	if (_sizes)
	{
		delete[] _sizes;
		_sizes = nullptr;
	}

	_dim = other._dim;
	if (!_dim)
		return *this;

	_sizes = new int[other._dim];
	for (unsigned i = 0; i < other._dim; i++)
		_sizes[i] = other._sizes[i];

	return *this;
}

void Shape::remove(int dim)
{
	if (!_dim)
		return;

	if (_dim == 1)
	{
		delete[] _sizes;
		_sizes = nullptr;
		_dim = 0;
		return;
	}

	// Modulo dim.
	dim = mod(dim, _dim);

	// Write new sizes.
	int* new_sizes = new int[_dim - 1];

	for (int i = 0; i < dim; i++)
		new_sizes[i] = _sizes[i];
	for (unsigned i = dim + 1; i < _dim; i++)
		new_sizes[i - 1] = _sizes[i];

	// Adopt new sizes.
	_dim--;
	delete[] _sizes;
	_sizes = new_sizes;
}

void Shape::add(int dim, int size)
{
	if (!_dim)
	{
		_dim = 1;
		_sizes = new int[1];
		_sizes[0] = size;
		return;
	}

	// Modulo dim.
	dim = mod(dim, _dim + 1);

	// Write new sizes.
	int* new_sizes = new int[_dim + 1];

	for (int i = 0; i < dim; i++)
		new_sizes[i] = _sizes[i];
	new_sizes[dim] = size;
	for (unsigned i = dim; i < _dim; i++)
		new_sizes[i + 1] = _sizes[i];

	// Adopt new sizes.
	_dim++;
	delete[] _sizes;
	_sizes = new_sizes;
}

const char* Shape::str(const char* fmt) const
{
	thread_local static char buffer[16][256] = {};
	thread_local static int next = 0;

	char* buf = buffer[(next++) % 16];
	int left = 64;

	*(buf++) = '('; left--;

	for (unsigned i = 0; i < _dim && left > 0; i++)
	{
		int added = snprintf(buf, left, fmt, _sizes[i]);
		buf += added, left -= added;

		if (i < _dim - 1 && left > 2)
		{
			*(buf++) = ','; left--;
			*(buf++) = ' '; left--;
		}
	}
	if (left > 1)
	{
		*(buf++) = ')'; left--;
		*(buf++) = '\0'; left--;
	}

	return buffer[(next - 1) % 16];
}

/*
--------------------------------------------------------------------------------------------------------------------------
 Constructor / Destructor
--------------------------------------------------------------------------------------------------------------------------
*/

Tensor::Tensor(const Tensor& other)
{
	if (other._internals)
	{
		_internals = other._internals;
		_internals->instances++;

		_view = other._view;
		_stride = other._stride;
		_is_no_grad = other._is_no_grad;
	}
}

Tensor::Tensor(const Shape& shape, const char* device, bool requires_grad)
{
	TENSOR_CHECK(shape.dim(),
		"If the tensor created will have zero dimensions please use the default constructor."
	);
	for (unsigned i = 0; i < shape.dim(); i++)
		TENSOR_CHECK(shape[i] >= 0,
			"A shape with negative values is not allowed for initialization."
		);

	_internals = new TensorInternals;
	_internals->instances++;

	// Copy view and initialize stride.
	_view = shape;
	_stride = Shape(shape.dim(), (int*)nullptr);

	// Set strides to size * offset of previous dimension.
	_stride[-1] = 1u;
	for (int i = shape.dim() - 2; i >= 0; i--)
		_stride[i] = shape[i + 1] * _stride[i + 1];

	// Get the total number of elements.
	_internals->_numel = _stride[0] * shape[0];

	// Get the total data size 64-byte aligned.
	_internals->_data_size = _internals->_numel * sizeof(float);
	if (_internals->_data_size % 64 != 0)
		_internals->_data_size += 64 - _internals->_data_size % 64;

	// Set strides to 0 for unitary dimensions.
	for (unsigned i = 0; i < shape.dim(); i++)
		if (_view[i] == 1) _stride[i] = 0;

	snprintf(_internals->device, sizeof(_internals->device), "%s", device);

	if (!strcmp(device, "cpu"))
	{
		_internals->is_gpu = false;
		// Allocate the data.
		_internals->_data = ::operator new[](_internals->_data_size, std::align_val_t(64));
		// Be clean and zero it out.
		memset(_internals->_data, 0, _internals->_data_size);
	}
	else if (!strcmp(device, "cuda"))
	{
		_internals->is_gpu = true;
		// Allocate clean data.
		_internals->_data = MemPool::allocate(_internals->_data_size);
	}
	else TENSOR_ERROR(
		"Unknown device string found \"%s\".\n"
		"Supported devices are \"cpu\" and \"cuda\".",
		device
	);

	if (requires_grad)
		_internals->gradient = new Tensor(shape, device, false);
}

// Reduces the instance count by one.

Tensor::~Tensor()
{
	reduce_instances_count();
}

/*
--------------------------------------------------------------------------------------------------------------------------
 User Functions
--------------------------------------------------------------------------------------------------------------------------
*/

float Tensor::item() const
{
	TENSOR_CHECK(_internals,
		"Trying to call item on an empty tensor is not allowed."
	);
	TENSOR_CHECK(numel() == 1,
		"Trying to call item on a tensor with %s elements.\n"
		"Item can only be called on single element tensors.", numel()
	);

	if (is_gpu())
	{
		float val;
		cuda::copy_gpu_to_cpu(&val, _internals->_data, sizeof(float));
		return val;
	}
	else return __data[0];
}

const char* Tensor::str() const
{
	TENSOR_CHECK(_internals,
		"Trying to get the string of an empty tensor is not allowed"
	);

	thread_local static char buffer[8][4096] = {};
	thread_local static int next = 0;

	snprintf(buffer[next % 8], 4096,
		"Shape:    %s\n"
		"Operator: %s\n"
		"Grad:     %s%s\n" 
		"Data:     \n%s",

		
		shape().str(), 
		get_operator(),
		has_grad() ? "\n" : "",
		has_grad() ? gradient().array_str() : "None",
		array_str()
	);

	return buffer[(next++) % 8];
}

const char* Tensor::array_str(const char* fmt) const
{
	constexpr unsigned truncation_size = 7;
	constexpr unsigned truncation_count = 3;

	thread_local static char buffer[8][4096] = {};
	thread_local static int next = 0;

	char* buf = buffer[(next++) % 8];
	int left = 4096;

	auto s_print = [&](const char* fmt, ...)
		{
			va_list ap;
			va_start(ap, fmt);
			int added = vsnprintf(buf, left, fmt, ap);
			va_end(ap);
			buf += added, left -= added;
		};

	Shape counting_shape(_view.dim(), (int*)nullptr);

	while (left > 0)
	{
		for (unsigned d = 0; d < dim(); d++)
		{
			bool opening = true;
			for (unsigned i = d; i < dim(); i++)
				if (counting_shape[i]) opening = false;

			if (opening)
			{
				if (_view[d] <= 1 || d == 0)
					s_print("(");
				else
					s_print("\n(");
			}
		}

		unsigned idx = 0;
		for (unsigned d = 0; d < dim(); d++)
			idx += counting_shape[d] * _stride[d];
		
		if (is_gpu())
		{
			float val;
			cuda::copy_gpu_to_cpu(&val, __data + idx, sizeof(float));
			s_print(fmt, val);
		}
		else
			s_print(fmt, __data[idx]);

		if (++counting_shape[-1] < _view[-1])
		{
			s_print(", ");
			if (_view[-1] > truncation_size && counting_shape[-1] == truncation_count)
			{
				counting_shape[-1] = _view[-1] - truncation_count;
				s_print("..., ");
			}
		}


		for (int d = dim() - 1; d > 0; d--)
		{
			if (counting_shape[d] >= _view[d])
			{
				counting_shape[d] -= _view[d];

				if (++counting_shape[d - 1] < _view[d - 1])
				{
					s_print("), ");
					if (_view[d - 1] > truncation_size && counting_shape[d - 1] == truncation_count)
					{
						counting_shape[d - 1] = _view[d - 1] - truncation_count;

						if (_view[d] <= 1)
							s_print("..., ");
						else
							s_print("\n ... ");
					}
				}
				else
				{
					if (_view[d] <= 1)
						s_print(")");
					else
						s_print(")\n");
				}

			}
		}

		if (counting_shape[0] >= _view[0])
		{
			s_print(")");
			break;
		}
	}

	return buffer[(next - 1) % 8];
}

const char* Tensor::device() const
{
	TENSOR_CHECK(_internals,
		"Trying to get the device on an empty tensor is not allowed."
	);

	// Return internal device string.
	return _internals->device;
}

bool Tensor::has_grad() const
{
	return _internals && !_is_no_grad && _internals->gradient != nullptr;
}

void Tensor::backward()
{
	// Sanity checks.
	TENSOR_CHECK(_internals,
		"Trying to call backwards on an empty tensor."
	);
	TENSOR_CHECK(has_grad(),
		"Trying to call backward on a tensor with no gradient. This would be a no-op.\n"
		"Please make sure all the relevant tensors in your network have gradient."
	);
	for (unsigned i = 0; i < dim(); i++)
		TENSOR_CHECK(size(i) == 1u,
			"Calling backpropagation on a tensor with invalid shape.\n"
			"Backward() may only be called on single element tensors.\n"
			"Found shape: %s.", shape().str()
		);

	// First set your gradient to 1.
	if (_internals->is_gpu)
		cuda::set_to_one(_internals->gradient->_internals->_data);
	else
		((float*)_internals->gradient->_internals->_data)[0] = 1.f;

	// Create the topological graph.
	Tensor** list = nullptr;
	unsigned count = 0u;
	add_to_backward_list(&list, &count);

	// Backprop and reset list.
	for (unsigned i = 0; i < count; i++)
	{
		list[i]->_internals->added_to_backward = false;
		list[i]->_internals->op->_backward();
	}

	// Clean before you leave.
	if (list)
		delete[] list;
}

void Tensor::zero_grad()
{
	// Sanity check.
	if (!has_grad())
		return;

	// Do changes on GPU tensors.
	if (_internals->is_gpu)
		cuda::zero_data(_internals->gradient->_internals->_data, _internals->_data_size);

	// Zero the gradient on the CPU.
	else
		memset(_internals->gradient->_internals->_data, 0, _internals->_data_size);
}

const char* Tensor::get_operator() const
{
	if (_internals && _internals->op)
		return _internals->op->_type;

	return "None";
}

// Returns a copy of the tensor in the specified device. If the tensor had gradient it also 
// copies the gradient. Backpropagation data is lost, so you must not use inside a forward pass.

Tensor Tensor::to(const char* device, bool with_grad) const
{
	// Create output tensor.
	Tensor out(_view, device, with_grad);

	void* ten_data = _internals->_data;
	void* out_data = out._internals->_data;
	unsigned data_size = _internals->_data_size;

	// Distinguish different cases.
	if (out.is_gpu())
	{
		if (is_gpu())	cuda::copy_gpu_to_gpu(out_data, ten_data, data_size);
		else			cuda::copy_cpu_to_gpu(out_data, ten_data, data_size);
	}
	else
	{
		if (is_gpu())	cuda::copy_gpu_to_cpu(out_data, ten_data, data_size);
		else						   memcpy(out_data, ten_data, data_size);
	}

	// If both have gradient copy it too.
	if (has_grad() && with_grad)
	{
		void* ten_grad = _internals->gradient->_internals->_data;
		void* out_grad = out._internals->gradient->_internals->_data;

		if (out.is_gpu())
		{
			if (is_gpu())	cuda::copy_gpu_to_gpu(out_grad, ten_grad, data_size);
			else			cuda::copy_cpu_to_gpu(out_grad, ten_grad, data_size);
		}
		else
		{
			if (is_gpu())	cuda::copy_gpu_to_cpu(out_grad, ten_grad, data_size);
			else						   memcpy(out_grad, ten_grad, data_size);
		}
	}

	// Return tensor.
	return out;
}

const Tensor& Tensor::gradient() const
{
	// Sanity checks.
	TENSOR_CHECK(has_grad(),
		"Trying to get the gradient on an tensor with no gradient is not allowed."
	);

	// Set the gradient view and offsets to your own.
	_internals->gradient->_view = _view;
	_internals->gradient->_stride = _stride;

	// Return reference to its gradient tensor.
	return *_internals->gradient;
}

// Returns a tensor with the same data but that says it has no gradient, this helps to 
// do operations with no gradient with almost no extra overhead.

Tensor Tensor::no_grad() const
{
	if (!has_grad())
		return *this;

	Tensor no_grad_tensor{ *this };
	no_grad_tensor._is_no_grad = true;

	return no_grad_tensor;
}

/*
--------------------------------------------------------------------------------------------------------------------------
 Private Functions
--------------------------------------------------------------------------------------------------------------------------
*/

// When the tensor is transformed or destroyed this method is called to reduce the count 
// of instances to a specific tensor data. Deletes the data if that count is zero.

void Tensor::reduce_instances_count()
{
	// Sanity check.
	if (!_internals) return;

	// Reduce the count.
	_internals->instances--;

	// If no more instances delete everything.
	if (!_internals->instances)
	{
		// If GPU free data on the GPU.
		if (_internals->is_gpu)
			MemPool::free(_internals->_data);

		// If CPU tensor simply delete all memory stored in data.
		else
			::operator delete[](_internals->_data, std::align_val_t(64));

		// Delete operator if exists.
		if (_internals->op)
			delete _internals->op;

		// Delete gradient if exists.
		if (_internals->gradient)
			delete _internals->gradient;

		// Delete the struct itself.
		delete _internals;
	}
}

// Function that to add itself and its relatives to the backward pass.

void Tensor::add_to_backward_list(Tensor*** p_list, unsigned* count)
{
	// Only add to the list if the tensor
	// can backpropagate, this ensures grad too.
	if (!_internals->op || _internals->added_to_backward)
		return;

	// Add relatives to the list first.
	for (Tensor* t : _internals->op->_relatives)
		if (t) t->add_to_backward_list(p_list, count);

	// Add yourself at the front of the list and increase count.
	Tensor** new_list = new Tensor*[*count + 1];
	new_list[0] = this;
	for (unsigned i = 0; i < *count; i++)
		new_list[i + 1] = (*p_list)[i];

	if (*p_list)
		delete[] *p_list;
	*p_list = new_list;
	(*count)++;
	_internals->added_to_backward = true;
}
