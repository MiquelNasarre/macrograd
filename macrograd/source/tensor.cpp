#include "macrograd.h"
#include "macrograd_error.h"
#include "cuda_backend.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <new>

/*
--------------------------------------------------------------------------------------------------------------------------
 Shape functions
--------------------------------------------------------------------------------------------------------------------------
*/

// Pointer initializer. Creates a shape with the given number of dimensions and copies them 
// from the pointer if it is not null. Else the dimensions are zero-initialized.

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

// Copy operator, copies the sizes of the other shape.

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

// Checks if all individual dimensions match.

bool Shape::operator==(const Shape& other) const
{
	if (_dim != other._dim) 
		return false;

	for (unsigned i = 0; i < _dim; i++)
		if (_sizes[i] != other._sizes[i])
			return false;

	return true;
}

// Removes the specified dimension. Supports negative indexing.

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

// Adds a new dimension on the specified spot with the given value.
// Supports negative indexing (modulo dim() + 1).

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

// Returns a string representation of the shape.

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
 VectorInt functions
--------------------------------------------------------------------------------------------------------------------------
*/

// Reduces the instance count by one. If none are left, it deletes the data.

void VectorInt::reduce_instance_count()
{
	if (!_internals) 
		return;

	_internals->_instances--;
	if (!_internals->_instances)
	{
		if (_internals->_is_gpu)MemPool::free(_internals->_data);
		else ::operator delete[](_internals->_data);
		delete _internals;
	}
}

// Constructor, creates a zero-initialized vector with the given length on the specified device.

VectorInt::VectorInt(unsigned length, const char* device)
{
	MACROGRAD_CHECK(length,
		"To initialize an empty VectorInt please use the default constructor."
	);

	_internals = new VecInternals;
	_internals->_instances++;
	_length = length;
	_offset = 0;

	snprintf(_internals->_device, sizeof(_internals->_device), "%s", device);

	if (!strcmp(device, "cpu"))
	{
		_internals->_is_gpu = false;
		_internals->_data = ::operator new[](length * sizeof(int));
		// Be clean and zero it out.
		memset(_internals->_data, 0, length * sizeof(int));
	}
	else if (!strcmp(device, "cuda"))
	{
		_internals->_is_gpu = true;
		_internals->_data = MemPool::allocate(length * sizeof(int));
	}
	else 
		
		(
		"Unknown device string found \"%s\".\n"
		"Supported devices are \"cpu\" and \"cuda\".",
		device
	);
}

// Arange constructor, creates a vector with values in the range [a,b) with the specified stride. 
// The distance between a and b must be divisible by the stride and must have the same sign.

VectorInt::VectorInt(int a, int b, int stride, const char* device)
{
	MACROGRAD_CHECK((b - a > 0 && stride > 0) || (b - a < 0 && stride < 0),
		"To generate an aranged VectorInt the direction a -> b must match the stride sign.\n"
		"Found values | a: %i | b: %i | stride: %i", a, b, stride
	);

	MACROGRAD_CHECK((b - a) % stride == 0,
		"Invalid values found for an aranged VectorInt initialization, stride must divide 'b - a'.\n"
		"Found values | a: %i | b: %i | stride: %i", a, b, stride
	);

	unsigned length = unsigned((b - a) / stride);
	*this = VectorInt(length, device);

	if (is_gpu())
		kernel_ops::arange(_internals->_data, a, stride, length);

	else for (unsigned i = 0; i < length; i++)
		((int*)_internals->_data)[i] = a + i * stride;
}

// Copy operator, gets a view on the other vector's data.

VectorInt& VectorInt::operator=(const VectorInt& other)
{
	if (other._internals)
		other._internals->_instances++;

	reduce_instance_count();

	_length = other._length;
	_internals = other._internals;
	_offset = other._offset;

	return *this;
}

// Returns an element-wise copy of the vector in the specified device.

VectorInt VectorInt::to(const char* device) const
{
	MACROGRAD_CHECK(_internals,
		"'to()' function on an empty VectorInt is not allowed"
	);

	VectorInt out(_length, device);

	if (_internals->_is_gpu)
	{
		if (out._internals->_is_gpu)
			cuda::copy_gpu_to_gpu(out.data(), data(), _length * sizeof(int));
		else
			cuda::copy_gpu_to_cpu(out.data(), data(), _length * sizeof(int));
	}
	else
	{
		if (out._internals->_is_gpu)
			cuda::copy_cpu_to_gpu(out.data(), data(), _length * sizeof(int));
		else
			memcpy(out.data(), data(), _length * sizeof(int));
	}

	return out;
}

// Permutation operator. Returns a new vector containing the data 
// of the current vector, reordered as specified by the indices.

VectorInt VectorInt::operator[](const VectorInt& idxs) const
{
	MACROGRAD_CHECK(idxs.len(),
		"Trying to use VectorInt::operator[] with empty indices is not allowed."
	);
	MACROGRAD_CHECK(len(),
		"Trying to use operator[] on an empty VectorInt is not allowed."
	);
	MACROGRAD_CHECK(idxs.is_gpu() == is_gpu(),
		"Trying to call operator[] with indices and VectorInt in different devices."
	);

	// Create output vector.
	VectorInt out(idxs.len(), device());

	// Get necessary data.
	int* out_data = out.data();
	unsigned length = idxs.len();
	unsigned size0 = len();
	const int* idxs_data = idxs.data();
	const int* vec_data = data();

	// Now we indexate the output vector.
	if (out.is_gpu())
		kernel_ops::vector_bracket_op(out_data, vec_data, idxs_data, length, size0);
	else
	{
		// Iterate through the entire length.
		for (unsigned i = 0; i < length; i++)
		{
			int idx = idxs_data[i];
			MACROGRAD_CHECK(idx >= 0 && (unsigned)idx < size0,
				"Idx out of bounds find during an operator [] call.\n"
				"Make sure your indices are in the range [0, len() - 1]. Idx found: %i", idx
			);
			out_data[i] = vec_data[idx];
		}
	}

	// Return output vector.
	return out;
}

// Access operator, returns a reference to the corresponding element of 
// the vector. Supports negative and circular indexing. These operators 
// are only allowed on CPU vectors, for CUDA use get()/set() instead.

int& VectorInt::operator[](int i)
{
	MACROGRAD_CHECK(_internals,
		"Operator [] on an empty VectorInt is not allowed"
	);
	MACROGRAD_CHECK(!is_gpu(),
		"Operator [] on a GPU VectorInt is not allowed"
	);

	int idx = mod(i, (int)_length);
	return data()[idx];
}

// Access operator, returns a reference to the corresponding element of 
// the vector. Supports negative and circular indexing. These operators 
// are only allowed on CPU vectors, for CUDA use get()/set() instead.

const int& VectorInt::operator[](int i) const
{
	MACROGRAD_CHECK(_internals,
		"Operator [] on an empty VectorInt is not allowed"
	);
	MACROGRAD_CHECK(!is_gpu(),
		"Operator [] on a GPU VectorInt is not allowed"
	);

	int idx = mod(i, (int)_length);
	return data()[idx];
}

// Returns the integer at the i-th position in the vector. 
// Supports negative and circular indexing.

int VectorInt::get(int i) const
{
	MACROGRAD_CHECK(_internals,
		"Get function on an empty VectorInt is not allowed"
	);

	int idx = mod(i, (int)_length);
	int val;
	if (is_gpu()) cuda::copy_gpu_to_cpu(&val, data() + idx, sizeof(int));
	else val = data()[idx];

	return val;
}

// Sets the integer at the i-th position in the vector to the 
// given value. Supports negative and circular indexing.

void VectorInt::set(int i, int val)
{
	MACROGRAD_CHECK(_internals,
		"Set function on an empty VectorInt is not allowed"
	);

	int idx = mod(i, (int)_length);

	if (is_gpu()) cuda::copy_cpu_to_gpu(data() + idx, &val, sizeof(int));
	else data()[idx] = val;
}

// Copies the indices from the pointer to the [a,b) range
// of the vector. This being a total of (b-a) elements.

void VectorInt::set(int a, int b, const int* values, bool is_gpu_ptr)
{
	MACROGRAD_CHECK(_internals,
		"Set function on an empty VectorInt is not allowed"
	);

	int idx_a = mod(a, (int)_length);
	int idx_b = mod(b - 1, (int)_length) + 1;

	MACROGRAD_CHECK(idx_b >= idx_a,
		"Trying to set a list of values on a VectorInt while the indices provided are reversed.\n"
		"Modulo index 'a': %i | Modulo index 'b': %i", idx_a, idx_b
	);

	if (idx_a == idx_b)
		return;

	if (is_gpu())
	{
		if (is_gpu_ptr)
			cuda::copy_gpu_to_gpu(data() + idx_a, values, (idx_b - idx_a) * sizeof(int));
		else
			cuda::copy_cpu_to_gpu(data() + idx_a, values, (idx_b - idx_a) * sizeof(int));
	}
	else
	{
		if (is_gpu_ptr)
			cuda::copy_gpu_to_cpu(data() + idx_a, values, (idx_b - idx_a) * sizeof(int));
		else
			memcpy(data() + idx_a, values, (idx_b - idx_a) * sizeof(int));
	}
		
}

// Returns a view of the current vector with its elements in the range [a,b).

VectorInt VectorInt::subset(int a, int b) const
{
	MACROGRAD_CHECK(_internals,
		"Subset function on an empty VectorInt is not allowed"
	);

	int idx_a = mod(a, (int)_length);
	int idx_b = mod(b - 1, (int)_length) + 1;

	MACROGRAD_CHECK(idx_b >= idx_a,
		"Trying to get a subset of a VectorInt while the indices provided are reversed.\n"
		"Modulo index 'a': %i | Modulo index 'b': %i", idx_a, idx_b
	);
	if (idx_a == idx_b)
		return VectorInt();

	VectorInt out = *this;
	out._length = idx_b - idx_a;
	out._offset += idx_a;

	return out;
}

// Returns an element-wise copy of the vector.

VectorInt VectorInt::copy() const
{
	if (!_internals)
		return VectorInt();

	VectorInt out(_length, _internals->_device);

	if (_internals->_is_gpu)
		cuda::copy_gpu_to_gpu(out.data(), data(), _length * sizeof(int));
	else
		memcpy(out.data(), data(), _length * sizeof(int));

	return out;
}

// Returns a string representation of the vector.

const char* VectorInt::str(const char* fmt) const
{
	constexpr unsigned truncation_size = 16;
	constexpr unsigned truncation_count = 8;

	thread_local static char buffer[16][256] = {};
	thread_local static int next = 0;

	char* buf = buffer[next % 16];
	int left = 256;

	auto s_print = [&](const char* fmt, ...)
	{
		va_list ap;
		va_start(ap, fmt);
		int added = vsnprintf(buf, left, fmt, ap);
		va_end(ap);
		buf += added, left -= added;
	};

	s_print("[");
	for (unsigned i = 0; i < _length; i++)
	{
		if (i == truncation_count && _length > truncation_size)
		{
			i = _length - truncation_count;
			s_print("..., ");
		}
		s_print(fmt, get(i));

		if (i < _length - 1)
			s_print(", ");
	}
	s_print("]");
	return buffer[(next++) % 16];
}

/*
--------------------------------------------------------------------------------------------------------------------------
 Copy Functions
--------------------------------------------------------------------------------------------------------------------------
*/

// Returns a copy of the tensor with the specified configuration.

Tensor Tensor::internal_copy(bool with_grad, bool copy_grad) const
{
	if (!_internals)
		return Tensor();

	Tensor out(_view, device(), with_grad);

	if (is_gpu())
		cuda::copy_gpu_to_gpu(out.internal_data(), internal_data(), data_size());
	else
		memcpy(out.internal_data(), internal_data(), data_size());

	if (with_grad && has_grad() && copy_grad && _internals->gradient)
	{
		if (is_gpu())
			cuda::copy_gpu_to_gpu(out.internal_gradient().internal_data(), gradient().internal_data(), data_size());
		else
			memcpy(out.internal_gradient().internal_data(), gradient().internal_data(), data_size());
	}
	return out;
}

// Returns a copy of the tensor in the specified device. If the tensor had gradient it also 
// copies the gradient. Backpropagation data is lost, so you must not use inside a forward pass.

Tensor Tensor::to(const char* device, bool with_grad) const
{
	// Create output tensor.
	Tensor out(_view, device, with_grad);

	const void* ten_data = internal_data();
	void* out_data = out.internal_data();
	unsigned _data_size = data_size();

	// Distinguish different cases.
	if (out.is_gpu())
	{
		if (is_gpu())	cuda::copy_gpu_to_gpu(out_data, ten_data, _data_size);
		else			cuda::copy_cpu_to_gpu(out_data, ten_data, _data_size);
	}
	else
	{
		if (is_gpu())	cuda::copy_gpu_to_cpu(out_data, ten_data, _data_size);
		else						   memcpy(out_data, ten_data, _data_size);
	}

	// If both have gradient copy it too.
	if (has_grad() && with_grad && _internals->gradient)
	{
		const void* ten_grad = gradient().internal_data();
		void* out_grad = out.internal_gradient().internal_data();

		if (out.is_gpu())
		{
			if (is_gpu())	cuda::copy_gpu_to_gpu(out_grad, ten_grad, _data_size);
			else			cuda::copy_cpu_to_gpu(out_grad, ten_grad, _data_size);
		}
		else
		{
			if (is_gpu())	cuda::copy_gpu_to_cpu(out_grad, ten_grad, _data_size);
			else						   memcpy(out_grad, ten_grad, _data_size);
		}
	}

	// Return tensor.
	return out;
}

/*
--------------------------------------------------------------------------------------------------------------------------
 Constructor Functions
--------------------------------------------------------------------------------------------------------------------------
*/

// Copy constructor, creates a tensor with the same view that 
// shares the pointer to internal data.

Tensor::Tensor(const Tensor& other)
{
	if (other._internals)
	{
		_internals = other._internals;
		_internals->instances++;

		_view = other._view;
		_stride = other._stride;
		_requires_grad = other._requires_grad;
	}
}

// Shape constructor, creates a new tensor with the specified shape 
// on the provided device and with gradient if specified.

Tensor::Tensor(const Shape& shape, const char* device, bool requires_grad)
{
	MACROGRAD_CHECK(shape.dim(),
		"If the tensor created will have zero dimensions please use the default constructor."
	);
	for (unsigned i = 0; i < shape.dim(); i++)
		MACROGRAD_CHECK(shape[i] >= 0,
			"A shape with negative values is not allowed for initialization."
		);

	_internals = new TensorInternals;
	_internals->instances++;

	// Copy view and initialize stride.
	_view = shape;
	_stride = Shape(shape.dim(), (int*)nullptr);
	_requires_grad = requires_grad;

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
	else MACROGRAD_ERROR(
		"Unknown device string found \"%s\".\n"
		"Supported devices are \"cpu\" and \"cuda\".",
		device
	);
}

// Equality operator. Takes the same data pointer as the other tensor and increases the 
// instances by one. If it was initialized it reduces the instances count on the old data.

Tensor& Tensor::operator=(const Tensor& other)
{
	// First increase the others instances just in case.
	if (other._internals)
		other._internals->instances++;

	// Reduce your count.
	reduce_instances_count();

	// Adopt other's data.
	_internals = other._internals;

	// Also copy no-grad status and view.
	_requires_grad = other._requires_grad;
	_view = other._view;
	_stride = other._stride;

	return *this;
}

/*
--------------------------------------------------------------------------------------------------------------------------
 User Functions
--------------------------------------------------------------------------------------------------------------------------
*/

// If the tensor has a single element it returns the value of that element.
// If the tensor is on CUDA this will force a synchronization.

float Tensor::item() const
{
	MACROGRAD_CHECK(_internals,
		"Trying to call item on an empty tensor is not allowed."
	);
	MACROGRAD_CHECK(numel() == 1,
		"Trying to call item on a tensor with %s elements.\n"
		"Item can only be called on single element tensors.", numel()
	);

	if (is_gpu())
	{
		float val;
		cuda::copy_gpu_to_cpu(&val, _internals->_data, sizeof(float));
		return val;
	}
	else return *internal_data();
}

// Returns a string representation of the entire tensor. This includes shape,
// device, operation, gradient and internal data. If the tensor is on CUDA
// this will force a synchronization.

const char* Tensor::str() const
{
	MACROGRAD_CHECK(_internals,
		"Trying to get the string of an empty tensor is not allowed"
	);

	thread_local static char buffer[8][4096] = {};
	thread_local static int next = 0;

	snprintf(buffer[next % 8], 4096,
		"Shape:    %s\n"
		"Device:   %s\n"
		"Operator: %s\n"
		"Grad:     %s%s\n" 
		"Data:     \n%s",

		
		shape().str(), 
		device(),
		get_operator(),
		has_grad() ? "\n" : "",
		has_grad() ? gradient().array_str() : "None",
		array_str()
	);

	return buffer[(next++) % 8];
}

// Returns a string representation of the tensor data, separated by dimensions
// according to the tensor view. If the tensor is on CUDA this will force a 
// synchronization.

const char* Tensor::array_str(const char* fmt) const
{
	constexpr unsigned truncation_size = 7;
	constexpr unsigned truncation_count = 3;

	thread_local static char buffer[8][4096] = {};
	thread_local static int next = 0;

	char* buf = buffer[next % 8];
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
			cuda::copy_gpu_to_cpu(&val, internal_data() + idx, sizeof(float));
			s_print(fmt, val);
		}
		else
			s_print(fmt, internal_data()[idx]);

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

	return buffer[(next++) % 8];
}

// The backward pass can only be called on single element tensors that have gradient. 
// It first sets its gradient to one, then generates the topological graph of the 
// backward pass, and finally calls the internal _backward() functions of each tensor 
// operation in topological order.

void Tensor::backward()
{
	// Sanity checks.
	MACROGRAD_CHECK(_internals,
		"Trying to call backwards on an empty tensor."
	);
	MACROGRAD_CHECK(has_grad(),
		"Trying to call backward on a tensor with no gradient. This would be a no-op.\n"
		"Please make sure all the relevant tensors in your network have gradient."
	);
	for (unsigned i = 0; i < dim(); i++)
		MACROGRAD_CHECK(size(i) == 1u,
			"Calling backpropagation on a tensor with invalid shape.\n"
			"Backward() may only be called on single element tensors.\n"
			"Found shape: %s.", shape().str()
		);

	// First set your gradient to 1.
	if (_internals->is_gpu)
		cuda::set_to_one(internal_gradient().internal_data());
	else
		internal_gradient().internal_data()[0] = 1.f;

	// Create the topological graph.
	Tensor** list = nullptr;
	unsigned count = 0u;
	add_to_backward_list(&list, &count);

	// Backprop and reset list.
	for (unsigned i = 0; i < count; i++)
	{
		list[i]->_internals->already_added = false;
		list[i]->_internals->op->_backward();
	}

	// Clean before you leave.
	if (list)
		delete[] list;
}

// If the tensor has gradient it sets all the gradient values to zero.

void Tensor::zero_grad()
{
	// Sanity check.
	if (!has_grad() || !_internals->gradient)
		return;

	// Do changes on GPU tensors.
	if (_internals->is_gpu)
		cuda::zero_data(internal_gradient().internal_data(), data_size());

	// Zero the gradient on the CPU.
	else
		memset(internal_gradient().internal_data(), 0, data_size());
}

// If it has gradient, returns a constant reference to the gradient tensor.

const Tensor& Tensor::gradient() const
{
	// Sanity checks.
	MACROGRAD_CHECK(has_grad(),
		"Trying to get the gradient on an tensor with no gradient is not allowed."
	);

	// If the data does not have a gradient tensor create it.
	if (!_internals->gradient)
		_internals->gradient = new Tensor(_view, device(), false);

	// Set the gradient view and offsets to your own.
	_internals->gradient->_view = _view;
	_internals->gradient->_stride = _stride;

	// Return reference to its gradient tensor.
	return *_internals->gradient;
}

// If has gradient, returns a reference to the gradient tensor.

Tensor& Tensor::internal_gradient()
{
	// Sanity checks.
	MACROGRAD_CHECK(has_grad(),
		"Trying to get the gradient on an tensor with no gradient is not allowed."
	);

	// If the data does not have a gradient tensor create it.
	if (!_internals->gradient)
		_internals->gradient = new Tensor(_view, device(), false);

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
	MACROGRAD_CHECK(is_init(),
		"Trying to call no_grad() on an uninitialized tensor is not allowed."
	);

	if (!has_grad())
		return *this;

	Tensor no_grad_tensor{ *this };
	no_grad_tensor._requires_grad = false;

	return no_grad_tensor;
}

// Returns a tensor with the same data marked as having gradient, this enables the tensor
// to participate in backpropatation, opposite to what no_grad() does.

Tensor Tensor::with_grad() const
{
	MACROGRAD_CHECK(is_init(),
		"Trying to call with_grad() on an uninitialized tensor is not allowed."
	);

	if (has_grad())
		return *this;

	Tensor with_grad_tensor{ *this };
	with_grad_tensor._requires_grad = true;

	return with_grad_tensor;
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

// Function to add itself and its relatives to the backward pass.

void Tensor::add_to_backward_list(Tensor*** p_list, unsigned* count)
{
	// Only add to the list if the tensor
	// can backpropagate, this ensures grad too.
	if (!_internals->op || _internals->already_added)
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
	_internals->already_added = true;
}
