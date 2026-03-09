#include "macrograd.h"
#include "macrograd_error.h"
#include "cuda_backend.h"

#include <math.h>
#include <limits>

/*
--------------------------------------------------------------------------------------------------------------------------
 Internal Operators.
--------------------------------------------------------------------------------------------------------------------------
*/

Tensor& Tensor::internal_gradient()
{
	// Sanity checks.
	MACROGRAD_CHECK(has_grad(),
		"Trying to get the gradient on an tensor with no gradient is not allowed."
	);

	// Set the gradient view and offsets to your own.
	_internals->gradient->_view = _view;
	_internals->gradient->_stride = _stride;

	// Return reference to its gradient tensor.
	return *_internals->gradient;
}

Tensor Tensor::internal_copy(bool with_grad, bool copy_grad) const
{
	if (!_internals)
		return Tensor();

	Tensor out(_view, device(), false);

	if (is_gpu())
		cuda::copy_gpu_to_gpu(out._internals->_data, _internals->_data, _internals->_data_size);
	else
		memcpy(out._internals->_data, _internals->_data, _internals->_data_size);

	if (with_grad)
	{
		out._internals->gradient = new Tensor(_view, device(), false);
		if (has_grad() && copy_grad)
		{
			if (is_gpu())
				cuda::copy_gpu_to_gpu(out._internals->gradient->_internals->_data, _internals->gradient->_internals->_data, _internals->_data_size);

			else
				memcpy(out._internals->gradient->_internals->_data, _internals->gradient->_internals->_data, _internals->_data_size);
		}
	}

	return out;
}

// Returns a tensor containing the leading dimensions with the indices specified. Does not have grad.

Tensor Tensor::operator[](const VectorInt& idxs) const
{
	MACROGRAD_CHECK(idxs.len(),
		"Trying to use Tensor::operator[] with empty indices is not allowed."
	);
	MACROGRAD_CHECK(is_init(),
		"Trying to use operator[] on an empty tensor is not allowed."
	);
	MACROGRAD_CHECK(idxs.is_gpu() == is_gpu(),
		"Trying to call operator[] with indices and tensor in different devices."
	);
	MACROGRAD_CHECK(!has_grad(),
		"Tensor::operator[] being called from a tensor that has gradient.\n"
		"Backpropagation is not implemented for this operator, consider using subset() instead.\n"
		"If you still want to use [] please call no_grad() right before to avoid this message."
	);

	// Create shape with leading dimension changed.
	Shape out_shape = _view;
	out_shape[0] = idxs.len();

	// Initialize output tensor.
	Tensor out(out_shape, device(), false);

	// Extract relevant data.
	const int* idxs_data = idxs.data();
	unsigned length = idxs.len();
	unsigned size0 = _view[0];
	unsigned stride = _stride[0];
	float* out_data = out.internal_data();
	const float* ten_data = internal_data();

	// Now let's indexate this tensor.
	if (out.is_gpu())
		kernel_ops::tensor_bracket_op(out_data, ten_data, idxs_data, length, stride, size0);
	else
	{
		// Iterate through the indices and memcpy.
		for (unsigned i = 0; i < length; i++)
		{
			int idx = idxs_data[i];
			MACROGRAD_CHECK(idx >= 0 && (unsigned)idx < size0,
				"Idx out of bounds find during an operator [] call.\n"
				"Make sure your indices are in the range [0, size(0) - 1]. Idx found: %i", idx
			);
			memcpy(out_data + stride * i, ten_data + stride * idx, stride * sizeof(float));
		}
	}
	// Return the tensor.
	return out;
}

void Tensor::internal_add(const float* val, bool gpu, float factor)
{
	MACROGRAD_CHECK(is_init(),
		"Trying to internally add to an empty tensor."
	);
	MACROGRAD_CHECK(!gpu || is_gpu(),
		"Trying to internally add a float stored in CUDA to a CPU tensor."
	);

	float* data = internal_data();
	const unsigned numel = this->numel();

	if (is_gpu())
		kernel_ops::add_scalar(data, data, val, numel, gpu, factor);

	else
	{
		float scalar = *val * factor;
		for (unsigned idx = 0; idx < numel; idx++)
			data[idx] += scalar;
	}
}

void Tensor::internal_multiply(const float* val, bool gpu, float factor)
{
	MACROGRAD_CHECK(is_init(),
		"Trying to internally multiply to an empty tensor."
	);
	MACROGRAD_CHECK(!gpu || is_gpu(),
		"Trying to internally multiply a float stored in CUDA to a CPU tensor."
	);

	float* data = internal_data();
	const unsigned numel = this->numel();

	if (is_gpu())
		kernel_ops::multiply_scalar(data, data, val, numel, gpu, factor);

	else
	{
		float scalar = *val * factor;
		for (unsigned idx = 0; idx < numel; idx++)
			data[idx] *= scalar;
	}
}

void Tensor::internal_set(const float* val, bool gpu, float factor)
{
	MACROGRAD_CHECK(is_init(),
		"Trying to internally set an empty tensor."
	);
	MACROGRAD_CHECK(!gpu || is_gpu(),
		"Trying to internally set a float stored in CUDA to a CPU tensor."
	);

	float* data = internal_data();
	const unsigned numel = this->numel();

	if (is_gpu())
		kernel_ops::set_scalar(data, val, numel, gpu, factor);

	else
	{
		float scalar = *val * factor;
		for (unsigned idx = 0; idx < numel; idx++)
			data[idx] = scalar;
	}
}

void Tensor::internal_add(const Tensor& other)
{
	MACROGRAD_CHECK(is_init(),
		"Trying to internally add to an empty tensor."
	);
	MACROGRAD_CHECK(other.is_init(),
		"Trying to internally add an empty tensor."
	);
	MACROGRAD_CHECK(this->numel() == other.numel(),
		"Trying to internally add a tensor with a different number of elements."
	);
	MACROGRAD_CHECK(is_gpu() == other.is_gpu(),
		"Trying to internally add two tensors on different devices."
	);

	float* data = internal_data();
	const float* other_data = other.internal_data();
	const int numel = this->numel();

	if (is_gpu())
		kernel_ops::add_tensor(data, data, other_data, numel);

	else
	{
		int idx = -1;
		while (++idx < numel)
			data[idx] += other_data[idx];
	}

}

void Tensor::internal_add_prod(float val, const Tensor& other)
{
	MACROGRAD_CHECK(is_init(),
		"Trying to internally add to an empty tensor."
	);
	MACROGRAD_CHECK(other.is_init(),
		"Trying to internally add an empty tensor."
	);
	MACROGRAD_CHECK(this->numel() == other.numel(),
		"Trying to internally add a tensor with a different number of elements."
	);
	MACROGRAD_CHECK(is_gpu() == other.is_gpu(),
		"Trying to internally add two tensors on different devices."
	);

	float* data = internal_data();
	const float* other_data = other.internal_data();
	const int numel = this->numel();

	if (is_gpu())
		kernel_ops::add_multiply_scalar_tensor(data, val, other_data, numel);
	else
	{
		int idx = -1;
		while (++idx < numel)
			data[idx] += val * other_data[idx];
	}
}

void Tensor::internal_subtract(const Tensor& other)
{
	MACROGRAD_CHECK(is_init(),
		"Trying to internally subtract to an empty tensor."
	);
	MACROGRAD_CHECK(other.is_init(),
		"Trying to internally subtract an empty tensor."
	);
	MACROGRAD_CHECK(this->numel() == other.numel(),
		"Trying to internally subtract a tensor with a different number of elements."
	);
	MACROGRAD_CHECK(is_gpu() == other.is_gpu(),
		"Trying to internally subtract two tensors on different devices."
	);

	float* data = internal_data();
	const float* other_data = other.internal_data();
	const int numel = this->numel();

	if (is_gpu())
		kernel_ops::subtract_tensor(data, data, other_data, numel);
	else
	{
		int idx = -1;
		while (++idx < numel)
			data[idx] -= other_data[idx];
	}
}

void Tensor::internal_multiply(const Tensor& other)
{
	MACROGRAD_CHECK(is_init(),
		"Trying to internally multiply an empty tensor."
	);
	MACROGRAD_CHECK(other.is_init(),
		"Trying to internally multiply by an empty tensor."
	);
	MACROGRAD_CHECK(this->numel() == other.numel(),
		"Trying to internally multiply a tensor with a different number of elements."
	);
	MACROGRAD_CHECK(is_gpu() == other.is_gpu(),
		"Trying to internally multiply two tensors on different devices."
	);

	float* data = internal_data();
	const float* other_data = other.internal_data();
	const int numel = this->numel();

	if (is_gpu())
		kernel_ops::multiply_tensor(data, data, other_data, numel);
	else
	{
		int idx = -1;
		while (++idx < numel)
			data[idx] *= other_data[idx];
	}
}

void Tensor::internal_set_value(const Shape& route, float value)
{
	MACROGRAD_CHECK(is_init(),
		"Trying to set a value on an empty tensor."
	);
	MACROGRAD_CHECK(numel(),
		"Trying to set a value on a tensor with no values.\n"
		"The tensor shape is %s", _view.str()
	);

	float* ptr = internal_data();
	for (unsigned d = 0; d < dim(); d++)
		ptr += ((route[d] + _view[d] * (2 - route[d] / _view[d])) % _view[d]) * _stride[d];

	if (is_gpu())
		cuda::copy_cpu_to_gpu(ptr, &value, sizeof(float));

	else
		*ptr = value;
}

float Tensor::internal_get_value(const Shape& route) const
{
	MACROGRAD_CHECK(is_init(),
		"Trying to get a value on an empty array."
	);
	MACROGRAD_CHECK(numel(),
		"Trying to get a value on an array with no values.\n"
		"The array shape is %s", _view.str()
	);

	// Get data pointer.
	const float* ptr = internal_data();

	// Compute pointer rout.
	for (unsigned d = 0; d < dim(); d++)
		ptr += ((route[d] + _view[d] * (2 - route[d] / _view[d])) % _view[d]) * _stride[d];

	if (is_gpu())
	{
		float val;
		cuda::copy_gpu_to_cpu(&val, ptr, sizeof(float));
		return val;
	}
	else
		return *ptr;
}

void Tensor::internal_set_vector(const Shape& route, const float* values)
{
	MACROGRAD_CHECK(is_init(),
		"Trying to set a vector on an empty tensor."
	);
	MACROGRAD_CHECK(numel(),
		"Trying to set a vector on a tensor with no values.\n"
		"The tensor shape is %s", _view.str()
	);

	// Get idx given route. Modulo for negative numbers.
	unsigned idx = 0;
	for (unsigned i = 0; i < route.dim(); i++)
		idx += ((route[i] + _view[i] * (2 - route[i] / _view[i])) % _view[i]) * _stride[i];

	// Get full data expected size.
	unsigned _data_size = sizeof(float);
	for (unsigned i = route.dim(); i < _view.dim(); i++)
		_data_size *= _view[i];

	if (is_gpu())
		cuda::copy_cpu_to_gpu(internal_data() + idx, values, _data_size);

	else
		memcpy(internal_data() + idx, values, _data_size);
}

float* Tensor::internal_get_vector(const Shape& route)
{
	MACROGRAD_CHECK(is_init(),
		"Trying to get a vector of an empty tensor."
	);
	MACROGRAD_CHECK(numel(),
		"Trying to get a vector of a tensor with no values.\n"
		"The tensor shape is %s", _view.str()
	);

	// Get idx given route. Modulo for negative numbers.
	unsigned idx = 0;
	for (unsigned i = 0; i < route.dim(); i++)
		idx += ((route[i] + _view[i] * (2 - route[i] / _view[i])) % _view[i]) * _stride[i];

	// Return data ptr.
	return internal_data() + idx;
}

/*
--------------------------------------------------------------------------------------------------------------------------
 No Allocation Operators
--------------------------------------------------------------------------------------------------------------------------
*/

// Equality operator. Takes the same data pointer as the other tensor and increases the instances by one. 
// If it was holding data, after takeing the new pointer, it decreases the count on the old one.

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
	_is_no_grad = other._is_no_grad;
	_view = other._view;
	_stride = other._stride;

	return *this;
}

// Creates a new tensor with the same data but different view.

Tensor Tensor::view(const Shape& shape) const
{
	MACROGRAD_CHECK(is_init(),
		"Trying to call view on an empty tensor."
	);
	MACROGRAD_CHECK(shape.dim(),
		"Trying to change the view of a tensor with an empty shape."
	);

	Shape new_shape = shape;

	// Deal with formatted shapes with -1 sizes.
	int neg_one = -1;
	unsigned new_total_dim = 1;
	for (unsigned i = 0; i < new_shape.dim(); i++)
	{
		if (new_shape[i] == -1)
		{
			MACROGRAD_CHECK(neg_one == -1,
				"Ambiguous shape found inside a view call.\n"
				"Make sure you only have one unknown dimension marked as -1 to avoid ambiguity.\n"
				"Old Shape: %s | View Input Shape: %s.", _view.str(), shape.str()
			);
			neg_one = i;
		}
		else new_total_dim *= new_shape[i];
	}
	if (neg_one != -1)
	{
		MACROGRAD_CHECK(new_total_dim != 0,
			"Ambiguous shape find inside a view call.\n"
			"It is not allowed to have an unknown dimension -1 while there is a size 0.\n"
			"Old Shape: %s | View Input Shape: %s.", _view.str(), shape.str()
		);
		MACROGRAD_CHECK(numel() % new_total_dim == 0,
			"Unreconcileable shapes found inside a view call, total sizes are not divisible.\n"
			"Old Shape: %s | View Input Shape: %s.", _view.str(), shape.str()
		);
		new_shape[neg_one] = numel() / new_total_dim;
	}
	else if (new_total_dim != numel())
		
		
		("Incompatible shapes found during a view call. Make sure the total dimensionality matches.\n"
			"Old Shape: %s | View Input Shape: %s.", _view.str(), shape.str()
		);

	// Create new stride shape.
	Shape new_stride(new_shape.dim(), (int*)nullptr);
	new_stride[-1] = 1;
	for (int i = new_shape.dim() - 2; i >= 0; i--)
		new_stride[i] = new_shape[i + 1] * new_stride[i + 1];
	
	// Create output with new shape and stride.
	Tensor out = *this;
	out._view = new_shape;
	out._stride = new_stride;

	return out;
}

// Returns a tensor with the same data reduced to a single vector.

Tensor Tensor::flatten() const
{
	// Tensor must be initialized.
	
	(is_init(),
		"Trying to flatten an empty tensor."
	);
	
	// Create output tensor with flat view.
	Tensor out = *this;
	out._view = Shape(numel());
	out._stride = Shape(1);

	// Return out.
	return out;
}

// Returns a tensor with the specified dimension removed, must be unitary.

Tensor Tensor::squeeze(int dim) const
{
	MACROGRAD_CHECK(is_init(),
		"Trying to call squeeze on an empty tensor."
	);
	MACROGRAD_CHECK(_view.dim() > 1,
		"Trying to squeeze a tensor with only one dimension left is not allowed."
	);
	MACROGRAD_CHECK(_view[dim] == 1,
		"Trying to squeeze a non unitary dimension on a tensor.\n"
		"Please make sure you call squeeze on dimensions of size 1."
	);

	Tensor copied = *this;
	copied._view.remove(dim);
	copied._stride.remove(dim);

	return copied;
}

// Returns a tensor with an added dimension 1 in the specified spot.

Tensor Tensor::unsqueeze(int dim) const
{
	MACROGRAD_CHECK(is_init(),
		"Trying to call unsqueeze on an empty tensor."
	);

	// Modulo dim with an extra dim considered.
	dim = ((dim % (_view.dim() + 1)) + (_view.dim() + 1)) % (_view.dim() + 1);

	Tensor copied = *this;
	copied._view.add(dim, 1);
	copied._stride.add(dim, (dim < int(_stride.dim())) ? _stride[dim] : 1);

	return copied;
}

/*
--------------------------------------------------------------------------------------------------------------------------
 Shape Operators
--------------------------------------------------------------------------------------------------------------------------
*/

// Returns a tensor with the specified dimensions transposed.

Tensor Tensor::transpose(int dim0, int dim1) const
{
	// Transposition tensor operator for backpropagation.
	class TransOp : public Tensor::TensorInternals::TensorOp
	{
		// Tensor copy storage for the input.
		Tensor in;

		// Storage for transposed dimensions.
		int dim0, dim1;

	public:
		// Constructor, stores all the data of the operation.
		TransOp(const Tensor& _in, const Tensor& _out, int _dim0, int _dim1) : TensorOp{ "Transposition", _out },
			in{ _in }, dim0{ _dim0 }, dim1{ _dim1 }
		{
			_relatives[0] = &in;
		}

		// Backpropagation. Transpose back before adding.
		void _backward() override
		{
			in.internal_gradient() += out.gradient().transpose(dim0, dim1);
		}
	};

	// Tensor must be initialized.
	MACROGRAD_CHECK(is_init(),
		"Trying to transpose an empty tensor."
	);

	// Modulo dimensions.
	dim0 = unsigned(dim0 + dim() * (2 - dim0 / int(dim()))) % dim();
	dim1 = unsigned(dim1 + dim() * (2 - dim1 / int(dim()))) % dim();

	// If its the same dimension just return.
	if (dim0 == dim1)
		return *this;

	// Find out which dimension is the first one and set it to dim0.
	if (dim0 > dim1)
	{
		int temp = dim0;
		dim0 = dim1;
		dim1 = temp;
	}

	// Create output shape.
	Shape out_shape = _view;
	out_shape[dim0] = _view[dim1];
	out_shape[dim1] = _view[dim0];

	// Create output with the transposed shape.
	Tensor out(out_shape, device(), has_grad());

	// Extract the data.
	float* out_data = out.internal_data();
	const float* ten_data = internal_data();
	// Store the length of the dimensions.
	const unsigned A = _view[dim0];
	const unsigned B = _view[dim1];
	// Tensor sizes data.
	const unsigned inner_size = _stride[dim1];
	const unsigned middle_size = _stride[dim0] / (B * inner_size);
	const unsigned outter_size = numel() / (A * _stride[dim0]);
	// Tensor strides data.
	const unsigned out_stride = out._stride[dim0];
	const unsigned ten_stride = _stride[dim0];

	const unsigned outter_stride = A * ten_stride;


	// Now we actually transpose the tensor.
	if (out.is_gpu())
		kernel_ops::transpose(out_data, ten_data, A, B, outter_size, middle_size, inner_size, ten_stride, out_stride);
	else
	{
		for (unsigned outter = 0; outter < outter_size; outter++)
		for (unsigned middle = 0; middle < middle_size; middle++)
		for (unsigned inner  = 0; inner  <  inner_size;  inner++)
		{
			unsigned ten_idx = outter * outter_stride + middle * inner_size * B + inner;
			unsigned out_idx = outter * outter_stride + middle * inner_size * A + inner;

			for (unsigned i = 0; i < A; i++)
			{
				unsigned running_ten_idx = ten_idx;
				unsigned running_out_idx = out_idx;

				for (unsigned j = 0; j < B; j++)
				{
					out_data[running_out_idx] = ten_data[running_ten_idx];
					running_ten_idx += inner_size;
					running_out_idx += out_stride;
				}
				ten_idx += ten_stride;
				out_idx += inner_size;
			}
		}
	}

	// If it was a gradient operation store a TransOp instance.
	if (out.has_grad())
		out._internals->op = new TransOp(*this, out, dim0, dim1);

	// Return out.
	return out;
}

// Returns a subset of the tensor with the specified shape starting from the specified indices.

Tensor Tensor::subset(const Shape& shape, const Shape& start_indices) const
{
	// Subset tensor operator for backpropagation.
	class SubsetOp : public Tensor::TensorInternals::TensorOp
	{
		// Tensor copy storage for the input.
		Tensor in;

		// Storage for subset data.
		Shape start_indices;

	public:
		// Constructor, stores all the data of the operation.
		SubsetOp(const Tensor& _in, const Tensor& _out, const Shape& _start_indices) : TensorOp{ "Subset", _out },
			in{ _in }, start_indices{ _start_indices }
		{
			_relatives[0] = &in;
		}

		// Backpropagation. Modify only subset region.
		void _backward() override
		{
			in.internal_gradient() += Tensor(in.shape(), in.device(), false).modify(out.gradient(), start_indices);
		}
	};

	// Tensor must be initialized.
	MACROGRAD_CHECK(is_init(),
		"Trying to subset an empty tensor."
	);

	// Shapes for processing
	Shape new_shape = shape;
	Shape start = start_indices;

	// Modulo indices.
	for (unsigned i = 0; i < start.dim(); i++)
		start[i] = unsigned(start_indices[i] + _view[i] * (2 - start_indices[i] / int(_view[i]))) % _view[i];

	// If missing assume starts at 0.
	while (start.dim() < dim())
		start.add(-1, 0);

	// If -1 assume full size.
	for (unsigned i = 0; i < new_shape.dim(); i++)
		if (new_shape[i] == -1)
			new_shape[i] = _view[i];

	// If missing dimensions assume full size.
	while (new_shape.dim() < dim())
		new_shape.add(-1, _view[new_shape.dim()]);

	// Sanity checks.
	MACROGRAD_CHECK(new_shape.dim() == dim(),
		"Trying to call subset with a shape of different dimensionality.\n"
		"Make sure you don't introduce a shape larger than the original.\n"
		"Tensor Shape: %s | Subset Shape: %s", _view.str(), shape.str()
	);
	MACROGRAD_CHECK(start.dim() == dim(),
		"Trying to call subset with a start_indices shape of different dimensionality.\n"
		"Make sure you don't introduce an indices shape larger than the original.\n"
		"Tensor Shape: %s | Start Indices: %s", _view.str(), start_indices.str()
	);
	for (unsigned i = 0; i < dim(); i++)
		MACROGRAD_CHECK(new_shape[i] >= 0,
			"Negative dimensions in a subset shape call.\n"
			"Please make sure all dimensions are positive or -1 (full size) to avoid ambiguity.\n"
			"Tensor Shape: %s | Subset Shape: %s", _view.str(), shape.str()
		);

	for (unsigned i = 0; i < dim(); i++)
		MACROGRAD_CHECK(new_shape[i] + start[i] <= _view[i],
			"Out of bounds dimension for a subset call.\n"
			"Start indices are not compatible with subset and input shape.\n"
			"Tensor Shape: %s | Processed Subset Shape: %s | Processed Start Indices: %s", _view.str(), new_shape.str(), start.str()
		);

	// If you subset the full thing just return yourself.
	if (new_shape == _view)
		return *this;

	// Create output with the subset shape.
	Tensor out(new_shape, device(), has_grad());

	// Extract the data.
	float* out_data = out.internal_data();
	const float* ten_data = internal_data();

	// Now we actually subset the tensor.
	if (out.is_gpu())
		kernel_ops::subset(new_shape, _view, start, out_data, ten_data);
	else
	{
		// Find the last non-unitary dimension to iterate through.
		unsigned last_long_dim = 0;
		for (unsigned i = 0; i < new_shape.dim(); i++)
			if (new_shape[i] > 1)
				last_long_dim = i;
		// Store the length of the longest dim
		const unsigned vector_len = new_shape[last_long_dim];
		// Create a running shape to count.
		Shape counting_shape(dim(), (int*)nullptr);
		// Create a refenrece shape with the long dimension removed.
		Shape reference = new_shape;
		reference[last_long_dim] = 1;
		// Get the stride for both tensors in the long dimension.
		unsigned ten_stride = _stride[last_long_dim];
		unsigned out_stride = out._stride[last_long_dim];

		// Iterate through vectors.
		while (true)
		{
			unsigned out_idx = 0, ten_idx = 0;
			for (unsigned d = 0; d < counting_shape.dim(); d++)
			{
				out_idx += counting_shape[d] * out._stride[d];
				ten_idx += (counting_shape[d] + start[d]) * _stride[d];
			}

			for (unsigned count = 0u; count < vector_len; count++)
			{
				out_data[out_idx] = ten_data[ten_idx];
				out_idx += out_stride, ten_idx += ten_stride;
			}

			counting_shape[-1]++;
			for (int d = dim() - 1; d > 0; d--)
				if (counting_shape[d] >= reference[d])
				{
					counting_shape[d] -= reference[d];
					counting_shape[d - 1]++;
				}

			// If you reach the end of the leading dimension you're done.
			if (counting_shape[0] >= reference[0])
				break;
		}
	}

	// If it was a gradient operation store a SubsetOp instance.
	if (out.has_grad())
		out._internals->op = new SubsetOp(*this, out, start);

	// Return out.
	return out;
}

// Returns a tensor with the same shape but with a subset substituted by the specified tensor.

Tensor Tensor::modify(const Tensor& other, const Shape& start_indices) const
{
	// Modified tensor operator for backpropagation.
	class ModiOp : public Tensor::TensorInternals::TensorOp
	{
		// Tensor copy storage for the input, modifier.
		Tensor in, mod;

		// Storage for subset data.
		Shape start_indices;

	public:
		// Constructor, stores all the data of the operation.
		ModiOp(const Tensor& _in, const Tensor& _mod, const Tensor& _out, const Shape& _start_indices) : TensorOp{ "Modify", _out },
			in{ _in }, mod{ _mod }, start_indices{ _start_indices }
		{
			if(in.has_grad()) _relatives[0] = &in;
			if(mod.has_grad()) _relatives[1] = &mod;
		}

		// Backpropagation. Route the gradient to the correct subset regions.
		void _backward() override
		{
			// Substitute region by an empty tensor.
			if (in.has_grad()) in.internal_gradient() += out.gradient().modify(Tensor(mod.shape(), in.device()), start_indices);
			// Add the gradient subset region.
			if (mod.has_grad()) mod.internal_gradient() += out.gradient().subset(mod.shape(), start_indices);
		}
	};

	// Tensor must be initialized.
	MACROGRAD_CHECK(is_init(),
		"Trying to modify an empty tensor."
	);
	MACROGRAD_CHECK(other.is_init(),
		"Trying to modify with an empty other tensor."
	);
	// Both tensors must be on the same device
	MACROGRAD_CHECK(is_gpu() == other.is_gpu(),
		"Trying to add madify a tensor with another tensor on a different device."
	);

	// Get new variables to adjust shapes.
	Shape shape = other._view;
	Shape stride = other._stride;
	Shape out_shape = _view;
	Shape out_stride = _stride;
	Shape start = start_indices;

	// Modulo indices.
	for (unsigned i = 0; i < start.dim(); i++)
		start[i] = ((start_indices[i] % _view[i]) + _view[i]) % _view[i];

	// If missing assume starts at 0.
	while (start.dim() < dim())
		start.add(-1, 0);

	// If missing dimensions assume single element from prev dimensions.
	while (shape.dim() < dim())
	{
		shape.add(0, 1);
		stride.add(0, stride[0]);
	}

	MACROGRAD_CHECK(shape.dim() == dim(),
		"Trying to call modify with modification tensor of different dimensionality.\n"
		"Make sure the number of dimensions is equal or smaller to the original tensor.\n"
		"Tensor Shape: %s | Modifier Shape: %s", _view.str(), other._view.str()
	);
	MACROGRAD_CHECK(start.dim() == dim(),
		"Trying to call subset with a start_indices shape of different dimensionality.\n"
		"Make sure the number of idx dimensions is equal or smaller to the original tensor.\n"
		"Tensor Shape: %s | Start Indices: %s", _view.str(), start_indices.str()
	);
	for (unsigned i = 0; i < dim(); i++)
		MACROGRAD_CHECK(shape[i] + start[i] <= _view[i],
			"Out of bounds dimension for a modify call.\n"
			"Start indices are not compatible with modifier and input shape.\n"
			"Tensor Shape: %s | Processed Modifier Shape: %s | Processed Start Indices: %s", _view.str(), shape.str(), start.str()
		);

	// If youre substituting the whole thin just return other.
	if (shape == out_shape)
		return other;

	// Get gradient data.
	bool requires_grad = has_grad() || other.has_grad();

	// Create output same as input.
	Tensor out = internal_copy(requires_grad, false);

	// Extract the data.
	float* out_data = out.internal_data();
	const float* mod_data = other.internal_data();

	// Discard unitary dimensions and move out pointer forward.
	while (shape[0] == 1 && shape.dim() > 1)
	{
		out_data += out_stride[0] * start[0];
		shape.remove(0);
		stride.remove(0);
		out_shape.remove(0);
		out_stride.remove(0);
		start.remove(0);
	}

	// Now we actually modify the tensor.
	if (out.is_gpu())
		kernel_ops::modify(out_shape, shape, out_stride, start, out_data, mod_data);
	else
	{
		// Find the last non-unitary dimension to iterate through.
		unsigned last_long_dim = 0;
		for (unsigned i = 0; i < shape.dim(); i++)
			if (shape[i] > 1)
				last_long_dim = i;
		// Store the length of the longest dim
		const unsigned vector_len = shape[last_long_dim];
		// Create a running shape to count.
		Shape counting_shape(shape.dim(), (int*)nullptr);
		// Create a refenrece shape with the long dimension removed.
		Shape reference = shape;
		reference[last_long_dim] = 1;
		// Get the stride for both tensors in the long dimension.
		unsigned mod_str = stride[last_long_dim];
		unsigned out_str = out_stride[last_long_dim];

		// Iterate through vectors.
		while (true)
		{
			unsigned out_idx = 0, mod_idx = 0;
			for (unsigned d = 0; d < shape.dim(); d++)
			{
				out_idx += (counting_shape[d] + start[d]) * out_stride[d];
				mod_idx += counting_shape[d] * stride[d];
			}

			for(unsigned count = 0u; count< vector_len; count++)
			{
				out_data[out_idx] = mod_data[mod_idx];
				out_idx += out_str, mod_idx += mod_str;
			}

			counting_shape[-1]++;
			for (int d = shape.dim() - 1; d > 0; d--)
				if (counting_shape[d] >= reference[d])
				{
					counting_shape[d] -= reference[d];
					counting_shape[d - 1]++;
				}

			// If you reach the end of the leading dimension you're done.
			if (counting_shape[0] >= reference[0])
				break;
		}
	}

	// If it was a gradient operation store a ModiOp instance.
	if (requires_grad)
		out._internals->op = new ModiOp(*this, other, out, start);

	// Return out.
	return out;
}

// Returns a tensor with repeated dimensions of out_shape = shape * repetitions.

Tensor Tensor::repeat(int dim, unsigned repetitions) const
{
	// Repeat tensor operator for backpropagation.
	class RepOp : public Tensor::TensorInternals::TensorOp
	{
		// Tensor copy storage for the input.
		Tensor in;

		// Dimension that was repeated.
		int dim;

	public:
		// Constructor, stores all the data of the operation.
		RepOp(const Tensor& _in, const Tensor& _out, int _dim) : TensorOp{ "Repeat", _out },
			in{ _in }, dim{ _dim }
		{
			_relatives[0] = &in;
		}

		// Backpropagation. Sum dimension back before adding.
		void _backward() override
		{
			in.internal_gradient() += out.gradient().sum(dim, true);
		}
	};

	// Tensor must be initialized.
	MACROGRAD_CHECK(is_init(),
		"Trying to add repetitions to an empty tensor."
	);
	// Repeated shape must be unitary.
	MACROGRAD_CHECK(_view[dim] == 1,
		"Trying to repeat a tensor on a non-unitary dimension.\n"
		"Make sure the dimension you are repeating is of size 1."
	);

	// If you're not repeating don't repeat
	if (repetitions == 1) 
		return *this;

	// Modulo dimension.
	dim = unsigned(dim + _view.dim() * (2 - dim / int(_view.dim()))) % _view.dim();

	// Create output with the repeated dimension.
	Shape out_shape = _view;
	out_shape[dim] = repetitions;
	Tensor out(out_shape, device(), has_grad());

	// Extract the data.
	float* out_data = out.internal_data();
	const float* ten_data = internal_data();
	// Get relevant sizes.
	unsigned outer_size = numel() / _stride[dim];
	unsigned inner_size = _stride[dim];
	// Get relevant strides.
	const unsigned outer_stride = inner_size * repetitions;

	// Now we actually repeat the tensor.
	if (out.is_gpu())
		kernel_ops::repeat(out_data, ten_data, outer_size, inner_size, repetitions);
	else
	{
		// Iterate through all elements to repeat them.
		for (unsigned outer = 0; outer < outer_size; outer++)
		for (unsigned inner = 0; inner < inner_size; inner++)
		{
			unsigned ten_idx = outer * inner_size + inner;
			unsigned out_idx = outer * outer_stride + inner;

			for (unsigned count = 0; count < repetitions; count++)
			{
				out_data[out_idx] = ten_data[ten_idx];
				out_idx += inner_size;
			}
		}
	}

	// If it was a gradient operation store a RepOp instance.
	if (has_grad())
		out._internals->op = new RepOp(*this, out, dim);

	// Return out.
	return out;
}

/*
--------------------------------------------------------------------------------------------------------------------------
 Function Operators
--------------------------------------------------------------------------------------------------------------------------
*/

Tensor Tensor::sign() const
{
	// Tensor must be initialized.
	MACROGRAD_CHECK(is_init(),
		"Trying to get the sign of an empty tensor."
	);

	// Create output with the same shape as tensor, no gradient.
	Tensor out(shape(), device(), false);

	// Extract the data.
	float* out_data = out.internal_data();
	const float* ten_data = internal_data();

	// Get the element count.
	unsigned _numel = out.numel();

	// Now we actually add signs to the tensor.
	if (out.is_gpu())
		kernel_ops::sign(out_data, ten_data, _numel);

	else for (unsigned idx = 0; idx < _numel; idx++)
		out_data[idx] = (ten_data[idx] > 0.f) ? 1.f : (ten_data[idx] < 0.f) ? -1.f : 0.f;

	// Return out.
	return out;
}

Tensor Tensor::exp() const
{
	// Exponential tensor operator for backpropagation.
	class ExpOp : public Tensor::TensorInternals::TensorOp
	{
		// Tensor copy storage for the base.
		Tensor base;

	public:
		// Constructor, stores all the data of the operation.
		ExpOp(const Tensor& _base, const Tensor& _out) : TensorOp{ "Exponential", _out },
			base{ _base }
		{
			_relatives[0] = &base;
		}

		// Backpropagation. For exponentiation the derivative is the output.
		void _backward() override
		{
			base.internal_gradient() += out.gradient() * out.no_grad();
		}
	};

	// Tensor must be initialized.
	MACROGRAD_CHECK(is_init(),
		"Trying to exponentiate an empty tensor."
	);

	// Create output with the same shape as tensor.
	Tensor out(shape(), device(), has_grad());

	// Extract the data.
	float* out_data = out.internal_data();
	const float* ten_data = internal_data();

	// Get the element count.
	unsigned _numel = out.numel();

	// Now we actually exponentiate the tensor.
	if (out.is_gpu())
		kernel_ops::exp(out_data, ten_data, out.numel());

	else for (unsigned idx = 0; idx < _numel; idx++)
		out_data[idx] = expf(ten_data[idx]);

	// If it was a gradient operation store a ExpOp instance.
	if (has_grad())
		out._internals->op = new ExpOp(*this, out);

	// Return out.
	return out;
}

Tensor Tensor::log() const
{
	// Logaritmic tensor operator for backpropagation.
	class LogOp : public Tensor::TensorInternals::TensorOp
	{
		// Tensor copy storage for the input.
		Tensor in;

	public:
		// Constructor, stores all the data of the operation.
		LogOp(const Tensor& _in, const Tensor& _out) : TensorOp{ "Logaritmic", _out },
			in{ _in }
		{
			_relatives[0] = &in;
		}

		// Backpropagation. For logaritmic the derivative is one over input.
		void _backward() override
		{
			in.internal_gradient() += out.gradient() / in.no_grad();
		}
	};

	// Tensor must be initialized.
	MACROGRAD_CHECK(is_init(),
		"Trying to log an empty tensor."
	);

	// Create output with the same shape as tensor.
	Tensor out(shape(), device(), has_grad());

	// Extract the data.
	float* out_data = out.internal_data();
	const float* ten_data = internal_data();

	// Get the element count.
	unsigned _numel = out.numel();

	// Now we actually log the tensors.
	if (out.is_gpu())
		kernel_ops::log(out_data, ten_data, out.numel());

	else for (unsigned idx = 0; idx < _numel; idx++)
		out_data[idx] = logf(ten_data[idx]);

	// If it was a gradient operation store a LogOp instance.
	if (has_grad())
		out._internals->op = new LogOp(*this, out);

	// Return out.
	return out;
}

Tensor Tensor::relu() const
{
	// ReLU tensor operator for backpropagation.
	class ReLUOp : public Tensor::TensorInternals::TensorOp
	{
		// Tensor copy storage for the input.
		Tensor in;

	public:
		// Constructor, stores all the data of the operation.
		ReLUOp(const Tensor& _in, const Tensor& _out) : TensorOp{ "ReLU", _out },
			in{ _in }
		{
			_relatives[0] = &in;
		}

		// Backpropagation. Zero for negative input one for positive.
		void _backward() override
		{
			in.internal_gradient() += out.gradient() * (in > 0.f);
		}
	};

	// Tensor must be initialized.
	MACROGRAD_CHECK(is_init(),
		"Trying to apply ReLU to an empty tensor."
	);

	// Create output with the same shape as tensor.
	Tensor out(shape(), device(), has_grad());

	// Extract the data.
	float* out_data = out.internal_data();
	const float* ten_data = internal_data();

	// Get the element count.
	unsigned _numel = out.numel();

	// Now we actually ReLU the tensors.
	if (out.is_gpu())
		kernel_ops::relu(out_data, ten_data, out.numel());

	else for (unsigned idx = 0; idx < _numel; idx++)
			out_data[idx] = (ten_data[idx] > 0.f) ? ten_data[idx] : 0.f;

	// If it was a gradient operation store a ReLUOp instance.
	if (has_grad())
		out._internals->op = new ReLUOp(*this, out);

	// Return out.
	return out;
}

Tensor Tensor::silu() const
{
	// SiLU tensor operator for backpropagation.
	class SiLUOp : public Tensor::TensorInternals::TensorOp
	{
		// Tensor copy storage for the input.
		Tensor in;

	public:
		// Constructor, stores all the data of the operation.
		SiLUOp(const Tensor& _in, const Tensor& _out) : TensorOp{ "SiLU", _out },
			in{ _in }
		{
			_relatives[0] = &in;
		}

		// Backpropagation. For the derivative is sig * (1 + x - out).
		void _backward() override
		{
			in.internal_gradient() += out.gradient() * (out.no_grad() + in.no_grad().sigmoid() * (1.f - out.no_grad()));
		}
	};

	// Tensor must be initialized.
	MACROGRAD_CHECK(is_init(),
		"Trying to apply SiLU to an empty tensor."
	);

	// Create output with the same shape as tensor.
	Tensor out(shape(), device(), has_grad());

	// Extract the data.
	float* out_data = out.internal_data();
	const float* ten_data = internal_data();

	// Get the element count.
	unsigned _numel = out.numel();

	// Now we actually SiLU the tensors.
	if (out.is_gpu())
		kernel_ops::silu(out_data, ten_data, out.numel());

	else for (unsigned idx = 0; idx < _numel; idx++)
			out_data[idx] = ten_data[idx] / (1.f + expf(-ten_data[idx]));

	// If it was a gradient operation store a SiLUOp instance.
	if (has_grad())
		out._internals->op = new SiLUOp(*this, out);

	// Return out.
	return out;
}

Tensor Tensor::gelu() const
{
	// GELU tensor operator for backpropagation.
	class GELUOp : public Tensor::TensorInternals::TensorOp
	{
		// Tensor copy storage for the input and Phi.
		Tensor in, Phi;

	public:
		// Constructor, stores all the data of the operation.
		GELUOp(const Tensor& _in, const Tensor& _Phi, const Tensor& _out) : TensorOp{ "GELU", _out },
			in{ _in }, Phi{ _Phi }
		{
			_relatives[0] = &in;
		}

		// Backpropagation. For the derivative is Phi + in * phi(in).
		void _backward() override
		{
			constexpr float inv_sqrt_2pi = 0.3989422804f;
			Tensor phi = inv_sqrt_2pi * (-0.5f * in.no_grad() * in.no_grad()).exp();
			in.internal_gradient() += out.gradient() * (Phi + in.no_grad() * phi);
		}
	};

	// Tensor must be initialized.
	MACROGRAD_CHECK(is_init(),
		"Trying to apply GELU to an empty tensor."
	);

	// Create intermediate tensor as erf of x for backprop.
	Tensor Phi(shape(), device(), false);

	// Create output with the same shape as tensor.
	Tensor out(shape(), device(), has_grad());

	// Extract the data.
	float* out_data = out.internal_data();
	float* Phi_data = Phi.internal_data();
	const float* ten_data = internal_data();

	// Get the element count.
	unsigned _numel = out.numel();

	// Now we actually GELU the tensor.
	if (out.is_gpu())
		kernel_ops::gelu(out_data, ten_data, out.numel());

	else for (unsigned idx = 0; idx < _numel; idx++)
	{
		const float x = ten_data[idx];
		constexpr float sq2 = 1.4142135624f;
		Phi_data[idx] = 0.5f * (1.f + erff(x / sq2));
		out_data[idx] = x * Phi_data[idx];
	}

	// If it was a gradient operation store a GELUOp instance.
	if (has_grad())
		out._internals->op = new GELUOp(*this, Phi, out);

	// Return out.
	return out;
}

Tensor Tensor::sigmoid() const
{
	// Sigmoid tensor operator for backpropagation.
	class SigOp : public Tensor::TensorInternals::TensorOp
	{
		// Tensor copy storage for the input.
		Tensor in;

	public:
		// Constructor, stores all the data of the operation.
		SigOp(const Tensor& _in, const Tensor& _out) : TensorOp{ "Sigmoid", _out },
			in{ _in }
		{
			_relatives[0] = &in;
		}

		// Backpropagation. For sigmoid the derivative is out * (1 - out).
		void _backward() override
		{
			in.internal_gradient() += out.gradient() * out.no_grad() * (1.f - out.no_grad());
		}
	};

	// Tensor must be initialized.
	MACROGRAD_CHECK(is_init(),
		"Trying to apply sigmoid to an empty tensor."
	);

	// Create output with the same shape as tensor.
	Tensor out(shape(), device(), has_grad());

	// Extract the data.
	float* out_data = out.internal_data();
	const float* ten_data = internal_data();

	// Get the element count.
	unsigned _numel = out.numel();

	// Now we actually sigmoid the tensor.
	if (out.is_gpu())
		kernel_ops::sigmoid(out_data, ten_data, out.numel());

	else for (unsigned idx = 0; idx < _numel; idx++)
		out_data[idx] = 1.f / (1.f + expf(-ten_data[idx]));

	// If it was a gradient operation store a SigOp instance.
	if (has_grad())
		out._internals->op = new SigOp(*this, out);

	// Return out.
	return out;
}

Tensor Tensor::tanh() const
{
	// Tanh tensor operator for backpropagation.
	class TanOp : public Tensor::TensorInternals::TensorOp
	{
		// Tensor copy storage for the input.
		Tensor in;

	public:
		// Constructor, stores all the data of the operation.
		TanOp(const Tensor& _in, const Tensor& _out) : TensorOp{ "Tanh", _out },
			in{ _in }
		{
			_relatives[0] = &in;
		}

		// Backpropagation. For tanh the derivative is 1 - out**2.
		void _backward() override
		{
			in.internal_gradient() += out.gradient() * (1.f - out.no_grad().square());
		}
	};

	// Tensor must be initialized.
	MACROGRAD_CHECK(is_init(),
		"Trying to apply tanh to an empty tensor."
	);

	// Create output with the same shape as tensor.
	Tensor out(shape(), device(), has_grad());

	// Extract the data.
	float* out_data = out.internal_data();
	const float* ten_data = internal_data();

	// Get the element count.
	unsigned _numel = out.numel();

	// Now we actually tanh the tensor.
	if (out.is_gpu())
		kernel_ops::tanh(out_data, ten_data, out.numel());

	else for (unsigned idx = 0; idx < _numel; idx++)
	{
		const float exp2 = expf(2 * ten_data[idx]);
		out_data[idx] = (exp2 - 1.f) / (exp2 + 1.f);
	}

	// If it was a gradient operation store a TanOp instance.
	if (has_grad())
		out._internals->op = new TanOp(*this, out);

	// Return out.
	return out;
}

Tensor Tensor::sqrt() const
{
	// Square root tensor operator for backpropagation.
	class SqrtOp : public Tensor::TensorInternals::TensorOp
	{
		// Tensor copy storage for the input.
		Tensor in;

	public:
		// Constructor, stores all the data of the operation.
		SqrtOp(const Tensor& _in, const Tensor& _out) : TensorOp{ "Square Root", _out },
			in{ _in }
		{
			_relatives[0] = &in;
		}

		// Backpropagation. For square root the derivative is 1/(2*out).
		void _backward() override
		{
			in.internal_gradient() += out.gradient() * 0.5f / out.no_grad();
		}
	};

	// Tensor must be initialized.
	MACROGRAD_CHECK(is_init(),
		"Trying to get the square root of an empty tensor."
	);

	// Create output with the same shape as tensor.
	Tensor out(shape(), device(), has_grad());

	// Extract the data.
	float* out_data = out.internal_data();
	const float* ten_data = internal_data();

	// Get the element count.
	unsigned _numel = out.numel();

	// Now we actually sqrt the tensor.
	if (out.is_gpu())
		kernel_ops::sqrt(out_data, ten_data, out.numel());

	else for (unsigned idx = 0; idx < _numel; idx++)
		out_data[idx] = sqrtf(ten_data[idx]);

	// If it was a gradient operation store a SqrtOp instance.
	if (has_grad())
		out._internals->op = new SqrtOp(*this, out);

	// Return out.
	return out;
}

Tensor Tensor::square() const
{
	// Square tensor operator for backpropagation.
	class SqOp : public Tensor::TensorInternals::TensorOp
	{
		// Tensor copy storage for the input.
		Tensor in;

	public:
		// Constructor, stores all the data of the operation.
		SqOp(const Tensor& _in, const Tensor& _out) : TensorOp{ "Square", _out },
			in{ _in }
		{
			_relatives[0] = &in;
		}

		// Backpropagation. For squaring the derivative is 2*in.
		void _backward() override
		{
			in.internal_gradient() += out.gradient() * 2.f * in.no_grad();
		}
	};

	// Tensor must be initialized.
	MACROGRAD_CHECK(is_init(),
		"Trying to square an empty tensor."
	);

	// Create output with the same shape as tensor.
	Tensor out(shape(), device(), has_grad());

	// Extract the data.
	float* out_data = out.internal_data();
	const float* ten_data = internal_data();

	// Get the element count.
	unsigned _numel = out.numel();

	// Now we actually square the tensor.
	if (out.is_gpu())
		kernel_ops::square(out_data, ten_data, out.numel());

	else for (unsigned idx = 0; idx < _numel; idx++)
		out_data[idx] = ten_data[idx] * ten_data[idx];

	// If it was a gradient operation store a SqOp instance.
	if (has_grad())
		out._internals->op = new SqOp(*this, out);

	// Return out.
	return out;
}

Tensor Tensor::pow(float exp) const
{
	// Power tensor operator for backpropagation.
	class PowOp : public Tensor::TensorInternals::TensorOp
	{
		// Tensor copy storage for the input.
		Tensor in;

		// Variable to store exponent.
		float exp;

	public:
		// Constructor, stores all the data of the operation.
		PowOp(const Tensor& _in, const Tensor& _out, float _exp) : TensorOp{ "Power", _out },
			in{ _in }, exp{ _exp }
		{
			_relatives[0] = &in;
		}

		// Backpropagation. For power the derivative is 'exp*in**(exp-1) = exp*out/in'.
		void _backward() override
		{
			if (exp != 0.f)
				in.internal_gradient() += out.gradient() * exp * out.no_grad() / in.no_grad();
		}
	};

	// Tensor must be initialized.
	MACROGRAD_CHECK(is_init(),
		"Trying to square an empty tensor."
	);

	// Create output with the same shape as tensor.
	Tensor out(shape(), device(), has_grad());

	// Extract the data.
	float* out_data = out.internal_data();
	const float* ten_data = internal_data();

	// Get the element count.
	unsigned _numel = out.numel();

	// Now we actually power the tensor.
	if (out.is_gpu())
		kernel_ops::pow(out_data, ten_data, exp, out.numel());

	else for (unsigned idx = 0; idx < _numel; idx++)
		out_data[idx] = powf(ten_data[idx], exp);

	// If it was a gradient operation store a PowOp instance.
	if (has_grad())
		out._internals->op = new PowOp(*this, out, exp);

	// Return out.
	return out;
}

Tensor Tensor::sum(int dim, bool keepdim) const
{
	// Sum tensor operator for backpropagation.
	class SumOp : public Tensor::TensorInternals::TensorOp
	{
		// Tensor copy storage for the input.
		Tensor in;

	public:
		// Constructor, stores all the data of the operation.
		SumOp(const Tensor& _in, const Tensor& _out) : TensorOp{ "Sum", _out },
			in{ _in }
		{
			_relatives[0] = &in;
		}

		// Backpropagation. Broadcasts to entire input.
		void _backward() override
		{
			in.internal_gradient() += out.gradient();
		}
	};

	// Tensor must be initialized.
	MACROGRAD_CHECK(is_init(),
		"Trying to apply sum to an empty tensor."
	);

	// Modulo dimension.
	dim = unsigned(dim + _view.dim() * (2 - dim / int(_view.dim()))) % _view.dim();

	// Get the initial dimension size.
	unsigned _size = size(dim);
	// Compute output shape.
	Shape out_shape = shape();
	out_shape[dim] = 1;

	// Create output with the same shape as tensor.
	Tensor out(out_shape, device(), has_grad());

	// Extract the data.
	float* out_data = out.internal_data();
	const float* ten_data = internal_data();
	// Get relevant strides.
	const unsigned elem_stride = _stride[dim];
	const unsigned prev_stride = elem_stride * _size;
	// Get outer size.
	unsigned prev_size = out.numel() / elem_stride;

	// Now we actually sum the tensor.
	if (out.is_gpu())
		kernel_ops::sum(out_data, ten_data, elem_stride, _size, out.numel());
	else
	{
		// Iterate through all vectors to compute sum.
		for (unsigned prev_count = 0; prev_count < prev_size; prev_count++)
		for (unsigned post_count = 0; post_count < elem_stride; post_count++)
		{
			unsigned ten_idx = prev_count * prev_stride + post_count;
			unsigned out_idx = prev_count * elem_stride + post_count;

			for (unsigned count = 0u; count < _size; count++)
			{
				out_data[out_idx] += ten_data[ten_idx];
				ten_idx += elem_stride;
			}
		}
	}

	// If it was a gradient operation store a SumOp instance.
	if (has_grad())
		out._internals->op = new SumOp(*this, out);

	// Return out. Squeeze if necessary.
	if (!keepdim && out.dim() > 1)
		return out.squeeze(dim);
	return out;
}

Tensor Tensor::mean(int dim, bool keepdim) const
{
	// Mean tensor operator for backpropagation.
	class MeanOp : public Tensor::TensorInternals::TensorOp
	{
		// Tensor copy storage for the input.
		Tensor in;

		// Storage for initial dimension size.
		unsigned size;

	public:
		// Constructor, stores all the data of the operation.
		MeanOp(const Tensor& _in, const Tensor& _out, unsigned _size) : TensorOp{ "Mean", _out },
			in{ _in }, size{ _size }
		{
			_relatives[0] = &in;
		}

		// Backpropagation. Broadcasts to entire input downscaled by dimension size.
		void _backward() override
		{
			in.internal_gradient() += out.gradient() / float(size);
		}
	};

	// Tensor must be initialized.
	MACROGRAD_CHECK(is_init(),
		"Trying to apply mean to an empty tensor."
	);

	// Modulo dimension.
	dim = unsigned(dim + _view.dim() * (2 - dim / int(_view.dim()))) % _view.dim();

	// Get the initial dimension size.
	const unsigned _size = size(dim);
	// Compute output shape.
	Shape out_shape = shape();
	out_shape[dim] = 1;

	// Create output with the same shape as tensor.
	Tensor out(out_shape, device(), has_grad());

	// Extract the data.
	float* out_data = out.internal_data();
	const float* ten_data = internal_data();
	// Get relevant strides.
	const unsigned elem_stride = _stride[dim];
	const unsigned prev_stride = elem_stride * _size;
	// Get outer size.
	unsigned prev_size = out.numel() / elem_stride;

	// Now we actually average the tensor.
	if (out.is_gpu())
		kernel_ops::mean(out_data, ten_data, elem_stride, _size, out.numel());
	else
	{
		// Iterate through all vectors to compute mean.
		for (unsigned prev_count = 0; prev_count < prev_size; prev_count++)
		for (unsigned post_count = 0; post_count < elem_stride; post_count++)
		{
			unsigned ten_idx = prev_count * prev_stride + post_count;
			unsigned out_idx = prev_count * elem_stride + post_count;
				
			for (unsigned count = 0u; count < _size; count++)
			{
				out_data[out_idx] += ten_data[ten_idx];
				ten_idx += elem_stride;
			}
			out_data[out_idx] /= _size;
		}
	}

	// If it was a gradient operation store a MeanOp instance.
	if (has_grad())
		out._internals->op = new MeanOp(*this, out, _size);

	// Return out. Squeeze if necessary.
	if (!keepdim && out.dim() > 1)
		return out.squeeze(dim);
	return out;
}

Tensor Tensor::var(int dim, bool keepdim) const
{
	// Variance tensor operator for backpropagation.
	class VarOp : public Tensor::TensorInternals::TensorOp
	{
		// Tensor copy storage for the input.
		Tensor in;

		// Storage for initial dimension and size.
		unsigned dim, size;

	public:
		// Constructor, stores all the data of the operation.
		VarOp(const Tensor& _in, const Tensor& _out, unsigned _size, unsigned _dim) : TensorOp{ "Variance", _out },
			in{ _in }, size{ _size }, dim{ _dim }
		{
			_relatives[0] = &in;
		}

		// Backpropagation. Computes the mean and uses 2/N * (in - mean). 
		// We set output gradient at the end for proper broadcasting.
		void _backward() override
		{
			in.internal_gradient() += (2.f / size) * (in.no_grad() - in.no_grad().mean(dim, true)) * out.gradient();
		}
	};

	// Tensor must be initialized.
	MACROGRAD_CHECK(is_init(),
		"Trying to apply variance to an empty tensor."
	);

	// Modulo dimension.
	dim = unsigned(dim + _view.dim() * (2 - dim / int(_view.dim()))) % _view.dim();

	// Get the initial dimension size.
	unsigned _size = size(dim);
	// Compute output shape.
	Shape out_shape = shape();
	out_shape[dim] = 1;

	// Create output with the same shape as tensor.
	Tensor out(out_shape, device(), has_grad());

	// Extract the data.
	float* out_data = out.internal_data();
	const float* ten_data = internal_data();
	// Get relevant strides.
	const unsigned elem_stride = _stride[dim];
	const unsigned prev_stride = elem_stride * _size;
	// Get outer size.
	unsigned prev_size = out.numel() / elem_stride;

	// Now we actually compute the tensor variance.
	if (out.is_gpu())
		kernel_ops::var(out_data, ten_data, elem_stride, _size, out.numel());
	else
	{
		// Iterate through all vectors to compute variance.
		for (unsigned prev_count = 0; prev_count < prev_size; prev_count++)
		for (unsigned post_count = 0; post_count < elem_stride; post_count++)
		{
			unsigned ten_idx = prev_count * prev_stride + post_count;
			unsigned out_idx = prev_count * elem_stride + post_count;

			// Welford's algorithm for stability.
			float mean = 0.f, M2 = 0.f, old_mean;
			for (unsigned n = 1u; n < _size + 1u; n++)
			{
				old_mean = mean;
				mean += (ten_data[ten_idx] - mean) / n;
				M2 += (ten_data[ten_idx] - old_mean) * (ten_data[ten_idx] - mean);
				ten_idx += elem_stride;
			}
			out_data[out_idx] = M2 / _size;
		}
	}

	// If it was a gradient operation store a VarOp instance.
	if (has_grad())
		out._internals->op = new VarOp(*this, out, _size, dim);

	// Return out. Squeeze if necessary.
	if (!keepdim && out.dim() > 1)
		return out.squeeze(dim);
	return out;
}

Tensor Tensor::std(int dim, bool keepdim) const
{
	// STD tensor operator for backpropagation.
	class StdOp : public Tensor::TensorInternals::TensorOp
	{
		// Tensor copy storage for the input.
		Tensor in;

		// Storage for initial dimension and size.
		unsigned dim, size;

	public:
		// Constructor, stores all the data of the operation.
		StdOp(const Tensor& _in, const Tensor& _out, unsigned _size, unsigned _dim) : TensorOp{ "Standard Deviation", _out },
			in{ _in }, size{ _size }, dim{ _dim }
		{
			_relatives[0] = &in;
		}

		// Backpropagation. Computes the mean and uses (in - mean) / (N * std). 
		// We set output gradient at the end for proper broadcasting.
		void _backward() override
		{
			in.internal_gradient() += (in.no_grad() - in.no_grad().mean(dim, true)) / (float(size) * out.no_grad()) * out.gradient();
		}
	};

	// Tensor must be initialized.
	MACROGRAD_CHECK(is_init(),
		"Trying to apply standard deviation to an empty tensor."
	);

	// Modulo dimension.
	dim = unsigned(dim + _view.dim() * (2 - dim / int(_view.dim()))) % _view.dim();

	// Get the initial dimension size.
	unsigned _size = size(dim);
	// Compute output shape.
	Shape out_shape = shape();
	out_shape[dim] = 1;

	// Create output with the same shape as tensor.
	Tensor out(out_shape, device(), has_grad());

	// Extract the data.
	float* out_data = out.internal_data();
	const float* ten_data = internal_data();
	// Get relevant strides.
	const unsigned elem_stride = _stride[dim];
	const unsigned prev_stride = elem_stride * _size;
	// Get outer size.
	unsigned prev_size = out.numel() / elem_stride;

	// Now we actually compute the tensor STD.
	if (out.is_gpu())
		kernel_ops::std(out_data, ten_data, elem_stride, _size, out.numel());
	else
	{
		// Iterate through all vectors to compute STD.
		for (unsigned prev_count = 0; prev_count < prev_size; prev_count++)
		for (unsigned post_count = 0; post_count < elem_stride; post_count++)
		{
			unsigned ten_idx = prev_count * prev_stride + post_count;
			unsigned out_idx = prev_count * elem_stride + post_count;

			// Welford's algorithm for stability.
			float mean = 0.f, M2 = 0.f, old_mean;
			for(unsigned n = 1u; n < _size + 1u; n++)
			{
				old_mean = mean;
				mean += (ten_data[ten_idx] - mean) / n;
				M2 += (ten_data[ten_idx] - old_mean) * (ten_data[ten_idx] - mean);
				ten_idx += elem_stride;
			}
			out_data[out_idx] = sqrtf(M2 / _size);
		}
	}

	// If it was a gradient operation store a StdOp instance.
	if (has_grad())
		out._internals->op = new StdOp(*this, out, _size, dim);

	// Return out. Squeeze if necessary.
	if (!keepdim && out.dim() > 1)
		return out.squeeze(dim);
	return out;
}

Tensor Tensor::softmax(int dim) const
{
	// Softmax tensor operator for backpropagation.
	class SoftOp : public Tensor::TensorInternals::TensorOp
	{
		// Tensor copy storage for the input.
		Tensor in;

		// Storage for initial dimension and size.
		unsigned dim;

	public:
		// Constructor, stores all the data of the operation.
		SoftOp(const Tensor& _in, const Tensor& _out, unsigned _dim) : TensorOp{ "Softmax", _out },
			in{ _in }, dim{ _dim }
		{
			_relatives[0] = &in;
		}

		// Backpropagation. Softmax derivative is the sum of all contributions from all 
		// elements: dSi/di = out_i * (1 - out_i). dSj/di = -out_i * out_j
		// dL/di = out_i * (grad_i - sum(outs * grads)).
		void _backward() override
		{
			in.internal_gradient() += out.no_grad() * (out.gradient() - (out.no_grad() * out.gradient()).sum(dim, true));
		}
	};

	// Tensor must be initialized.
	MACROGRAD_CHECK(is_init(),
		"Trying to apply softmax to an empty tensor."
	);

	// Modulo dimension.
	dim = unsigned(dim + _view.dim() * (2 - dim / int(_view.dim()))) % _view.dim();

	// Get the initial dimension size.
	unsigned _size = size(dim);
	// Create output with the same shape as tensor.
	Tensor out(shape(), device(), has_grad());

	// Extract the data.
	float* out_data = out.internal_data();
	const float* ten_data = internal_data();
	// Get relevant strides.
	const unsigned elem_stride = _stride[dim];
	const unsigned prev_stride = elem_stride * _size;
	// Get outer size.
	unsigned prev_size = out.numel() / prev_stride;

	// Now we actually apply softmax to the tensor.
	if (out.is_gpu())
		kernel_ops::softmax(out_data, ten_data, elem_stride, _size, out.numel() / _size);
	else
	{
		// Iterate through all vectors to compute softmax.
		for (unsigned prev_count = 0; prev_count < prev_size; prev_count++)
		for (unsigned post_count = 0; post_count < elem_stride; post_count++)
		{
			// Compute initial and final idx for this vector.
			const unsigned idx_0 = prev_count * prev_stride + post_count;
			const unsigned idx_f = (prev_count + 1) * prev_stride + post_count;
				
			// First pass find maximum.
			float max = ten_data[idx_0];
			for (unsigned idx = idx_0; idx < idx_f; idx += elem_stride)
				if (ten_data[idx] > max)
					max = ten_data[idx];
			// Second pass compute exponential and accumulate sum.
			float sum = 0.f;
			for (unsigned idx = idx_0; idx < idx_f; idx += elem_stride)
			{
				out_data[idx] = expf(ten_data[idx] - max);
				sum += out_data[idx];
			}
			// Third pass divide by sum.
			for (unsigned idx = idx_0; idx < idx_f; idx+=elem_stride)
				out_data[idx] /= sum;
		}
	}

	// If it was a gradient operation store a SoftOp instance.
	if (has_grad())
		out._internals->op = new SoftOp(*this, out, dim);

	// Return out.
	return out;
}

Tensor Tensor::max(int dim, bool keepdim) const
{
	// Max tensor operator for backpropagation.
	class MaxOp : public Tensor::TensorInternals::TensorOp
	{
		// Tensor copy storage for the input. and one-hot encoded maxs.
		Tensor in, one_hot;

	public:
		// Constructor, stores all the data of the operation.
		MaxOp(const Tensor& _in, const Tensor& _one_hot, const Tensor& _out) : TensorOp{ "Max", _out },
			in{ _in }, one_hot{ _one_hot }
		{
			_relatives[0] = &in;
		}

		// Backpropagation. Broadcasts to one-hot elements of input.
		void _backward() override
		{
			in.internal_gradient() += one_hot * out.gradient();
		}
	};

	// Tensor must be initialized.
	MACROGRAD_CHECK(is_init(),
		"Trying to apply max to an empty tensor."
	);

	// Modulo dimension.
	dim = ((dim % (int)_view.dim()) + _view.dim()) % _view.dim();

	// Get the initial dimension size.
	unsigned _size = size(dim);
	// Compute output shape.
	Shape out_shape = shape();
	out_shape[dim] = 1;

	// Create output with the new shape.
	Tensor out(out_shape, device(), has_grad());

	// Create one-hot encoded tensor to store maxs for backprop.
	Tensor one_hot = Tensor();
	if (has_grad())
		one_hot = Tensor(_view, device(), false);

	// Extract the data.
	float* out_data = out.internal_data();
	const float* ten_data = internal_data();
	float* hot_data = has_grad() ? one_hot.internal_data() : nullptr;

	// Get relevant strides.
	const unsigned elem_stride = _stride[dim];
	const unsigned prev_stride = elem_stride * _size;
	// Get outer size.
	unsigned prev_size = out.numel() / elem_stride;

	// Now we actually find the max values.
	if (out.is_gpu())
		kernel_ops::max(out_data, hot_data, ten_data, prev_size, _size, elem_stride);
	else
	{
		// Iterate through all vectors to compute max.
		for (unsigned prev_count = 0; prev_count < prev_size; prev_count++)
		for (unsigned post_count = 0; post_count < elem_stride; post_count++)
		{
			unsigned ten_idx = prev_count * prev_stride + post_count;
			unsigned out_idx = prev_count * elem_stride + post_count;

			float max = ten_data[ten_idx];
			int argmax = 0;

			// Compare all elements.
			for (unsigned k = 1u; k < _size; k++)
			{
				const float val = ten_data[ten_idx + k * elem_stride];
				if (val > max)
				{
					max = val;
					argmax = k;
				}
			}
			// Write the max to output.
			out_data[out_idx] = max;

			// Store argmax if grad.
			if (hot_data)
				hot_data[ten_idx + argmax * elem_stride] = 1.f;
		}
	}

	// If it was a gradient operation store a MaxOp instance.
	if (has_grad())
		out._internals->op = new MaxOp(*this, one_hot, out);

	// Return out. Squeeze if necessary.
	if (!keepdim && out.dim() > 1)
		return out.squeeze(dim);
	return out;
}

Tensor Tensor::min(int dim, bool keepdim) const
{
	// Min tensor operator for backpropagation.
	class MinOp : public Tensor::TensorInternals::TensorOp
	{
		// Tensor copy storage for the input. and one-hot encoded mins.
		Tensor in, one_hot;

	public:
		// Constructor, stores all the data of the operation.
		MinOp(const Tensor& _in, const Tensor& _one_hot, const Tensor& _out) : TensorOp{ "Min", _out },
			in{ _in }, one_hot{ _one_hot }
		{
			_relatives[0] = &in;
		}

		// Backpropagation. Broadcasts to one-hot elements of input.
		void _backward() override
		{
			in.internal_gradient() += one_hot * out.gradient();
		}
	};

	// Tensor must be initialized.
	MACROGRAD_CHECK(is_init(),
		"Trying to apply min to an empty tensor."
	);

	// Modulo dimension.
	dim = ((dim % (int)_view.dim()) + _view.dim()) % _view.dim();

	// Get the initial dimension size.
	unsigned _size = size(dim);
	// Compute output shape.
	Shape out_shape = shape();
	out_shape[dim] = 1;

	// Create output with the new shape.
	Tensor out(out_shape, device(), has_grad());

	// Create one-hot encoded tensor to store mins for backprop.
	Tensor one_hot = Tensor();
	if (has_grad())
		one_hot = Tensor(_view, device(), false);

	// Extract the data.
	float* out_data = out.internal_data();
	const float* ten_data = internal_data();
	float* hot_data = has_grad() ? one_hot.internal_data() : nullptr;

	// Get relevant strides.
	const unsigned elem_stride = _stride[dim];
	const unsigned prev_stride = elem_stride * _size;
	// Get outer size.
	unsigned prev_size = out.numel() / elem_stride;

	// Now we actually find the min values.
	if (out.is_gpu())
		kernel_ops::min(out_data, hot_data, ten_data, prev_size, _size, elem_stride);
	else
	{
		// Iterate through all vectors to compute min.
		for (unsigned prev_count = 0; prev_count < prev_size; prev_count++)
			for (unsigned post_count = 0; post_count < elem_stride; post_count++)
			{
				unsigned ten_idx = prev_count * prev_stride + post_count;
				unsigned out_idx = prev_count * elem_stride + post_count;

				float min = ten_data[ten_idx];
				int argmin = 0;

				// Compare all elements.
				for (unsigned k = 1u; k < _size; k++)
				{
					const float val = ten_data[ten_idx + k * elem_stride];
					if (val < min)
					{
						min = val;
						argmin = k;
					}
				}
				// Write the min to output.
				out_data[out_idx] = min;

				// Store argmin if grad.
				if (hot_data)
					hot_data[ten_idx + argmin * elem_stride] = 1.f;
			}
	}

	// If it was a gradient operation store a MinOp instance.
	if (has_grad())
		out._internals->op = new MinOp(*this, one_hot, out);

	// Return out. Squeeze if necessary.
	if (!keepdim && out.dim() > 1)
		return out.squeeze(dim);
	return out;
}

VectorInt Tensor::argmax(bool last_dim) const
{
	// Tensor must be initialized.
	MACROGRAD_CHECK(is_init(),
		"Trying to get argmax from an empty tensor."
	);
	// Tensor must have leq two dimensions.
	MACROGRAD_CHECK(dim() <= 2,
		"Argmax can only be called on two dimensional or single dimensional tensors.\n"
		"Use view/squeeze/flatten to reshape your tensor accordingly."
	);

	// Get the dimension size.
	unsigned _size = size(last_dim ? -1 : 0);

	// Get the length of the output vector.
	unsigned cases = (dim() > 1) ? size(last_dim ? 0 : -1) : 1;

	// Create output vector.
	VectorInt args(cases, device());

	// Extract the data.
	int* args_data = args.data();
	const float* ten_data = internal_data();

	// Get relevant strides.
	const unsigned elem_stride = last_dim ? 1 : cases;
	const unsigned case_stride = last_dim ? _size : 1;

	// Now we actually find the max args.
	if (args.is_gpu())
		kernel_ops::argmax(args_data, ten_data, cases, _size, case_stride, elem_stride);
	else
	{
		// Iterate through all vectors to find argmax.
		for (unsigned row = 0; row < cases; row++)
		{
			unsigned ten_idx = row * case_stride;
			float max = ten_data[ten_idx];
			int argmax = 0;

			// Compare all elements.
			for (unsigned k = 1u; k < _size; k++)
			{
				const float val = ten_data[ten_idx + k * elem_stride];
				if (val > max)
				{
					max = val;
					argmax = k;
				}
			}
			// Write the argmax to output.
			args_data[row] = argmax;
		}
	}
	// Return args.
	return args;
}

VectorInt Tensor::argmin(bool last_dim) const
{
	// Tensor must be initialized.
	MACROGRAD_CHECK(is_init(),
		"Trying to get argmin from an empty tensor."
	);
	// Tensor must have leq two dimensions.
	MACROGRAD_CHECK(dim() <= 2,
		"Argmin can only be called on two dimensional or single dimensional tensors.\n"
		"Use view/squeeze/flatten to reshape your tensor accordingly."
	);

	// Get the dimension size.
	unsigned _size = size(last_dim ? -1 : 0);

	// Get the length of the output vector.
	unsigned cases = (dim() > 1) ? size(last_dim ? 0 : -1) : 1;

	// Create output vector.
	VectorInt args(cases, device());

	// Extract the data.
	int* args_data = args.data();
	const float* ten_data = internal_data();

	// Get relevant strides.
	const unsigned elem_stride = last_dim ? 1 : cases;
	const unsigned case_stride = last_dim ? _size : 1;

	// Now we actually find the min args.
	if (args.is_gpu())
		kernel_ops::argmin(args_data, ten_data, cases, _size, case_stride, elem_stride);
	else
	{
		// Iterate through all vectors to find argmin.
		for (unsigned row = 0; row < cases; row++)
		{
			unsigned ten_idx = row * case_stride;
			float min = ten_data[ten_idx];
			int argmin = 0;

			// Compare all elements.
			for (unsigned k = 1u; k < _size; k++)
			{
				const float val = ten_data[ten_idx + k * elem_stride];
				if (val < min)
				{
					min = val;
					argmin = k;
				}
			}
			// Write the argmin to output.
			args_data[row] = argmin;
		}
	}
	// Return args.
	return args;
}

/*
--------------------------------------------------------------------------------------------------------------------------
 Regular Operators
--------------------------------------------------------------------------------------------------------------------------
*/

Tensor operator+(const Tensor& ten0, const Tensor& ten1)
{
	// Addition tensor operator for backpropagation.
	class AddOp : public Tensor::TensorInternals::TensorOp
	{
		// Tensor copy storage for the summands.
		Tensor sum0, sum1;

	public:
		// Constructor, stores all the data of the operation.
		AddOp(const Tensor& _sum0, const Tensor& _sum1, const Tensor& _out) : TensorOp{ "Addition", _out },
			sum0{ _sum0.has_grad() ? _sum0 : Tensor() }, sum1{ _sum1.has_grad() ? _sum1 : Tensor() }
		{
			// Unsqueeze so that we can broacdast properly during backprop.
			while (sum1.is_init() && out.dim() > sum1.dim()) sum1 = sum1.unsqueeze(0);
			if (sum0.has_grad()) _relatives[0] = &sum0;
			if (sum1.has_grad()) _relatives[1] = &sum1;
		}

		// Backpropagation. For addition the gradient gets routed equally.
		// The broadcasting logic of the '+=' operator handles shapes.
		void _backward() override
		{
			if (sum0.has_grad()) sum0.internal_gradient() += out.gradient();
			if (sum1.has_grad()) sum1.internal_gradient() += out.gradient();
		}
	};

	// --- Sanity checks ---
	
	// Both tensor must be initialized.
	MACROGRAD_CHECK(ten0.is_init(),
		"Trying to add two tensors while the first tensor is empty."
	);
	MACROGRAD_CHECK(ten1.is_init(),
		"Trying to add two tensors while the second tensor is empty."
	);
	// Both tensors must be on the same device
	MACROGRAD_CHECK(ten0.is_gpu() == ten1.is_gpu(),
		"Trying to add two tensors in different devices is not allowed."
	);
	// Ten0 must have more or equal the amount of dimensions of ten1 for broadcasting.
	MACROGRAD_CHECK(ten0.dim() >= ten1.dim(),
		"Trying to add two tensors with incompatible dimensions.\n"
		"The dimensions of the tensors must match or the second tensor must broadcast to the first one.\n"
		"Found shapes | Tensor0: %s | Tensor1: %s", ten0._view.str(), ten1._view.str()
	);
	// Both tensors must either have the same dimension sizes or one of them have size one.
	unsigned offset = ten0.dim() - ten1.dim();
	for (unsigned i = offset; i < ten0.dim(); i++)
		MACROGRAD_CHECK(ten0._view[i] == ten1._view[i - offset] || ten1._view[i - offset] == 1 || ten0._view[i] == 1,
			"Trying to add two tensors with incompatible shapes for broadcasting.\n"
			"Make sure shapes are compatible, meaning they have the same sizes or one of them is unitary.\n"
			"Found shapes | Tensor0: %s | Tensor1: %s", ten0._view.str(), ten1._view.str()
		);

	// The output will have gradient if either of the inputs has.
	bool requires_grad = ten0.has_grad() || ten1.has_grad();

	// Create output with the same shape as first tensor.
	Tensor out(ten0.shape(), ten0.device(), requires_grad);

	// Extract the data.
	float* out_data = out.internal_data();
	const float* ten0_data = ten0.internal_data();
	const float* ten1_data = ten1.internal_data();

	// Now we actually add the tensors.
	if (out.is_gpu())
		kernel_ops::shaped_add(out_data, ten0_data, ten1_data, out._view, ten1._view);
	else
	{
		// Unsqueeze and sum dimensions on ten1 if necessary.
		Tensor t1 = ten1.no_grad();
		while (ten0.dim() > t1.dim()) t1 = t1.unsqueeze(0);
		for (unsigned i = 0; i < ten0.dim(); i++)
			if (ten0._view[i] == 1 && t1._view[i] != 1)
				t1 = t1.sum(i, true);

		// Extract data.
		const float* t1_data = t1.internal_data();

		// Exptract the shapes.
		Shape t0_shape = ten0._view;
		Shape t1_shape = t1._view;
		// Find the last non-unitary dimension to iterate through.
		unsigned last_long_dim = 0;
		for (unsigned i = 0; i < t0_shape.dim(); i++)
			if (t0_shape[i] > 1)
				last_long_dim = i;
		// Find the stride, will be 0 if broadcasting on last dim.
		const unsigned dt1 = (t1._view[last_long_dim] != 1) ? 1 : 0;
		// Find the vector length to iterate.
		const unsigned vector_len = t0_shape[last_long_dim];
		// Create a running shape to count.
		Shape counting_shape(last_long_dim, (int*)nullptr);
		// Iterate through vectors.
		while (true)
		{
			unsigned out_idx = 0, t1_idx = 0;
			for (unsigned d = 0; d < counting_shape.dim(); d++)
			{
				if (out._view[d] != 1) out_idx += counting_shape[d] * out._stride[d];
				if (t1._view[d] != 1) t1_idx += counting_shape[d] * t1._stride[d];
			}

			for (unsigned count = 0u; count < vector_len; count++)
			{
				out_data[out_idx] = ten0_data[out_idx] + t1_data[t1_idx];
				out_idx++, t1_idx += dt1;
			}

			if (!counting_shape.dim())
				break;

			counting_shape[-1]++;
			for (int d = last_long_dim - 1; d > 0; d--)
				if (counting_shape[d] >= t0_shape[d])
				{
					counting_shape[d] -= t0_shape[d];
					counting_shape[d - 1]++;
				}

			// If you reach the end of the leading dimension you're done.
			if (counting_shape[0] >= t0_shape[0])
				break;
		}
	}

	// If it was a gradient operation store a AddOp instance.
	if (requires_grad)
		out._internals->op = new AddOp(ten0, ten1, out);

	// Return out.
	return out;
}

Tensor operator-(const Tensor& ten0, const Tensor& ten1)
{
	// Subtraction tensor operator for backpropagation.
	class SubOp : public Tensor::TensorInternals::TensorOp
	{
		// Tensor copy storage for the operands.
		Tensor sum, sub;

	public:
		// Constructor, stores all the data of the operation.
		SubOp(const Tensor& _sum, const Tensor& _sub, const Tensor& _out) : TensorOp{ "Subtraction", _out },
			sum{ _sum.has_grad() ? _sum : Tensor() }, sub{ _sub.has_grad() ? _sub : Tensor() }
		{
			// Unsqueeze so that we can broacdast properly during backprop.
			while (sub.is_init() && out.dim() > sub.dim()) sub = sub.unsqueeze(0);
			if (sum.has_grad()) _relatives[0] = &sum;
			if (sub.has_grad()) _relatives[1] = &sub;
		}

		// Backpropagation. For subtraction the gradient gets routed equally with the sign changed 
		// for the subtractor. The broadcasting logic of the '-=/+=' operators handle shapes.
		void _backward() override
		{
			if (sum.has_grad()) sum.internal_gradient() += out.gradient();
			if (sub.has_grad()) sub.internal_gradient() -= out.gradient();
		}
	};

	// --- Sanity checks ---

	// Both tensor must be initialized.
	MACROGRAD_CHECK(ten0.is_init(),
		"Trying to subtract two tensors while the first tensor is empty."
	);
	MACROGRAD_CHECK(ten1.is_init(),
		"Trying to subtract two tensors while the second tensor is empty."
	);
	// Both tensors must be on the same device
	MACROGRAD_CHECK(ten0.is_gpu() == ten1.is_gpu(),
		"Trying to subtract two tensors in different devices is not allowed."
	);
	// Ten0 must have more or equal the amount of dimensions of ten1 for broadcasting.
	MACROGRAD_CHECK(ten0.dim() >= ten1.dim(),
		"Trying to subtract two tensors with incompatible dimensions.\n"
		"The dimensions of the tensors must match or the second tensor must broadcast to the first one.\n"
		"Found shapes | Tensor0: %s | Tensor1: %s", ten0._view.str(), ten1._view.str()
	);
	// Both tensors must either have the same dimension sizes or one of them have size one.
	unsigned offset = ten0.dim() - ten1.dim();
	for (unsigned i = offset; i < ten0.dim(); i++)
		MACROGRAD_CHECK(ten0._view[i] == ten1._view[i - offset] || ten1._view[i - offset] == 1 || ten0._view[i] == 1,
			"Trying to subtract two tensors with incompatible shapes for broadcasting.\n"
			"Make sure shapes are compatible, meaning they have the same sizes or one of them is unitary.\n"
			"Found shapes | Tensor0: %s | Tensor1: %s", ten0._view.str(), ten1._view.str()
		);

	// The output will have gradient if either of the inputs has.
	bool requires_grad = ten0.has_grad() || ten1.has_grad();

	// Create output with the same shape as first tensor.
	Tensor out(ten0.shape(), ten0.device(), requires_grad);

	// Extract the data.
	float* out_data = out.internal_data();
	const float* ten0_data = ten0.internal_data();
	const float* ten1_data = ten1.internal_data();

	// Now we actually subtract the tensors.
	if (out.is_gpu())
		kernel_ops::shaped_subtract(out_data, ten0_data, ten1_data, out._view, ten1._view);
	else
	{
		// Unsqueeze and sum dimensions on ten1 if necessary.
		Tensor t1 = ten1.no_grad();
		while (ten0.dim() > t1.dim()) t1 = t1.unsqueeze(0);
		for (unsigned i = 0; i < ten0.dim(); i++)
			if (ten0._view[i] == 1 && t1._view[i] != 1)
				t1 = t1.sum(i, true);

		// Extract data.
		const float* t1_data = t1.internal_data();

		// Exptract the shapes.
		Shape t0_shape = ten0._view;
		Shape t1_shape = t1._view;
		// Find the last non-unitary dimension to iterate through.
		unsigned last_long_dim = 0;
		for (unsigned i = 0; i < t0_shape.dim(); i++)
			if (t0_shape[i] > 1)
				last_long_dim = i;
		// Find the stride, will be 0 if broadcasting on last dim.
		const unsigned dt1 = (t1._view[last_long_dim] != 1) ? 1 : 0;
		// Find the vector length to iterate.
		const unsigned vector_len = t0_shape[last_long_dim];
		// Create a running shape to count.
		Shape counting_shape(last_long_dim, (int*)nullptr);
		// Iterate through vectors.
		while (true)
		{
			unsigned out_idx = 0, t1_idx = 0;
			for (unsigned d = 0; d < counting_shape.dim(); d++)
			{
				if (out._view[d] != 1) out_idx += counting_shape[d] * out._stride[d];
				if (t1._view[d] != 1) t1_idx += counting_shape[d] * t1._stride[d];
			}

			for (unsigned count = 0u; count < vector_len; count++)
			{
				out_data[out_idx] = ten0_data[out_idx] - t1_data[t1_idx];
				out_idx++, t1_idx += dt1;
			}

			if (!counting_shape.dim())
				break;

			counting_shape[-1]++;
			for (int d = last_long_dim - 1; d > 0; d--)
				if (counting_shape[d] >= t0_shape[d])
				{
					counting_shape[d] -= t0_shape[d];
					counting_shape[d - 1]++;
				}

			// If you reach the end of the leading dimension you're done.
			if (counting_shape[0] >= t0_shape[0])
				break;
		}
	}

	// If it was a gradient operation store a SubOp instance.
	if (requires_grad)
		out._internals->op = new SubOp(ten0, ten1, out);

	// Return out.
	return out;
}

Tensor operator*(const Tensor& ten0, const Tensor& ten1)
{
	// Multiplication tensor operator for backpropagation.
	class MulOp : public Tensor::TensorInternals::TensorOp
	{
		// Tensor copy storage for the factors.
		Tensor fac0, fac1;

	public:
		// Constructor, stores all the data of the operation.
		MulOp(const Tensor& _fac0, const Tensor& _fac1, const Tensor& _out) : TensorOp{ "Multiplication", _out },
			fac0{ _fac0 }, fac1{ _fac1 }
		{
			// Unsqueeze so that we can broacdast properly during backprop.
			while (out.dim() > fac1.dim()) fac1 = fac1.unsqueeze(0);
			if (fac0.has_grad()) _relatives[0] = &fac0;
			if (fac1.has_grad()) _relatives[1] = &fac1;
		}

		// Backpropagation. For multiplication the gradient gets multiplied by the other tensor.
		// The broadcasting logic of the '+=/*' operators handle shapes.
		void _backward() override
		{
			if (fac0.has_grad()) fac0.internal_gradient() += out.gradient() * fac1.no_grad();
			if (fac1.has_grad()) fac1.internal_gradient() += out.gradient() * fac0.no_grad();
		}
	};

	// --- Sanity checks ---

	// Both tensor must be initialized.
	MACROGRAD_CHECK(ten0.is_init(),
		"Trying to multiply two tensors while the first tensor is empty."
	);
	MACROGRAD_CHECK(ten1.is_init(),
		"Trying to multiply two tensors while the second tensor is empty."
	);
	// Both tensors must be on the same device
	MACROGRAD_CHECK(ten0.is_gpu() == ten1.is_gpu(),
		"Trying to multiply two tensors in different devices is not allowed."
	);
	// Ten0 must have more or equal the amount of dimensions of ten1 for broadcasting.
	MACROGRAD_CHECK(ten0.dim() >= ten1.dim(),
		"Trying to multiply two tensors with incompatible dimensions.\n"
		"The dimensions of the tensors must match or the second tensor must broadcast to the first one.\n"
		"Found shapes | Tensor0: %s | Tensor1: %s", ten0._view.str(), ten1._view.str()
	);
	// Both tensors must either have the same dimension sizes or one of them have size one.
	unsigned offset = ten0.dim() - ten1.dim();
	for (unsigned i = offset; i < ten0.dim(); i++)
		MACROGRAD_CHECK(ten0._view[i] == ten1._view[i - offset] || ten1._view[i - offset] == 1 || ten0._view[i] == 1,
			"Trying to multiply two tensors with incompatible shapes for broadcasting.\n"
			"Make sure shapes are compatible, meaning they have the same sizes or one of them is unitary.\n"
			"Found shapes | Tensor0: %s | Tensor1: %s", ten0._view.str(), ten1._view.str()
		);

	// The output will have gradient if either of the inputs has.
	bool requires_grad = ten0.has_grad() || ten1.has_grad();

	// Create output with the same shape as first tensor.
	Tensor out(ten0.shape(), ten0.device(), requires_grad);

	// Extract the data.
	float* out_data = out.internal_data();
	const float* ten0_data = ten0.internal_data();
	const float* ten1_data = ten1.internal_data();

	// Now we actually multiply the tensors.
	if (out.is_gpu())
		kernel_ops::shaped_multiply(out_data, ten0_data, ten1_data, out._view, ten1._view);
	else
	{
		// Unsqueeze and sum dimensions on ten1 if necessary.
		Tensor t1 = ten1.no_grad();
		while (ten0.dim() > t1.dim()) t1 = t1.unsqueeze(0);
		for (unsigned i = 0; i < ten0.dim(); i++)
			if (ten0._view[i] == 1 && t1._view[i] != 1)
				t1 = t1.sum(i, true);

		// Extract data.
		const float* t1_data = t1.internal_data();

		// Exptract the shapes.
		Shape t0_shape = ten0._view;
		Shape t1_shape = t1._view;
		// Find the last non-unitary dimension to iterate through.
		unsigned last_long_dim = 0;
		for (unsigned i = 0; i < t0_shape.dim(); i++)
			if (t0_shape[i] > 1)
				last_long_dim = i;
		// Find the stride, will be 0 if broadcasting on last dim.
		const unsigned dt1 = (t1._view[last_long_dim] != 1) ? 1 : 0;
		// Find the vector length to iterate.
		const unsigned vector_len = t0_shape[last_long_dim];
		// Create a running shape to count.
		Shape counting_shape(last_long_dim, (int*)nullptr);
		// Iterate through vectors.
		while (true)
		{
			unsigned out_idx = 0, t1_idx = 0;
			for (unsigned d = 0; d < counting_shape.dim(); d++)
			{
				if (out._view[d] != 1) out_idx += counting_shape[d] * out._stride[d];
				if (t1._view[d] != 1) t1_idx += counting_shape[d] * t1._stride[d];
			}

			for (unsigned count = 0u; count < vector_len; count++)
			{
				out_data[out_idx] = ten0_data[out_idx] * t1_data[t1_idx];
				out_idx++, t1_idx += dt1;
			}

			if (!counting_shape.dim())
				break;

			counting_shape[-1]++;
			for (int d = last_long_dim - 1; d > 0; d--)
				if (counting_shape[d] >= t0_shape[d])
				{
					counting_shape[d] -= t0_shape[d];
					counting_shape[d - 1]++;
				}

			// If you reach the end of the leading dimension you're done.
			if (counting_shape[0] >= t0_shape[0])
				break;
		}
	}

	// If it was a gradient operation store a MulOp instance.
	if (requires_grad)
		out._internals->op = new MulOp(ten0, ten1, out);

	// Return out.
	return out;
}

Tensor operator/(const Tensor& ten0, const Tensor& ten1)
{
	// Division tensor operator for backpropagation.
	class DivOp : public Tensor::TensorInternals::TensorOp
	{
		// Tensor copy storage for the operators.
		Tensor num, den;

	public:
		// Constructor, stores all the data of the operation.
		DivOp(const Tensor& _num, const Tensor& _den, const Tensor& _out) : TensorOp{ "Division", _out },
			num{ _num }, den{ _den }
		{
			// Unsqueeze so that we can broacdast properly during backprop.
			while (out.dim() > den.dim()) den = den.unsqueeze(0);
			if (num.has_grad()) _relatives[0] = &num;
			if (den.has_grad()) _relatives[1] = &den;
		}

		// Backpropagation. For division the gradients are 'dn = dout * 1/d' and 'dd = dout * (-n/d**2) = -dout * (out/d)'.
		// The broadcasting logic of the '+=/*' operators handle shapes.
		void _backward() override
		{
			if (num.has_grad()) num.internal_gradient() += out.gradient() / den.no_grad();
			if (den.has_grad()) den.internal_gradient() -= out.gradient() * (out.no_grad() / den.no_grad());
		}
	};

	// --- Sanity checks ---

	// Both tensor must be initialized.
	MACROGRAD_CHECK(ten0.is_init(),
		"Trying to divide two tensors while the first tensor is empty."
	);
	MACROGRAD_CHECK(ten1.is_init(),
		"Trying to divide two tensors while the second tensor is empty."
	);
	// Both tensors must be on the same device
	MACROGRAD_CHECK(ten0.is_gpu() == ten1.is_gpu(),
		"Trying to divide two tensors in different devices is not allowed."
	);
	// Ten0 must have more or equal the amount of dimensions of ten1 for broadcasting.
	MACROGRAD_CHECK(ten0.dim() >= ten1.dim(),
		"Trying to divide two tensors with incompatible dimensions.\n"
		"The dimensions of the tensors must match or the second tensor must broadcast to the first one.\n"
		"Found shapes | Tensor0: %s | Tensor1: %s", ten0._view.str(), ten1._view.str()
	);
	// Both tensors must either have the same dimension sizes or one of them have size one.
	unsigned offset = ten0.dim() - ten1.dim();
	for (unsigned i = offset; i < ten0.dim(); i++)
		MACROGRAD_CHECK(ten0._view[i] == ten1._view[i - offset] || ten1._view[i - offset] == 1 || ten0._view[i] == 1,
			"Trying to divide two tensors with incompatible shapes for broadcasting.\n"
			"Make sure shapes are compatible, meaning they have the same sizes or one of them is unitary.\n"
			"Found shapes | Tensor0: %s | Tensor1: %s", ten0._view.str(), ten1._view.str()
		);

	// The output will have gradient if either of the inputs has.
	bool requires_grad = ten0.has_grad() || ten1.has_grad();

	// Create output with the same shape as first tensor.
	Tensor out(ten0.shape(), ten0.device(), requires_grad);

	// Extract the data.
	float* out_data = out.internal_data();
	const float* ten0_data = ten0.internal_data();
	const float* ten1_data = ten1.internal_data();

	// Now we actually divide the tensors.
	if (out.is_gpu())
		kernel_ops::shaped_divide(out_data, ten0_data, ten1_data, out._view, ten1._view);
	else
	{
		// Unsqueeze and sum dimensions on ten1 if necessary.
		Tensor t1 = ten1.no_grad();
		while (ten0.dim() > t1.dim()) t1 = t1.unsqueeze(0);
		for (unsigned i = 0; i < ten0.dim(); i++)
			if (ten0._view[i] == 1 && t1._view[i] != 1)
				t1 = t1.sum(i, true);

		// Extract data.
		const float* t1_data = t1.internal_data();

		// Exptract the shapes.
		Shape t0_shape = ten0._view;
		Shape t1_shape = t1._view;
		// Find the last non-unitary dimension to iterate through.
		unsigned last_long_dim = 0;
		for (unsigned i = 0; i < t0_shape.dim(); i++)
			if (t0_shape[i] > 1)
				last_long_dim = i;
		// Find the stride, will be 0 if broadcasting on last dim.
		const unsigned dt1 = (t1._view[last_long_dim] != 1) ? 1 : 0;
		// Find the vector length to iterate.
		const unsigned vector_len = t0_shape[last_long_dim];
		// Create a running shape to count.
		Shape counting_shape(last_long_dim, (int*)nullptr);
		// Iterate through vectors.
		while (true)
		{
			unsigned out_idx = 0, t1_idx = 0;
			for (unsigned d = 0; d < counting_shape.dim(); d++)
			{
				if (out._view[d] != 1) out_idx += counting_shape[d] * out._stride[d];
				if (t1._view[d] != 1) t1_idx += counting_shape[d] * t1._stride[d];
			}

			for (unsigned count = 0u; count < vector_len; count++)
			{
				out_data[out_idx] = ten0_data[out_idx] / t1_data[t1_idx];
				out_idx++, t1_idx += dt1;
			}

			if (!counting_shape.dim())
				break;

			counting_shape[-1]++;
			for (int d = last_long_dim - 1; d > 0; d--)
				if (counting_shape[d] >= t0_shape[d])
				{
					counting_shape[d] -= t0_shape[d];
					counting_shape[d - 1]++;
				}

			// If you reach the end of the leading dimension you're done.
			if (counting_shape[0] >= t0_shape[0])
				break;
		}
	}

	// If it was a gradient operation store a DivOp instance.
	if (requires_grad)
		out._internals->op = new DivOp(ten0, ten1, out);

	// Return out.
	return out;
}

Tensor operator+(const Tensor& ten, float val)
{
	// Scalar addition tensor operator for backpropagation.
	class ScaAddOp : public Tensor::TensorInternals::TensorOp
	{
		// Tensor copy storage for the summand.
		Tensor sum;

	public:
		// Constructor, stores all the data of the operation.
		ScaAddOp(const Tensor& _sum, const Tensor& _out) : TensorOp{ "Scalar Addition", _out },
			sum{ _sum }
		{
			_relatives[0] = &sum;
		}

		// Backpropagation. For scalar addition the gradient gets routed.
		void _backward() override
		{
			sum.internal_gradient() += out.gradient();
		}
	};

	// Tensor must be initialized.
	MACROGRAD_CHECK(ten.is_init(),
		"Trying to do scalar addition with an empty tensor."
	);

	// Create output with the same shape as tensor.
	Tensor out(ten.shape(), ten.device(), ten.has_grad());

	// Extract the data.
	float* out_data = out.internal_data();
	const float* ten_data = ten.internal_data();

	// Get element count.
	unsigned numel = ten.numel();

	// Now we actually add the scalar.
	if (out.is_gpu())
		kernel_ops::add_scalar(out_data, ten_data, &val, numel, false, 1.f);

	else for (unsigned i = 0u; i< numel; i++)
		out_data[i] = ten_data[i] + val;

	// If it was a gradient operation store a ScaAddOp instance.
	if (out.has_grad())
		out._internals->op = new ScaAddOp(ten, out);

	// Return out.
	return out;
}

Tensor operator-(const Tensor& ten, float val)
{
	return ten + (-val);
}

Tensor operator*(const Tensor& ten, float val)
{
	// Scalar multiplication tensor operator for backpropagation.
	class ScaMulOp : public Tensor::TensorInternals::TensorOp
	{
		// Tensor copy storage for the factor.
		Tensor fac;

		// Value storage.
		float val;

	public:
		// Constructor, stores all the data of the operation.
		ScaMulOp(const Tensor& _fac, const Tensor& _out, float _val) : TensorOp{ "Scalar Multiplication", _out },
			fac{ _fac }, val{ _val }
		{
			_relatives[0] = &fac;
		}

		// Backpropagation. For scalar multiplication the gradient gets scaled and routed.
		void _backward() override
		{
			fac.internal_gradient() += out.gradient() * val;
		}
	};

	// Tensor must be initialized.
	MACROGRAD_CHECK(ten.is_init(),
		"Trying to do scalar multiplication with an empty tensor."
	);

	// Create output with the same shape as tensor.
	Tensor out(ten.shape(), ten.device(), ten.has_grad());

	// Extract the data.
	float* out_data = out.internal_data();
	const float* ten_data = ten.internal_data();

	// Get element count.
	unsigned numel = ten.numel();

	// Now we actually multiply the scalar.
	if (out.is_gpu())
		kernel_ops::multiply_scalar(out_data, ten_data, &val, numel, false, 1.f);

	else for (unsigned i = 0u; i < numel; i++)
		out_data[i] = ten_data[i] * val;

	// If it was a gradient operation store a ScaMulOp instance.
	if (out.has_grad())
		out._internals->op = new ScaMulOp(ten, out, val);

	// Return out.
	return out;
}

Tensor operator/(const Tensor& ten, float val)
{
	return ten * (1.f/val);
}

Tensor operator+(float val, const Tensor& ten)
{
	return ten + val;
}

Tensor operator-(float val, const Tensor& ten)
{
	// Scalar subtraction tensor operator for backpropagation.
	class ScaSubOp : public Tensor::TensorInternals::TensorOp
	{
		// Tensor copy storage for the subtractor and output.
		Tensor sub;

	public:
		// Constructor, stores all the data of the operation.
		ScaSubOp(const Tensor& _sub, const Tensor& _out) : TensorOp{ "Scalar Subtraction", _out },
			sub{ _sub }
		{
			_relatives[0] = &sub;
		}

		// Backpropagation. For scalar subtraction the gradient gets negated and routed.
		void _backward() override
		{
			sub.internal_gradient() -= out.gradient();
		}
	};

	// Tensor must be initialized.
	MACROGRAD_CHECK(ten.is_init(),
		"Trying to do scalar subtraction with an empty tensor."
	);

	// Create output with the same shape as tensor.
	Tensor out(ten.shape(), ten.device(), ten.has_grad());

	// Extract the data.
	float* out_data = out.internal_data();
	const float* ten_data = ten.internal_data();

	// Get element count.
	unsigned numel = ten.numel();

	// Now we actually subtract from scalar.
	if (out.is_gpu())
		kernel_ops::subtract_from_scalar(out_data, ten_data, val, numel);

	else for (unsigned i = 0u; i < numel; i++)
		out_data[i] = val - ten_data[i];

	// If it was a gradient operation store a ScaSubOp instance.
	if (out.has_grad())
		out._internals->op = new ScaSubOp(ten, out);

	// Return out.
	return out;
}

Tensor operator*(float val, const Tensor& ten)
{
	return ten * val;
}

Tensor operator/(float val, const Tensor& ten)
{
	// Scalar division tensor operator for backpropagation.
	class ScaDivOp : public Tensor::TensorInternals::TensorOp
	{
		// Tensor copy storage for the divisor.
		Tensor div;

	public:
		// Constructor, stores all the data of the operation.
		ScaDivOp(const Tensor& _div, const Tensor& _out) : TensorOp{ "Scalar Division", _out },
			div{ _div }
		{
			_relatives[0] = &div;
		}

		// Backpropagation. For scalar division the gradient is 'dd = dout * (-val/d**2) = -dout * (out/d)'
		void _backward() override
		{
			div.internal_gradient() += out.gradient() * (out.no_grad() / div.no_grad());
		}
	};

	// Tensor must be initialized.
	MACROGRAD_CHECK(ten.is_init(),
		"Trying to do scalar division with an empty tensor."
	);

	// Create output with the same shape as tensor.
	Tensor out(ten.shape(), ten.device(), ten.has_grad());

	// Extract the data.
	float* out_data = out.internal_data();
	const float* ten_data = ten.internal_data();

	// Get element count.
	unsigned numel = ten.numel();

	// Now we actually divide from scalar.
	if (out.is_gpu())
		kernel_ops::divide_from_scalar(out_data, ten_data, val, numel);

	else for (unsigned i = 0u; i < numel; i++)
		out_data[i] = val / ten_data[i];

	// If it was a gradient operation store a ScaDivOp instance.
	if (out.has_grad())
		out._internals->op = new ScaDivOp(ten, out);

	// Return out.
	return out;
}

Tensor Tensor::operator-() const
{
	// Negation tensor operator for backpropagation.
	class NegOp : public Tensor::TensorInternals::TensorOp
	{
		// Tensor copy storage for the negator.
		Tensor neg;

	public:
		// Constructor, stores all the data of the operation.
		NegOp(const Tensor& _neg, const Tensor& _out) : TensorOp{ "Negation", _out },
			neg{ _neg }
		{
			_relatives[0] = &neg;
		}

		// Backpropagation. For negation the gradient gets negated and routed.
		void _backward() override
		{
			neg.internal_gradient() -= out.gradient();
		}
	};

	// Tensor must be initialized.
	MACROGRAD_CHECK(is_init(),
		"Trying to negate an empty tensor."
	);

	// Create output with the same shape as tensor.
	Tensor out(shape(), device(), has_grad());

	// Extract the data.
	float* out_data = out.internal_data();
	const float* ten_data = internal_data();

	// Get element count.
	unsigned numel = out.numel();

	// Now we actually negate the tensor.
	if (out.is_gpu())
	{
		float neg1 = -1.f;
		kernel_ops::multiply_scalar(out_data, ten_data, &neg1, numel, false, 1.f);
	}
	else for (unsigned i = 0u; i < numel; i++)
		out_data[i] = -ten_data[i];

	// If it was a gradient operation store a NegOp instance.
	if (out.has_grad())
		out._internals->op = new NegOp(*this, out);

	// Return out.
	return out;
}

/*
--------------------------------------------------------------------------------------------------------------------------
 Comparisson Operators
--------------------------------------------------------------------------------------------------------------------------
*/

Tensor operator<(const Tensor& ten0, const Tensor& ten1)
{
	// --- Sanity checks ---

	// Both tensor must be initialized.
	MACROGRAD_CHECK(ten0.is_init(),
		"Trying to compare two tensors while the first tensor is empty."
	);
	MACROGRAD_CHECK(ten1.is_init(),
		"Trying to compare two tensors while the second tensor is empty."
	);
	// Both tensors must be on the same device
	MACROGRAD_CHECK(ten0.is_gpu() == ten1.is_gpu(),
		"Trying to compare two tensors in different devices is not allowed."
	);
	// Ten0 must have more or equal the amount of dimensions of ten1 for broadcasting.
	MACROGRAD_CHECK(ten0.dim() >= ten1.dim(),
		"Trying to compare two tensors with incompatible dimensions.\n"
		"The dimensions of the tensors must match or the second tensor must cleanly broadcast to the first one.\n"
		"Found shapes | Tensor0: %s | Tensor1: %s", ten0._view.str(), ten1._view.str()
	);
	// Both tensors must either have the same dimension sizes or the second one have size one.
	bool has_dim = false;
	unsigned offset = ten0.dim() - ten1.dim();
	for (unsigned i = offset; i < ten0.dim(); i++)
	{
		unsigned j = i - offset;
		MACROGRAD_CHECK(ten0._view[i] == ten1._view[j] || (ten1._view[j] == 1 && !has_dim),
			"Trying to compare two tensors with incompatible shapes for broadcasting.\n"
			"Make sure shapes are compatible, meaning they have the same sizes or the second one is unitary.\n"
			"Found shapes | Tensor0: %s | Tensor1: %s", ten0._view.str(), ten1._view.str()
		);
		// When dimensions start matching they must keep matching.
		if (ten1._view[j] > 1) has_dim = true;
	}

	// Create output with the same shape as first tensor.
	Tensor out(ten0.shape(), ten0.device(), false);

	// Extract the data.
	float* out_data = out.internal_data();
	const float* ten0_data = ten0.internal_data();
	const float* ten1_data = ten1.internal_data();

	// Get counts.
	unsigned numel0 = ten0.numel();
	unsigned numel1 = ten1.numel();

	// Now we actually compare the tensors.
	if (out.is_gpu())
		kernel_ops::less_than(out_data, ten0_data, ten1_data, numel0, numel1);

	else for (unsigned idx = 0; idx < numel0; idx++)
		out_data[idx] = (ten0_data[idx] < ten1_data[idx % numel1]) ? 1.f : 0.f;

	// Return out.
	return out;
}

Tensor operator>(const Tensor& ten0, const Tensor& ten1)
{
	// --- Sanity checks ---

	// Both tensor must be initialized.
	MACROGRAD_CHECK(ten0.is_init(),
		"Trying to compare two tensors while the first tensor is empty."
	);
	MACROGRAD_CHECK(ten1.is_init(),
		"Trying to compare two tensors while the second tensor is empty."
	);
	// Both tensors must be on the same device
	MACROGRAD_CHECK(ten0.is_gpu() == ten1.is_gpu(),
		"Trying to compare two tensors in different devices is not allowed."
	);
	// Ten0 must have more or equal the amount of dimensions of ten1 for broadcasting.
	MACROGRAD_CHECK(ten0.dim() >= ten1.dim(),
		"Trying to compare two tensors with incompatible dimensions.\n"
		"The dimensions of the tensors must match or the second tensor must cleanly broadcast to the first one.\n"
		"Found shapes | Tensor0: %s | Tensor1: %s", ten0._view.str(), ten1._view.str()
	);
	// Both tensors must either have the same dimension sizes or the second one have size one.
	bool has_dim = false;
	unsigned offset = ten0.dim() - ten1.dim();
	for (unsigned i = offset; i < ten0.dim(); i++)
	{
		unsigned j = i - offset;
		MACROGRAD_CHECK(ten0._view[i] == ten1._view[j] || (ten1._view[j] == 1 && !has_dim),
			"Trying to compare two tensors with incompatible shapes for broadcasting.\n"
			"Make sure shapes are compatible, meaning they have the same sizes or the second one is unitary.\n"
			"Found shapes | Tensor0: %s | Tensor1: %s", ten0._view.str(), ten1._view.str()
		);
		// When dimensions start matching they must keep matching.
		if (ten1._view[j] > 1) has_dim = true;
	}

	// Create output with the same shape as first tensor.
	Tensor out(ten0.shape(), ten0.device(), false);

	// Extract the data.
	float* out_data = out.internal_data();
	const float* ten0_data = ten0.internal_data();
	const float* ten1_data = ten1.internal_data();

	// Get counts.
	unsigned numel0 = ten0.numel();
	unsigned numel1 = ten1.numel();

	// Now we actually compare the tensors.
	if (out.is_gpu())
		kernel_ops::more_than(out_data, ten0_data, ten1_data, numel0, numel1);

	else for (unsigned idx = 0; idx < numel0; idx++)
		out_data[idx] = (ten0_data[idx] > ten1_data[idx % numel1]) ? 1.f : 0.f;

	// Return out.
	return out;
}

Tensor operator<=(const Tensor& ten0, const Tensor& ten1)
{
	// --- Sanity checks ---

// Both tensor must be initialized.
	MACROGRAD_CHECK(ten0.is_init(),
		"Trying to compare two tensors while the first tensor is empty."
	);
	MACROGRAD_CHECK(ten1.is_init(),
		"Trying to compare two tensors while the second tensor is empty."
	);
	// Both tensors must be on the same device
	MACROGRAD_CHECK(ten0.is_gpu() == ten1.is_gpu(),
		"Trying to compare two tensors in different devices is not allowed."
	);
	// Ten0 must have more or equal the amount of dimensions of ten1 for broadcasting.
	MACROGRAD_CHECK(ten0.dim() >= ten1.dim(),
		"Trying to compare two tensors with incompatible dimensions.\n"
		"The dimensions of the tensors must match or the second tensor must cleanly broadcast to the first one.\n"
		"Found shapes | Tensor0: %s | Tensor1: %s", ten0._view.str(), ten1._view.str()
	);
	// Both tensors must either have the same dimension sizes or the second one have size one.
	bool has_dim = false;
	unsigned offset = ten0.dim() - ten1.dim();
	for (unsigned i = offset; i < ten0.dim(); i++)
	{
		unsigned j = i - offset;
		MACROGRAD_CHECK(ten0._view[i] == ten1._view[j] || (ten1._view[j] == 1 && !has_dim),
			"Trying to compare two tensors with incompatible shapes for broadcasting.\n"
			"Make sure shapes are compatible, meaning they have the same sizes or the second one is unitary.\n"
			"Found shapes | Tensor0: %s | Tensor1: %s", ten0._view.str(), ten1._view.str()
		);
		// When dimensions start matching they must keep matching.
		if (ten1._view[j] > 1) has_dim = true;
	}

	// Create output with the same shape as first tensor.
	Tensor out(ten0.shape(), ten0.device(), false);

	// Extract the data.
	float* out_data = out.internal_data();
	const float* ten0_data = ten0.internal_data();
	const float* ten1_data = ten1.internal_data();

	// Get counts.
	unsigned numel0 = ten0.numel();
	unsigned numel1 = ten1.numel();

	// Now we actually compare the tensors.
	if (out.is_gpu())
		kernel_ops::leq_than(out_data, ten0_data, ten1_data, numel0, numel1);

	else for (unsigned idx = 0; idx < numel0; idx++)
		out_data[idx] = (ten0_data[idx] <= ten1_data[idx % numel1]) ? 1.f : 0.f;

	// Return out.
	return out;
}

Tensor operator>=(const Tensor& ten0, const Tensor& ten1)
{
	// --- Sanity checks ---

// Both tensor must be initialized.
	MACROGRAD_CHECK(ten0.is_init(),
		"Trying to compare two tensors while the first tensor is empty."
	);
	MACROGRAD_CHECK(ten1.is_init(),
		"Trying to compare two tensors while the second tensor is empty."
	);
	// Both tensors must be on the same device
	MACROGRAD_CHECK(ten0.is_gpu() == ten1.is_gpu(),
		"Trying to compare two tensors in different devices is not allowed."
	);
	// Ten0 must have more or equal the amount of dimensions of ten1 for broadcasting.
	MACROGRAD_CHECK(ten0.dim() >= ten1.dim(),
		"Trying to compare two tensors with incompatible dimensions.\n"
		"The dimensions of the tensors must match or the second tensor must cleanly broadcast to the first one.\n"
		"Found shapes | Tensor0: %s | Tensor1: %s", ten0._view.str(), ten1._view.str()
	);
	// Both tensors must either have the same dimension sizes or the second one have size one.
	bool has_dim = false;
	unsigned offset = ten0.dim() - ten1.dim();
	for (unsigned i = offset; i < ten0.dim(); i++)
	{
		unsigned j = i - offset;
		MACROGRAD_CHECK(ten0._view[i] == ten1._view[j] || (ten1._view[j] == 1 && !has_dim),
			"Trying to compare two tensors with incompatible shapes for broadcasting.\n"
			"Make sure shapes are compatible, meaning they have the same sizes or the second one is unitary.\n"
			"Found shapes | Tensor0: %s | Tensor1: %s", ten0._view.str(), ten1._view.str()
		);
		// When dimensions start matching they must keep matching.
		if (ten1._view[j] > 1) has_dim = true;
	}

	// Create output with the same shape as first tensor.
	Tensor out(ten0.shape(), ten0.device(), false);

	// Extract the data.
	float* out_data = out.internal_data();
	const float* ten0_data = ten0.internal_data();
	const float* ten1_data = ten1.internal_data();

	// Get counts.
	unsigned numel0 = ten0.numel();
	unsigned numel1 = ten1.numel();

	// Now we actually compare the tensors.
	if (out.is_gpu())
		kernel_ops::meq_than(out_data, ten0_data, ten1_data, numel0, numel1);

	else for (unsigned idx = 0; idx < numel0; idx++)
		out_data[idx] = (ten0_data[idx] >= ten1_data[idx % numel1]) ? 1.f : 0.f;

	// Return out.
	return out;
}

Tensor operator==(const Tensor& ten0, const Tensor& ten1)
{
	// --- Sanity checks ---

// Both tensor must be initialized.
	MACROGRAD_CHECK(ten0.is_init(),
		"Trying to compare two tensors while the first tensor is empty."
	);
	MACROGRAD_CHECK(ten1.is_init(),
		"Trying to compare two tensors while the second tensor is empty."
	);
	// Both tensors must be on the same device
	MACROGRAD_CHECK(ten0.is_gpu() == ten1.is_gpu(),
		"Trying to compare two tensors in different devices is not allowed."
	);
	// Ten0 must have more or equal the amount of dimensions of ten1 for broadcasting.
	MACROGRAD_CHECK(ten0.dim() >= ten1.dim(),
		"Trying to compare two tensors with incompatible dimensions.\n"
		"The dimensions of the tensors must match or the second tensor must cleanly broadcast to the first one.\n"
		"Found shapes | Tensor0: %s | Tensor1: %s", ten0._view.str(), ten1._view.str()
	);
	// Both tensors must either have the same dimension sizes or the second one have size one.
	bool has_dim = false;
	unsigned offset = ten0.dim() - ten1.dim();
	for (unsigned i = offset; i < ten0.dim(); i++)
	{
		unsigned j = i - offset;
		MACROGRAD_CHECK(ten0._view[i] == ten1._view[j] || (ten1._view[j] == 1 && !has_dim),
			"Trying to compare two tensors with incompatible shapes for broadcasting.\n"
			"Make sure shapes are compatible, meaning they have the same sizes or the second one is unitary.\n"
			"Found shapes | Tensor0: %s | Tensor1: %s", ten0._view.str(), ten1._view.str()
		);
		// When dimensions start matching they must keep matching.
		if (ten1._view[j] > 1) has_dim = true;
	}

	// Create output with the same shape as first tensor.
	Tensor out(ten0.shape(), ten0.device(), false);

	// Extract the data.
	float* out_data = out.internal_data();
	const float* ten0_data = ten0.internal_data();
	const float* ten1_data = ten1.internal_data();

	// Get counts.
	unsigned numel0 = ten0.numel();
	unsigned numel1 = ten1.numel();

	// Now we actually compare the tensors.
	if (out.is_gpu())
		kernel_ops::eq_than(out_data, ten0_data, ten1_data, numel0, numel1);

	else for (unsigned idx = 0; idx < numel0; idx++)
		out_data[idx] = (ten0_data[idx] == ten1_data[idx % numel1]) ? 1.f : 0.f;

	// Return out.
	return out;
}

Tensor operator!=(const Tensor& ten0, const Tensor& ten1)
{
	// --- Sanity checks ---

// Both tensor must be initialized.
	MACROGRAD_CHECK(ten0.is_init(),
		"Trying to compare two tensors while the first tensor is empty."
	);
	MACROGRAD_CHECK(ten1.is_init(),
		"Trying to compare two tensors while the second tensor is empty."
	);
	// Both tensors must be on the same device
	MACROGRAD_CHECK(ten0.is_gpu() == ten1.is_gpu(),
		"Trying to compare two tensors in different devices is not allowed."
	);
	// Ten0 must have more or equal the amount of dimensions of ten1 for broadcasting.
	MACROGRAD_CHECK(ten0.dim() >= ten1.dim(),
		"Trying to compare two tensors with incompatible dimensions.\n"
		"The dimensions of the tensors must match or the second tensor must cleanly broadcast to the first one.\n"
		"Found shapes | Tensor0: %s | Tensor1: %s", ten0._view.str(), ten1._view.str()
	);
	// Both tensors must either have the same dimension sizes or the second one have size one.
	bool has_dim = false;
	unsigned offset = ten0.dim() - ten1.dim();
	for (unsigned i = offset; i < ten0.dim(); i++)
	{
		unsigned j = i - offset;
		MACROGRAD_CHECK(ten0._view[i] == ten1._view[j] || (ten1._view[j] == 1 && !has_dim),
			"Trying to compare two tensors with incompatible shapes for broadcasting.\n"
			"Make sure shapes are compatible, meaning they have the same sizes or the second one is unitary.\n"
			"Found shapes | Tensor0: %s | Tensor1: %s", ten0._view.str(), ten1._view.str()
		);
		// When dimensions start matching they must keep matching.
		if (ten1._view[j] > 1) has_dim = true;
	}

	// Create output with the same shape as first tensor.
	Tensor out(ten0.shape(), ten0.device(), false);

	// Extract the data.
	float* out_data = out.internal_data();
	const float* ten0_data = ten0.internal_data();
	const float* ten1_data = ten1.internal_data();

	// Get counts.
	unsigned numel0 = ten0.numel();
	unsigned numel1 = ten1.numel();

	// Now we actually compare the tensors.
	if (out.is_gpu())
		kernel_ops::neq_than(out_data, ten0_data, ten1_data, numel0, numel1);

	else for (unsigned idx = 0; idx < numel0; idx++)
		out_data[idx] = (ten0_data[idx] != ten1_data[idx % numel1]) ? 1.f : 0.f;

	// Return out.
	return out;
}

Tensor operator<(const Tensor& ten, float val)
{
	// Tensor must be initialized.
	MACROGRAD_CHECK(ten.is_init(),
		"Trying to do scalar comparisson with an empty tensor."
	);

	// Create output with the same shape as tensor.
	Tensor out(ten.shape(), ten.device(), false);

	// Extract the data.
	float* out_data = out.internal_data();
	const float* ten_data = ten.internal_data();

	// Get count.
	unsigned numel = ten.numel();

	// Now we actually multiply the tensors.
	if (out.is_gpu())
		kernel_ops::less_than_scalar(out_data, ten_data, val, numel);

	else for (unsigned idx = 0; idx < numel; idx++)
		out_data[idx] = (ten_data[idx] < val) ? 1.f : 0.f;

	// Return out.
	return out;
}

Tensor operator>(const Tensor& ten, float val)
{
	// Tensor must be initialized.
	MACROGRAD_CHECK(ten.is_init(),
		"Trying to do scalar comparisson with an empty tensor."
	);

	// Create output with the same shape as tensor.
	Tensor out(ten.shape(), ten.device(), false);

	// Extract the data.
	float* out_data = out.internal_data();
	const float* ten_data = ten.internal_data();

	// Get count.
	unsigned numel = ten.numel();

	// Now we actually multiply the tensors.
	if (out.is_gpu())
		kernel_ops::more_than_scalar(out_data, ten_data, val, numel);

	else for (unsigned idx = 0; idx < numel; idx++)
		out_data[idx] = (ten_data[idx] > val) ? 1.f : 0.f;

	// Return out.
	return out;
}

Tensor operator<=(const Tensor& ten, float val)
{
	// Tensor must be initialized.
	MACROGRAD_CHECK(ten.is_init(),
		"Trying to do scalar comparisson with an empty tensor."
	);

	// Create output with the same shape as tensor.
	Tensor out(ten.shape(), ten.device(), false);

	// Extract the data.
	float* out_data = out.internal_data();
	const float* ten_data = ten.internal_data();

	// Get count.
	unsigned numel = ten.numel();

	// Now we actually multiply the tensors.
	if (out.is_gpu())
		kernel_ops::leq_than_scalar(out_data, ten_data, val, numel);

	else for (unsigned idx = 0; idx < numel; idx++)
		out_data[idx] = (ten_data[idx] <= val) ? 1.f : 0.f;

	// Return out.
	return out;
}

Tensor operator>=(const Tensor& ten, float val)
{
	// Tensor must be initialized.
	MACROGRAD_CHECK(ten.is_init(),
		"Trying to do scalar comparisson with an empty tensor."
	);

	// Create output with the same shape as tensor.
	Tensor out(ten.shape(), ten.device(), false);

	// Extract the data.
	float* out_data = out.internal_data();
	const float* ten_data = ten.internal_data();

	// Get count.
	unsigned numel = ten.numel();

	// Now we actually multiply the tensors.
	if (out.is_gpu())
		kernel_ops::meq_than_scalar(out_data, ten_data, val, numel);

	else for (unsigned idx = 0; idx < numel; idx++)
		out_data[idx] = (ten_data[idx] >= val) ? 1.f : 0.f;

	// Return out.
	return out;
}

Tensor operator==(const Tensor& ten, float val)
{
	// Tensor must be initialized.
	MACROGRAD_CHECK(ten.is_init(),
		"Trying to do scalar comparisson with an empty tensor."
	);

	// Create output with the same shape as tensor.
	Tensor out(ten.shape(), ten.device(), false);

	// Extract the data.
	float* out_data = out.internal_data();
	const float* ten_data = ten.internal_data();

	// Get count.
	unsigned numel = ten.numel();

	// Now we actually multiply the tensors.
	if (out.is_gpu())
		kernel_ops::eq_than_scalar(out_data, ten_data, val, numel);

	else for (unsigned idx = 0; idx < numel; idx++)
		out_data[idx] = (ten_data[idx] == val) ? 1.f : 0.f;

	// Return out.
	return out;
}

Tensor operator!=(const Tensor& ten, float val)
{
	// Tensor must be initialized.
	MACROGRAD_CHECK(ten.is_init(),
		"Trying to do scalar comparisson with an empty tensor."
	);

	// Create output with the same shape as tensor.
	Tensor out(ten.shape(), ten.device(), false);

	// Extract the data.
	float* out_data = out.internal_data();
	const float* ten_data = ten.internal_data();

	// Get count.
	unsigned numel = ten.numel();

	// Now we actually multiply the tensors.
	if (out.is_gpu())
		kernel_ops::neq_than_scalar(out_data, ten_data, val, numel);

	else for (unsigned idx = 0; idx < numel; idx++)
		out_data[idx] = (ten_data[idx] != val) ? 1.f : 0.f;

	// Return out.
	return out;
}

Tensor operator<(float val, const Tensor& ten)
{
	return (ten > val);
}

Tensor operator>(float val, const Tensor& ten)
{
	return (ten < val);
}

Tensor operator<=(float val, const Tensor& ten)
{
	return (ten >= val);
}

Tensor operator>=(float val, const Tensor& ten)
{
	return (ten <= val);
}

Tensor operator==(float val, const Tensor& ten)
{
	return (ten == val);
}

Tensor operator!=(float val, const Tensor& ten)
{
	return (ten != val);
}


/*
--------------------------------------------------------------------------------------------------------------------------
 In-Place Operators
--------------------------------------------------------------------------------------------------------------------------
*/

Tensor& Tensor::operator+=(const Tensor& other)
{
	// If there is gradient the middle man must exist. Maintain gradient.
	if (has_grad())
	{
		Tensor out = *this + other;
		out.internal_gradient() = gradient().internal_copy(false, false);
		return *this = out;
	}
	// If there is another instance make sure to not modify it.
	if (_internals && _internals->instances > 1)
		return *this = *this + other.no_grad();

	// --- Sanity checks ---

	// Both tensors must be initialized.
	MACROGRAD_CHECK(is_init(),
		"Trying to add two tensors while the first tensor is empty."
	);
	MACROGRAD_CHECK(other.is_init(),
		"Trying to add two tensors while the second tensor is empty."
	);
	// Both tensors must be on the same device
	MACROGRAD_CHECK(is_gpu() == other.is_gpu(),
		"Trying to add two tensors in different devices is not allowed."
	);
	// Ten0 must have more or equal the amount of dimensions of other for broadcasting.
	MACROGRAD_CHECK(dim() >= other.dim(),
		"Trying to add two tensors with incompatible dimensions.\n"
		"The dimensions of the tensors must match or the second tensor must broadcast to the first one.\n"
		"Found shapes | This: %s | Other: %s", _view.str(), other._view.str()
	);
	// Both tensors must either have the same dimension sizes or one of them have size one.
	unsigned offset = dim() - other.dim();
	for (unsigned i = offset; i < dim(); i++)
		MACROGRAD_CHECK(_view[i] == other._view[i - offset] || other._view[i - offset] == 1 || _view[i] == 1,
			"Trying to add two tensors with incompatible shapes for broadcasting.\n"
			"Make sure shapes are compatible, meaning they have the same sizes or one of them is unitary.\n"
			"Found shapes | This: %s | Other: %s", _view.str(), other._view.str()
		);

	// Extract the data.
	float* out_data = internal_data();
	const float* ten_data = other.internal_data();

	// Now we actually add the tensors.
	if (is_gpu())
		kernel_ops::shaped_add(out_data, out_data, ten_data, _view, other._view);
	else
	{
		// Unsqueeze and sum dimensions on other if necessary.
		Tensor t1 = other.no_grad();
		while (dim() > t1.dim()) t1 = t1.unsqueeze(0);
		for (unsigned i = 0; i < dim(); i++)
			if (_view[i] == 1 && t1._view[i] != 1)
				t1 = t1.sum(i, true);

		// Extract data.
		const float* t1_data = t1.internal_data();

		// Find the last non-unitary dimension to iterate through.
		unsigned last_long_dim = 0;
		for (unsigned i = 0; i < _view.dim(); i++)
			if (_view[i] > 1)
				last_long_dim = i;
		// Find the stride, will be 0 if broadcasting on last dim.
		const unsigned dt1 = (t1._view[last_long_dim] != 1) ? 1 : 0;
		// Find the vector length to iterate.
		const unsigned vector_len = _view[last_long_dim];
		// Create a running shape to count.
		Shape counting_shape(last_long_dim, (int*)nullptr);
		// Iterate through vectors.
		while (true)
		{
			unsigned out_idx = 0, t1_idx = 0;
			for (unsigned d = 0; d < counting_shape.dim(); d++)
			{
				if (_view[d] != 1) out_idx += counting_shape[d] * _stride[d];
				if (t1._view[d] != 1) t1_idx += counting_shape[d] * t1._stride[d];
			}

			for (unsigned count = 0u; count < vector_len; count++)
			{
				out_data[out_idx] += t1_data[t1_idx];
				out_idx++, t1_idx += dt1;
			}

			if (!counting_shape.dim())
				break;

			counting_shape[-1]++;
			for (int d = last_long_dim - 1; d > 0; d--)
				if (counting_shape[d] >= _view[d])
				{
					counting_shape[d] -= _view[d];
					counting_shape[d - 1]++;
				}

			// If you reach the end of the leading dimension you're done.
			if (counting_shape[0] >= _view[0])
				break;
		}
	}

	// Return self.
	return *this;
}

Tensor& Tensor::operator-=(const Tensor& other)
{
	// If there is gradient the middle man must exist. Maintain gradient.
	if (has_grad())
	{
		Tensor out = *this - other;
		out.internal_gradient() = gradient().internal_copy(false, false);
		return *this = out;
	}
	// If there is another instance make sure to not modify it.
	if (_internals && _internals->instances > 1)
		return *this = *this - other.no_grad();

	// --- Sanity checks ---

	// Both tensors must be initialized.
	MACROGRAD_CHECK(is_init(),
		"Trying to subtract two tensors while the first tensor is empty."
	);
	MACROGRAD_CHECK(other.is_init(),
		"Trying to subtract two tensors while the second tensor is empty."
	);
	// Both tensors must be on the same device
	MACROGRAD_CHECK(is_gpu() == other.is_gpu(),
		"Trying to subtract two tensors in different devices is not allowed."
	);
	// Ten0 must have more or equal the amount of dimensions of other for broadcasting.
	MACROGRAD_CHECK(dim() >= other.dim(),
		"Trying to subtract two tensors with incompatible dimensions.\n"
		"The dimensions of the tensors must match or the second tensor must broadcast to the first one.\n"
		"Found shapes | This: %s | Other: %s", _view.str(), other._view.str()
	);
	// Both tensors must either have the same dimension sizes or one of them have size one.
	unsigned offset = dim() - other.dim();
	for (unsigned i = offset; i < dim(); i++)
		MACROGRAD_CHECK(_view[i] == other._view[i - offset] || other._view[i - offset] == 1 || _view[i] == 1,
			"Trying to subtract two tensors with incompatible shapes for broadcasting.\n"
			"Make sure shapes are compatible, meaning they have the same sizes or one of them is unitary.\n"
			"Found shapes | This: %s | Other: %s", _view.str(), other._view.str()
		);

	// Extract the data.
	float* out_data = internal_data();
	const float* ten_data = other.internal_data();

	// Now we actually subtract the tensors.
	if (is_gpu())
		kernel_ops::shaped_subtract(out_data, out_data, ten_data, _view, other._view);
	else
	{
		// Unsqueeze and sum dimensions on other if necessary.
		Tensor t1 = other.no_grad();
		while (dim() > t1.dim()) t1 = t1.unsqueeze(0);
		for (unsigned i = 0; i < dim(); i++)
			if (_view[i] == 1 && t1._view[i] != 1)
				t1 = t1.sum(i, true);

		// Extract data.
		const float* t1_data = t1.internal_data();

		// Find the last non-unitary dimension to iterate through.
		unsigned last_long_dim = 0;
		for (unsigned i = 0; i < _view.dim(); i++)
			if (_view[i] > 1)
				last_long_dim = i;
		// Find the stride, will be 0 if broadcasting on last dim.
		const unsigned dt1 = (t1._view[last_long_dim] != 1) ? 1 : 0;
		// Find the vector length to iterate.
		const unsigned vector_len = _view[last_long_dim];
		// Create a running shape to count.
		Shape counting_shape(last_long_dim, (int*)nullptr);
		// Iterate through vectors.
		while (true)
		{
			unsigned out_idx = 0, t1_idx = 0;
			for (unsigned d = 0; d < counting_shape.dim(); d++)
			{
				if (_view[d] != 1) out_idx += counting_shape[d] * _stride[d];
				if (t1._view[d] != 1) t1_idx += counting_shape[d] * t1._stride[d];
			}

			for (unsigned count = 0u; count < vector_len; count++)
			{
				out_data[out_idx] -= t1_data[t1_idx];
				out_idx++, t1_idx += dt1;
			}

			if (!counting_shape.dim())
				break;

			counting_shape[-1]++;
			for (int d = last_long_dim - 1; d > 0; d--)
				if (counting_shape[d] >= _view[d])
				{
					counting_shape[d] -= _view[d];
					counting_shape[d - 1]++;
				}

			// If you reach the end of the leading dimension you're done.
			if (counting_shape[0] >= _view[0])
				break;
		}
	}

	// Return self.
	return *this;
}

Tensor& Tensor::operator*=(const Tensor& other)
{
	// If there is gradient the middle man must exist. Maintain gradient.
	if (has_grad())
	{
		Tensor out = *this * other;
		out.internal_gradient() = gradient().internal_copy(false, false);
		return *this = out;
	}
	// If there is another instance make sure to not modify it.
	if (_internals && _internals->instances > 1)
		return *this = *this * other.no_grad();

	// --- Sanity checks ---

	// Both tensors must be initialized.
	MACROGRAD_CHECK(is_init(),
		"Trying to multiply two tensors while the first tensor is empty."
	);
	MACROGRAD_CHECK(other.is_init(),
		"Trying to multiply two tensors while the second tensor is empty."
	);
	// Both tensors must be on the same device
	MACROGRAD_CHECK(is_gpu() == other.is_gpu(),
		"Trying to multiply two tensors in different devices is not allowed."
	);
	// Ten0 must have more or equal the amount of dimensions of other for broadcasting.
	MACROGRAD_CHECK(dim() >= other.dim(),
		"Trying to multiply two tensors with incompatible dimensions.\n"
		"The dimensions of the tensors must match or the second tensor must broadcast to the first one.\n"
		"Found shapes | This: %s | Other: %s", _view.str(), other._view.str()
	);
	// Both tensors must either have the same dimension sizes or one of them have size one.
	unsigned offset = dim() - other.dim();
	for (unsigned i = offset; i < dim(); i++)
		MACROGRAD_CHECK(_view[i] == other._view[i - offset] || other._view[i - offset] == 1 || _view[i] == 1,
			"Trying to multiply two tensors with incompatible shapes for broadcasting.\n"
			"Make sure shapes are compatible, meaning they have the same sizes or one of them is unitary.\n"
			"Found shapes | This: %s | Other: %s", _view.str(), other._view.str()
		);

	// Extract the data.
	float* out_data = internal_data();
	const float* ten_data = other.internal_data();

	// Now we actually multiply the tensors.
	if (is_gpu())
		kernel_ops::shaped_multiply(out_data, out_data, ten_data, _view, other._view);
	else
	{
		// Unsqueeze and sum dimensions on other if necessary.
		Tensor t1 = other.no_grad();
		while (dim() > t1.dim()) t1 = t1.unsqueeze(0);
		for (unsigned i = 0; i < dim(); i++)
			if (_view[i] == 1 && t1._view[i] != 1)
				t1 = t1.sum(i, true);

		// Extract data.
		const float* t1_data = t1.internal_data();

		// Find the last non-unitary dimension to iterate through.
		unsigned last_long_dim = 0;
		for (unsigned i = 0; i < _view.dim(); i++)
			if (_view[i] > 1)
				last_long_dim = i;
		// Find the stride, will be 0 if broadcasting on last dim.
		const unsigned dt1 = (t1._view[last_long_dim] != 1) ? 1 : 0;
		// Find the vector length to iterate.
		const unsigned vector_len = _view[last_long_dim];
		// Create a running shape to count.
		Shape counting_shape(last_long_dim, (int*)nullptr);
		// Iterate through vectors.
		while (true)
		{
			unsigned out_idx = 0, t1_idx = 0;
			for (unsigned d = 0; d < counting_shape.dim(); d++)
			{
				if (_view[d] != 1) out_idx += counting_shape[d] * _stride[d];
				if (t1._view[d] != 1) t1_idx += counting_shape[d] * t1._stride[d];
			}

			for (unsigned count = 0u; count < vector_len; count++)
			{
				out_data[out_idx] *= t1_data[t1_idx];
				out_idx++, t1_idx += dt1;
			}

			if (!counting_shape.dim())
				break;

			counting_shape[-1]++;
			for (int d = last_long_dim - 1; d > 0; d--)
				if (counting_shape[d] >= _view[d])
				{
					counting_shape[d] -= _view[d];
					counting_shape[d - 1]++;
				}

			// If you reach the end of the leading dimension you're done.
			if (counting_shape[0] >= _view[0])
				break;
		}
	}

	// Return self.
	return *this;
}

Tensor& Tensor::operator/=(const Tensor& other)
{
	// If there is gradient the middle man must exist. Maintain gradient.
	if (has_grad())
	{
		Tensor out = *this / other;
		out.internal_gradient() = gradient().internal_copy(false, false);
		return *this = out;
	}
	// If there is another instance make sure to not modify it.
	if (_internals && _internals->instances > 1)
		return *this = *this / other.no_grad();

	// --- Sanity checks ---

	// Both tensors must be initialized.
	MACROGRAD_CHECK(is_init(),
		"Trying to divide two tensors while the first tensor is empty."
	);
	MACROGRAD_CHECK(other.is_init(),
		"Trying to divide two tensors while the second tensor is empty."
	);
	// Both tensors must be on the same device
	MACROGRAD_CHECK(is_gpu() == other.is_gpu(),
		"Trying to divide two tensors in different devices is not allowed."
	);
	// Ten0 must have more or equal the amount of dimensions of other for broadcasting.
	MACROGRAD_CHECK(dim() >= other.dim(),
		"Trying to divide two tensors with incompatible dimensions.\n"
		"The dimensions of the tensors must match or the second tensor must broadcast to the first one.\n"
		"Found shapes | This: %s | Other: %s", _view.str(), other._view.str()
	);
	// Both tensors must either have the same dimension sizes or one of them have size one.
	unsigned offset = dim() - other.dim();
	for (unsigned i = offset; i < dim(); i++)
		MACROGRAD_CHECK(_view[i] == other._view[i - offset] || other._view[i - offset] == 1 || _view[i] == 1,
			"Trying to divide two tensors with incompatible shapes for broadcasting.\n"
			"Make sure shapes are compatible, meaning they have the same sizes or one of them is unitary.\n"
			"Found shapes | This: %s | Other: %s", _view.str(), other._view.str()
		);

	// Extract the data.
	float* out_data = internal_data();
	const float* ten_data = other.internal_data();

	// Now we actually divide the tensors.
	if (is_gpu())
		kernel_ops::shaped_divide(out_data, out_data, ten_data, _view, other._view);
	else
	{
		// Unsqueeze and sum dimensions on other if necessary.
		Tensor t1 = other.no_grad();
		while (dim() > t1.dim()) t1 = t1.unsqueeze(0);
		for (unsigned i = 0; i < dim(); i++)
			if (_view[i] == 1 && t1._view[i] != 1)
				t1 = t1.sum(i, true);

		// Extract data.
		const float* t1_data = t1.internal_data();

		// Find the last non-unitary dimension to iterate through.
		unsigned last_long_dim = 0;
		for (unsigned i = 0; i < _view.dim(); i++)
			if (_view[i] > 1)
				last_long_dim = i;
		// Find the stride, will be 0 if broadcasting on last dim.
		const unsigned dt1 = (t1._view[last_long_dim] != 1) ? 1 : 0;
		// Find the vector length to iterate.
		const unsigned vector_len = _view[last_long_dim];
		// Create a running shape to count.
		Shape counting_shape(last_long_dim, (int*)nullptr);
		// Iterate through vectors.
		while (true)
		{
			unsigned out_idx = 0, t1_idx = 0;
			for (unsigned d = 0; d < counting_shape.dim(); d++)
			{
				if (_view[d] != 1) out_idx += counting_shape[d] * _stride[d];
				if (t1._view[d] != 1) t1_idx += counting_shape[d] * t1._stride[d];
			}

			for (unsigned count = 0u; count < vector_len; count++)
			{
				out_data[out_idx] /= t1_data[t1_idx];
				out_idx++, t1_idx += dt1;
			}

			if (!counting_shape.dim())
				break;

			counting_shape[-1]++;
			for (int d = last_long_dim - 1; d > 0; d--)
				if (counting_shape[d] >= _view[d])
				{
					counting_shape[d] -= _view[d];
					counting_shape[d - 1]++;
				}

			// If you reach the end of the leading dimension you're done.
			if (counting_shape[0] >= _view[0])
				break;
		}
	}

	// Return self.
	return *this;
}

Tensor& Tensor::operator+=(float val)
{
	// If there is another instance middle man must exist.
	// Gradient does not matter since the derivative is the same.
	// And if someone had used this tensor in a gradient operation there would be more instances.
	if (_internals && _internals->instances > 1)
	{
		Tensor out = *this + val;
		if (has_grad())
			out.internal_gradient() = gradient().internal_copy(false, false);
		return *this = out;
	}

	// Tensor must be initialized.
	MACROGRAD_CHECK(is_init(),
		"Trying to add two tensors while the first tensor is empty."
	);

	// Extract the data.
	float* out_data = internal_data();

	// Get number of elements.
	unsigned _numel = numel();

	// Now we actually sum the tensor.
	if (is_gpu())
		kernel_ops::add_scalar(out_data, out_data, &val, _numel, false, 1.f);

	else for (unsigned i = 0u; i < _numel; i++)
		out_data[i] += val;

	// Return self.
	return *this;
}

Tensor& Tensor::operator-=(float val)
{
	return *this += -val;
}

Tensor& Tensor::operator*=(float val)
{
	// If there is gradient the middle man must exist.
	if (has_grad() || (_internals && _internals->instances > 1))
	{
		Tensor out = *this * val;
		if (has_grad())
			out.internal_gradient() = gradient().internal_copy(false, false);
		return *this = out;
	}

	// Tensor must be initialized.
	MACROGRAD_CHECK(is_init(),
		"Trying to multiply by a scalar on an empty tensor."
	);

	// Extract the data.
	float* out_data = internal_data();

	// Get number of elements.
	unsigned _numel = numel();

	// Now we actually multiply the tensor.
	if (is_gpu())
		kernel_ops::multiply_scalar(out_data, out_data, &val, _numel, false, 1.f);

	else for (unsigned i = 0u; i < _numel; i++)
		out_data[i] *= val;

	// Return self.
	return *this;
}

Tensor& Tensor::operator/=(float val)
{
	return *this *= 1.f / val;
}

/*
--------------------------------------------------------------------------------------------------------------------------
 Functional Namespace Operators
--------------------------------------------------------------------------------------------------------------------------
*/

Tensor Functional::matmul(const Tensor& mat0, const Tensor& mat1, bool transA, bool transB)
{
	// Matrix multiplication tensor operator for backpropagation.
	class MatMulOp : public Tensor::TensorInternals::TensorOp
	{
		// Tensor copy storage for the input matrices.
		Tensor mat0, mat1;

		// Storage for transposition information.
		bool transA, transB;

	public:
		// Constructor, stores all the data of the operation.
		MatMulOp(const Tensor& _mat0, const Tensor& _mat1, const Tensor& _out, bool _transA, bool _transB) : TensorOp{ "Matrix Multiplication", _out },
			mat0{ _mat0 }, mat1{ _mat1 }, transA{ _transA }, transB{ _transB }
		{
			_relatives[0] = &mat0;
			_relatives[1] = &mat1;
		}

		// Backpropagation. For matmul the gradient gets multiplied by the transposed other.
		// The broadcasting logic of matmul and the '+=' operator handles shapes.
		void _backward() override
		{
			if (mat0.has_grad())
			{
				if (transA)	mat0.internal_gradient() += matmul(mat1.no_grad(), out.gradient(),  transB,  true);
				else		mat0.internal_gradient() += matmul(out.gradient(), mat1.no_grad(), false, !transB);

			}
			if (mat1.has_grad())
			{
				if (transB) mat1.internal_gradient() += matmul(out.gradient(), mat0.no_grad(),  true,  transA);
				else		mat1.internal_gradient() += matmul(mat0.no_grad(), out.gradient(), !transA, false);
			}
		}
	};

	// Both tensor must be initialized.
	MACROGRAD_CHECK(mat0.is_init(),
		"Trying to matrix multiply two tensors while the first tensor is empty."
	);
	MACROGRAD_CHECK(mat1.is_init(),
		"Trying to matrix multiply two tensors while the second tensor is empty."
	);
	// Both tensors must be on the same device
	MACROGRAD_CHECK(mat0.is_gpu() == mat1.is_gpu(),
		"Trying to matrix multiply two tensors in different devices is not allowed."
	);
	MACROGRAD_CHECK(!transA || mat0.dim() > 1,
		"Addint a transposition to a single dimensional tensor for a matmul call is not allowed.\n"
		"Make sure your tensor has at least 2 dimensions if you set 'transA' as true."
	);
	MACROGRAD_CHECK(!transB || mat1.dim() > 1,
		"Addint a transposition to a single dimensional tensor for a matmul call is not allowed.\n"
		"Make sure your tensor has at least 2 dimensions if you set 'transB' as true."
	);

	// Gradiend data.
	bool requires_grad = mat0.has_grad() || mat1.has_grad();

	// Copy the tensors and let's reshape them.
	Tensor A = mat0;
	Tensor B = mat1;

	// Make sure we have at least 2D.
	bool a_was_1d = (mat0.dim() == 1);
	bool b_was_1d = (mat1.dim() == 1);
	if (a_was_1d) A = A.unsqueeze(0);
	if (b_was_1d) B = B.unsqueeze(-1);

	// Make sure both of them have the same number of dimensions.
	while (A.dim() > B.dim())
		B = B.unsqueeze(0);
	while (B.dim() > A.dim())
		A = A.unsqueeze(0);

	// Make sure dimensions are compatible.
	MACROGRAD_CHECK(A._view[transA ? -2 : -1] == B._view[transB ? -1 : -2],
		"Incompatible dimensions found inside a matmul() call.\n"
		"Please make sure your tensors follow proper matrix multiplication logic (...,M,K) @ (...,K,N) = (...,M,N).\n"
		"Matrix0 shape: %s | Matrix1 shape: %s", mat0._view.str(), mat1._view.str()
	);

	// Make sure other dimensions are broadcastable.
	for (unsigned i = 0; i < A.dim() - 2; i++)
		MACROGRAD_CHECK(A._view[i] == B._view[i] || A._view[i] == 1 || B._view[i] == 1,
			"Trying to matrix multiply two tensors with incompatible shapes.\n"
			"Make sure leading shapes are compatible, meaning they have the same sizes or one of them is unitary for broadcasting.\n"
			"Found shapes | Unsqueezed Matrix0: %s | Unsqueezed Matrix1: %s", A._view.str(), B._view.str()
		);

	// Prepare output shape.
	Shape out_shape(A.dim(), (int*)nullptr);
	out_shape[-1] = transB ? B._view[-2] : B._view[-1];
	out_shape[-2] = transA ? A._view[-1] : A._view[-2];
	for (unsigned i = 0; i < A.dim() - 2; i++)
		out_shape[i] = A._view[i] > B._view[i] ? A._view[i] : B._view[i];

	// Create output tensor with this shape.
	Tensor out(out_shape, A.device(), requires_grad);

	// Get matrix multiplication data.
	unsigned M = out_shape[-2];
	unsigned N = out_shape[-1];
	unsigned K = transA ? A._view[-2] : A._view[-1];

	// Extract the data.
	float* out_data = out.internal_data();
	const float* A_data = A.internal_data();
	const float* B_data = B.internal_data();

	// Now we actually multiply the matrices.
	if (out.is_gpu())
		kernel_ops::matmul(out_data, A_data, B_data, out_shape, A._view, B._view, transA, transB);
	else
	{
		// Create a running shape to count.
		Shape counting_shape(out_shape.dim() > 2 ? out_shape.dim() - 2 : 1, (int*)nullptr);
		// Create reference shape.
		Shape reference = out_shape;
		if (reference.dim() > 2)
		{
			reference.remove(-1);
			reference.remove(-1);
		}
		else reference = Shape(1);

		// Important data for transposition.
		const unsigned A_cols = A._view[-1];
		const unsigned B_cols = B._view[-1];

		// Iterate through vectors.
		while (true)
		{
			// Get initial pointer idxs given running shape.
			unsigned out_idx = 0, A_idx = 0, B_idx = 0;
			for (unsigned d = 0; d < counting_shape.dim(); d++)
			{
				if (out._view[d] != 1) out_idx += counting_shape[d] * out._stride[d];
				if (A._view[d]   != 1) A_idx   += counting_shape[d] *   A._stride[d];
				if (B._view[d]   != 1) B_idx   += counting_shape[d] *   B._stride[d];
			}
			// Get matrix pointers.
			float* out_matrix = out_data + out_idx; // M x N values
			const float* A_matrix = A_data + A_idx;		// M x K values
			const float* B_matrix = B_data + B_idx;		// K x N values

			// Here we actually multiply the matrices.
			{
				for (unsigned i0 = 0; i0 < M; i0 += 64)
				{
					const unsigned i1 = (i0 + 64 < M) ? (i0 + 64) : M;

					for (unsigned j0 = 0; j0 < N; j0 += 64)
					{
						const unsigned j1 = (j0 + 64 < N) ? (j0 + 64) : N;

						for (unsigned k0 = 0; k0 < K; k0 += 64)
						{
							const unsigned k1 = (k0 + 64 < K) ? (k0 + 64) : K;

							for (unsigned i = i0; i < i1; ++i)
							{
								float* c_row = out_matrix + i * N;

								for (unsigned k = k0; k < k1; ++k)
								{
									const float a = transA
										? A_matrix[k * A_cols + i]
										: A_matrix[i * A_cols + k];

									for (unsigned j = j0; j < j1; ++j)
									{
										const float b = transB
											? B_matrix[j * B_cols + k]
											: B_matrix[k * B_cols + j];

										c_row[j] += a * b;
									}
								}
							}
						}
					}
				}
			}

			// Add one to the shape count.
			counting_shape[-1]++;
			for (int d = counting_shape.dim() - 1; d > 0; d--)
				if (counting_shape[d] >= reference[d])
				{
					counting_shape[d] -= reference[d];
					counting_shape[d - 1]++;
				}

			// If you reach the end of the leading dimension you're done.
			if (counting_shape[0] >= reference[0])
				break;
		}
	}

	// If it was a gradient operation store a MatMulOp instance.
	if (requires_grad)
		out._internals->op = new MatMulOp(A, B, out, transA, transB);

	// Squeeze back if necessary.
	if (a_was_1d) out = out.squeeze(-2);
	if (b_was_1d) out = out.squeeze(-1);

	// Return out.
	return out;
}

Tensor Functional::matmul(const Tensor& mat0, const Tensor& mat1, const Tensor& bias, bool transA, bool transB)
{
	// Matrix multiplication with bias tensor operator for backpropagation.
	class MatMulBiasOp : public Tensor::TensorInternals::TensorOp
	{
		// Tensor copy storage for the input matrices.
		Tensor mat0, mat1, bias;

		// Storage for transposition information.
		bool transA, transB;

	public:
		// Constructor, stores all the data of the operation.
		MatMulBiasOp(const Tensor& _mat0, const Tensor& _mat1, const Tensor& _bias, const Tensor& _out, bool _transA, bool _transB) : TensorOp{ "Matrix Multiplication with Bias", _out },
			mat0{ _mat0 }, mat1{ _mat1 }, bias{ _bias }, transA{ _transA }, transB{ _transB }
		{
			if (mat0.has_grad()) _relatives[0] = &mat0;
			if (mat1.has_grad()) _relatives[1] = &mat1;
			if (bias.has_grad()) _relatives[2] = &bias;
		}

		// Backpropagation. For matmul the gradient gets multiplied by the transposed other.
		// The broadcasting logic of matmul and the '+=' operator handles shapes.
		void _backward() override
		{
			if (mat0.has_grad())
			{
				if (transA)	mat0.internal_gradient() += matmul(mat1.no_grad(), out.gradient(),  transB,  true);
				else		mat0.internal_gradient() += matmul(out.gradient(), mat1.no_grad(), false, !transB);

			}
			if (mat1.has_grad())
			{
				if (transB) mat1.internal_gradient() += matmul(out.gradient(), mat0.no_grad(),  true,  transA);
				else		mat1.internal_gradient() += matmul(mat0.no_grad(), out.gradient(), !transA, false);
			}
			if (bias.has_grad()) bias.internal_gradient() += out.gradient();
		}
	};

	// Both tensor must be initialized.
	MACROGRAD_CHECK(mat0.is_init(),
		"Trying to matrix multiply with bias tensors while the first tensor is empty."
	);
	MACROGRAD_CHECK(mat1.is_init(),
		"Trying to matrix multiply with bias tensors while the second tensor is empty."
	);
	MACROGRAD_CHECK(bias.is_init(),
		"Trying to matrix multiply with bias tensors while the bias tensor is empty."
	);
	// Both tensors must be on the same device
	MACROGRAD_CHECK(mat0.is_gpu() == mat1.is_gpu() && mat1.is_gpu() == bias.is_gpu(),
		"Trying to matrix multiply with bias tensors in different devices is not allowed."
	);
	MACROGRAD_CHECK(!transA || mat0.dim() > 1,
		"Addint a transposition to a single dimensional tensor for a matmul call is not allowed.\n"
		"Make sure your tensor has at least 2 dimensions if you set 'transA' as true."
	);
	MACROGRAD_CHECK(!transB || mat1.dim() > 1,
		"Addint a transposition to a single dimensional tensor for a matmul call is not allowed.\n"
		"Make sure your tensor has at least 2 dimensions if you set 'transB' as true."
	);

	// Gradiend data.
	bool requires_grad = mat0.has_grad() || mat1.has_grad() || bias.has_grad();

	// Copy the tensors and let's reshape them.
	Tensor A = mat0;
	Tensor B = mat1;
	Tensor b = bias;

	// Make sure we have at least 2D.
	bool A_was_1d = (mat0.dim() == 1);
	bool B_was_1d = (mat1.dim() == 1);
	if (A_was_1d) A = A.unsqueeze(0);
	if (B_was_1d) B = B.unsqueeze(-1);

	// Make sure both of them have the same number of dimensions.
	while (A.dim() > B.dim())
		B = B.unsqueeze(0);
	while (B.dim() > A.dim())
		A = A.unsqueeze(0);
	while (A.dim() > b.dim())
		b = b.unsqueeze(0);

	// Make sure dimensions are compatible.
	MACROGRAD_CHECK(A._view[transA ? -2 : -1] == B._view[transB ? -1 : -2],
		"Incompatible dimensions found inside a matmul() call.\n"
		"Please make sure your tensors follow proper matrix multiplication logic (...,M,K) @ (...,K,N) = (...,M,N).\n"
		"Matrix0 shape: %s | Matrix1 shape: %s", mat0._view.str(), mat1._view.str()
	);

	// Make sure other dimensions are broadcastable.
	for (unsigned i = 0; i < A.dim() - 2; i++)
		MACROGRAD_CHECK(A._view[i] == B._view[i] || A._view[i] == 1 || B._view[i] == 1,
			"Trying to matrix multiply two tensors with incompatible shapes.\n"
			"Make sure leading shapes are compatible, meaning they have the same sizes or one of them is unitary for broadcasting.\n"
			"Found shapes | Unsqueezed Matrix0: %s | Unsqueezed Matrix1: %s", A._view.str(), B._view.str()
		);

	// Prepare output shape.
	Shape out_shape(A.dim(), (int*)nullptr);
	out_shape[-1] = transB ? B._view[-2] : B._view[-1];
	out_shape[-2] = transA ? A._view[-1] : A._view[-2];
	for (unsigned i = 0; i < A.dim() - 2; i++)
		out_shape[i] = A._view[i] > B._view[i] ? A._view[i] : B._view[i];

	// Make sure bias can be broadcasted to output.
	MACROGRAD_CHECK(b.dim() <= out_shape.dim(),
		"Trying to add a bias to a matmul tensor with more dimensions than the matmul output.\n"
		"Make sure the bias is broadcastable to the matmul output, bias dimensionality must be less or equal.\n"
		"Found shapes | Matmul output: %s | Bias: %s", out_shape.str(), b._view.str()
	);
	for (unsigned i = 0; i < out_shape.dim(); i++)
		MACROGRAD_CHECK(out_shape[i] == b._view[i] || b._view[i] == 1,
			"Trying to add a bias to a matmul tensor with incompatible shapes.\n"
			"Make sure leading shapes are compatible, meaning they have the same sizes or the bias one is unitary for broadcasting.\n"
			"Found shapes | Matmul output: %s | Unsqueezed bias: %s", out_shape.str(), b._view.str()
		);

	// Create output tensor with this shape.
	Tensor out(out_shape, A.device(), requires_grad);

	// Get matrix multiplication data.
	unsigned M = out_shape[-2];
	unsigned N = out_shape[-1];
	unsigned K = transA ? A._view[-2] : A._view[-1];

	// Extract the data.
	float* out_data = out.internal_data();
	const float* A_data = A.internal_data();
	const float* B_data = B.internal_data();
	const float* b_data = b.internal_data();

	// Now we actually multiply the matrices.
	if (out.is_gpu())
		kernel_ops::matmul_bias(out_data, A_data, B_data, b_data, out_shape, A._view, B._view, b._view, transA, transB);
	else
	{
		// Get bias last dimensions strides.
		unsigned bm_stride = (b._view[-2] == 1) ? 0u : b._stride[-2];
		unsigned bn_stride = (b._view[-1] == 1) ? 0u : b._stride[-1];
		// Create a running shape to count.
		Shape counting_shape(out_shape.dim() > 2 ? out_shape.dim() - 2 : 1, (int*)nullptr);
		// Create reference shape.
		Shape reference = out_shape;
		if (reference.dim() > 2)
		{
			reference.remove(-1);
			reference.remove(-1);
		}
		else reference = Shape(1);

		// Important data for transposition.
		const unsigned A_cols = A._view[-1];
		const unsigned B_cols = B._view[-1];

		// Iterate through vectors.
		while (true)
		{
			// Get initial pointer idxs given running shape.
			unsigned out_idx = 0, A_idx = 0, B_idx = 0, b_idx = 0;
			for (unsigned d = 0; d < counting_shape.dim(); d++)
			{
				if (out._view[d] != 1) out_idx += counting_shape[d] * out._stride[d];
				if (A._view[d]   != 1) A_idx   += counting_shape[d] *   A._stride[d];
				if (B._view[d]   != 1) B_idx   += counting_shape[d] *   B._stride[d];
				if (b._view[d]   != 1) b_idx   += counting_shape[d] *   b._stride[d];
			}
			// Get matrix pointers.
			float* out_matrix = out_data + out_idx; // M x N values
			const float* A_matrix = A_data + A_idx;		// M x K values
			const float* B_matrix = B_data + B_idx;		// K x N values
			const float* b_matrix = b_data + b_idx;		// M? x N? values

			// Here we actually multiply the matrices.
			{
				for (unsigned i0 = 0; i0 < M; i0 += 64)
				{
					const unsigned i1 = (i0 + 64 < M) ? (i0 + 64) : M;

					for (unsigned j0 = 0; j0 < N; j0 += 64)
					{
						const unsigned j1 = (j0 + 64 < N) ? (j0 + 64) : N;

						// add bias.
						for (unsigned i = i0; i < i1; ++i)
						{
							float* c_row = out_matrix + i * N;
							const float* bias_row = b_matrix + i * bm_stride;

							for (unsigned j = j0; j < j1; ++j)
								c_row[j] = bias_row[j * bn_stride];
						}

						for (unsigned k0 = 0; k0 < K; k0 += 64)
						{
							const unsigned k1 = (k0 + 64 < K) ? (k0 + 64) : K;

							for (unsigned i = i0; i < i1; ++i)
							{
								float* c_row = out_matrix + i * N;

								for (unsigned k = k0; k < k1; ++k)
								{
									const float a = transA
										? A_matrix[k * A_cols + i]
										: A_matrix[i * A_cols + k];

									for (unsigned j = j0; j < j1; ++j)
									{
										const float b = transB
											? B_matrix[j * B_cols + k]
											: B_matrix[k * B_cols + j];

										c_row[j] += a * b;
									}
								}
							}
						}
					}
				}
			}

			// Add one to the shape count.
			counting_shape[-1]++;
			for (int d = counting_shape.dim() - 1; d > 0; d--)
				if (counting_shape[d] >= reference[d])
				{
					counting_shape[d] -= reference[d];
					counting_shape[d - 1]++;
				}

			// If you reach the end of the leading dimension you're done.
			if (counting_shape[0] >= reference[0])
				break;
		}
	}

	// If it was a gradient operation store a MatMulBiasOp instance.
	if (requires_grad)
		out._internals->op = new MatMulBiasOp(A, B, b, out, transA, transB);

	// Squeeze back if necessary.
	if (A_was_1d) out = out.squeeze(-2);
	if (B_was_1d) out = out.squeeze(-1);

	// Return out.
	return out;
}

Tensor Functional::cat(const Tensor& ten0, const Tensor& ten1, int dim)
{
	// Concatenation tensor operator for backpropagation.
	class CatOp : public Tensor::TensorInternals::TensorOp
	{
		// Tensor copy storage.
		Tensor ten0, ten1;

		// Storage for the concatenation dimension.
		Shape s0, s1;

	public:
		// Constructor, stores all the data of the operation.
		CatOp(const Tensor& _ten0, const Tensor& _ten1, const Tensor& _out, int _dim) : TensorOp{ "Concatenation", _out },
			ten0{ _ten0.has_grad() ? _ten0 : Tensor() }, ten1{ _ten1.has_grad() ? _ten1 : Tensor() }, 
			s0{ _out.dim(), (int*)nullptr }, s1{ _out.dim(), (int*)nullptr }
		{
			if (ten0.has_grad()) _relatives[0] = &ten0;
			if (ten1.has_grad()) _relatives[1] = &ten1;
			s1[_dim] = ten0._view[_dim];
		}

		// Backpropagation. Route the gradient from the correct output subsets.
		void _backward() override
		{
			if (ten0.has_grad()) ten0.internal_gradient() += out.gradient().subset(ten0.shape(), s0);
			if (ten1.has_grad()) ten1.internal_gradient() += out.gradient().subset(ten1.shape(), s1);
		}
	};

	// --- Sanity checks ---

	// Both tensor must be initialized.
	MACROGRAD_CHECK(ten0.is_init(),
		"Trying to concatenate two tensors while the first tensor is empty."
	);
	MACROGRAD_CHECK(ten1.is_init(),
		"Trying to concatenate two tensors while the second tensor is empty."
	);
	// Both tensors must be on the same device
	MACROGRAD_CHECK(ten0.is_gpu() == ten1.is_gpu(),
		"Trying to concatenate two tensors in different devices is not allowed."
	);
	// Both tensor must have the same dimensionality to avoid ambiguity.
	MACROGRAD_CHECK(ten0.dim() == ten1.dim(),
		"Trying to concatenate two tensors with different dimensions.\n"
		"Found shapes | Tensor0: %s | Tensor1: %s", ten0._view.str(), ten1._view.str()
	);

	// Modulo dimension.
	dim = unsigned(dim + ten0.dim() * (2 - dim / int(ten0.dim()))) % ten0.dim();
	// Both tensors must have the same dimension sizes except at dim.
	for (unsigned i = 0; i < ten0.dim(); i++)
		MACROGRAD_CHECK(i == dim || ten0.size(i) == ten1.size(i),
			"Trying to concatenate two tensors with incompatible shapes.\n"
			"Make sure they have the same sizes in all dimensions except the concatenated one.\n"
			"Found shapes | Tensor0: %s | Tensor1: %s", ten0._view.str(), ten1._view.str()
		);

	// The output will have gradient if either of the inputs has.
	bool requires_grad = ten0.has_grad() || ten1.has_grad();

	// Get concatenated dimension size.
	const unsigned _size0 = ten0._view[dim];
	const unsigned _size1 = ten1._view[dim];

	// Prepare output shape.
	Shape out_shape = ten0._view;
	out_shape[dim] = _size0 + _size1;

	// Create output with the concatenated shape.
	Tensor out(out_shape, ten0.device(), requires_grad);

	// Extract the data.
	float* out_data = out.internal_data();
	const float* ten0_data = ten0.internal_data();
	const float* ten1_data = ten1.internal_data();
	// Get outer stride.
	const unsigned outer_stride_out = out._stride[dim] * out_shape[dim];
	const unsigned outer_stride_in0 = out._stride[dim] * _size0;
	const unsigned outer_stride_in1 = out._stride[dim] * _size1;
	// Get relevant sizes.
	unsigned outer_size = out.numel() / outer_stride_out;
	unsigned inner_size = out._stride[dim];

	// Now we actually concatenate the tensors.
	if (out.is_gpu())
		kernel_ops::cat(out_data, ten0_data, ten1_data, inner_size, outer_size, _size0, _size1);
	else
	{
		// Iterate through entire dimension size to write to output.
		for (unsigned outer = 0; outer < outer_size; outer++)
		for (unsigned inner = 0; inner < inner_size; inner++)
		{
			unsigned in0_idx = outer * outer_stride_in0 + inner;
			unsigned in1_idx = outer * outer_stride_in1 + inner;
			unsigned out_idx = outer * outer_stride_out + inner;

			// Copy first tensor data.
			for (unsigned i = 0; i < _size0; i++)
			{
				out_data[out_idx] = ten0_data[in0_idx];
				out_idx += inner_size, in0_idx += inner_size;
			}
			// Copy second tensor data.
			for (unsigned i = 0; i < _size1; i++)
			{
				out_data[out_idx] = ten1_data[in1_idx];
				out_idx += inner_size, in1_idx += inner_size;
			}
		}
	}

	// If it was a gradient operation store a CatOp instance.
	if (requires_grad)
		out._internals->op = new CatOp(ten0, ten1, out, dim);

	// Return out.
	return out;
}

Tensor Functional::mean_squared_error(const Tensor& ten0, const Tensor& ten1)
{
	// MSE tensor operator for backpropagation.
	class MseOp : public Tensor::TensorInternals::TensorOp
	{
		// Tensor copy storage.
		Tensor y0, y1;

		// Total number of elements storage.
		unsigned count;

	public:
		// Constructor, stores all the data of the operation.
		MseOp(const Tensor& _y0, const Tensor& _y1, const Tensor& _out, unsigned _count) : TensorOp{ "Mean Squared Error", _out },
			y0{ _y0 }, y1{ _y1 }, count{ _count }
		{
			if (y0.has_grad()) _relatives[0] = &y0;
			if (y1.has_grad()) _relatives[1] = &y1;
		}

		// Backpropagation. Gradient is +- 2 (y0 - y1).
		void _backward() override
		{
			Tensor derivative = y0.no_grad() - y1.no_grad();
			derivative.internal_multiply(out.gradient().internal_data(), out.is_gpu(), 2.f / count);

			if (y0.has_grad()) y0.internal_gradient() += derivative;
			if (y1.has_grad()) y1.internal_gradient() -= derivative;
		}
	};

	// --- Sanity checks ---

	// Both tensor must be initialized.
	MACROGRAD_CHECK(ten0.is_init(),
		"Trying to compute MSE of two tensors while the first tensor is empty."
	);
	MACROGRAD_CHECK(ten1.is_init(),
		"Trying to compute MSE of two tensors while the second tensor is empty."
	);
	// Both tensors must be on the same device
	MACROGRAD_CHECK(ten0.is_gpu() == ten1.is_gpu(),
		"Trying to compute MSE of two tensors in different devices is not allowed."
	);
	// Both tensor must have the same number of elements.
	MACROGRAD_CHECK(ten0.numel() == ten1.numel(),
		"Trying to compute MSE of two tensors with different nomber of elements.\n"
		"Found shapes | Tensor0: %s | Tensor1: %s", ten0._view.str(), ten1._view.str()
	);

	// The output will have gradient if either of the inputs has.
	bool requires_grad = ten0.has_grad() || ten1.has_grad();

	// Flatten the tensors.
	Tensor y0 = ten0.flatten();
	Tensor y1 = ten1.flatten();

	// Get size.
	int size = y0.numel();
	MACROGRAD_CHECK(size,
		"Trying to compute MSE of two tensors of zero elements is not allowed."
	);

	// Create output with single element.
	Tensor out(Shape(1), ten0.device(), requires_grad);

	// Extract the data.
	float* out_data = out.internal_data();
	float* y0_data = y0.internal_data();
	float* y1_data = y1.internal_data();

	// Now we actually compute the MSE.
	if (out.is_gpu())
		kernel_ops::mse(out_data, y0_data, y1_data, size);
	else
	{
		// Iterate through entire length.
		for (int idx = 0; idx < size; idx++)
		{
			float d = y0_data[idx] - y1_data[idx];
			*out_data += d * d;
		}
		// Divide at the end.
		*out_data /= size;
	}

	// If it was a gradient operation store a CatOp instance.
	if (requires_grad)
		out._internals->op = new MseOp(y0, y1, out, size);

	// Return out.
	return out;
}

Tensor Functional::cross_entropy_loss(const Tensor& logits, const VectorInt& labels)
{
	// Cross-Entropy tensor operator for backpropagation.
	class CelOp : public Tensor::TensorInternals::TensorOp
	{
		// Tensor copy storage.
		Tensor logits;

		// Probabilities for backprop and one-hot encoded labels.
		Tensor probs, labels;

		// Total number of samples.
		unsigned size;

	public:
		// Constructor, stores all the data of the operation.
		CelOp(const Tensor& _logits, const Tensor& _probs, const Tensor& _out, const VectorInt& _labels) : TensorOp{ "Cross-Entropy Loss", _out },
			logits{ _logits }, probs{ _probs }, labels{ Functional::one_hot(_labels, _probs._view[-1])}, size{_logits.size(0)}
		{
			_relatives[0] = &logits;
		}

		// Backpropagation. Gradient is prob_k - y_k.
		void _backward() override
		{
			Tensor derivative = probs - labels;
			derivative.internal_multiply(out.gradient().internal_data(), out.is_gpu(), 1.f / size);

			logits.internal_gradient() += derivative;
		}
	};

	// Tensor must be initialized.
	MACROGRAD_CHECK(logits.is_init(),
		"Trying to apply softmax to an empty tensor."
	);
	// Make sure shape is correct.
	MACROGRAD_CHECK(logits.dim() == 2,
		"Invalid shape found in logits for cross-entropy loss.\n"
		"Make sure your logits have shape (n_cases, n_classes) to avoid any ambiguity.\n"
		"Logits shape found: %s", logits._view.str()
	);
	MACROGRAD_CHECK(logits._view[-1] > 1,
		"Invalid shape found in logits for cross-entropy loss.\n"
		"Make sure your logits have shape (n_cases, n_classes).\n" 
		"Found only one class, this makes the operation a no-op and is most likely unintentional.\n"
		"Logits shape found: %s", logits._view.str()
	);
	MACROGRAD_CHECK(logits._view[0] != 0,
		"Found zero number of cases for cross-entropy loss.\n"
		"There must be at least one case, void tensors are not allowed.\n"
		"Logits shape found: %s", logits._view.str()
	);
	MACROGRAD_CHECK(labels.is_gpu() == logits.is_gpu(),
		"Logits and labels found in different devices inside a cross-entropy loss call."
	);
	MACROGRAD_CHECK((int)labels.len() >= logits._view[0],
		"Insuficient amount of labels found inside a cross-entropy loss call.\n"
		"Make sure the length of the labels vector is at least the amount of cases.\n"
		"Logits shape found: %s | Labels count: %u", logits._view.str(), labels.len()
	);

	// First do softmax without gradient.
	Tensor probs(logits.shape(), logits.device(), false);

	// Create single element output.
	Tensor out(Shape(1), logits.device(), logits.has_grad());

	// Get number of cases.
	unsigned size = logits._view[0];

	// Extract the data.
	float* out_data = out.internal_data();
	float* probs_data = probs.internal_data();
	const float* logits_data = logits.internal_data();
	const int* labels_data = labels.data();

	// Get relevant stride.
	const unsigned num_classes = logits._view[-1];

	// Now compute the actual cross-entropy loss.
	if (out.is_gpu())
		kernel_ops::cross_entropy_loss(out_data, probs_data, logits_data, labels_data, size, num_classes);
	else
	{
		// Add all loss values and compute all probs.
		for (unsigned i = 0; i < size; i++)
		{
			// Get the correct label.
			int label = labels_data[i];

			float* p_probs = probs_data + i * num_classes;
			const float* p_logits = logits_data + i * num_classes;
			
			MACROGRAD_CHECK(label < (int)num_classes && label >= 0,
				"Label out of range found inside cross-entropy loss computation.\n"
				"Make sure your labels are in the range [0, num_classes - 1]."
			);

			// First pass find maximum.
			float max = *p_logits;
			for (unsigned idx = 1; idx < num_classes; idx++)
				if (p_logits[idx] > max)
					max = p_logits[idx];
			// Second pass compute exponential and accumulate sum.
			float sum = 0.f;
			for (unsigned idx = 0; idx < num_classes; idx++)
			{
				p_probs[idx] = expf(p_logits[idx] - max);
				sum += p_probs[idx];
			}
			// Now compute loss.
			*out_data += -p_logits[label] + max + logf(sum);

			// Third pass divide by sum.
			for (unsigned idx = 0; idx < num_classes; idx++)
				p_probs[idx] /= sum;
		}
		// Divide by number of cases.
		*out_data /= size;
	}

	// If it was a gradient operation store a CelOp instance.
	if (logits.has_grad())
		out._internals->op = new CelOp(logits, probs, out, labels);

	// Return out.
	return out;
}

Tensor Functional::negative_log_likelihood(const Tensor& probs, const VectorInt& labels)
{
	// NLL tensor operator for backpropagation.
	class NllOp : public Tensor::TensorInternals::TensorOp
	{
		// Probabilities for backprop and one-hot encoded labels.
		Tensor probs, labels;

		// Total number of samples.
		unsigned size;

	public:
		// Constructor, stores all the data of the operation.
		NllOp(const Tensor& _probs, const Tensor& _out, const VectorInt& _labels) : TensorOp{ "Negative Log Likelihood", _out },
			probs{ _probs }, labels{ Functional::one_hot(_labels, _probs._view[-1])}, size{_probs.size(0)}
		{
			_relatives[0] = &probs;
		}

		// Backpropagation. Gradient is -out_grad/prob_j if j is the correct label, else 0.
		void _backward() override
		{
			Tensor derivative = labels / probs.no_grad();
			derivative.internal_multiply(out.gradient().internal_data(), out.is_gpu(), 1.f / size);

			probs.internal_gradient() -= derivative;
		}
	};

	// Tensor must be initialized.
	MACROGRAD_CHECK(probs.is_init(),
		"Trying to compute NLL with an empty tensor."
	);
	// Make sure shape is correct.
	MACROGRAD_CHECK(probs.dim() == 2,
		"Invalid shape found in probabilities for negative log likelihood.\n"
		"Make sure your probs have shape (n_cases, n_classes) to avoid any ambiguity.\n"
		"Probs shape found: %s", probs._view.str()
	);
	MACROGRAD_CHECK(probs._view[-1] > 1,
		"Invalid shape found in probabilities for negative log likelihood.\n"
		"Make sure your probs have shape (n_cases, n_classes).\n"
		"Found only one class, this makes the operation a no-op and is most likely unintentional.\n"
		"Probs shape found: %s", probs._view.str()
	);
	MACROGRAD_CHECK(probs._view[0] != 0,
		"Found zero number of cases for negative log likelihood.\n"
		"There must be at least one case, void tensors are not allowed.\n"
		"Probs shape found: %s", probs._view.str()
	);
	MACROGRAD_CHECK(labels.is_gpu() == probs.is_gpu(),
		"Probabilities and labels found in different devices inside a negative log likelihood call."
	);
	MACROGRAD_CHECK((int)labels.len() >= probs._view[0],
		"Insuficient amount of labels found inside a negative log likelihood call.\n"
		"Make sure the length of the labels vector is at least the amount of cases.\n"
		"Probs shape found: %s | Labels count: %u", probs._view.str(), labels.len()
	);

	// Create single element output.
	Tensor out(Shape(1), probs.device(), probs.has_grad());

	// Get number of cases.
	unsigned size = probs._view[0];

	// Extract the data.
	float* out_data = out.internal_data();
	const float* probs_data = probs.internal_data();
	const int* labels_data = labels.data();

	// Get relevant stride.
	const unsigned num_classes = probs._view[-1];

	// Now compute the actual cross-entropy loss.
	if (out.is_gpu())
		kernel_ops::negative_log_likelihood(out_data, probs_data, labels_data, size, num_classes);
	else
	{
		// Add all loss values.
		for (unsigned i = 0; i < size; i++)
		{
			// Get the label.
			int label = labels_data[i];

			MACROGRAD_CHECK(label < (int)num_classes && label >= 0,
				"Label out of range found inside negative log likelihood computation.\n"
				"Make sure your labels are in the range [0, num_classes - 1]."
			);
			*out_data += -logf(probs_data[i * num_classes + label]);
		}

		// Divide by number of cases.
		*out_data /= size;
	}

	// If it was a gradient operation store a NllOp instance.
	if (probs.has_grad())
		out._internals->op = new NllOp(probs, out, labels);

	// Return out.
	return out;
}

Tensor Functional::one_hot(const VectorInt& labels, unsigned num_classes)
{
	// Size must be correct.
	MACROGRAD_CHECK(num_classes > 1,
		"Invalid number of classes found in one-hot encoding. Found num_classes less than two.\n"
		"This makes the operation a no-op and is most likely unintentional.\n"
	);
	MACROGRAD_CHECK(labels.len(),
		"Uninitialized label vector can not be used to generate a one-hot encoding.\n"
		"Please make sure your labels vector has at least one case.\n"
	);

	// Generate output shape.
	Shape shape(labels.len(), num_classes);

	// Create output tensor.
	Tensor out(shape, labels.device(), false);

	// Extract the data.
	float* out_data = out.internal_data();
	const int* labels_data = labels.data();
	// Get relevant stride.
	const unsigned n_cases = labels.len();

	// Now we one-hot these vectors.
	if (out.is_gpu())
		kernel_ops::one_hot(out_data, labels_data, n_cases, num_classes);
	else
	{
		// Itearate through all cases.
		for (unsigned i = 0; i < n_cases; i++)
		{
			// Get the label.
			int label = labels_data[i];

			MACROGRAD_CHECK(label < (int)num_classes && label >= 0,
				"Label out of range found inside a one-hot encoding.\n"
				"Make sure your labels are in the range [0, num_classes - 1]."
			);
			out_data[i * num_classes + label] = 1.f;
		}
	}

	// Return out.
	return out;
}

Tensor Functional::causal_mask(unsigned L, const char* device)
{
	// Create mask tensor.
	Tensor mask(Shape(L, L), device, false);

	// Extract the data.
	float* out_data = mask.internal_data();

	// Set values to -inf.
	if (mask.is_gpu())
		kernel_ops::causal_mast(out_data, L);
	else
	{
		// Our negative infinity.
		constexpr float neg_inf = -std::numeric_limits<float>::infinity();

		// Iterate through mask.
		for (unsigned i = 0    ; i < L; ++i)
		for (unsigned j = i + 1; j < L; ++j)
			out_data[i * L + j] = neg_inf;
	}

	// Return mask.
	return mask;
}

/*
--------------------------------------------------------------------------------------------------------------------------
 Random Namespace Operators
--------------------------------------------------------------------------------------------------------------------------
*/

// Global random seed.
static inline unsigned long long _seed = 42ull;
// Mixes up the randomizer seed.
static inline unsigned long long splitmix()
{
	_seed += 0x9E3779B97F4A7C15ull;
	_seed = (_seed ^ (_seed >> 30)) * 0xBF58476D1CE4E5B9ull;
	_seed = (_seed ^ (_seed >> 27)) * 0x94D049BB133111EBull;
	_seed ^= (_seed >> 31);
	return _seed;
}
// Generates random values between 0 and 1.
static inline float random_0_1()
{
	return (splitmix() >> 8) * (1.0f / 72057594037927936.0f); // 2^56
}
// Generates random values with the specified normal distribution.
static inline float random_norm()
{
	float rand_1 = (splitmix() >> 8) * (1.0f / 72057594037927936.0f); // Random value between 0 and 1
	float rand_2 = (splitmix() >> 8) * (1.0f / 72057594037927936.0f); // Random value between 0 and 1

	return sqrtf(-2.0f * logf(rand_1)) * cosf(2.0f * 3.1415926535f * rand_2); // Normal distribution N(0,1) via Box-Muller
}
// Generates a random natural number between begin and end (included).
static inline unsigned random_unsigned(unsigned begin, unsigned end)
{
	return splitmix() % (end + 1 - begin) + begin;
}

// Random seed initialization.

void Random::set_seed(unsigned long long seed)
{
	_seed = seed;
}

void Random::set_cuda_seed(unsigned long long seed)
{
	kernel_ops::set_seed(seed);
}

// Uses a randomizer to generate a shuffled set of integers from 0 to size.

void Random::shuffle(VectorInt& values)
{
	unsigned len = values.len();
	int* data = values.data();

	if (len <= 1)
		return;

	if (values.is_gpu())
		kernel_ops::shuffle(data, len);

	// Fisher–Yates
	else for (unsigned i = len; i > 1; --i) 
	{
		unsigned j = (unsigned)(splitmix() % i); // 0..i-1
		int tmp = data[i - 1];
		data[i - 1] = data[j];
		data[j] = tmp;
	}
}

float Random::rand_normal(float mean, float std)
{
	return random_norm() * std - mean;
}

float Random::rand_uniform(float min, float max)
{
	return random_0_1() * (max - min) + min;
}

int Random::rand_int(int min, int max)
{
	return int(random_unsigned(0, max - min)) + min;
}

/*
--------------------------------------------------------------------------------------------------------------------------
 Initialization Namespace Operators
--------------------------------------------------------------------------------------------------------------------------
*/

void Initialization::normal(Tensor& tensor, float mean, float std)
{
	MACROGRAD_CHECK(tensor.is_init(),
		"Found an empty tensor inside unifor initialization, please make sure your tensor is initialized."
	);

	float* data = tensor.internal_data();
	const unsigned numel = tensor.numel();

	if (tensor.is_gpu())
		kernel_ops::normal(data, mean, std, numel);
	else
	{
		unsigned idx = 0;
		while (idx < numel)
			data[idx++] = random_norm() * std + mean;
	}
}

void Initialization::uniform(Tensor& tensor, float min, float max)
{
	MACROGRAD_CHECK(tensor.is_init(),
		"Found an empty tensor inside unifor initialization, please make sure your tensor is initialized."
	);

	float* data = tensor.internal_data();
	const unsigned numel = tensor.numel();

	if (tensor.is_gpu())
		kernel_ops::uniform(data, min, max, numel);
	else
	{
		unsigned idx = 0;
		while(idx < numel)
			data[idx++] = random_0_1() * (max - min) + min;
	}
}
