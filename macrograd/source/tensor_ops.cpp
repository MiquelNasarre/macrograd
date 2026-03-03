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
	TENSOR_CHECK(has_grad(),
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

	memcpy(out._internals->_data, _internals->_data, _internals->_data_size);

	if (with_grad)
	{
		out._internals->gradient = new Tensor(_view, device(), false);
		if (has_grad() && copy_grad)
			memcpy(out._internals->gradient->_internals->_data, _internals->gradient->_internals->_data, _internals->_data_size);
	}

	return out;
}

void Tensor::internal_add(float val)
{
	TENSOR_CHECK(_internals,
		"Trying to internally add to an empty tensor."
	);
	
	if (_internals->is_gpu)
	{
		TENSOR_ERROR("CUDA backend not implemented yet.");
	}
	else
	{
		float* data = _internals->_data;
		const unsigned numel = this->numel();

		unsigned idx = 0;
		while (idx < numel)
			data[idx++] += val;
	}
}

void Tensor::internal_multiply(float val)
{
	TENSOR_CHECK(_internals,
		"Trying to internally multiply to an empty tensor."
	);

	if (_internals->is_gpu)
	{
		TENSOR_ERROR("CUDA backend not implemented yet.");
	}
	else
	{
		float* data = _internals->_data;
		const unsigned numel = this->numel();

		unsigned idx = 0;
		while (idx < numel)
			data[idx++] *= val;
	}
}

void Tensor::internal_set(float val)
{
	TENSOR_CHECK(_internals,
		"Trying to internally set an empty tensor."
	);

	if (_internals->is_gpu)
	{
		TENSOR_ERROR("CUDA backend not implemented yet.");
	}
	else
	{
		float* data = _internals->_data;
		const unsigned numel = this->numel();

		unsigned idx = 0;
		while (idx < numel)
			data[idx++] = val;
	}
}

void Tensor::internal_add(const Tensor& other)
{
	TENSOR_CHECK(_internals,
		"Trying to internally add to an empty tensor."
	);
	TENSOR_CHECK(other._internals,
		"Trying to internally add an empty tensor."
	);
	TENSOR_CHECK(this->numel() == other.numel(),
		"Trying to internally add a tensor with a different number of elements."
	);

	float* data = _internals->_data;
	float* other_data = other._internals->_data;
	const int numel = this->numel();

	int idx = -1;
	while (++idx < numel)
		data[idx] += other_data[idx];
}

void Tensor::internal_add_prod(float val, const Tensor& other)
{
	TENSOR_CHECK(_internals,
		"Trying to internally add to an empty tensor."
	);
	TENSOR_CHECK(other._internals,
		"Trying to internally add an empty tensor."
	);
	TENSOR_CHECK(this->numel() == other.numel(),
		"Trying to internally add a tensor with a different number of elements."
	);

	float* data = _internals->_data;
	float* other_data = other._internals->_data;
	const int numel = this->numel();

	int idx = -1;
	while (++idx < numel)
		data[idx] += val * other_data[idx];
}

void Tensor::internal_subtract(const Tensor& other)
{
	TENSOR_CHECK(_internals,
		"Trying to internally subtract to an empty tensor."
	);
	TENSOR_CHECK(other._internals,
		"Trying to internally subtract an empty tensor."
	);
	TENSOR_CHECK(this->numel() == other.numel(),
		"Trying to internally subtract a tensor with a different number of elements."
	);

	float* data = _internals->_data;
	float* other_data = other._internals->_data;
	const int numel = this->numel();

	int idx = -1;
	while (++idx < numel)
		data[idx] -= other_data[idx];
}

void Tensor::internal_multiply(const Tensor& other)
{
	TENSOR_CHECK(_internals,
		"Trying to internally multiply an empty tensor."
	);
	TENSOR_CHECK(other._internals,
		"Trying to internally multiply by an empty tensor."
	);
	TENSOR_CHECK(this->numel() == other.numel(),
		"Trying to internally multiply a tensor with a different number of elements."
	);

	float* data = _internals->_data;
	float* other_data = other._internals->_data;
	const int numel = this->numel();

	int idx = -1;
	while (++idx < numel)
		data[idx] *= other_data[idx];
}

void Tensor::internal_set_value(const Shape& route, float value)
{
	TENSOR_CHECK(_internals,
		"Trying to set a value on an empty tensor."
	);
	TENSOR_CHECK(numel(),
		"Trying to set a value on a tensor with no values.\n"
		"The tensor shape is %s", _view.str()
	);

	float* ptr = _internals->_data;

	for (unsigned d = 0; d < dim(); d++)
		ptr += ((route[d] + _view[d] * (2 - route[d] / _view[d])) % _view[d]) * _stride[d];

	*ptr = value;
}

float Tensor::internal_get_value(const Shape& route)
{
	TENSOR_CHECK(_internals,
		"Trying to get a value on an empty array."
	);
	TENSOR_CHECK(numel(),
		"Trying to get a value on an array with no values.\n"
		"The array shape is %s", _view.str()
	);

	float* ptr = _internals->_data;

	for (unsigned d = 0; d < dim(); d++)
		ptr += ((route[d] + _view[d] * (2 - route[d] / _view[d])) % _view[d]) * _stride[d];

	return *ptr;
}

void Tensor::internal_set_vector(const Shape& route, const float* values)
{
	TENSOR_CHECK(_internals,
		"Trying to set a vector on an empty tensor."
	);
	TENSOR_CHECK(numel(),
		"Trying to set a vector on a tensor with no values.\n"
		"The tensor shape is %s", _view.str()
	);

	// Get idx given route. Modulo for negative numbers.
	unsigned idx = 0;
	for (unsigned i = 0; i < route.dim(); i++)
		idx += (route[i] + _view[i] * (2 - route[i] / _view[i]) % _view[i]) * _stride[i];

	// Get full data expected size.
	unsigned _data_size = sizeof(float);
	for (unsigned i = route.dim(); i < _view.dim(); i++)
		_data_size *= _view[i];

	// Copy given data.
	memcpy(_internals->_data + idx, values, _data_size);
}

float* Tensor::internal_get_vector(const Shape& route)
{
	TENSOR_CHECK(_internals,
		"Trying to get a vector of an empty tensor."
	);
	TENSOR_CHECK(numel(),
		"Trying to fet a vector of a tensor with no values.\n"
		"The tensor shape is %s", _view.str()
	);

	// Get idx given route. Modulo for negative numbers.
	unsigned idx = 0;
	for (unsigned i = 0; i < route.dim(); i++)
		idx += (route[i] + _view[i] * (2 - route[i] / _view[i]) % _view[i]) * _stride[i];

	// Return data ptr.
	return _internals->_data + idx;
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
	TENSOR_CHECK(_internals,
		"Trying to call view on an empty tensor."
	);
	TENSOR_CHECK(shape.dim(),
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
			TENSOR_CHECK(neg_one == -1,
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
		TENSOR_CHECK(new_total_dim != 0,
			"Ambiguous shape find inside a view call.\n"
			"It is not allowed to have an unknown dimension -1 while there is a size 0.\n"
			"Old Shape: %s | View Input Shape: %s.", _view.str(), shape.str()
		);
		TENSOR_CHECK(numel() % new_total_dim == 0,
			"Unreconcileable shapes found inside a view call, total sizes are not divisible.\n"
			"Old Shape: %s | View Input Shape: %s.", _view.str(), shape.str()
		);
		new_shape[neg_one] = numel() / new_total_dim;
	}
	else if (new_total_dim != numel())
		TENSOR_ERROR("Incompatible shapes found during a view call. Make sure the total dimensionality matches.\n"
			"Old Shape: %s | View Input Shape: %s.", _view.str(), shape.str()
		);

	// Create new stride shape.
	Shape new_stride(new_shape.dim(), (int*)nullptr);
	new_stride[-1] = 1;
	for (int i = new_shape.dim() - 2; i >= 0; i--)
		new_stride[i] = new_shape[i + 1] * new_stride[i + 1];
	for (unsigned i = 0; i < new_shape.dim(); i++)
		if (new_shape[i] == 1)
			new_stride[i] = 0;
	
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
	TENSOR_CHECK(_internals,
		"Trying to flatten an empty tensor."
	);
	
	// Create output tensor with flat view.
	Tensor out = *this;
	out._view = Shape(numel());
	out._stride = Shape(numel() > 1u ? 1u : 0u);

	// Return out.
	return out;
}

// Returns a tensor with the specified dimension removed, must be unitary.

Tensor Tensor::squeeze(int dim) const
{
	TENSOR_CHECK(_internals,
		"Trying to call squeeze on an empty tensor."
	);
	TENSOR_CHECK(_view.dim() > 1,
		"Trying to squeeze a tensor with only one dimension left is not allowed."
	);
	TENSOR_CHECK(_view[dim] == 1,
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
	TENSOR_CHECK(_internals,
		"Trying to call unsqueeze on an empty tensor."
	);

	Tensor copied = *this;
	copied._view.add(dim, 1);
	copied._stride.add(dim, 0);

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
	TENSOR_CHECK(_internals,
		"Trying to transpose an empty tensor."
	);

	// Modulo dimensions.
	dim0 = unsigned(dim0 + dim() * (2 - dim0 / int(dim()))) % dim();
	dim1 = unsigned(dim1 + dim() * (2 - dim1 / int(dim()))) % dim();

	// Find out which dimension is the longest and set it to dim0.
	if (_view[dim1] > _view[dim0])
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

	// Now we actually transpose the tensor.
	if (out._internals->is_gpu)
	{
		TENSOR_ERROR("CUDA backend not implemented yet.");
	}
	else
	{
		// Extract the data.
		float* out_data = out._internals->_data;
		float* ten_data = _internals->_data;
		// Store the length of the longest dim
		const unsigned vector_len = _view[dim0];
		// Create a running shape to count.
		Shape counting_shape(dim(), (int*)nullptr);
		// Create a refenrece shape with the longest dimension removed.
		Shape reference = _view;
		reference[dim0] = 1;
		// Get the stride for both tensors in the long dimension.
		unsigned ten_stride = _stride[dim0];
		unsigned out_stride = out._stride[dim1];

		// Iterate through vectors.
		while (true)
		{
			unsigned ten_idx = 0;
			for (unsigned d = 0; d < counting_shape.dim(); d++)
				ten_idx += counting_shape[d] * _stride[d];

			unsigned out_idx = 0;
			for (unsigned d = 0; d < counting_shape.dim(); d++)
			{
				if (d == dim1)	out_idx += counting_shape[d] * out._stride[dim0];
				else			out_idx += counting_shape[d] * out._stride[d];
			}

			unsigned count = 0u;
			while (count++ < vector_len)
			{
				out_data[out_idx] = ten_data[ten_idx];
				out_idx += out_stride, ten_idx += ten_stride;
			}

			counting_shape[-1]++;

			for (int d = dim() - 1; d > 0; d--)
			{
				if (counting_shape[d] >= reference[d])
				{
					counting_shape[d] -= reference[d];
					counting_shape[d - 1]++;
				}
			}
			// If you reach the end of the leading dimension you're done.
			if (counting_shape[0] >= reference[0])
				break;
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
	TENSOR_CHECK(_internals,
		"Trying to subset an empty tensor."
	);
	TENSOR_CHECK(shape.dim() == dim(),
		"Trying to call subset with a shape of different dimensionality.\n"
		"Make sure the number of dimensions matches, later you can squeeze out unitary ones.\n"
		"Tensor Shape: %s | Subset Shape: %s", _view.str(), shape.str()
	);
	TENSOR_CHECK(start_indices.dim() == dim(),
		"Trying to call subset with a start_indices shape of different dimensionality.\n"
		"Make sure you indicate the starting index of all dimensions without ambiguity.\n"
		"Tensor Shape: %s | Start Indices: %s", _view.str(), start_indices.str()
	);
	for (unsigned i = 0; i < dim(); i++)
		TENSOR_CHECK(shape[i] >= 0,
			"Negative dimensions in a subset shape call.\n"
			"Please make sure all dimensions are positive to avoid ambiguity.\n"
			"Tensor Shape: %s | Subset Shape: %s", _view.str(), shape.str()
		);

	// Modulo indices.
	Shape start = start_indices;
	for (unsigned i = 0; i < dim(); i++)
		start[i] = unsigned(start_indices[i] + _view[i] * (2 - start_indices[i] / int(_view[i]))) % _view[i];

	for (unsigned i = 0; i < dim(); i++)
		TENSOR_CHECK(shape[i] + start[i] <= _view[i],
			"Out of bounds dimension for a subset call.\n"
			"Start indices are not compatible with subset and input shape.\n"
			"Tensor Shape: %s | Subset Shape: %s | Modulo Start Indices: %s", _view.str(), shape.str(), start.str()
		);

	// Create output with the subset shape.
	Tensor out(shape, device(), has_grad());

	// Now we actually subset the tensor.
	if (out._internals->is_gpu)
	{
		TENSOR_ERROR("CUDA backend not implemented yet.");
	}
	else
	{
		// Extract the data.
		float* out_data = out._internals->_data;
		float* ten_data = _internals->_data;
		// Find the last non-unitary dimension to iterate through.
		unsigned last_long_dim = 0;
		for (unsigned i = 0; i < shape.dim(); i++)
			if (shape[i] > 1)
				last_long_dim = i;
		// Store the length of the longest dim
		const unsigned vector_len = shape[last_long_dim];
		// Create a running shape to count.
		Shape counting_shape(dim(), (int*)nullptr);
		// Create a refenrece shape with the long dimension removed.
		Shape reference = shape;
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

			unsigned count = 0u;
			while (count++ < vector_len)
			{
				out_data[out_idx] = ten_data[ten_idx];
				out_idx += out_stride, ten_idx += ten_stride;
			}

			counting_shape[-1]++;

			for (int d = dim() - 1; d > 0; d--)
			{
				if (counting_shape[d] >= reference[d])
				{
					counting_shape[d] -= reference[d];
					counting_shape[d - 1]++;
				}
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
	TENSOR_CHECK(_internals,
		"Trying to modify an empty tensor."
	);
	TENSOR_CHECK(other._internals,
		"Trying to modify with an empty other tensor."
	);
	// Both tensors must be on the same device
	TENSOR_CHECK(_internals->is_gpu == other._internals->is_gpu,
		"Trying to add madify a tensor with another tensor on a different device."
	);
	TENSOR_CHECK(other._view.dim() == dim(),
		"Trying to call modify with modification tensor of different dimensionality.\n"
		"Make sure the number of dimensions matches, unsqueeze unitary ones as needed.\n"
		"Tensor Shape: %s | Modifier Shape: %s", _view.str(), other._view.str()
	);
	TENSOR_CHECK(start_indices.dim() == dim(),
		"Trying to call subset with a start_indices shape of different dimensionality.\n"
		"Make sure you indicate the starting index of all dimensions without ambiguity.\n"
		"Tensor Shape: %s | Start Indices: %s", _view.str(), start_indices.str()
	);

	// Modulo indices.
	Shape start = start_indices;
	for (unsigned i = 0; i < dim(); i++)
		start[i] = unsigned(start_indices[i] + _view[i] * (2 - start_indices[i] / int(_view[i]))) % _view[i];

	for (unsigned i = 0; i < dim(); i++)
		TENSOR_CHECK(other._view[i] + start[i] <= _view[i],
			"Out of bounds dimension for a modify call.\n"
			"Start indices are not compatible with modifier and input shape.\n"
			"Tensor Shape: %s | Modifier Shape: %s | Modulo Start Indices: %s", _view.str(), other._view.str(), start.str()
		);

	// Get gradient data.
	bool requires_grad = has_grad() || other.has_grad();

	// Create output same as input.
	Tensor out = internal_copy(requires_grad, false);
	out._view = _view;
	out._stride = _stride;

	// Now we actually modify the tensor.
	if (out._internals->is_gpu)
	{
		TENSOR_ERROR("CUDA backend not implemented yet.");
	}
	else
	{
		// Extract the data.
		float* out_data = out._internals->_data;
		float* mod_data = other._internals->_data;
		// Find the last non-unitary dimension to iterate through.
		unsigned last_long_dim = 0;
		for (unsigned i = 0; i < dim(); i++)
			if (other._view[i] > 1)
				last_long_dim = i;
		// Store the length of the longest dim
		const unsigned vector_len = other._view[last_long_dim];
		// Create a running shape to count.
		Shape counting_shape(dim(), (int*)nullptr);
		// Create a refenrece shape with the long dimension removed.
		Shape reference = other._view;
		reference[last_long_dim] = 1;
		// Get the stride for both tensors in the long dimension.
		unsigned mod_stride = other._stride[last_long_dim];
		unsigned out_stride = out._stride[last_long_dim];

		// Iterate through vectors.
		while (true)
		{
			unsigned out_idx = 0, mod_idx = 0;
			for (unsigned d = 0; d < counting_shape.dim(); d++)
			{
				out_idx += (counting_shape[d] + start[d]) * out._stride[d];
				mod_idx += counting_shape[d] * other._stride[d];
			}

			unsigned count = 0u;
			while (count++ < vector_len)
			{
				out_data[out_idx] = mod_data[mod_idx];
				out_idx += out_stride, mod_idx += mod_stride;
			}

			counting_shape[-1]++;

			for (int d = dim() - 1; d > 0; d--)
			{
				if (counting_shape[d] >= reference[d])
				{
					counting_shape[d] -= reference[d];
					counting_shape[d - 1]++;
				}
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
	TENSOR_CHECK(_internals,
		"Trying to add repetitions to an empty tensor."
	);
	// Repeated shape must be unitary.
	TENSOR_CHECK(_view[dim] == 1,
		"Trying to repeat a tensor on a non-unitary dimension.\n"
		"Make sure the dimension you are repeating is of size 1."
	);

	// Modulo dimension.
	dim = unsigned(dim + _view.dim() * (2 - dim / int(_view.dim()))) % _view.dim();

	// Create output with the repeated dimension.
	Shape out_shape = _view;
	out_shape[dim] = repetitions;
	Tensor out(out_shape, device(), has_grad());

	// Now we actually repeat the tensor.
	if (out._internals->is_gpu)
	{
		TENSOR_ERROR("CUDA backend not implemented yet.");
	}
	else
	{
		// Extract the data.
		float* out_data = out._internals->_data;
		float* ten_data = _internals->_data;
		// Get relevant strides.
		const unsigned elem_stride = out._stride[dim];
		const unsigned post_stride = elem_stride * repetitions;
		// Get relevant sizes.
		unsigned prev_size = 1u;
		unsigned post_size = 1u;
		for (int i = 0; i < dim; i++)
			post_size *= _view[i];
		for (unsigned i = dim + 1; i < _view.dim(); i++)
			prev_size *= _view[i];
		// Iterate through all vectors to compute mean.
		for (unsigned post_count = 0; post_count < post_size; post_count++)
			for (unsigned prev_count = 0; prev_count < prev_size; prev_count++)
			{
				unsigned ten_idx = post_count * elem_stride + prev_count;
				unsigned out_idx = post_count * post_stride + prev_count;

				unsigned count = 0;
				while (count++ < repetitions)
				{
					out_data[out_idx] = ten_data[ten_idx];
					out_idx += elem_stride;
				}
			}
	}

	// If it was a gradient operation store a RepOp instance.
	if (has_grad())
		out._internals->op = new RepOp(*this, out, dim);

	// Return out.
	return out;
}

// Returns an exact copy of the tensor. This includes array, view and gradient if exist.

Tensor Tensor::copy(const char* device, bool grad) const
{
	// Copy tensor operator for backpropagation.
	class CopyOp : public Tensor::TensorInternals::TensorOp
	{
		// Tensor copy storage for the input.
		Tensor in;

	public:
		// Constructor, stores all the data of the operation.
		CopyOp(const Tensor& _in, const Tensor& _out) : TensorOp{ "Copy", _out },
			in{ _in }
		{
			_relatives[0] = &in;
		}

		// Backpropagation. Route the gradient.
		void _backward() override
		{
			in.internal_gradient() += out.gradient();
		}
	};

	// Tensor must be initialized.
	TENSOR_CHECK(_internals,
		"Trying to call copy on an empty tensor."
	);

	// If both have gradient copy it and set operator.
	Tensor out = internal_copy(false, false);
	out._view = _view;
	out._stride = _stride;
	if (grad)
	{
		if (has_grad())
		{
			out._internals->gradient = new Tensor(_internals->gradient->internal_copy(false, false));
			out._internals->op = new CopyOp(*this, out);
		}
		else out._internals->gradient = new Tensor(_view, device, false);
	}
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
	TENSOR_CHECK(_internals,
		"Trying to get the sign of an empty tensor."
	);

	// Create output with the same shape as tensor, no gradient.
	Tensor out(shape(), device(), false);

	// Now we actually add signs to the tensor.
	if (out._internals->is_gpu)
	{
		TENSOR_ERROR("CUDA backend not implemented yet.");
	}
	else
	{
		// Extract the data.
		float* out_data = out._internals->_data;
		float* ten_data = _internals->_data;
		// Iterate through data.
		int idx = -1, _numel = int(numel());
		while (++idx < _numel)
			out_data[idx] = (ten_data[idx] > 0.f) ? 1.f : 0.f;
	}

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
	TENSOR_CHECK(_internals,
		"Trying to exponentiate an empty tensor."
	);

	// Create output with the same shape as tensor.
	Tensor out(shape(), device(), has_grad());

	// Now we actually exponentiate the tensor.
	if (out._internals->is_gpu)
	{
		TENSOR_ERROR("CUDA backend not implemented yet.");
	}
	else
	{
		// Extract the data.
		float* out_data = out._internals->_data;
		float* ten_data = _internals->_data;
		// Iterate through all elements.
		int idx = -1, _numel = int(numel());
		while (++idx < _numel)
			out_data[idx] = expf(ten_data[idx]);
	}

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
	TENSOR_CHECK(_internals,
		"Trying to log an empty tensor."
	);

	// Create output with the same shape as tensor.
	Tensor out(shape(), device(), has_grad());

	// Now we actually log the tensors.
	if (out._internals->is_gpu)
	{
		TENSOR_ERROR("CUDA backend not implemented yet.");
	}
	else
	{
		// Extract the data.
		float* out_data = out._internals->_data;
		float* ten_data = _internals->_data;
		// Iterate through all elements.
		int idx = -1, _numel = int(numel());
		while (++idx < _numel)
			out_data[idx] = logf(ten_data[idx]);
	}

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
			in.internal_gradient() += out.gradient() * in.sign();
		}
	};

	// Tensor must be initialized.
	TENSOR_CHECK(_internals,
		"Trying to apply ReLU to an empty tensor."
	);

	// Create output with the same shape as tensor.
	Tensor out(shape(), device(), has_grad());

	// Now we actually ReLU the tensors.
	if (out._internals->is_gpu)
	{
		TENSOR_ERROR("CUDA backend not implemented yet.");
	}
	else
	{
		// Extract the data.
		float* out_data = out._internals->_data;
		float* ten_data = _internals->_data;
		// Iterate through all elements.
		int idx = -1, _numel = int(numel());
		while (++idx < _numel)
			out_data[idx] = (ten_data[idx] > 0.f) ? ten_data[idx] : 0.f;
	}

	// If it was a gradient operation store a ReLUOp instance.
	if (has_grad())
		out._internals->op = new ReLUOp(*this, out);

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
	TENSOR_CHECK(_internals,
		"Trying to apply sigmoid to an empty tensor."
	);

	// Create output with the same shape as tensor.
	Tensor out(shape(), device(), has_grad());

	// Now we actually sigmoid the tensor.
	if (out._internals->is_gpu)
	{
		TENSOR_ERROR("CUDA backend not implemented yet.");
	}
	else
	{
		// Extract the data.
		float* out_data = out._internals->_data;
		float* ten_data = _internals->_data;
		// Iterate through all elements.
		int idx = -1, _numel = int(numel());
		while (++idx < _numel)
			out_data[idx] = 1.f / 1.f + expf(-ten_data[idx]);
	}

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
	TENSOR_CHECK(_internals,
		"Trying to apply tanh to an empty tensor."
	);

	// Create output with the same shape as tensor.
	Tensor out(shape(), device(), has_grad());

	// Now we actually tanh the tensor.
	if (out._internals->is_gpu)
	{
		TENSOR_ERROR("CUDA backend not implemented yet.");
	}
	else
	{
		// Extract the data.
		float* out_data = out._internals->_data;
		float* ten_data = _internals->_data;
		// Iterate through all elements.
		int idx = -1, _numel = int(numel());
		while (++idx < _numel)
		{
			const float exp2 = expf(2 * ten_data[idx]);
			out_data[idx] = (exp2 - 1.f) / (exp2 + 1.f);
		}
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
	TENSOR_CHECK(_internals,
		"Trying to get the square root of an empty tensor."
	);

	// Create output with the same shape as tensor.
	Tensor out(shape(), device(), has_grad());

	// Now we actually sqrt the tensor.
	if (out._internals->is_gpu)
	{
		TENSOR_ERROR("CUDA backend not implemented yet.");
	}
	else
	{
		// Extract the data.
		float* out_data = out._internals->_data;
		float* ten_data = _internals->_data;
		// Iterate through all elements.
		int idx = -1, _numel = int(numel());
		while (++idx < _numel)
			out_data[idx] = sqrtf(ten_data[idx]);
	}

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
	TENSOR_CHECK(_internals,
		"Trying to square an empty tensor."
	);

	// Create output with the same shape as tensor.
	Tensor out(shape(), device(), has_grad());

	// Now we actually square the tensor.
	if (out._internals->is_gpu)
	{
		TENSOR_ERROR("CUDA backend not implemented yet.");
	}
	else
	{
		// Extract the data.
		float* out_data = out._internals->_data;
		float* ten_data = _internals->_data;
		// Iterate through all elements.
		int idx = -1, _numel = int(numel());
		while (++idx < _numel)
			out_data[idx] = ten_data[idx] * ten_data[idx];
	}

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
	TENSOR_CHECK(_internals,
		"Trying to square an empty tensor."
	);

	// Create output with the same shape as tensor.
	Tensor out(shape(), device(), has_grad());

	// Now we actually power the tensor.
	if (out._internals->is_gpu)
	{
		TENSOR_ERROR("CUDA backend not implemented yet.");
	}
	else
	{
		// Extract the data.
		float* out_data = out._internals->_data;
		float* ten_data = _internals->_data;
		// Iterate through all elements.
		int idx = -1, _numel = int(numel());
		while (++idx < _numel)
			out_data[idx] = powf(ten_data[idx], exp);
	}

	// If it was a gradient operation store a PowOp instance.
	if (has_grad())
		out._internals->op = new PowOp(*this, out, exp);

	// Return out.
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
	TENSOR_CHECK(_internals,
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

	// Now we actually average the tensor.
	if (out._internals->is_gpu)
	{
		TENSOR_ERROR("CUDA backend not implemented yet.");
	}
	else
	{
		// Extract the data.
		float* out_data = out._internals->_data;
		float* ten_data = _internals->_data;
		// Get relevant strides.
		const unsigned elem_stride = _stride[dim];
		const unsigned post_stride = elem_stride * _size;
		// Get relevant sizes.
		unsigned prev_size = 1u;
		unsigned post_size = 1u;
		for (int i = 0; i < dim; i++)
			post_size *= _view[i];
		for (unsigned i = dim + 1; i < _view.dim(); i++)
			prev_size *= _view[i];
		// Iterate through all vectors to compute mean.
		for (unsigned post_count = 0; post_count < post_size; post_count++)
			for (unsigned prev_count = 0; prev_count < prev_size; prev_count++)
			{
				unsigned ten_idx = post_count * post_stride + prev_count;
				unsigned out_idx = post_count * elem_stride + prev_count;
				
				unsigned count = 0;
				while (count++ < _size)
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
	TENSOR_CHECK(_internals,
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

	// Now we actually get the tensor variance.
	if (out._internals->is_gpu)
	{
		TENSOR_ERROR("CUDA backend not implemented yet.");
	}
	else
	{
		// Extract the data.
		float* out_data = out._internals->_data;
		float* ten_data = _internals->_data;
		// Get relevant strides.
		const unsigned elem_stride = _stride[dim];
		const unsigned post_stride = elem_stride * _size;
		// Get relevant sizes.
		unsigned prev_size = 1u;
		unsigned post_size = 1u;
		for (int i = 0; i < dim; i++)
			post_size *= _view[i];
		for (unsigned i = dim + 1; i < _view.dim(); i++)
			prev_size *= _view[i];
		// Iterate through all vectors to compute variance.
		for (unsigned post_count = 0; post_count < post_size; post_count++)
			for (unsigned prev_count = 0; prev_count < prev_size; prev_count++)
			{
				unsigned ten_idx = post_count * post_stride + prev_count;
				unsigned out_idx = post_count * elem_stride + prev_count;

				// Welford's algorithm for stability.
				float mean = 0.f, M2 = 0.f, old_mean;
				unsigned n = 0u;

				unsigned count = 0u;
				while (count++ < _size)
				{
					n++;
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
	TENSOR_CHECK(_internals,
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

	// Now we actually get the tensor deviation.
	if (out._internals->is_gpu)
	{
		TENSOR_ERROR("CUDA backend not implemented yet.");
	}
	else
	{
		// Extract the data.
		float* out_data = out._internals->_data;
		float* ten_data = _internals->_data;
		// Get relevant strides.
		const unsigned elem_stride = _stride[dim];
		const unsigned post_stride = elem_stride * _size;
		// Get relevant sizes.
		unsigned prev_size = 1u;
		unsigned post_size = 1u;
		for (int i = 0; i < dim; i++)
			post_size *= _view[i];
		for (unsigned i = dim + 1; i < _view.dim(); i++)
			prev_size *= _view[i];
		// Iterate through all vectors to compute std.
		for (unsigned post_count = 0; post_count < post_size; post_count++)
			for (unsigned prev_count = 0; prev_count < prev_size; prev_count++)
			{
				unsigned ten_idx = post_count * post_stride + prev_count;
				unsigned out_idx = post_count * elem_stride + prev_count;

				// Welford's algorithm for stability.
				float mean = 0.f, M2 = 0.f, old_mean;
				unsigned n = 0u;

				unsigned count = 0u;
				while (count++ < _size)
				{
					n++;
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
	TENSOR_CHECK(_internals,
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

	// Now we actually sum the tensor.
	if (out._internals->is_gpu)
	{
		TENSOR_ERROR("CUDA backend not implemented yet.");
	}
	else
	{
		// Extract the data.
		float* out_data = out._internals->_data;
		float* ten_data = _internals->_data;
		// Get relevant strides.
		const unsigned elem_stride = _stride[dim];
		const unsigned post_stride = elem_stride * _size;
		// Get relevant sizes.
		unsigned prev_size = 1u;
		unsigned post_size = 1u;
		for (int i = 0; i < dim; i++)
			post_size *= _view[i];
		for (unsigned i = dim + 1; i < _view.dim(); i++)
			prev_size *= _view[i];
		// Iterate through all vectors to compute sum.
		for (unsigned post_count = 0; post_count < post_size; post_count++)
			for (unsigned prev_count = 0; prev_count < prev_size; prev_count++)
			{
				unsigned ten_idx = post_count * post_stride + prev_count;
				unsigned out_idx = post_count * elem_stride + prev_count;

				unsigned count = 0;
				while (count++ < _size)
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
	TENSOR_CHECK(_internals,
		"Trying to apply softmax to an empty tensor."
	);

	// Modulo dimension.
	dim = unsigned(dim + _view.dim() * (2 - dim / int(_view.dim()))) % _view.dim();

	// Get the initial dimension size.
	unsigned _size = size(dim);
	// Create output with the same shape as tensor.
	Tensor out(shape(), device(), has_grad());
	// Compute output shape.
	Shape out_shape = shape();

	// Now we actually apply softmax to the tensor.
	if (out._internals->is_gpu)
	{
		TENSOR_ERROR("CUDA backend not implemented yet.");
	}
	else
	{
		// Extract the data.
		float* out_data = out._internals->_data;
		float* ten_data = _internals->_data;
		// Get relevant strides.
		const unsigned elem_stride = _stride[dim];
		const unsigned post_stride = elem_stride * _size;
		// Get relevant sizes.
		unsigned prev_size = 1u;
		unsigned post_size = 1u;
		for (int i = 0; i < dim; i++)
			post_size *= _view[i];
		for (unsigned i = dim + 1; i < _view.dim(); i++)
			prev_size *= _view[i];
		// Iterate through all vectors to compute softmax.
		for (unsigned post_count = 0; post_count < post_size; post_count++)
			for (unsigned prev_count = 0; prev_count < prev_size; prev_count++)
			{
				// Compute initial and final idx for this vector.
				const unsigned idx_0 = post_count * post_stride + prev_count;
				const unsigned idx_f = (post_count + 1) * post_stride + prev_count;
				
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
	TENSOR_CHECK(ten0._internals,
		"Trying to add two tensors while the first tensor is empty."
	);
	TENSOR_CHECK(ten1._internals,
		"Trying to add two tensors while the second tensor is empty."
	);
	// Both tensors must be on the same device
	TENSOR_CHECK(ten0._internals->is_gpu == ten1._internals->is_gpu,
		"Trying to add two tensors in different devices is not allowed."
	);
	// Both tensor must have the same dimensionality to avoid ambiguity.
	TENSOR_CHECK(ten0.dim() == ten1.dim(),
		"Trying to add two tensors with different dimensions.\n"
		"For broadcast addition please use view() or squeeze()/unsqueeze() to make dimensions match.\n"
		"Found shapes | Tensor0: %s | Tensor1: %s", ten0._view.str(), ten1._view.str()
	);
	// Both tensors must either have the same dimension sizes or one of them have size one.
	// So that they can be broadcasted or summed without ambiguity.
	for (unsigned i = 0; i < ten0.dim(); i++)
		TENSOR_CHECK(ten0.size(i) == 1 || ten1.size(i) == 1 || ten0.size(i) == ten1.size(i),
			"Trying to add two tensors with incompatible shapes.\n"
			"Make sure shapes are compatible, meaning they have the same sizes or one of them is unitary for broadcasting.\n"
			"Found shapes | Tensor0: %s | Tensor1: %s", ten0._view.str(), ten1._view.str()
		);

	// The output will have gradient if either of the inputs has.
	bool requires_grad = ten0.has_grad() || ten1.has_grad();

	// Create output with the same shape as first tensor.
	Tensor out(ten0.shape(), ten0.device(), requires_grad);
	
	// Sum dimensions on ten1 if necessary.
	Tensor ten1_summed = ten1;
	for (unsigned i = 0; i < ten0.dim(); i++)
		if (ten0.size(i) == 1 && ten1_summed.size(i) != 1)
			ten1_summed = ten1_summed.sum(i, true);

	// Now we actually sum the tensors.
	if (out._internals->is_gpu)
	{
		TENSOR_ERROR("CUDA backend not implemented yet.");
	}
	else
	{
		// Extract the data.
		float* out_data = out._internals->_data;
		float* ten0_data = ten0._internals->_data;
		float* ten1_data = ten1_summed._internals->_data;
		// Exptract the shapes.
		Shape out_shape = out.shape();
		Shape ten0_shape = ten0.shape();
		Shape ten1_shape = ten1_summed.shape();
		// Find the last non-unitary dimension to iterate through.
		unsigned last_long_dim = 0;
		for (unsigned i = 0; i < out_shape.dim(); i++)
			if (out_shape[i] > 1)
				last_long_dim = i;
		// Find the stride, will be 1 except for ten1 if broadcasting.
		const unsigned dout = out._stride[last_long_dim];
		const unsigned dten0 = ten0._stride[last_long_dim];
		const unsigned dten1 = ten1_summed._stride[last_long_dim];
		// Find the vector length to iterate.
		const unsigned vector_len = out_shape[last_long_dim];
		// Create a running shape to count.
		Shape counting_shape(last_long_dim, (int*)nullptr);
		// Iterate through vectors.
		while (true)
		{
			unsigned out_idx = 0, ten0_idx = 0, ten1_idx = 0;
			for (unsigned d = 0; d < counting_shape.dim(); d++)
			{
				out_idx += counting_shape[d] * out._stride[d];
				ten0_idx += counting_shape[d] * ten0._stride[d];
				ten1_idx += counting_shape[d] * ten1_summed._stride[d];
			}

			unsigned count = 0u;
			while (count++ < vector_len)
			{
				out_data[out_idx] = ten0_data[ten0_idx] + ten1_data[ten1_idx];
				out_idx += dout, ten0_idx += dten0, ten1_idx += dten1;
			}

			if (counting_shape.dim())
			{
				counting_shape[last_long_dim - 1]++;

				for (int d = last_long_dim - 1; d > 0; d--)
				{
					if (counting_shape[d] >= out_shape[d])
					{
						counting_shape[d] -= out_shape[d];
						counting_shape[d - 1]++;
					}
				}
				// If you reach the end of the leading dimension you're done.
				if (counting_shape[0] >= out_shape[0])
					break;
			}
			else
				break;
		}
	}

	// If it was a gradient operation store a AddOp instance.
	if (requires_grad)
		out._internals->op = new AddOp(ten0, ten1_summed, out);

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
	TENSOR_CHECK(ten0._internals,
		"Trying to subtract two tensors while the first tensor is empty."
	);
	TENSOR_CHECK(ten1._internals,
		"Trying to subtract two tensors while the second tensor is empty."
	);
	// Both tensors must be on the same device
	TENSOR_CHECK(ten0._internals->is_gpu == ten1._internals->is_gpu,
		"Trying to subtract two tensors in different devices is not allowed."
	);
	// Both tensor must have the same dimensionality to avoid ambiguity.
	TENSOR_CHECK(ten0.dim() == ten1.dim(),
		"Trying to subtract two tensors with different dimensions.\n"
		"For broadcast addition please use view() or squeeze()/unsqueeze() to make dimensions match.\n"
		"Found shapes | Tensor0: %s | Tensor1: %s", ten0._view.str(), ten1._view.str()
	);
	// Both tensors must either have the same dimension sizes or one of them have size one.
	// So that they can be broadcasted or summed without ambiguity.
	for (unsigned i = 0; i < ten0.dim(); i++)
		TENSOR_CHECK(ten0.size(i) == 1 || ten1.size(i) == 1 || ten0.size(i) == ten1.size(i),
			"Trying to subtract two tensors with incompatible shapes.\n"
			"Make sure shapes are compatible, meaning they have the same sizes or one of them is unitary for broadcasting.\n"
			"Found shapes | Tensor0: %s | Tensor1: %s", ten0._view.str(), ten1._view.str()
		);

	// The output will have gradient if either of the inputs has.
	bool requires_grad = ten0.has_grad() || ten1.has_grad();

	// Create output with the same shape as first tensor.
	Tensor out(ten0.shape(), ten0.device(), requires_grad);

	// Sum dimensions on ten1 if necessary.
	Tensor ten1_summed = ten1;
	for (unsigned i = 0; i < ten0.dim(); i++)
		if (ten0.size(i) == 1 && ten1_summed.size(i) != 1)
			ten1_summed = ten1_summed.sum(i, true);

	// Now we actually sum the tensors.
	if (out._internals->is_gpu)
	{
		TENSOR_ERROR("CUDA backend not implemented yet.");
	}
	else
	{
		// Extract the data.
		float* out_data = out._internals->_data;
		float* ten0_data = ten0._internals->_data;
		float* ten1_data = ten1_summed._internals->_data;
		// Exptract the shapes.
		Shape out_shape = out.shape();
		Shape ten0_shape = ten0.shape();
		Shape ten1_shape = ten1_summed.shape();
		// Find the last non-unitary dimension to iterate through.
		unsigned last_long_dim = 0;
		for (unsigned i = 0; i < out_shape.dim(); i++)
			if (out_shape[i] > 1)
				last_long_dim = i;
		// Find the stride, will be 1 except for ten1 if broadcasting.
		const unsigned dout = out._stride[last_long_dim];
		const unsigned dten0 = ten0._stride[last_long_dim];
		const unsigned dten1 = ten1_summed._stride[last_long_dim];
		// Find the vector length to iterate.
		const unsigned vector_len = out_shape[last_long_dim];
		// Create a running shape to count.
		Shape counting_shape(last_long_dim, (int*)nullptr);
		// Iterate through vectors.
		while (true)
		{
			unsigned out_idx = 0, ten0_idx = 0, ten1_idx = 0;
			for (unsigned d = 0; d < counting_shape.dim(); d++)
			{
				out_idx += counting_shape[d] * out._stride[d];
				ten0_idx += counting_shape[d] * ten0._stride[d];
				ten1_idx += counting_shape[d] * ten1_summed._stride[d];
			}

			unsigned count = 0u;
			while (count++ < vector_len)
			{
				out_data[out_idx] = ten0_data[ten0_idx] - ten1_data[ten1_idx];
				out_idx += dout, ten0_idx += dten0, ten1_idx += dten1;
			}

			if (counting_shape.dim())
			{
				counting_shape[last_long_dim - 1]++;

				for (int d = last_long_dim - 1; d > 0; d--)
				{
					if (counting_shape[d] >= out_shape[d])
					{
						counting_shape[d] -= out_shape[d];
						counting_shape[d - 1]++;
					}
				}
				// If you reach the end of the leading dimension you're done.
				if (counting_shape[0] >= out_shape[0])
					break;
			}
			else
				break;
		}
	}

	// If it was a gradient operation store a SubOp instance.
	if (requires_grad)
		out._internals->op = new SubOp(ten0, ten1_summed, out);

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
	TENSOR_CHECK(ten0._internals,
		"Trying to multiply two tensors while the first tensor is empty."
	);
	TENSOR_CHECK(ten1._internals,
		"Trying to multiply two tensors while the second tensor is empty."
	);
	// Both tensors must be on the same device
	TENSOR_CHECK(ten0._internals->is_gpu == ten1._internals->is_gpu,
		"Trying to multiply two tensors in different devices is not allowed."
	);
	// Both tensor must have the same dimensionality to avoid ambiguity.
	TENSOR_CHECK(ten0.dim() == ten1.dim(),
		"Trying to multiply two tensors with different dimensions.\n"
		"For broadcast addition please use view() or squeeze()/unsqueeze() to make dimensions match.\n"
		"Found shapes | Tensor0: %s | Tensor1: %s", ten0._view.str(), ten1._view.str()
	);
	// Both tensors must either have the same dimension sizes or one of them have size one.
	// So that they can be broadcasted or summed without ambiguity.
	for (unsigned i = 0; i < ten0.dim(); i++)
		TENSOR_CHECK(ten0.size(i) == 1 || ten1.size(i) == 1 || ten0.size(i) == ten1.size(i),
			"Trying to multiply two tensors with incompatible shapes.\n"
			"Make sure shapes are compatible, meaning they have the same sizes or one of them is unitary for broadcasting.\n"
			"Found shapes | Tensor0: %s | Tensor1: %s", ten0._view.str(), ten1._view.str()
		);

	// The output will have gradient if either of the inputs has.
	bool requires_grad = ten0.has_grad() || ten1.has_grad();

	// Create output with the same shape as first tensor.
	Tensor out(ten0.shape(), ten0.device(), requires_grad);

	// Sum dimensions on ten1 if necessary.
	Tensor ten1_summed = ten1;
	for (unsigned i = 0; i < ten0.dim(); i++)
		if (ten0.size(i) == 1 && ten1_summed.size(i) != 1)
			ten1_summed = ten1_summed.sum(i, true);

	// Now we actually sum the tensors.
	if (out._internals->is_gpu)
	{
		TENSOR_ERROR("CUDA backend not implemented yet.");
	}
	else
	{
		// Extract the data.
		float* out_data = out._internals->_data;
		float* ten0_data = ten0._internals->_data;
		float* ten1_data = ten1_summed._internals->_data;
		// Exptract the shapes.
		Shape out_shape = out.shape();
		Shape ten0_shape = ten0.shape();
		Shape ten1_shape = ten1_summed.shape();
		// Find the last non-unitary dimension to iterate through.
		unsigned last_long_dim = 0;
		for (unsigned i = 0; i < out_shape.dim(); i++)
			if (out_shape[i] > 1)
				last_long_dim = i;
		// Find the stride, will be 1 except for ten1 if broadcasting.
		const unsigned dout = out._stride[last_long_dim];
		const unsigned dten0 = ten0._stride[last_long_dim];
		const unsigned dten1 = ten1_summed._stride[last_long_dim];
		// Find the vector length to iterate.
		const unsigned vector_len = out_shape[last_long_dim];
		// Create a running shape to count.
		Shape counting_shape(last_long_dim, (int*)nullptr);
		// Iterate through vectors.
		while (true)
		{
			unsigned out_idx = 0, ten0_idx = 0, ten1_idx = 0;
			for (unsigned d = 0; d < counting_shape.dim(); d++)
			{
				out_idx += counting_shape[d] * out._stride[d];
				ten0_idx += counting_shape[d] * ten0._stride[d];
				ten1_idx += counting_shape[d] * ten1_summed._stride[d];
			}

			unsigned count = 0u;
			while (count++ < vector_len)
			{
				out_data[out_idx] = ten0_data[ten0_idx] * ten1_data[ten1_idx];
				out_idx += dout, ten0_idx += dten0, ten1_idx += dten1;
			}

			if (counting_shape.dim())
			{
				counting_shape[last_long_dim - 1]++;

				for (int d = last_long_dim - 1; d > 0; d--)
				{
					if (counting_shape[d] >= out_shape[d])
					{
						counting_shape[d] -= out_shape[d];
						counting_shape[d - 1]++;
					}
				}
				// If you reach the end of the leading dimension you're done.
				if (counting_shape[0] >= out_shape[0])
					break;
			}
			else
				break;
		}
	}

	// If it was a gradient operation store a MulOp instance.
	if (requires_grad)
		out._internals->op = new MulOp(ten0, ten1_summed, out);

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
	TENSOR_CHECK(ten0._internals,
		"Trying to add two tensors while the first tensor is empty."
	);
	TENSOR_CHECK(ten1._internals,
		"Trying to add two tensors while the second tensor is empty."
	);
	// Both tensors must be on the same device
	TENSOR_CHECK(ten0._internals->is_gpu == ten1._internals->is_gpu,
		"Trying to add two tensors in different devices is not allowed."
	);
	// Both tensor must have the same dimensionality to avoid ambiguity.
	TENSOR_CHECK(ten0.dim() == ten1.dim(),
		"Trying to add two tensors with different dimensions.\n"
		"For broadcast addition please use view() or squeeze()/unsqueeze() to make dimensions match.\n"
		"Found shapes | Tensor0: %s | Tensor1: %s", ten0._view.str(), ten1._view.str()
	);
	// Both tensors must either have the same dimension sizes or one of them have size one.
	// So that they can be broadcasted or summed without ambiguity.
	for (unsigned i = 0; i < ten0.dim(); i++)
		TENSOR_CHECK(ten0.size(i) == 1 || ten1.size(i) == 1 || ten0.size(i) == ten1.size(i),
			"Trying to add two tensors with incompatible shapes.\n"
			"Make sure shapes are compatible, meaning they have the same sizes or one of them is one for broadcasting.\n"
			"Found shapes | Tensor0: %s | Tensor1: %s", ten0._view.str(), ten1._view.str()
		);

	// The output will have gradient if either of the inputs has.
	bool requires_grad = ten0.has_grad() || ten1.has_grad();

	// Create output with the same shape as first tensor.
	Tensor out(ten0.shape(), ten0.device(), requires_grad);

	// Sum dimensions on ten1 if necessary.
	Tensor ten1_summed = ten1;
	for (unsigned i = 0; i < ten0.dim(); i++)
		if (ten0.size(i) == 1 && ten1_summed.size(i) != 1)
			ten1_summed = ten1_summed.sum(i, true);

	// Now we actually divide the tensors.
	if (out._internals->is_gpu)
	{
		TENSOR_ERROR("CUDA backend not implemented yet.");
	}
	else
	{
		// Extract the data.
		float* out_data = out._internals->_data;
		float* ten0_data = ten0._internals->_data;
		float* ten1_data = ten1_summed._internals->_data;
		// Exptract the shapes.
		Shape out_shape = out.shape();
		Shape ten0_shape = ten0.shape();
		Shape ten1_shape = ten1_summed.shape();
		// Find the last non-unitary dimension to iterate through.
		unsigned last_long_dim = 0;
		for (unsigned i = 0; i < out_shape.dim(); i++)
			if (out_shape[i] > 1)
				last_long_dim = i;
		// Find the stride, will be 1 except for ten1 if broadcasting.
		const unsigned dout = out._stride[last_long_dim];
		const unsigned dten0 = ten0._stride[last_long_dim];
		const unsigned dten1 = ten1_summed._stride[last_long_dim];
		// Find the vector length to iterate.
		const unsigned vector_len = out_shape[last_long_dim];
		// Create a running shape to count.
		Shape counting_shape(last_long_dim, (int*)nullptr);
		// Iterate through vectors.
		while (true)
		{
			unsigned out_idx = 0, ten0_idx = 0, ten1_idx = 0;
			for (unsigned d = 0; d < counting_shape.dim(); d++)
			{
				out_idx += counting_shape[d] * out._stride[d];
				ten0_idx += counting_shape[d] * ten0._stride[d];
				ten1_idx += counting_shape[d] * ten1_summed._stride[d];
			}

			unsigned count = 0u;
			while (count++ < vector_len)
			{
				out_data[out_idx] = ten0_data[ten0_idx] / ten1_data[ten1_idx];
				out_idx += dout, ten0_idx += dten0, ten1_idx += dten1;
			}

			if (counting_shape.dim())
			{
				counting_shape[last_long_dim - 1]++;

				for (int d = last_long_dim - 1; d > 0; d--)
				{
					if (counting_shape[d] >= out_shape[d])
					{
						counting_shape[d] -= out_shape[d];
						counting_shape[d - 1]++;
					}
				}
				// If you reach the end of the leading dimension you're done.
				if (counting_shape[0] >= out_shape[0])
					break;
			}
			else
				break;
		}
	}

	// If it was a gradient operation store a DivOp instance.
	if (requires_grad)
		out._internals->op = new DivOp(ten0, ten1_summed, out);

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
	TENSOR_CHECK(ten._internals,
		"Trying to do scalar addition with an empty tensor."
	);

	// Create output with the same shape as tensor.
	Tensor out(ten.shape(), ten.device(), ten.has_grad());

	// Now we actually sum the tensors.
	if (out._internals->is_gpu)
	{
		TENSOR_ERROR("CUDA backend not implemented yet.");
	}
	else
	{
		// Extract the data.
		float* out_data = out._internals->_data;
		float* ten_data = ten._internals->_data;
		// Iterate through all elements.
		int idx = -1, _numel = int(ten.numel());
		while (++idx < _numel)
			out_data[idx] = ten_data[idx] + val;
	}

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
	TENSOR_CHECK(ten._internals,
		"Trying to do scalar multiplication with an empty tensor."
	);

	// Create output with the same shape as tensor.
	Tensor out(ten.shape(), ten.device(), ten.has_grad());

	// Now we actually multiply the tensors.
	if (out._internals->is_gpu)
	{
		TENSOR_ERROR("CUDA backend not implemented yet.");
	}
	else
	{
		// Extract the data.
		float* out_data = out._internals->_data;
		float* ten_data = ten._internals->_data;
		// Iterate through all elements.
		int idx = -1, _numel = int(ten.numel());
		while (++idx < _numel)
			out_data[idx] = ten_data[idx] * val;
	}

	// If it was a gradient operation store a ScaAddOp instance.
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
	TENSOR_CHECK(ten._internals,
		"Trying to do scalar subtraction with an empty tensor."
	);

	// Create output with the same shape as tensor.
	Tensor out(ten.shape(), ten.device(), ten.has_grad());

	// Now we actually subtract the tensors.
	if (out._internals->is_gpu)
	{
		TENSOR_ERROR("CUDA backend not implemented yet.");
	}
	else
	{
		// Extract the data.
		float* out_data = out._internals->_data;
		float* ten_data = ten._internals->_data;
		// Iterate through all elements.
		int idx = -1, _numel = int(ten.numel());
		while (++idx < _numel)
			out_data[idx] = val - ten_data[idx];
	}

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
	TENSOR_CHECK(ten._internals,
		"Trying to do scalar division with an empty tensor."
	);

	// Create output with the same shape as tensor.
	Tensor out(ten.shape(), ten.device(), ten.has_grad());

	// Now we actually divide the tensors.
	if (out._internals->is_gpu)
	{
		TENSOR_ERROR("CUDA backend not implemented yet.");
	}
	else
	{
		// Extract the data.
		float* out_data = out._internals->_data;
		float* ten_data = ten._internals->_data;
		// Iterate through all elements.
		int idx = -1, _numel = int(ten.numel());
		while (++idx < _numel)
			out_data[idx] = val / ten_data[idx];
	}

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
	TENSOR_CHECK(_internals,
		"Trying to negate an empty tensor."
	);

	// Create output with the same shape as tensor.
	Tensor out(shape(), device(), has_grad());

	// Now we actually negate the tensor.
	if (out._internals->is_gpu)
	{
		TENSOR_ERROR("CUDA backend not implemented yet.");
	}
	else
	{
		// Extract the data.
		float* out_data = out._internals->_data;
		float* ten_data = _internals->_data;
		// Iterate through all elements.
		int idx = -1, _numel = int(numel());
		while (++idx < _numel)
			out_data[idx] = -ten_data[idx];
	}

	// If it was a gradient operation store a NegOp instance.
	if (out.has_grad())
		out._internals->op = new NegOp(*this, out);

	// Return out.
	return out;
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
		out.internal_gradient() = gradient().copy(device(), false);
		return *this = out;
	}
	// If there is another instance make sure to not modify it.
	if (_internals && _internals->instances > 1)
		return *this = *this + other.no_grad();

	// --- Sanity checks ---
	
	// Both tensors must be initialized.
	TENSOR_CHECK(_internals,
		"Trying to add two tensors while the first tensor is empty."
	);
	TENSOR_CHECK(other._internals,
		"Trying to add two tensors while the second tensor is empty."
	);
	// Both tensors must be on the same device
	TENSOR_CHECK(_internals->is_gpu == other._internals->is_gpu,
		"Trying to add two tensors in different devices is not allowed."
	);
	// Both tensor must have the same dimensionality to avoid ambiguity.
	TENSOR_CHECK(dim() == other.dim(),
		"Trying to add two tensors with different dimensions.\n"
		"For broadcast addition please use view() or squeeze()/unsqueeze() to make dimensions match.\n"
		"Found shapes | Tensor0: %s | Tensor1: %s", _view.str(), other._view.str()
	);
	// Both tensors must either have the same dimension sizes or one of them have size one.
	// So that they can be broadcasted or summed without ambiguity.
	for (unsigned i = 0; i < dim(); i++)
		TENSOR_CHECK(size(i) == 1 || other.size(i) == 1 || size(i) == other.size(i),
			"Trying to add two tensors with incompatible shapes.\n"
			"Make sure shapes are compatible, meaning they have the same sizes or one of them is one for broadcasting.\n"
			"Found shapes | Tensor0: %s | Tensor1: %s", _view.str(), other._view.str()
		);

	// Sum dimensions on ten1 if necessary.
	Tensor other_summed = other.no_grad();
	for (unsigned i = 0; i < dim(); i++)
		if (size(i) == 1 && other_summed.size(i) != 1)
			other_summed = other_summed.sum(i, true);

	// Now we actually sum the tensors.
	if (_internals->is_gpu)
	{
		TENSOR_ERROR("CUDA backend not implemented yet.");
	}
	else
	{
		// Extract the data.
		float* out_data = _internals->_data;
		float* oth_data = other_summed._internals->_data;
		// Exptract the shapes.
		Shape out_shape = shape();
		Shape oth_shape = other_summed.shape();
		// Find the last non-unitary dimension to iterate through.
		unsigned last_long_dim = 0;
		for (unsigned i = 0; i < out_shape.dim(); i++)
			if (out_shape[i] > 1)
				last_long_dim = i;
		// Find the stride, will be 1 except for ten1 if broadcasting.
		const unsigned dout = _stride[last_long_dim];
		const unsigned doth = other_summed._stride[last_long_dim];
		// Find the vector length to iterate.
		const unsigned vector_len = out_shape[last_long_dim];
		// Create a running shape to count.
		Shape counting_shape(last_long_dim, (int*)nullptr);
		// Iterate through vectors.
		while (true)
		{
			unsigned out_idx = 0, oth_idx = 0;
			for (unsigned d = 0; d < counting_shape.dim(); d++)
			{
				out_idx += counting_shape[d] * _stride[d];
				oth_idx += counting_shape[d] * other_summed._stride[d];
			}

			unsigned count = 0u;
			while (count++ < vector_len)
			{
				out_data[out_idx] += oth_data[oth_idx];
				out_idx += dout, oth_idx += doth;
			}

			if (counting_shape.dim())
			{
				counting_shape[last_long_dim - 1]++;

				for (int d = last_long_dim - 1; d > 0; d--)
				{
					if (counting_shape[d] >= out_shape[d])
					{
						counting_shape[d] -= out_shape[d];
						counting_shape[d - 1]++;
					}
				}
				// If you reach the end of the leading dimension you're done.
				if (counting_shape[0] >= out_shape[0])
					break;
			}
			else
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
		out.internal_gradient() = gradient().copy(device(), false);
		return *this = out;
	}
	// If there is another instance make sure to not modify it.
	if (_internals && _internals->instances > 1)
		return *this = *this - other.no_grad();

	// --- Sanity checks ---

	// Both tensors must be initialized.
	TENSOR_CHECK(_internals,
		"Trying to subtract two tensors while the first tensor is empty."
	);
	TENSOR_CHECK(other._internals,
		"Trying to subtract two tensors while the second tensor is empty."
	);
	// Both tensors must be on the same device
	TENSOR_CHECK(_internals->is_gpu == other._internals->is_gpu,
		"Trying to subtract two tensors in different devices is not allowed."
	);
	// Both tensor must have the same dimensionality to avoid ambiguity.
	TENSOR_CHECK(dim() == other.dim(),
		"Trying to subtract two tensors with different dimensions.\n"
		"For broadcast addition please use view() or squeeze()/unsqueeze() to make dimensions match.\n"
		"Found shapes | Tensor0: %s | Tensor1: %s", _view.str(), other._view.str()
	);
	// Both tensors must either have the same dimension sizes or one of them have size one.
	// So that they can be broadcasted or summed without ambiguity.
	for (unsigned i = 0; i < dim(); i++)
		TENSOR_CHECK(size(i) == 1 || other.size(i) == 1 || size(i) == other.size(i),
			"Trying to subtract two tensors with incompatible shapes.\n"
			"Make sure shapes are compatible, meaning they have the same sizes or one of them is one for broadcasting.\n"
			"Found shapes | Tensor0: %s | Tensor1: %s", _view.str(), other._view.str()
		);

	// Sum dimensions on ten1 if necessary.
	Tensor other_summed = other.no_grad();
	for (unsigned i = 0; i < dim(); i++)
		if (size(i) == 1 && other_summed.size(i) != 1)
			other_summed = other_summed.sum(i, true);

	// Now we actually subtract the tensors.
	if (_internals->is_gpu)
	{
		TENSOR_ERROR("CUDA backend not implemented yet.");
	}
	else
	{
		// Extract the data.
		float* out_data = _internals->_data;
		float* oth_data = other_summed._internals->_data;
		// Exptract the shapes.
		Shape out_shape = shape();
		Shape oth_shape = other_summed.shape();
		// Find the last non-unitary dimension to iterate through.
		unsigned last_long_dim = 0;
		for (unsigned i = 0; i < out_shape.dim(); i++)
			if (out_shape[i] > 1)
				last_long_dim = i;
		// Find the stride, will be 1 except for ten1 if broadcasting.
		const unsigned dout = _stride[last_long_dim];
		const unsigned doth = other_summed._stride[last_long_dim];
		// Find the vector length to iterate.
		const unsigned vector_len = out_shape[last_long_dim];
		// Create a running shape to count.
		Shape counting_shape(last_long_dim, (int*)nullptr);
		// Iterate through vectors.
		while (true)
		{
			unsigned out_idx = 0, oth_idx = 0;
			for (unsigned d = 0; d < counting_shape.dim(); d++)
			{
				out_idx += counting_shape[d] * _stride[d];
				oth_idx += counting_shape[d] * other_summed._stride[d];
			}

			unsigned count = 0u;
			while (count++ < vector_len)
			{
				out_data[out_idx] -= oth_data[oth_idx];
				out_idx += dout, oth_idx += doth;
			}

			if (counting_shape.dim())
			{
				counting_shape[last_long_dim - 1]++;

				for (int d = last_long_dim - 1; d > 0; d--)
				{
					if (counting_shape[d] >= out_shape[d])
					{
						counting_shape[d] -= out_shape[d];
						counting_shape[d - 1]++;
					}
				}
				// If you reach the end of the leading dimension you're done.
				if (counting_shape[0] >= out_shape[0])
					break;
			}
			else
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
		out.internal_gradient() = gradient().copy(device(), false);
		return *this = out;
	}
	// If there is another instance make sure to not modify it.
	if (_internals && _internals->instances > 1)
		return *this = *this * other.no_grad();

	// --- Sanity checks ---

	// Both tensors must be initialized.
	TENSOR_CHECK(_internals,
		"Trying to multiply two tensors while the first tensor is empty."
	);
	TENSOR_CHECK(other._internals,
		"Trying to multiply two tensors while the second tensor is empty."
	);
	// Both tensors must be on the same device
	TENSOR_CHECK(_internals->is_gpu == other._internals->is_gpu,
		"Trying to multiply two tensors in different devices is not allowed."
	);
	// Both tensor must have the same dimensionality to avoid ambiguity.
	TENSOR_CHECK(dim() == other.dim(),
		"Trying to multiply two tensors with different dimensions.\n"
		"For broadcast addition please use view() or squeeze()/unsqueeze() to make dimensions match.\n"
		"Found shapes | Tensor0: %s | Tensor1: %s", _view.str(), other._view.str()
	);
	// Both tensors must either have the same dimension sizes or one of them have size one.
	// So that they can be broadcasted or summed without ambiguity.
	for (unsigned i = 0; i < dim(); i++)
		TENSOR_CHECK(size(i) == 1 || other.size(i) == 1 || size(i) == other.size(i),
			"Trying to multiply two tensors with incompatible shapes.\n"
			"Make sure shapes are compatible, meaning they have the same sizes or one of them is one for broadcasting.\n"
			"Found shapes | Tensor0: %s | Tensor1: %s", _view.str(), other._view.str()
		);

	// Sum dimensions on ten1 if necessary.
	Tensor other_summed = other.no_grad();
	for (unsigned i = 0; i < dim(); i++)
		if (size(i) == 1 && other_summed.size(i) != 1)
			other_summed = other_summed.sum(i, true);

	// Now we actually sum the tensors.
	if (_internals->is_gpu)
	{
		TENSOR_ERROR("CUDA backend not implemented yet.");
	}
	else
	{
		// Extract the data.
		float* out_data = _internals->_data;
		float* oth_data = other_summed._internals->_data;
		// Exptract the shapes.
		Shape out_shape = shape();
		Shape oth_shape = other_summed.shape();
		// Find the last non-unitary dimension to iterate through.
		unsigned last_long_dim = 0;
		for (unsigned i = 0; i < out_shape.dim(); i++)
			if (out_shape[i] > 1)
				last_long_dim = i;
		// Find the stride, will be 1 except for ten1 if broadcasting.
		const unsigned dout = _stride[last_long_dim];
		const unsigned doth = other_summed._stride[last_long_dim];
		// Find the vector length to iterate.
		const unsigned vector_len = out_shape[last_long_dim];
		// Create a running shape to count.
		Shape counting_shape(last_long_dim, (int*)nullptr);
		// Iterate through vectors.
		while (true)
		{
			unsigned out_idx = 0, oth_idx = 0;
			for (unsigned d = 0; d < counting_shape.dim(); d++)
			{
				out_idx += counting_shape[d] * _stride[d];
				oth_idx += counting_shape[d] * other_summed._stride[d];
			}

			unsigned count = 0u;
			while (count++ < vector_len)
			{
				out_data[out_idx] *= oth_data[oth_idx];
				out_idx += dout, oth_idx += doth;
			}

			if (counting_shape.dim())
			{
				counting_shape[last_long_dim - 1]++;

				for (int d = last_long_dim - 1; d > 0; d--)
				{
					if (counting_shape[d] >= out_shape[d])
					{
						counting_shape[d] -= out_shape[d];
						counting_shape[d - 1]++;
					}
				}
				// If you reach the end of the leading dimension you're done.
				if (counting_shape[0] >= out_shape[0])
					break;
			}
			else
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
		out.internal_gradient() = gradient().copy(device(), false);
		return *this = out;
	}
	// If there is another instance make sure to not modify it.
	if (_internals && _internals->instances > 1)
		return *this = *this / other.no_grad();

	// --- Sanity checks ---

// Both tensors must be initialized.
	TENSOR_CHECK(_internals,
		"Trying to divide two tensors while the first tensor is empty."
	);
	TENSOR_CHECK(other._internals,
		"Trying to divide two tensors while the second tensor is empty."
	);
	// Both tensors must be on the same device
	TENSOR_CHECK(_internals->is_gpu == other._internals->is_gpu,
		"Trying to divide two tensors in different devices is not allowed."
	);
	// Both tensor must have the same dimensionality to avoid ambiguity.
	TENSOR_CHECK(dim() == other.dim(),
		"Trying to divide two tensors with different dimensions.\n"
		"For broadcast addition please use view() or squeeze()/unsqueeze() to make dimensions match.\n"
		"Found shapes | Tensor0: %s | Tensor1: %s", _view.str(), other._view.str()
	);
	// Both tensors must either have the same dimension sizes or one of them have size one.
	// So that they can be broadcasted or summed without ambiguity.
	for (unsigned i = 0; i < dim(); i++)
		TENSOR_CHECK(size(i) == 1 || other.size(i) == 1 || size(i) == other.size(i),
			"Trying to divide two tensors with incompatible shapes.\n"
			"Make sure shapes are compatible, meaning they have the same sizes or one of them is one for broadcasting.\n"
			"Found shapes | Tensor0: %s | Tensor1: %s", _view.str(), other._view.str()
		);

	// Sum dimensions on ten1 if necessary.
	Tensor other_summed = other.no_grad();
	for (unsigned i = 0; i < dim(); i++)
		if (size(i) == 1 && other_summed.size(i) != 1)
			other_summed = other_summed.sum(i, true);

	// Now we actually divide the tensors.
	if (_internals->is_gpu)
	{
		TENSOR_ERROR("CUDA backend not implemented yet.");
	}
	else
	{
		// Extract the data.
		float* out_data = _internals->_data;
		float* oth_data = other_summed._internals->_data;
		// Exptract the shapes.
		Shape out_shape = shape();
		Shape oth_shape = other_summed.shape();
		// Find the last non-unitary dimension to iterate through.
		unsigned last_long_dim = 0;
		for (unsigned i = 0; i < out_shape.dim(); i++)
			if (out_shape[i] > 1)
				last_long_dim = i;
		// Find the stride, will be 1 except for ten1 if broadcasting.
		const unsigned dout = _stride[last_long_dim];
		const unsigned doth = other_summed._stride[last_long_dim];
		// Find the vector length to iterate.
		const unsigned vector_len = out_shape[last_long_dim];
		// Create a running shape to count.
		Shape counting_shape(last_long_dim, (int*)nullptr);
		// Iterate through vectors.
		while (true)
		{
			unsigned out_idx = 0, oth_idx = 0;
			for (unsigned d = 0; d < counting_shape.dim(); d++)
			{
				out_idx += counting_shape[d] * _stride[d];
				oth_idx += counting_shape[d] * other_summed._stride[d];
			}

			unsigned count = 0u;
			while (count++ < vector_len)
			{
				out_data[out_idx] /= oth_data[oth_idx];
				out_idx += dout, oth_idx += doth;
			}

			if (counting_shape.dim())
			{
				counting_shape[last_long_dim - 1]++;

				for (int d = last_long_dim - 1; d > 0; d--)
				{
					if (counting_shape[d] >= out_shape[d])
					{
						counting_shape[d] -= out_shape[d];
						counting_shape[d - 1]++;
					}
				}
				// If you reach the end of the leading dimension you're done.
				if (counting_shape[0] >= out_shape[0])
					break;
			}
			else
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
			out.internal_gradient() = gradient().copy(device(), false);
		return *this = out;
	}

	// Tensor must be initialized.
	TENSOR_CHECK(_internals,
		"Trying to add two tensors while the first tensor is empty."
	);

	// Now we actually sum the tensors.
	if (_internals->is_gpu)
	{
		TENSOR_ERROR("CUDA backend not implemented yet.");
	}
	else
	{
		// Extract the data.
		float* out_data = _internals->_data;
		// Iterate through all elements.
		int idx = -1, _numel = int(numel());
		while (++idx < _numel)
			out_data[idx] += val;
	}

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
			out.internal_gradient() = gradient().copy(device(), false);
		return *this = out;
	}

	// Tensor must be initialized.
	TENSOR_CHECK(_internals,
		"Trying to multiply by a scalar on an empty tensor."
	);

	// Now we actually multiply the tensors.
	if (_internals->is_gpu)
	{
		TENSOR_ERROR("CUDA backend not implemented yet.");
	}
	else
	{
		// Extract the data.
		float* out_data = _internals->_data;
		// Iterate through all elements.
		int idx = -1, _numel = int(numel());
		while (++idx < _numel)
			out_data[idx] *= val;
	}

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

Tensor Functional::matmul(const Tensor& mat0, const Tensor& mat1)
{
	// Matrix multiplication tensor operator for backpropagation.
	class MatMulOp : public Tensor::TensorInternals::TensorOp
	{
		// Tensor copy storage for the input matrices.
		Tensor mat0, mat1;

	public:
		// Constructor, stores all the data of the operation.
		MatMulOp(const Tensor& _mat0, const Tensor& _mat1, const Tensor& _out) : TensorOp{ "Matrix Multiplication", _out },
			mat0{ _mat0 }, mat1{ _mat1 }
		{
			_relatives[0] = &mat0;
			_relatives[1] = &mat1;
		}

		// Backpropagation. For matmul the gradient gets multiplied by the transposed other.
		// The broadcasting logic of matmul and the '+=' operator handles shapes.
		void _backward() override
		{
			if (mat0.has_grad()) mat0.internal_gradient() += matmul(out.gradient(), mat1.no_grad().transpose(-1, -2));
			if (mat1.has_grad()) mat1.internal_gradient() += matmul(mat0.no_grad().transpose(-1, -2), out.gradient());
		}
	};

	// Both tensor must be initialized.
	TENSOR_CHECK(mat0._internals,
		"Trying to matrix multiply two tensors while the first tensor is empty."
	);
	TENSOR_CHECK(mat1._internals,
		"Trying to matrix multiply two tensors while the second tensor is empty."
	);
	// Both tensors must be on the same device
	TENSOR_CHECK(mat0._internals->is_gpu == mat1._internals->is_gpu,
		"Trying to matrix multiply two tensors in different devices is not allowed."
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
	TENSOR_CHECK(A._view[-1] == B._view[-2],
		"Incompatible dimensions found inside a matmul() call.\n"
		"Please make sure your tensors follow proper matrix multiplication logic (...,M,K) @ (...,K,N) = (...,M,N).\n"
		"Matrix0 shape: %s | Matrix1 shape: %s", mat0._view.str(), mat1._view.str()
	);

	// Make sure other dimensions are broadcastable.
	for (unsigned i = 0; i < A.dim() - 2; i++)
		TENSOR_CHECK(A._view[i] == B._view[i] || A._view[i] == 1 || B._view[i] == 1,
			"Trying to matrix multiply two tensors with incompatible shapes.\n"
			"Make sure leading shapes are compatible, meaning they have the same sizes or one of them is unitary for broadcasting.\n"
			"Found shapes | Unsqueezed Matrix0: %s | Unsqueezed Matrix1: %s", A._view.str(), B._view.str()
		);

	// Prepare output shape.
	Shape out_shape(A.dim(), (int*)nullptr);
	out_shape[-1] = B._view[-1];
	out_shape[-2] = A._view[-2];
	for (unsigned i = 0; i < A.dim() - 2; i++)
		out_shape[i] = A._view[i] > B._view[i] ? A._view[i] : B._view[i];

	// Create output tensor with this shape.
	Tensor out(out_shape, A.device(), requires_grad);

	// Get matrix multiplication data.
	unsigned M = out_shape[-2];
	unsigned N = out_shape[-1];
	unsigned K = A._view[-1];

	// Now we actually multiply the matrices.
	if (out._internals->is_gpu)
	{
		TENSOR_ERROR("CUDA backend not implemented yet.");
	}
	else
	{
		// Extract the data.
		float* out_data = out._internals->_data;
		float* A_data = A._internals->_data;
		float* B_data = B._internals->_data;
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

		// Iterate through vectors.
		while (true)
		{
			// Get initial pointer idxs given running shape.
			unsigned out_idx = 0, A_idx = 0, B_idx = 0;
			for (unsigned d = 0; d < counting_shape.dim(); d++)
			{
				out_idx += counting_shape[d] * out._stride[d];
				A_idx += counting_shape[d] * A._stride[d];
				B_idx += counting_shape[d] * B._stride[d];
			}
			// Get matrix pointers.
			float* out_matrix = out_data + out_idx; // M x N values
			float* A_matrix = A_data + A_idx;		// M x K values
			float* B_matrix = B_data + B_idx;		// K x N values

			// Here we actually multiply the matrices.
			{
				for (unsigned i0 = 0; i0 < M; i0 += 64)
				{
					const unsigned i1 = (i0 + 64 < M) ? (i0 + 64) : M;

					for (unsigned k0 = 0; k0 < K; k0 += 64)
					{
						const unsigned k1 = (k0 + 64 < K) ? (k0 + 64) : K;

						for (unsigned j0 = 0; j0 < N; j0 += 64)
						{
							const unsigned j1 = (j0 + 64 < N) ? (j0 + 64) : N;

							for (unsigned i = i0; i < i1; ++i)
							{
								float* c_row = out_matrix + i * N;

								for (unsigned k = k0; k < k1; ++k)
								{
									const float a = A_matrix[i * K + k];
									const float* b_row = B_matrix + k * N;

									for (unsigned j = j0; j < j1; ++j)
										c_row[j] += a * b_row[j];
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
		out._internals->op = new MatMulOp(A, B, out);

	// Squeeze back if necessary.
	if (a_was_1d) out = out.squeeze(-2);
	if (b_was_1d) out = out.squeeze(-1);

	// Return out.
	return out;
}

Tensor Functional::matmul(const Tensor& mat0, const Tensor& mat1, const Tensor& bias)
{
	// Matrix multiplication with bias tensor operator for backpropagation.
	class MatMulBiasOp : public Tensor::TensorInternals::TensorOp
	{
		// Tensor copy storage for the input matrices.
		Tensor mat0, mat1, bias;

	public:
		// Constructor, stores all the data of the operation.
		MatMulBiasOp(const Tensor& _mat0, const Tensor& _mat1, const Tensor& _bias, const Tensor& _out) : TensorOp{ "Matrix Multiplication with Bias", _out },
			mat0{ _mat0 }, mat1{ _mat1 }, bias{ _bias }
		{
			if (mat0.has_grad()) _relatives[0] = &mat0;
			if (mat1.has_grad()) _relatives[1] = &mat1;
			if (bias.has_grad()) _relatives[2] = &bias;
		}

		// Backpropagation. For matmul the gradient gets multiplied by the transposed other.
		// The broadcasting logic of matmul and the '+=' operator handles shapes.
		void _backward() override
		{
			if (mat0.has_grad()) mat0.internal_gradient() += matmul(out.gradient(), mat1.no_grad().transpose(-1, -2));
			if (mat1.has_grad()) mat1.internal_gradient() += matmul(mat0.no_grad().transpose(-1, -2), out.gradient());
			if (bias.has_grad()) bias.internal_gradient() += out.gradient();
		}
	};

	// Both tensor must be initialized.
	TENSOR_CHECK(mat0._internals,
		"Trying to matrix multiply with bias tensors while the first tensor is empty."
	);
	TENSOR_CHECK(mat1._internals,
		"Trying to matrix multiply with bias tensors while the second tensor is empty."
	);
	TENSOR_CHECK(bias._internals,
		"Trying to matrix multiply with bias tensors while the bias tensor is empty."
	);
	// Both tensors must be on the same device
	TENSOR_CHECK(mat0._internals->is_gpu == mat1._internals->is_gpu && mat1._internals->is_gpu == bias._internals->is_gpu,
		"Trying to matrix multiply with bias tensors in different devices is not allowed."
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
	TENSOR_CHECK(A._view[-1] == B._view[-2],
		"Incompatible dimensions found inside a matmul() call.\n"
		"Please make sure your tensors follow proper matrix multiplication logic (...,M,K) @ (...,K,N) = (...,M,N).\n"
		"Matrix0 shape: %s | Matrix1 shape: %s", mat0._view.str(), mat1._view.str()
	);

	// Make sure other dimensions are broadcastable.
	for (unsigned i = 0; i < A.dim() - 2; i++)
		TENSOR_CHECK(A._view[i] == B._view[i] || A._view[i] == 1 || B._view[i] == 1,
			"Trying to matrix multiply two tensors with incompatible shapes.\n"
			"Make sure leading shapes are compatible, meaning they have the same sizes or one of them is unitary for broadcasting.\n"
			"Found shapes | Unsqueezed Matrix0: %s | Unsqueezed Matrix1: %s", A._view.str(), B._view.str()
		);

	// Prepare output shape.
	Shape out_shape(A.dim(), (int*)nullptr);
	out_shape[-1] = B._view[-1];
	out_shape[-2] = A._view[-2];
	for (unsigned i = 0; i < A.dim() - 2; i++)
		out_shape[i] = A._view[i] > B._view[i] ? A._view[i] : B._view[i];

	// Make sure bias can be broadcasted to output.
	for (unsigned i = 0; i < out_shape.dim(); i++)
		TENSOR_CHECK(out_shape[i] == b._view[i] || b._view[i] == 1,
			"Trying to add a bias to a matmul tensor with incompatible shapes.\n"
			"Make sure leading shapes are compatible, meaning they have the same sizes or the bias one is unitary for broadcasting.\n"
			"Found shapes | Matmul output: %s | Unsqueezed bias: %s", out_shape.str(), b._view.str()
		);

	// Create output tensor with this shape.
	Tensor out(out_shape, A.device(), requires_grad);

	// Get matrix multiplication data.
	unsigned M = out_shape[-2];
	unsigned N = out_shape[-1];
	unsigned K = A._view[-1];

	// Now we actually multiply the matrices.
	if (out._internals->is_gpu)
	{
		TENSOR_ERROR("CUDA backend not implemented yet.");
	}
	else
	{
		// Extract the data.
		float* out_data = out._internals->_data;
		float* A_data = A._internals->_data;
		float* B_data = B._internals->_data;
		float* b_data = b._internals->_data;
		// Get bias last dimensions strides.
		unsigned bm_stride = b._stride[-2];
		unsigned bn_stride = b._stride[-1];
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

		// Iterate through vectors.
		while (true)
		{
			// Get initial pointer idxs given running shape.
			unsigned out_idx = 0, A_idx = 0, B_idx = 0, b_idx = 0;
			for (unsigned d = 0; d < counting_shape.dim(); d++)
			{
				out_idx += counting_shape[d] * out._stride[d];
				A_idx += counting_shape[d] * A._stride[d];
				B_idx += counting_shape[d] * B._stride[d];
				b_idx += counting_shape[d] * b._stride[d];
			}
			// Get matrix pointers.
			float* out_matrix = out_data + out_idx; // M x N values
			float* A_matrix = A_data + A_idx;		// M x K values
			float* B_matrix = B_data + B_idx;		// K x N values
			float* b_matrix = b_data + b_idx;		// M? x N? values

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
									const float a = A_matrix[i * K + k];
									const float* b_row = B_matrix + k * N;

									for (unsigned j = j0; j < j1; ++j)
										c_row[j] += a * b_row[j];
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
		out._internals->op = new MatMulBiasOp(A, B, b, out);

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
	TENSOR_CHECK(ten0._internals,
		"Trying to concatenate two tensors while the first tensor is empty."
	);
	TENSOR_CHECK(ten1._internals,
		"Trying to concatenate two tensors while the second tensor is empty."
	);
	// Both tensors must be on the same device
	TENSOR_CHECK(ten0._internals->is_gpu == ten1._internals->is_gpu,
		"Trying to concatenate two tensors in different devices is not allowed."
	);
	// Both tensor must have the same dimensionality to avoid ambiguity.
	TENSOR_CHECK(ten0.dim() == ten1.dim(),
		"Trying to concatenate two tensors with different dimensions.\n"
		"Found shapes | Tensor0: %s | Tensor1: %s", ten0._view.str(), ten1._view.str()
	);

	// Modulo dimension.
	dim = unsigned(dim + ten0.dim() * (2 - dim / int(ten0.dim()))) % ten0.dim();
	// Both tensors must have the same dimension sizes except at dim.
	for (unsigned i = 0; i < ten0.dim(); i++)
		TENSOR_CHECK(i == dim || ten0.size(i) == ten1.size(i),
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

	// Now we actually concatenate the tensors.
	if (out._internals->is_gpu)
	{
		TENSOR_ERROR("CUDA backend not implemented yet.");
	}
	else
	{
		// Extract the data.
		float* out_data = out._internals->_data;
		float* ten0_data = ten0._internals->_data;
		float* ten1_data = ten1._internals->_data;
		// Get relevant sizes.
		unsigned prev_size = 1u;
		unsigned post_size = 1u;
		for (int i = 0; i < dim; i++)
			post_size *= out._view[i];
		for (unsigned i = dim + 1; i < out._view.dim(); i++)
			prev_size *= out._view[i];
		// Get relevant strides.
		const unsigned elem_stride = out._stride[dim];
		const unsigned post_stride_out = elem_stride * out_shape[dim];

		// Start with tensor 0.
		{
			// Get tensor strides.
			const unsigned post_stride_ten = elem_stride * _size0;
			// Iterate thruough entire dimension size to write to output.
			for (unsigned i = 0; i < _size0; i++)
			{
				for (unsigned post_count = 0; post_count < post_size; post_count++)
				{
					unsigned ten_idx = i * elem_stride + post_count * post_stride_ten;
					unsigned out_idx = i * elem_stride + post_count * post_stride_out;

					for (unsigned prev_count = 0; prev_count < prev_size; prev_count++)
						out_data[out_idx++] = ten0_data[ten_idx++];
				}
			}
		}
		// Follow with tensor 1.
		{
			// Get tensor strides.
			const unsigned post_stride_ten = elem_stride * _size1;
			// Iterate thruough entire dimension size to write to output.
			for (unsigned i = 0; i < _size1; i++)
			{
				for (unsigned post_count = 0; post_count < post_size; post_count++)
				{
					unsigned ten_idx = i * elem_stride + post_count * post_stride_ten;
					unsigned out_idx = (i + _size0) * elem_stride + post_count * post_stride_out;

					for (unsigned prev_count = 0; prev_count < prev_size; prev_count++)
						out_data[out_idx++] = ten1_data[ten_idx++];
				}
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
			derivative *= 2.f * out.gradient().item() / count;
			if (y0.has_grad()) y0.internal_gradient() += derivative;
			if (y1.has_grad()) y1.internal_gradient() -= derivative;
		}
	};

	// --- Sanity checks ---

	// Both tensor must be initialized.
	TENSOR_CHECK(ten0._internals,
		"Trying to compute MSE of two tensors while the first tensor is empty."
	);
	TENSOR_CHECK(ten1._internals,
		"Trying to compute MSE of two tensors while the second tensor is empty."
	);
	// Both tensors must be on the same device
	TENSOR_CHECK(ten0._internals->is_gpu == ten1._internals->is_gpu,
		"Trying to compute MSE of two tensors in different devices is not allowed."
	);
	// Both tensor must have the same number of elements.
	TENSOR_CHECK(ten0.numel() == ten1.numel(),
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
	TENSOR_CHECK(size,
		"Trying to compute MSE of two tensors of zero elements is not allowed."
	);

	// Create output with single element.
	Tensor out(Shape(1), ten0.device(), requires_grad);

	// Now we actually compute the MSE.
	if (out._internals->is_gpu)
	{
		TENSOR_ERROR("CUDA backend not implemented yet.");
	}
	else
	{
		// Extract the data.
		float& out_data = *out._internals->_data;
		float* y0_data = y0._internals->_data;
		float* y1_data = y1._internals->_data;
		// Iterate through entire length.
		int idx = -1;
		while (++idx < size)
		{
			float d = y0_data[idx] - y1_data[idx];
			out_data += d * d;
		}
		// Divide at the end.
		out_data /= size;
	}

	// If it was a gradient operation store a CatOp instance.
	if (requires_grad)
		out._internals->op = new MseOp(y0, y1, out, size);

	// Return out.
	return out;
}

Tensor Functional::cross_entropy_loss(const Tensor& logits, unsigned* labels)
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
		CelOp(const Tensor& _logits, const Tensor& _probs, const Tensor& _out, unsigned* _labels) : TensorOp{ "Cross-Entropy Loss", _out },
			logits{ _logits }, probs{ _probs }, labels{ Functional::one_hot(_logits.shape(), _labels, _logits.device()) }, size{ _logits.size(0) }
		{
			_relatives[0] = &logits;
		}

		// Backpropagation. Gradient is prob_k - y_k.
		void _backward() override
		{
			logits.internal_gradient() += (out.gradient().item() / size) * (probs - labels);
		}
	};

	// Tensor must be initialized.
	TENSOR_CHECK(logits._internals,
		"Trying to apply softmax to an empty tensor."
	);
	// Make sure shape is correct.
	TENSOR_CHECK(logits.dim() == 2,
		"Invalid shape found in logits for cross-entropy loss.\n"
		"Make sure your logits have shape (n_cases, n_labels) to avoid any ambiguity.\n"
		"Logits shape found: %s", logits._view.str()
	);
	TENSOR_CHECK(logits._view[-1] > 1,
		"Invalid shape found in logits for cross-entropy loss.\n"
		"Make sure your logits have shape (n_cases, n_labels).\n" 
		"Found only one label, this makes the operation a no-op and is most likely unintentional.\n"
		"Logits shape found: %s", logits._view.str()
	);
	TENSOR_CHECK(logits._view[0] != 0,
		"Found zero number of cases for cross-entropy loss.\n"
		"There must be at least one case, void tensors are not allowed.\n"
		"Logits shape found: %s", logits._view.str()
	);

	// First do softmax without gradient.
	Tensor probs(logits.shape(), logits.device(), false);

	// Create single element output.
	Tensor out(Shape(1), logits.device(), logits.has_grad());

	// Get number of cases.
	unsigned size = logits._view[0];

	// Now compute the actual cross-entropy loss.
	if (out._internals->is_gpu)
	{
		TENSOR_ERROR("CUDA backend not implemented yet.");
	}
	else
	{
		// Extract the data.
		float& out_data = *out._internals->_data;
		float* probs_data = probs._internals->_data;
		float* logits_data = logits._internals->_data;
		// Get relevant stride.
		const unsigned n_labels = logits._view[-1];
		// Add all loss values and compute all probs.
		for (unsigned i = 0; i < size; i++)
		{
			float* p_probs = probs_data + i * n_labels;
			float* p_logits = logits_data + i * n_labels;
			
			TENSOR_CHECK(labels[i] < n_labels, 
				"Label out of range found inside cross-entropy loss computation.\n"
				"Make sure your labels are in the range [0, logits.size(-1) - 1]."
			);

			// First pass find maximum.
			float max = *p_logits;
			for (unsigned idx = 1; idx < n_labels; idx++)
				if (p_logits[idx] > max)
					max = p_logits[idx];
			// Second pass compute exponential and accumulate sum.
			float sum = 0.f;
			for (unsigned idx = 0; idx < n_labels; idx++)
			{
				p_probs[idx] = expf(p_logits[idx] - max);
				sum += p_probs[idx];
			}
			// Now compute loss.
			out_data += -p_logits[labels[i]] + max + logf(sum);

			// Third pass divide by sum.
			for (unsigned idx = 0; idx < n_labels; idx++)
				p_probs[idx] /= sum;
		}
		// Divide by number of cases.
		out_data /= size;
	}

	// If it was a gradient operation store a CelOp instance.
	if (logits.has_grad())
		out._internals->op = new CelOp(logits, probs, out, labels);

	// Return out.
	return out;
}

Tensor Functional::negative_log_likelihood(const Tensor& probs, unsigned* labels)
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
		NllOp(const Tensor& _probs, const Tensor& _out, unsigned* _labels) : TensorOp{ "Negative Log Likelihood", _out },
			probs{ _probs }, labels{ Functional::one_hot(_probs.shape(), _labels, _probs.device()) }, size{ _probs.size(0) }
		{
			_relatives[0] = &probs;
		}

		// Backpropagation. Gradient is -out_grad/prob_j if j is the correct label, else 0.
		void _backward() override
		{
			probs.internal_gradient() -= (out.gradient().item() / size) * (labels / probs.no_grad());
		}
	};

	// Tensor must be initialized.
	TENSOR_CHECK(probs._internals,
		"Trying to compute NLL with an empty tensor."
	);
	// Make sure shape is correct.
	TENSOR_CHECK(probs.dim() == 2,
		"Invalid shape found in probabilities for negative log likelihood.\n"
		"Make sure your probs have shape (n_cases, n_labels) to avoid any ambiguity.\n"
		"Probs shape found: %s", probs._view.str()
	);
	TENSOR_CHECK(probs._view[-1] > 1,
		"Invalid shape found in probabilities for negative log likelihood.\n"
		"Make sure your probs have shape (n_cases, n_labels).\n"
		"Found only one label, this makes the operation a no-op and is most likely unintentional.\n"
		"Probs shape found: %s", probs._view.str()
	);
	TENSOR_CHECK(probs._view[0] != 0,
		"Found zero number of cases for negative log likelihood.\n"
		"There must be at least one case, void tensors are not allowed.\n"
		"Probs shape found: %s", probs._view.str()
	);

	// Create single element output.
	Tensor out(Shape(1), probs.device(), probs.has_grad());

	// Get number of cases.
	unsigned size = probs._view[0];

	// Now compute the actual cross-entropy loss.
	if (out._internals->is_gpu)
	{
		TENSOR_ERROR("CUDA backend not implemented yet.");
	}
	else
	{
		// Extract the data.
		float& out_data = *out._internals->_data;
		float* probs_data = probs._internals->_data;
		// Get relevant stride.
		const unsigned n_labels = probs._view[-1];
		// Add all loss values.
		for (unsigned i = 0; i < size; i++)
		{
			TENSOR_CHECK(labels[i] < n_labels,
				"Label out of range found inside negative log likelihood computation.\n"
				"Make sure your labels are in the range [0, probs.size(-1) - 1]."
			);
			out_data += -logf(probs_data[i * n_labels + labels[i]]);
		}

		// Divide by number of cases.
		out_data /= size;
	}

	// If it was a gradient operation store a NllOp instance.
	if (probs.has_grad())
		out._internals->op = new NllOp(probs, out, labels);

	// Return out.
	return out;
}

Tensor Functional::one_hot(const Shape& shape, unsigned* labels, const char* device)
{
	// Size must be correct.
	TENSOR_CHECK(shape.dim() == 2,
		"Invalid shape found in one-hot encoding.\n"
		"Make sure your shape is (n_cases, n_labels) to avoid any ambiguity.\n"
		"Shape found: %s", shape.str()
	);
	TENSOR_CHECK(shape[-1] > 1,
		"Invalid shape found in one-hot encoding.\n"
		"Make sure your shape is (n_cases, n_labels).\n"
		"Found only one label, this makes the operation a no-op and is most likely unintentional.\n"
		"Shape found: %s", shape.str()
	);

	// Create output tensor.
	Tensor out(shape, device, false);

	// Now we one-hot these vectors.
	if (out._internals->is_gpu)
	{
		TENSOR_ERROR("CUDA backend not implemented yet.");
	}
	else
	{
		// Extract the data.
		float* out_data = out._internals->_data;
		// Get relevant stride.
		const unsigned n_labels = shape[-1];
		const unsigned n_cases = shape[0];
		// One-hot this.
		for (unsigned i = 0; i < n_cases; i++)
		{
			TENSOR_CHECK(labels[i] < n_labels,
				"Label out of range found inside a one-hot encoding.\n"
				"Make sure your labels are in the range [0, shape(-1) - 1]."
			);
			out_data[i * n_labels + labels[i]] = 1.f;
		}
	}

	// Return out.
	return out;
}

Tensor Functional::causal_mask(unsigned L, const char* device)
{
	// Create mask tensor.
	Tensor mask(Shape(L, L), device, false);

	// Set values to -inf.
	if (mask._internals->is_gpu)
	{
		TENSOR_ERROR("CUDA backend not implemented yet.");
	}
	else
	{
		// Extract the data.
		float* out_data = mask._internals->_data;

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

// Uses a randomizer to generate a shuffled set of integers from 0 to size.

void Random::shuffle(unsigned size, int* data)
{
	// Fisher–Yates
	for (unsigned i = size; i > 1; --i) {
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

const Tensor& Initialization::normal(const Tensor& tensor, float mean, float std)
{
	TENSOR_CHECK(tensor._internals,
		"Found an empty tensor inside unifor initialization, please make sure your tensor is initialized."
	);

	float* data = tensor._internals->_data;
	const unsigned numel = tensor.numel();

	unsigned idx = 0;
	while (idx < numel)
		data[idx++] = random_norm() * std + mean;

	return tensor;
}

const Tensor& Initialization::uniform(const Tensor& tensor, float min, float max)
{
	TENSOR_CHECK(tensor._internals,
		"Found an empty tensor inside unifor initialization, please make sure your tensor is initialized."
	);

	float* data = tensor._internals->_data;
	const unsigned numel = tensor.numel();

	unsigned idx = 0;
	while(idx < numel)
		data[idx++] = random_0_1() * (max - min) + min;
	
	return tensor;
}
