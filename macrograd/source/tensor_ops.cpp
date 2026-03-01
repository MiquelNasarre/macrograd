#include "macrograd.h"
#include "macrograd_error.h"
#include "cuda_backend.h"

#include <math.h>

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
	if (other._data)
		other._data->instances++;

	// Reduce your count.
	reduce_instances_count();

	// Adopt other's data.
	_data = other._data;

	// Also copy no-grad status and view.
	_is_no_grad = other._is_no_grad;
	_view = other._view;
	_offset = other._offset;

	return *this;
}

// Creates a new tensor with the same data but different view.

Tensor Tensor::view(const Shape& shape) const
{
	TENSOR_CHECK(_data,
		"Trying to call view on an empty tensor."
	);
	TENSOR_CHECK(shape.dim(),
		"Trying to change the view of a tensor with an empty shape."
	);

	Shape old_shape = _view;
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

		unsigned old_total_dim = 1;
		for (unsigned i = 0; i < old_shape.dim(); i++)
			old_total_dim *= old_shape[i];

		TENSOR_CHECK(old_total_dim % new_total_dim == 0,
			"Unreconcileable shapes found inside a view call, total sizes are not divisible.\n"
			"Old Shape: %s | View Input Shape: %s.", _view.str(), shape.str()
		);

		new_shape[neg_one] = old_total_dim / new_total_dim;
	}
	// Store final shape before altering it.
	Shape final_shape = new_shape;

	// Create old offset variable to remove empty dim.
	Shape old_offset = _offset;

	// Get rid of dummy dimensions.
	for (unsigned i = 0; i < new_shape.dim(); i++)
		if (new_shape[i] < 2 && new_shape.dim() > 1)
			new_shape.remove(i--);
	for (unsigned i = 0; i < old_shape.dim(); i++)
		if (old_shape[i] < 2 && old_shape.dim() > 1)
		{
			old_shape.remove(i);
			old_offset.remove(i--);
		}


	// Create shape to store new offsets.
	Shape new_offset(new_shape.dim(), (int*)nullptr);

	unsigned accum_new_shape = new_shape[0];
	unsigned accum_old_shape = old_shape[0];

	unsigned d_idx_old = 1;
	unsigned d_idx_new = 1;

	unsigned idx_old_shape = 0; 
	unsigned idx_new_shape = 0;

	while (idx_old_shape < old_shape.dim() && idx_new_shape < new_shape.dim())
	{
		if (accum_new_shape == accum_old_shape)
		{
			if (d_idx_new == d_idx_old)
			{
				new_offset[idx_new_shape] = old_offset[idx_old_shape];
			}
			else if (d_idx_new > d_idx_old)
			{
				unsigned accum_offset = old_offset[idx_old_shape];

				for (unsigned i = 0; i < d_idx_new; i++)
				{
					new_offset[idx_new_shape - i] = accum_offset;
					accum_offset = new_shape[idx_new_shape - i] * new_offset[idx_new_shape - i];
				}
			}
			else if (d_idx_old > d_idx_new)
			{
				for (unsigned i = 0; i < d_idx_old - 1; i++)
					TENSOR_CHECK(old_offset[idx_old_shape - i] * old_shape[idx_old_shape - i] == old_offset[idx_old_shape - i - 1],
						"Trying to change the view of a tensor by combining two dimensions which are incompatible due to data alignment.\n"
						"If you want to produce this reshape please use reshape instead.\n"
						"Old Shape: %s | View Input Shape: %s.", _view.str(), shape.str()
					);

				new_offset[idx_new_shape] = old_offset[idx_old_shape];
			}

			idx_old_shape++; idx_new_shape++;
			if (idx_new_shape == new_shape.dim() || idx_old_shape == old_shape.dim())
				break;

			accum_new_shape = new_shape[idx_new_shape];
			accum_old_shape = old_shape[idx_old_shape];

			d_idx_old = 1;
			d_idx_new = 1;
			continue;
		}


		if (accum_new_shape > accum_old_shape)
		{
			if (++idx_old_shape == old_shape.dim())
				break;
			accum_old_shape *= old_shape[idx_old_shape];
			d_idx_old++;

			TENSOR_CHECK(accum_new_shape >= accum_old_shape,
				"Ambiguous shapes found inside a view call, maybe you meant to use transpose.\n"
				"If you are sure about the reshaping please use reshape instead when there is ambiguity.\n"
				"Old Shape: %s | View Input Shape: %s.", _view.str(), shape.str()
			);
			continue;
		}
		if (accum_old_shape > accum_new_shape)
		{
			if (++idx_new_shape == new_shape.dim())
				break;
			accum_new_shape *= new_shape[idx_new_shape];
			d_idx_new++;

			TENSOR_CHECK(accum_old_shape >= accum_new_shape,
				"Ambiguous shapes found inside a view call, maybe you meant to use transpose.\n"
				"If you are sure about the reshaping please use reshape instead when there is ambiguity.\n"
				"Old Shape: %s | View Input Shape: %s.", _view.str(), shape.str()
			);
			continue;
		}
	}

	// If scan reached all the way to the end with no issues create new tensor and return.
	if (idx_old_shape == old_shape.dim() && idx_new_shape == new_shape.dim())
	{
		Tensor out(*this);
		out._view = final_shape;

		// Add removed dimensions from previously.
		for (unsigned i = 0; i < final_shape.dim(); i++)
			if (final_shape[i] < 2)
				new_offset.add(i, 0);
		
		out._offset = new_offset;
		return out;
	}

	// Else we have a problem.
	TENSOR_ERROR("Incompatible shapes found during a view call. Make sure the total dimensionality matches.\n"
		"Old Shape: %s | View Input Shape: %s.", _view.str(), shape.str()
	);
}

// Returns a tensor with the same data reduced to a single vector.

Tensor Tensor::flatten() const
{
	// Tensor must be initialized.
	TENSOR_CHECK(_data,
		"Trying to flatten an empty tensor."
	);
	
	// Create output tensor with flat view.
	Tensor out = *this;
	out._view = Shape(numel());
	out._offset = Shape(numel() > 1u ? 1u : 0u);

	// Return out.
	return out;
}

// Returns a tensor with the specified dimension removed, must be unitary.

Tensor Tensor::squeeze(int dim) const
{
	TENSOR_CHECK(_data,
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
	copied._offset.remove(dim);

	return copied;
}

// Returns a tensor with an added dimension 1 in the specified spot.

Tensor Tensor::unsqueeze(int dim) const
{
	TENSOR_CHECK(_data,
		"Trying to call unsqueeze on an empty tensor."
	);

	Tensor copied = *this;
	copied._view.add(dim, 1);
	copied._offset.add(dim, 0);

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
	class TransOp : public Tensor::TensorOp
	{
		// Tensor copy storage for the input and output.
		Tensor in, out;

		// Storage for transposed dimensions.
		int dim0, dim1;

	public:
		// Constructor, stores all the data of the operation.
		TransOp(const Tensor& _in, const Tensor& _out, int _dim0, int _dim1) : TensorOp{ "Transposition" },
			in{ _in }, out{ _out }, dim0{ _dim0 }, dim1{ _dim1 }
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
	TENSOR_CHECK(_data,
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
	if (out._data->is_gpu)
	{
		TENSOR_ERROR("CUDA backend not implemented yet.");
	}
	else
	{
		// Extract the data.
		float* out_data = out._data->array.data();
		float* ten_data = _data->array.data();
		// Store the length of the longest dim
		const unsigned vector_len = _view[dim0];
		// Create a running shape to count.
		Shape counting_shape(dim(), (int*)nullptr);
		// Create a refenrece shape with the longest dimension removed.
		Shape reference = _view;
		reference[dim0] = 1;
		// Get the offset for both tensors in the long dimension.
		unsigned ten_offset = _offset[dim0];
		unsigned out_offset = out._offset[dim1];

		// Iterate through vectors.
		while (true)
		{
			unsigned ten_idx = 0;
			for (unsigned d = 0; d < counting_shape.dim(); d++)
				ten_idx += counting_shape[d] * _offset[d];

			unsigned out_idx = 0;
			for (unsigned d = 0; d < counting_shape.dim(); d++)
			{
				if (d == dim1)	out_idx += counting_shape[d] * out._offset[dim0];
				else			out_idx += counting_shape[d] * out._offset[d];
			}

			unsigned count = 0u;
			while (count++ < vector_len)
			{
				out_data[out_idx] = ten_data[ten_idx];
				out_idx += out_offset, ten_idx += ten_offset;
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
		out._data->op = new TransOp(*this, out, dim0, dim1);

	// Return out.
	return out;
}

// Returns a tensor reshaped from to the specified dimensions if possible.

Tensor Tensor::reshape(const Shape& shape) const
{
	// Reshaping tensor operator for backpropagation.
	class ReshOp : public Tensor::TensorOp
	{
		// Tensor copy storage for the input and output.
		Tensor in, out;

	public:
		// Constructor, stores all the data of the operation.
		ReshOp(const Tensor& _in, const Tensor& _out) : TensorOp{ "Reshape" },
			in{ _in }, out{ _out }
		{
			_relatives[0] = &in;
		}

		// Backpropagation. Routes the gradient through the original shape.
		void _backward() override
		{
			in.internal_gradient() += out.gradient().reshape(in.shape());
		}
	};

	// Tensor must be initialized.
	TENSOR_CHECK(_data,
		"Trying to reshape an empty tensor."
	);

	Shape old_shape = _view;
	Shape new_shape = shape;

	// Deal with formatted shapes with -1 sizes.
	int neg_one = -1;
	unsigned new_total_dim = 1;
	unsigned old_total_dim = numel();
	for (unsigned i = 0; i < new_shape.dim(); i++)
	{
		if (new_shape[i] == -1)
		{
			TENSOR_CHECK(neg_one == -1,
				"Ambiguous shape found inside a reshape call.\n"
				"Make sure you only have one unknown dimension marked as -1 to avoid ambiguity.\n"
				"Old Shape: %s | Reshape: %s.", _view.str(), shape.str()
			);
			neg_one = i;
		}
		else new_total_dim *= new_shape[i];
	}
	if (neg_one != -1)
	{
		TENSOR_CHECK(new_total_dim != 0,
			"Ambiguous shape find inside a reshape call.\n"
			"It is not allowed to have an unknown dimension -1 while there is a size 0.\n"
			"Old Shape: %s | Reshape: %s.", _view.str(), shape.str()
		);

		TENSOR_CHECK(old_total_dim % new_total_dim == 0,
			"Unreconcileable shapes found inside a reshape call, total sizes are not divisible.\n"
			"Old Shape: %s | Reshape: %s.", _view.str(), shape.str()
		);

		new_shape[neg_one] = old_total_dim / new_total_dim;
	}
	else
		TENSOR_CHECK(new_total_dim == numel(),
			"Trying to reshape a tensor with an incompatible shape. Total size must match.\n"
			"Tensor Shape: %s | Reshape: %s", _view.str(), shape.str()
		);

	// Create tensor with a flat shape and output tensor.
	Tensor flat(Shape(new_total_dim), device(), false);
	Tensor out(new_shape, device(), has_grad());

	// Now we actually flatten the tensor.
	if (out._data->is_gpu)
	{
		TENSOR_ERROR("CUDA backend not implemented yet.");
	}
	else
	{
		// Extract the data.
		float* out_data = out._data->array.data();
		float* flat_data = flat._data->array.data();
		float* ten_data = _data->array.data();

		// First write the data to the flat tensor.
		{
			// Find the last non-unitary dimension to iterate through.
			unsigned last_long_dim = 0;
			for (unsigned i = 0; i < _view.dim(); i++)
				if (_view[i] > 1)
					last_long_dim = i;
			// Find the vector length to iterate.
			const unsigned vector_len = _view[last_long_dim];
			// Create a running shape to count.
			Shape counting_shape(last_long_dim, (int*)nullptr);
			// Running flat tensor idx.
			unsigned flat_idx = 0;
			// Iterate through vectors.
			while (true)
			{
				unsigned ten_idx = 0;
				for (unsigned d = 0; d < counting_shape.dim(); d++)
					ten_idx += counting_shape[d] * _offset[d];

				unsigned count = 0u;
				while (count++ < vector_len)
					flat_data[flat_idx++] = ten_data[ten_idx++];

				if (counting_shape.dim())
				{
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
				else
					break;
			}
		}
		// Now that we have the flat data let's do a second pass and write the data to the output.
		{
			// Find the last non-unitary dimension to iterate through.
			unsigned last_long_dim = 0;
			for (unsigned i = 0; i < out.dim(); i++)
				if (out._view[i] > 1)
					last_long_dim = i;
			// Find the vector length to iterate.
			const unsigned vector_len = out._view[last_long_dim];
			// Create a running shape to count.
			Shape counting_shape(last_long_dim, (int*)nullptr);
			// Running flat tensor idx.
			unsigned flat_idx = 0;
			// Iterate through vectors.
			while (true)
			{
				unsigned out_idx = 0;
				for (unsigned d = 0; d < counting_shape.dim(); d++)
					out_idx += counting_shape[d] * out._offset[d];

				unsigned count = 0u;
				while (count++ < vector_len)
					out_data[out_idx++] = flat_data[flat_idx++];

				if (counting_shape.dim())
				{
					counting_shape[-1]++;

					for (int d = last_long_dim - 1; d > 0; d--)
						if (counting_shape[d] >= out._view[d])
						{
							counting_shape[d] -= out._view[d];
							counting_shape[d - 1]++;
						}

					// If you reach the end of the leading dimension you're done.
					if (counting_shape[0] >= out._view[0])
						break;
				}
				else
					break;
			}
		}
	}

	// If it was a gradient operation store a ReshOp instance.
	if (has_grad())
		out._data->op = new ReshOp(*this, out);

	// Return out.
	return out;
}

// Returns a subset of the tensor with the specified shape starting from the specified indices.

Tensor Tensor::subset(const Shape& shape, const Shape& start_indices) const
{
	// Subset tensor operator for backpropagation.
	class SubsetOp : public Tensor::TensorOp
	{
		// Tensor copy storage for the input and output.
		Tensor in, out;

		// Storage for subset data.
		Shape start_indices;

	public:
		// Constructor, stores all the data of the operation.
		SubsetOp(const Tensor& _in, const Tensor& _out, const Shape& _start_indices) : TensorOp{ "Subset" },
			in{ _in }, out{ _out }, start_indices{ _start_indices }
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
	TENSOR_CHECK(_data,
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
	if (out._data->is_gpu)
	{
		TENSOR_ERROR("CUDA backend not implemented yet.");
	}
	else
	{
		// Extract the data.
		float* out_data = out._data->array.data();
		float* ten_data = _data->array.data();
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
		// Get the offset for both tensors in the long dimension.
		unsigned ten_offset = _offset[last_long_dim];
		unsigned out_offset = out._offset[last_long_dim];

		// Iterate through vectors.
		while (true)
		{
			unsigned out_idx = 0, ten_idx = 0;
			for (unsigned d = 0; d < counting_shape.dim(); d++)
			{
				out_idx += counting_shape[d] * out._offset[d];
				ten_idx += (counting_shape[d] + start[d]) * _offset[d];
			}

			unsigned count = 0u;
			while (count++ < vector_len)
			{
				out_data[out_idx] = ten_data[ten_idx];
				out_idx += out_offset, ten_idx += ten_offset;
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
		out._data->op = new SubsetOp(*this, out, start);

	// Return out.
	return out;
}

// Returns a tensor with the same shape but with a subset substituted by the specified tensor.

Tensor Tensor::modify(const Tensor& other, const Shape& start_indices) const
{
	// Modified tensor operator for backpropagation.
	class ModiOp : public Tensor::TensorOp
	{
		// Tensor copy storage for the input, modifier and output.
		Tensor in, mod, out;

		// Storage for subset data.
		Shape start_indices;

	public:
		// Constructor, stores all the data of the operation.
		ModiOp(const Tensor& _in, const Tensor& _mod, const Tensor& _out, const Shape& _start_indices) : TensorOp{ "Modify" },
			in{ _in }, mod{_mod}, out{_out}, start_indices{_start_indices}
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
	TENSOR_CHECK(_data,
		"Trying to modify an empty tensor."
	);
	TENSOR_CHECK(other._data,
		"Trying to modify with an empty other tensor."
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

	// Create output same as input. if non-contiguous deal with it.
	Tensor out(_data->array, device(), has_grad());
	out._view = _view;
	out._offset = _offset;

	// Now we actually modify the tensor.
	if (out._data->is_gpu)
	{
		TENSOR_ERROR("CUDA backend not implemented yet.");
	}
	else
	{
		// Extract the data.
		float* out_data = out._data->array.data();
		float* mod_data = other._data->array.data();
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
		// Get the offset for both tensors in the long dimension.
		unsigned mod_offset = other._offset[last_long_dim];
		unsigned out_offset = out._offset[last_long_dim];

		// Iterate through vectors.
		while (true)
		{
			unsigned out_idx = 0, mod_idx = 0;
			for (unsigned d = 0; d < counting_shape.dim(); d++)
			{
				out_idx += (counting_shape[d] + start[d]) * out._offset[d];
				mod_idx += counting_shape[d] * other._offset[d];
			}

			unsigned count = 0u;
			while (count++ < vector_len)
			{
				out_data[out_idx] = mod_data[mod_idx];
				out_idx += out_offset, mod_idx += mod_offset;
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
	if (out.has_grad())
		out._data->op = new ModiOp(*this, other, out, start);

	// Return out.
	return out;
}

// Returns a tensor with repeated dimensions of out_shape = shape * repetitions.

Tensor Tensor::repeat(int dim, unsigned repetitions) const
{
	// Repeat tensor operator for backpropagation.
	class RepOp : public Tensor::TensorOp
	{
		// Tensor copy storage for the input and output.
		Tensor in, out;

		// Dimension that was repeated.
		int dim;

	public:
		// Constructor, stores all the data of the operation.
		RepOp(const Tensor& _in, const Tensor& _out, int _dim) : TensorOp{ "Repeat" },
			in{ _in }, out{ _out }, dim{ _dim }
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
	TENSOR_CHECK(_data,
		"Trying to add repetitions to an empty tensor."
	);
	// Repeated shape must be unitary.
	TENSOR_CHECK(_view[dim] == 1,
		"Trying to repeat a tensor on a non-unitary dimension.\n"
		"Make sure the dimension you are repeating is of size 1."
	);

	// Create output with the repeated dimension.
	Shape out_shape = _view;
	out_shape[dim] = repetitions;
	Tensor out(out_shape, device(), has_grad());

	// Now we actually repeat the tensor.
	if (out._data->is_gpu)
	{
		TENSOR_ERROR("CUDA backend not implemented yet.");
	}
	else
	{
		// Extract the data.
		float* out_data = out._data->array.data();
		float* ten_data = _data->array.data();
		// Find the last non-unitary dimension to iterate through.
		unsigned last_long_dim = 0;
		for (unsigned i = 0; i < out_shape.dim(); i++)
			if (out_shape[i] > 1)
				last_long_dim = i;
		// Find the vector length to iterate.
		const unsigned vector_len = out_shape[last_long_dim];
		// Create a running shape to count.
		Shape counting_shape(last_long_dim, (int*)nullptr);
		// Get the input tensor offset on that dimension. 
		// If it is the repeating one it will be zero.
		unsigned ten_offset = _offset[last_long_dim];
		// Iterate through vectors.
		while (true)
		{
			unsigned out_idx = 0, ten_idx = 0;
			for (unsigned d = 0; d < counting_shape.dim(); d++)
			{
				out_idx += counting_shape[d] * out._offset[d];
				ten_idx += counting_shape[d] * _offset[d];
			}

			unsigned count = 0u;
			while (count++ < vector_len)
			{
				out_data[out_idx++] = ten_data[ten_idx];
				ten_idx += ten_offset;
			}

			if (counting_shape.dim())
			{
				counting_shape[-1]++;

				for (int d = last_long_dim - 1; d > 0; d--)
					if (counting_shape[d] >= out_shape[d])
					{
						counting_shape[d] -= out_shape[d];
						counting_shape[d - 1]++;
					}

				// If you reach the end of the leading dimension you're done.
				if (counting_shape[0] >= out_shape[0])
					break;
			}
			else
				break;
		}
	}

	// If it was a gradient operation store a RepOp instance.
	if (has_grad())
		out._data->op = new RepOp(*this, out, dim);

	// Return out.
	return out;
}

// Returns an exact copy of the tensor. This includes array, view and gradient if exist.

Tensor Tensor::copy(const char* device, bool grad) const
{
	// Copy tensor operator for backpropagation.
	class CopyOp : public Tensor::TensorOp
	{
		// Tensor copy storage for the input and output.
		Tensor in, out;

	public:
		// Constructor, stores all the data of the operation.
		CopyOp(const Tensor& _in, const Tensor& _out) : TensorOp{ "Copy" },
			in{ _in }, out{ _out }
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
	TENSOR_CHECK(_data,
		"Trying to call copy on an empty tensor."
	);

	// If no gradient keep it simple.
	if (!has_grad() || !grad)
	{
		Tensor out(_data->array, device, grad);
		out._view = _view;
		out._offset = _offset;
		return out;
	}

	// If both have gradient copy it and set operator.
	Tensor out(_data->array, device, false);
	out._view = _view;
	out._offset = _offset;
	out._data->gradient = new Tensor(_data->gradient->array(), device, false);
	out._data->op = new CopyOp(*this, out);

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
	TENSOR_CHECK(_data,
		"Trying to get the sign of an empty tensor."
	);

	// Create output with the same shape as tensor, no gradient.
	Tensor out(shape(), device(), false);

	// Now we actually add signs to the tensor.
	if (out._data->is_gpu)
	{
		TENSOR_ERROR("CUDA backend not implemented yet.");
	}
	else
	{
		// Extract the data.
		float* out_data = out._data->array.data();
		float* ten_data = _data->array.data();
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
	class ExpOp : public Tensor::TensorOp
	{
		// Tensor copy storage for the base and output.
		Tensor base, out;

	public:
		// Constructor, stores all the data of the operation.
		ExpOp(const Tensor& _base, const Tensor& _out) : TensorOp{ "Exponential" },
			base{ _base }, out{ _out }
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
	TENSOR_CHECK(_data,
		"Trying to exponentiate an empty tensor."
	);

	// Create output with the same shape as tensor.
	Tensor out(shape(), device(), has_grad());

	// Now we actually exponentiate the tensor.
	if (out._data->is_gpu)
	{
		TENSOR_ERROR("CUDA backend not implemented yet.");
	}
	else
	{
		// Extract the data.
		float* out_data = out._data->array.data();
		float* ten_data = _data->array.data();
		// Iterate through all elements.
		int idx = -1, _numel = int(numel());
		while (++idx < _numel)
			out_data[idx] = expf(ten_data[idx]);
	}

	// If it was a gradient operation store a ExpOp instance.
	if (has_grad())
		out._data->op = new ExpOp(*this, out);

	// Return out.
	return out;
}

Tensor Tensor::log() const
{
	// Logaritmic tensor operator for backpropagation.
	class LogOp : public Tensor::TensorOp
	{
		// Tensor copy storage for the input and output.
		Tensor in, out;

	public:
		// Constructor, stores all the data of the operation.
		LogOp(const Tensor& _in, const Tensor& _out) : TensorOp{ "Logaritmic" },
			in{ _in }, out{ _out }
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
	TENSOR_CHECK(_data,
		"Trying to log an empty tensor."
	);

	// Create output with the same shape as tensor.
	Tensor out(shape(), device(), has_grad());

	// Now we actually log the tensors.
	if (out._data->is_gpu)
	{
		TENSOR_ERROR("CUDA backend not implemented yet.");
	}
	else
	{
		// Extract the data.
		float* out_data = out._data->array.data();
		float* ten_data = _data->array.data();
		// Iterate through all elements.
		int idx = -1, _numel = int(numel());
		while (++idx < _numel)
			out_data[idx] = logf(ten_data[idx]);
	}

	// If it was a gradient operation store a LogOp instance.
	if (has_grad())
		out._data->op = new LogOp(*this, out);

	// Return out.
	return out;
}

Tensor Tensor::relu() const
{
	// ReLU tensor operator for backpropagation.
	class ReLUOp : public Tensor::TensorOp
	{
		// Tensor copy storage for the input and output.
		Tensor in, out;

	public:
		// Constructor, stores all the data of the operation.
		ReLUOp(const Tensor& _in, const Tensor& _out) : TensorOp{ "ReLU" },
			in{ _in }, out{ _out }
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
	TENSOR_CHECK(_data,
		"Trying to apply ReLU to an empty tensor."
	);

	// Create output with the same shape as tensor.
	Tensor out(shape(), device(), has_grad());

	// Now we actually ReLU the tensors.
	if (out._data->is_gpu)
	{
		TENSOR_ERROR("CUDA backend not implemented yet.");
	}
	else
	{
		// Extract the data.
		float* out_data = out._data->array.data();
		float* ten_data = _data->array.data();
		// Iterate through all elements.
		int idx = -1, _numel = int(numel());
		while (++idx < _numel)
			out_data[idx] = (ten_data[idx] > 0.f) ? ten_data[idx] : 0.f;
	}

	// If it was a gradient operation store a ReLUOp instance.
	if (has_grad())
		out._data->op = new ReLUOp(*this, out);

	// Return out.
	return out;
}

Tensor Tensor::sigmoid() const
{
	// Sigmoid tensor operator for backpropagation.
	class SigOp : public Tensor::TensorOp
	{
		// Tensor copy storage for the input and output.
		Tensor in, out;

	public:
		// Constructor, stores all the data of the operation.
		SigOp(const Tensor& _in, const Tensor& _out) : TensorOp{ "Sigmoid" },
			in{ _in }, out{ _out }
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
	TENSOR_CHECK(_data,
		"Trying to apply sigmoid to an empty tensor."
	);

	// Create output with the same shape as tensor.
	Tensor out(shape(), device(), has_grad());

	// Now we actually sigmoid the tensor.
	if (out._data->is_gpu)
	{
		TENSOR_ERROR("CUDA backend not implemented yet.");
	}
	else
	{
		// Extract the data.
		float* out_data = out._data->array.data();
		float* ten_data = _data->array.data();
		// Iterate through all elements.
		int idx = -1, _numel = int(numel());
		while (++idx < _numel)
			out_data[idx] = 1.f / 1.f + expf(-ten_data[idx]);
	}

	// If it was a gradient operation store a SigOp instance.
	if (has_grad())
		out._data->op = new SigOp(*this, out);

	// Return out.
	return out;
}

Tensor Tensor::tanh() const
{
	// Tanh tensor operator for backpropagation.
	class TanOp : public Tensor::TensorOp
	{
		// Tensor copy storage for the input and output.
		Tensor in, out;

	public:
		// Constructor, stores all the data of the operation.
		TanOp(const Tensor& _in, const Tensor& _out) : TensorOp{ "Tanh" },
			in{ _in }, out{ _out }
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
	TENSOR_CHECK(_data,
		"Trying to apply tanh to an empty tensor."
	);

	// Create output with the same shape as tensor.
	Tensor out(shape(), device(), has_grad());

	// Now we actually tanh the tensor.
	if (out._data->is_gpu)
	{
		TENSOR_ERROR("CUDA backend not implemented yet.");
	}
	else
	{
		// Extract the data.
		float* out_data = out._data->array.data();
		float* ten_data = _data->array.data();
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
		out._data->op = new TanOp(*this, out);

	// Return out.
	return out;
}

Tensor Tensor::sqrt() const
{
	// Square root tensor operator for backpropagation.
	class SqrtOp : public Tensor::TensorOp
	{
		// Tensor copy storage for the input and output.
		Tensor in, out;

	public:
		// Constructor, stores all the data of the operation.
		SqrtOp(const Tensor& _in, const Tensor& _out) : TensorOp{ "Square Root" },
			in{ _in }, out{ _out }
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
	TENSOR_CHECK(_data,
		"Trying to get the square root of an empty tensor."
	);

	// Create output with the same shape as tensor.
	Tensor out(shape(), device(), has_grad());

	// Now we actually sqrt the tensor.
	if (out._data->is_gpu)
	{
		TENSOR_ERROR("CUDA backend not implemented yet.");
	}
	else
	{
		// Extract the data.
		float* out_data = out._data->array.data();
		float* ten_data = _data->array.data();
		// Iterate through all elements.
		int idx = -1, _numel = int(numel());
		while (++idx < _numel)
			out_data[idx] = sqrtf(ten_data[idx]);
	}

	// If it was a gradient operation store a SqrtOp instance.
	if (has_grad())
		out._data->op = new SqrtOp(*this, out);

	// Return out.
	return out;
}

Tensor Tensor::square() const
{
	// Square tensor operator for backpropagation.
	class SqOp : public Tensor::TensorOp
	{
		// Tensor copy storage for the input and output.
		Tensor in, out;

	public:
		// Constructor, stores all the data of the operation.
		SqOp(const Tensor& _in, const Tensor& _out) : TensorOp{ "Square" },
			in{ _in }, out{ _out }
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
	TENSOR_CHECK(_data,
		"Trying to square an empty tensor."
	);

	// Create output with the same shape as tensor.
	Tensor out(shape(), device(), has_grad());

	// Now we actually square the tensor.
	if (out._data->is_gpu)
	{
		TENSOR_ERROR("CUDA backend not implemented yet.");
	}
	else
	{
		// Extract the data.
		float* out_data = out._data->array.data();
		float* ten_data = _data->array.data();
		// Iterate through all elements.
		int idx = -1, _numel = int(numel());
		while (++idx < _numel)
			out_data[idx] = ten_data[idx] * ten_data[idx];
	}

	// If it was a gradient operation store a SqOp instance.
	if (has_grad())
		out._data->op = new SqOp(*this, out);

	// Return out.
	return out;
}

Tensor Tensor::pow(float exp) const
{
	// Power tensor operator for backpropagation.
	class PowOp : public Tensor::TensorOp
	{
		// Tensor copy storage for the input and output.
		Tensor in, out;

		// Variable to store exponent.
		float exp;

	public:
		// Constructor, stores all the data of the operation.
		PowOp(const Tensor& _in, const Tensor& _out, float _exp) : TensorOp{ "Power" },
			in{ _in }, out{ _out }, exp{ _exp }
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
	TENSOR_CHECK(_data,
		"Trying to square an empty tensor."
	);

	// Create output with the same shape as tensor.
	Tensor out(shape(), device(), has_grad());

	// Now we actually power the tensor.
	if (out._data->is_gpu)
	{
		TENSOR_ERROR("CUDA backend not implemented yet.");
	}
	else
	{
		// Extract the data.
		float* out_data = out._data->array.data();
		float* ten_data = _data->array.data();
		// Iterate through all elements.
		int idx = -1, _numel = int(numel());
		while (++idx < _numel)
			out_data[idx] = powf(ten_data[idx], exp);
	}

	// If it was a gradient operation store a PowOp instance.
	if (has_grad())
		out._data->op = new PowOp(*this, out, exp);

	// Return out.
	return out;
}

Tensor Tensor::mean(int dim, bool keepdim) const
{
	// Mean tensor operator for backpropagation.
	class MeanOp : public Tensor::TensorOp
	{
		// Tensor copy storage for the input and output.
		Tensor in, out;

		// Storage for initial dimension size.
		unsigned size;

	public:
		// Constructor, stores all the data of the operation.
		MeanOp(const Tensor& _in, const Tensor& _out, unsigned _size) : TensorOp{ "Mean" },
			in{ _in }, out{ _out }, size{ _size }
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
	TENSOR_CHECK(_data,
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
	if (out._data->is_gpu)
	{
		TENSOR_ERROR("CUDA backend not implemented yet.");
	}
	else
	{
		// Extract the data.
		float* out_data = out._data->array.data();
		float* ten_data = _data->array.data();
		// Get relevant offsets.
		const unsigned elem_offset = _offset[dim];
		const unsigned post_offset = elem_offset * _size;
		// Get relevant sizes.
		unsigned prev_size = 1u;
		unsigned post_size = 1u;
		for (unsigned i = 0; i < dim; i++)
			post_size *= _view[dim];
		for (unsigned i = dim + 1; i < _view.dim(); i++)
			prev_size *= _view[dim];
		// Iterate through all vectors to compute mean.
		for (unsigned post_count = 0; post_count < post_size; post_count++)
			for (unsigned prev_count = 0; prev_count < prev_size; prev_count++)
			{
				unsigned ten_idx = post_count * post_offset + prev_count;
				unsigned out_idx = post_count * elem_offset + prev_count;
				
				unsigned count = 0;
				while (count++ < _size)
				{
					out_data[out_idx] += ten_data[ten_idx];
					ten_idx += elem_offset;
				}
				out_data[out_idx] /= _size;
			}
	}

	// If it was a gradient operation store a MeanOp instance.
	if (has_grad())
		out._data->op = new MeanOp(*this, out, _size);

	// Return out. Squeeze if necessary.
	if (!keepdim && out.dim() > 1)
		return out.squeeze(dim);
	return out;
}

Tensor Tensor::var(int dim, bool keepdim) const
{
	// Variance tensor operator for backpropagation.
	class VarOp : public Tensor::TensorOp
	{
		// Tensor copy storage for the input and output.
		Tensor in, out;

		// Storage for initial dimension and size.
		unsigned dim, size;

	public:
		// Constructor, stores all the data of the operation.
		VarOp(const Tensor& _in, const Tensor& _out, unsigned _size, unsigned _dim) : TensorOp{ "Variance" },
			in{ _in }, out{ _out }, size{ _size }, dim{ _dim }
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
	TENSOR_CHECK(_data,
		"Trying to apply variance to an empty tensor."
	);

	// Get the initial dimension size.
	unsigned _size = size(dim);
	// Compute output shape.
	Shape out_shape = shape();
	out_shape[dim] = 1;

	// Create output with the same shape as tensor.
	Tensor out(out_shape, device(), has_grad());

	// Now we actually get the tensor variance.
	if (out._data->is_gpu)
	{
		TENSOR_ERROR("CUDA backend not implemented yet.");
	}
	else
	{
		// Extract the data.
		float* out_data = out._data->array.data();
		float* ten_data = _data->array.data();
		// Get relevant offsets.
		const unsigned elem_offset = _offset[dim];
		const unsigned post_offset = elem_offset * _size;
		// Get relevant sizes.
		unsigned prev_size = 1u;
		unsigned post_size = 1u;
		for (unsigned i = 0; i < dim; i++)
			post_size *= _view[dim];
		for (unsigned i = dim + 1; i < _view.dim(); i++)
			prev_size *= _view[dim];
		// Iterate through all vectors to compute variance.
		for (unsigned post_count = 0; post_count < post_size; post_count++)
			for (unsigned prev_count = 0; prev_count < prev_size; prev_count++)
			{
				unsigned ten_idx = post_count * post_offset + prev_count;
				unsigned out_idx = post_count * elem_offset + prev_count;

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
					ten_idx += elem_offset;
				}
				out_data[out_idx] = M2 / _size;
			}
	}

	// If it was a gradient operation store a VarOp instance.
	if (has_grad())
		out._data->op = new VarOp(*this, out, _size, dim);

	// Return out. Squeeze if necessary.
	if (!keepdim && out.dim() > 1)
		return out.squeeze(dim);
	return out;
}

Tensor Tensor::std(int dim, bool keepdim) const
{
	// STD tensor operator for backpropagation.
	class StdOp : public Tensor::TensorOp
	{
		// Tensor copy storage for the input and output.
		Tensor in, out;

		// Storage for initial dimension and size.
		unsigned dim, size;

	public:
		// Constructor, stores all the data of the operation.
		StdOp(const Tensor& _in, const Tensor& _out, unsigned _size, unsigned _dim) : TensorOp{ "Standard Deviation" },
			in{ _in }, out{ _out }, size{ _size }, dim{ _dim }
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
	TENSOR_CHECK(_data,
		"Trying to apply standard deviation to an empty tensor."
	);

	// Get the initial dimension size.
	unsigned _size = size(dim);
	// Compute output shape.
	Shape out_shape = shape();
	out_shape[dim] = 1;

	// Create output with the same shape as tensor.
	Tensor out(out_shape, device(), has_grad());

	// Now we actually get the tensor deviation.
	if (out._data->is_gpu)
	{
		TENSOR_ERROR("CUDA backend not implemented yet.");
	}
	else
	{
		// Extract the data.
		float* out_data = out._data->array.data();
		float* ten_data = _data->array.data();
		// Get relevant offsets.
		const unsigned elem_offset = _offset[dim];
		const unsigned post_offset = elem_offset * _size;
		// Get relevant sizes.
		unsigned prev_size = 1u;
		unsigned post_size = 1u;
		for (unsigned i = 0; i < dim; i++)
			post_size *= _view[dim];
		for (unsigned i = dim + 1; i < _view.dim(); i++)
			prev_size *= _view[dim];
		// Iterate through all vectors to compute std.
		for (unsigned post_count = 0; post_count < post_size; post_count++)
			for (unsigned prev_count = 0; prev_count < prev_size; prev_count++)
			{
				unsigned ten_idx = post_count * post_offset + prev_count;
				unsigned out_idx = post_count * elem_offset + prev_count;

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
					ten_idx += elem_offset;
				}
				out_data[out_idx] = sqrtf(M2 / _size);
			}
	}

	// If it was a gradient operation store a StdOp instance.
	if (has_grad())
		out._data->op = new StdOp(*this, out, _size, dim);

	// Return out. Squeeze if necessary.
	if (!keepdim && out.dim() > 1)
		return out.squeeze(dim);
	return out;
}

Tensor Tensor::sum(int dim, bool keepdim) const
{
	// Sum tensor operator for backpropagation.
	class SumOp : public Tensor::TensorOp
	{
		// Tensor copy storage for the input and output.
		Tensor in, out;

	public:
		// Constructor, stores all the data of the operation.
		SumOp(const Tensor& _in, const Tensor& _out) : TensorOp{ "Sum" },
			in{ _in }, out{ _out }
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
	TENSOR_CHECK(_data,
		"Trying to apply sum to an empty tensor."
	);

	// Get the initial dimension size.
	unsigned _size = size(dim);
	// Compute output shape.
	Shape out_shape = shape();
	out_shape[dim] = 1;

	// Create output with the same shape as tensor.
	Tensor out(out_shape, device(), has_grad());

	// Now we actually sum the tensor.
	if (out._data->is_gpu)
	{
		TENSOR_ERROR("CUDA backend not implemented yet.");
	}
	else
	{
		// Extract the data.
		float* out_data = out._data->array.data();
		float* ten_data = _data->array.data();
		// Get relevant offsets.
		const unsigned elem_offset = _offset[dim];
		const unsigned post_offset = elem_offset * _size;
		// Get relevant sizes.
		unsigned prev_size = 1u;
		unsigned post_size = 1u;
		for (unsigned i = 0; i < dim; i++)
			post_size *= _view[dim];
		for (unsigned i = dim + 1; i < _view.dim(); i++)
			prev_size *= _view[dim];
		// Iterate through all vectors to compute sum.
		for (unsigned post_count = 0; post_count < post_size; post_count++)
			for (unsigned prev_count = 0; prev_count < prev_size; prev_count++)
			{
				unsigned ten_idx = post_count * post_offset + prev_count;
				unsigned out_idx = post_count * elem_offset + prev_count;

				unsigned count = 0;
				while (count++ < _size)
				{
					out_data[out_idx] += ten_data[ten_idx];
					ten_idx += elem_offset;
				}
			}
	}

	// If it was a gradient operation store a SumOp instance.
	if (has_grad())
		out._data->op = new SumOp(*this, out);

	// Return out. Squeeze if necessary.
	if (!keepdim && out.dim() > 1)
		return out.squeeze(dim);
	return out;
}

Tensor Tensor::softmax(int dim) const
{
	// Softmax tensor operator for backpropagation.
	class SoftOp : public Tensor::TensorOp
	{
		// Tensor copy storage for the input and output.
		Tensor in, out;

		// Storage for initial dimension and size.
		unsigned dim;

	public:
		// Constructor, stores all the data of the operation.
		SoftOp(const Tensor& _in, const Tensor& _out, unsigned _dim) : TensorOp{ "Softmax" },
			in{ _in }, out{ _out }, dim{ _dim }
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
	TENSOR_CHECK(_data,
		"Trying to apply softmax to an empty tensor."
	);

	// Get the initial dimension size.
	unsigned _size = size(dim);
	// Create output with the same shape as tensor.
	Tensor out(shape(), device(), has_grad());
	// Compute output shape.
	Shape out_shape = shape();

	// Now we actually apply softmax to the tensor.
	if (out._data->is_gpu)
	{
		TENSOR_ERROR("CUDA backend not implemented yet.");
	}
	else
	{
		// Extract the data.
		float* out_data = out._data->array.data();
		float* ten_data = _data->array.data();
		// Get relevant offsets.
		const unsigned elem_offset = _offset[dim];
		const unsigned post_offset = elem_offset * _size;
		// Get relevant sizes.
		unsigned prev_size = 1u;
		unsigned post_size = 1u;
		for (unsigned i = 0; i < dim; i++)
			post_size *= _view[dim];
		for (unsigned i = dim + 1; i < _view.dim(); i++)
			prev_size *= _view[dim];
		// Iterate through all vectors to compute softmax.
		for (unsigned post_count = 0; post_count < post_size; post_count++)
			for (unsigned prev_count = 0; prev_count < prev_size; prev_count++)
			{
				// Compute initial and final idx for this vector.
				const unsigned idx_0 = post_count * post_offset + prev_count;
				const unsigned idx_f = (post_count + 1) * post_offset + prev_count;
				
				// First pass find maximum.
				float max = ten_data[idx_0];
				for (unsigned idx = idx_0; idx < idx_f; idx += elem_offset)
					if (ten_data[idx] > max)
						max = ten_data[idx];
				// Second pass compute exponential and accumulate sum.
				float sum = 0.f;
				for (unsigned idx = idx_0; idx < idx_f; idx += elem_offset)
				{
					out_data[idx] = expf(ten_data[ten_idx] - max);
					sum += out_data[idx];
				}
				// Third pass divide by sum.
				for (unsigned idx = idx_0; idx < idx_f; idx+=elem_offset)
					out_data[idx] /= sum;
			}
	}

	// If it was a gradient operation store a SoftOp instance.
	if (has_grad())
		out._data->op = new SoftOp(*this, out, dim);

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
	class AddOp : public Tensor::TensorOp
	{
		// Tensor copy storage for the summands and output.
		Tensor sum0, sum1, out;

	public:
		// Constructor, stores all the data of the operation.
		AddOp(const Tensor& _sum0, const Tensor& _sum1, const Tensor& _out) : TensorOp{ "Addition" },
			sum0{ _sum0.has_grad() ? _sum0 : Tensor() }, sum1{ _sum1.has_grad() ? _sum1 : Tensor() }, out{ _out }
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
	TENSOR_CHECK(ten0._data,
		"Trying to add two tensors while the first tensor is empty."
	);
	TENSOR_CHECK(ten1._data,
		"Trying to add two tensors while the second tensor is empty."
	);
	// Both tensors must be on the same device
	TENSOR_CHECK(ten0._data->is_gpu == ten1._data->is_gpu,
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
			ten1_summed = Functional::sum(ten1_summed, i, true);

	// Now we actually sum the tensors.
	if (out._data->is_gpu)
	{
		TENSOR_ERROR("CUDA backend not implemented yet.");
	}
	else
	{
		// Extract the data.
		float* out_data = out._data->array.data();
		float* ten0_data = ten0._data->array.data();
		float* ten1_data = ten1_summed._data->array.data();
		// Exptract the shapes.
		Shape out_shape = out.shape();
		Shape ten0_shape = ten0.shape();
		Shape ten1_shape = ten1_summed.shape();
		// Find the last non-unitary dimension to iterate through.
		unsigned last_long_dim = 0;
		for (unsigned i = 0; i < out_shape.dim(); i++)
			if (out_shape[i] > 1)
				last_long_dim = i;
		// Find the offset, will be 1 except for ten1 if broadcasting.
		const unsigned dout = out._offset[last_long_dim];
		const unsigned dten0 = ten0._offset[last_long_dim];
		const unsigned dten1 = ten1_summed._offset[last_long_dim];
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
				out_idx += counting_shape[d] * out._offset[d];
				ten0_idx += counting_shape[d] * ten0._offset[d];
				ten1_idx += counting_shape[d] * ten1_summed._offset[d];
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
		out._data->op = new AddOp(ten0, ten1_summed, out);

	// Return out.
	return out;
}

Tensor operator-(const Tensor& ten0, const Tensor& ten1)
{
	// Subtraction tensor operator for backpropagation.
	class SubOp : public Tensor::TensorOp
	{
		// Tensor copy storage for the operands and output.
		Tensor sum, sub, out;

	public:
		// Constructor, stores all the data of the operation.
		SubOp(const Tensor& _sum, const Tensor& _sub, const Tensor& _out) : TensorOp{ "Subtraction" },
			sum{ _sum.has_grad() ? _sum : Tensor() }, sub{ _sub.has_grad() ? _sub : Tensor() }, out{ _out }
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
	TENSOR_CHECK(ten0._data,
		"Trying to subtract two tensors while the first tensor is empty."
	);
	TENSOR_CHECK(ten1._data,
		"Trying to subtract two tensors while the second tensor is empty."
	);
	// Both tensors must be on the same device
	TENSOR_CHECK(ten0._data->is_gpu == ten1._data->is_gpu,
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
			ten1_summed = Functional::sum(ten1_summed, i, true);

	// Now we actually sum the tensors.
	if (out._data->is_gpu)
	{
		TENSOR_ERROR("CUDA backend not implemented yet.");
	}
	else
	{
		// Extract the data.
		float* out_data = out._data->array.data();
		float* ten0_data = ten0._data->array.data();
		float* ten1_data = ten1_summed._data->array.data();
		// Exptract the shapes.
		Shape out_shape = out.shape();
		Shape ten0_shape = ten0.shape();
		Shape ten1_shape = ten1_summed.shape();
		// Find the last non-unitary dimension to iterate through.
		unsigned last_long_dim = 0;
		for (unsigned i = 0; i < out_shape.dim(); i++)
			if (out_shape[i] > 1)
				last_long_dim = i;
		// Find the offset, will be 1 except for ten1 if broadcasting.
		const unsigned dout = out._offset[last_long_dim];
		const unsigned dten0 = ten0._offset[last_long_dim];
		const unsigned dten1 = ten1_summed._offset[last_long_dim];
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
				out_idx += counting_shape[d] * out._offset[d];
				ten0_idx += counting_shape[d] * ten0._offset[d];
				ten1_idx += counting_shape[d] * ten1_summed._offset[d];
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
		out._data->op = new SubOp(ten0, ten1_summed, out);

	// Return out.
	return out;
}

Tensor operator*(const Tensor& ten0, const Tensor& ten1)
{
	// Multiplication tensor operator for backpropagation.
	class MulOp : public Tensor::TensorOp
	{
		// Tensor copy storage for the summands and output.
		Tensor fac0, fac1, out;

	public:
		// Constructor, stores all the data of the operation.
		MulOp(const Tensor& _fac0, const Tensor& _fac1, const Tensor& _out) : TensorOp{ "Multiplication" },
			fac0{ _fac0 }, fac1{ _fac1 }, out{ _out }
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
	TENSOR_CHECK(ten0._data,
		"Trying to multiply two tensors while the first tensor is empty."
	);
	TENSOR_CHECK(ten1._data,
		"Trying to multiply two tensors while the second tensor is empty."
	);
	// Both tensors must be on the same device
	TENSOR_CHECK(ten0._data->is_gpu == ten1._data->is_gpu,
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
			ten1_summed = Functional::sum(ten1_summed, i, true);

	// Now we actually sum the tensors.
	if (out._data->is_gpu)
	{
		TENSOR_ERROR("CUDA backend not implemented yet.");
	}
	else
	{
		// Extract the data.
		float* out_data = out._data->array.data();
		float* ten0_data = ten0._data->array.data();
		float* ten1_data = ten1_summed._data->array.data();
		// Exptract the shapes.
		Shape out_shape = out.shape();
		Shape ten0_shape = ten0.shape();
		Shape ten1_shape = ten1_summed.shape();
		// Find the last non-unitary dimension to iterate through.
		unsigned last_long_dim = 0;
		for (unsigned i = 0; i < out_shape.dim(); i++)
			if (out_shape[i] > 1)
				last_long_dim = i;
		// Find the offset, will be 1 except for ten1 if broadcasting.
		const unsigned dout = out._offset[last_long_dim];
		const unsigned dten0 = ten0._offset[last_long_dim];
		const unsigned dten1 = ten1_summed._offset[last_long_dim];
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
				out_idx += counting_shape[d] * out._offset[d];
				ten0_idx += counting_shape[d] * ten0._offset[d];
				ten1_idx += counting_shape[d] * ten1_summed._offset[d];
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
		out._data->op = new MulOp(ten0, ten1_summed, out);

	// Return out.
	return out;
}

Tensor operator/(const Tensor& ten0, const Tensor& ten1)
{
	// Division tensor operator for backpropagation.
	class DivOp : public Tensor::TensorOp
	{
		// Tensor copy storage for the summands and output.
		Tensor num, den, out;

	public:
		// Constructor, stores all the data of the operation.
		DivOp(const Tensor& _num, const Tensor& _den, const Tensor& _out) : TensorOp{ "Division" },
			num{ _num }, den{ _den }, out{ _out }
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
	TENSOR_CHECK(ten0._data,
		"Trying to add two tensors while the first tensor is empty."
	);
	TENSOR_CHECK(ten1._data,
		"Trying to add two tensors while the second tensor is empty."
	);
	// Both tensors must be on the same device
	TENSOR_CHECK(ten0._data->is_gpu == ten1._data->is_gpu,
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
			ten1_summed = Functional::sum(ten1_summed, i, true);

	// Now we actually divide the tensors.
	if (out._data->is_gpu)
	{
		TENSOR_ERROR("CUDA backend not implemented yet.");
	}
	else
	{
		// Extract the data.
		float* out_data = out._data->array.data();
		float* ten0_data = ten0._data->array.data();
		float* ten1_data = ten1_summed._data->array.data();
		// Exptract the shapes.
		Shape out_shape = out.shape();
		Shape ten0_shape = ten0.shape();
		Shape ten1_shape = ten1_summed.shape();
		// Find the last non-unitary dimension to iterate through.
		unsigned last_long_dim = 0;
		for (unsigned i = 0; i < out_shape.dim(); i++)
			if (out_shape[i] > 1)
				last_long_dim = i;
		// Find the offset, will be 1 except for ten1 if broadcasting.
		const unsigned dout = out._offset[last_long_dim];
		const unsigned dten0 = ten0._offset[last_long_dim];
		const unsigned dten1 = ten1_summed._offset[last_long_dim];
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
				out_idx += counting_shape[d] * out._offset[d];
				ten0_idx += counting_shape[d] * ten0._offset[d];
				ten1_idx += counting_shape[d] * ten1_summed._offset[d];
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
		out._data->op = new DivOp(ten0, ten1_summed, out);

	// Return out.
	return out;
}

Tensor operator+(const Tensor& ten, float val)
{
	// Scalar addition tensor operator for backpropagation.
	class ScaAddOp : public Tensor::TensorOp
	{
		// Tensor copy storage for the summands and output.
		Tensor sum, out;

	public:
		// Constructor, stores all the data of the operation.
		ScaAddOp(const Tensor& _sum, const Tensor& _out) : TensorOp{ "Scalar Addition" },
			sum{ _sum }, out{ _out }
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
	TENSOR_CHECK(ten._data,
		"Trying to do scalar addition with an empty tensor."
	);

	// Create output with the same shape as tensor.
	Tensor out(ten.shape(), ten.device(), ten.has_grad());

	// Now we actually sum the tensors.
	if (out._data->is_gpu)
	{
		TENSOR_ERROR("CUDA backend not implemented yet.");
	}
	else
	{
		// Extract the data.
		float* out_data = out._data->array.data();
		float* ten_data = ten._data->array.data();
		// Exptract the shapes.
		Shape out_shape = out.shape();
		// Find the last non-unitary dimension to iterate through.
		unsigned last_long_dim = 0;
		for (unsigned i = 0; i < out_shape.dim(); i++)
			if (out_shape[i] > 1)
				last_long_dim = i;
		// Find the vector length to iterate.
		const unsigned vector_len = out_shape[last_long_dim];
		// Create a running shape to count.
		Shape counting_shape(last_long_dim, (int*)nullptr);
		// Iterate through vectors.
		while (true)
		{
			unsigned out_idx = 0, ten_idx = 0;
			for (unsigned d = 0; d < counting_shape.dim(); d++)
			{
				out_idx += counting_shape[d] * out._offset[d];
				ten_idx += counting_shape[d] * ten._offset[d];
			}

			unsigned count = 0u;
			while (count++ < vector_len)
				out_data[out_idx++] = ten_data[ten_idx++] + val;

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

	// If it was a gradient operation store a ScaAddOp instance.
	if (out.has_grad())
		out._data->op = new ScaAddOp(ten, out);

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
	class ScaMulOp : public Tensor::TensorOp
	{
		// Tensor copy storage for the factor and output.
		Tensor fac, out;

		// Value storage.
		float val;

	public:
		// Constructor, stores all the data of the operation.
		ScaMulOp(const Tensor& _fac, const Tensor& _out, float _val) : TensorOp{ "Scalar Multiplication" },
			fac{ _fac }, out{ _out }, val{ _val }
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
	TENSOR_CHECK(ten._data,
		"Trying to do scalar multiplication with an empty tensor."
	);

	// Create output with the same shape as tensor.
	Tensor out(ten.shape(), ten.device(), ten.has_grad());

	// Now we actually multiply the tensors.
	if (out._data->is_gpu)
	{
		TENSOR_ERROR("CUDA backend not implemented yet.");
	}
	else
	{
		// Extract the data.
		float* out_data = out._data->array.data();
		float* ten_data = ten._data->array.data();
		// Exptract the shapes.
		Shape out_shape = out.shape();
		// Find the last non-unitary dimension to iterate through.
		unsigned last_long_dim = 0;
		for (unsigned i = 0; i < out_shape.dim(); i++)
			if (out_shape[i] > 1)
				last_long_dim = i;
		// Find the vector length to iterate.
		const unsigned vector_len = out_shape[last_long_dim];
		// Create a running shape to count.
		Shape counting_shape(last_long_dim, (int*)nullptr);
		// Iterate through vectors.
		while (true)
		{
			unsigned out_idx = 0, ten_idx = 0;
			for (unsigned d = 0; d < counting_shape.dim(); d++)
			{
				out_idx += counting_shape[d] * out._offset[d];
				ten_idx += counting_shape[d] * ten._offset[d];
			}

			unsigned count = 0u;
			while (count++ < vector_len)
				out_data[out_idx++] = ten_data[ten_idx++] * val;

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

	// If it was a gradient operation store a ScaAddOp instance.
	if (out.has_grad())
		out._data->op = new ScaMulOp(ten, out, val);

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
	class ScaSubOp : public Tensor::TensorOp
	{
		// Tensor copy storage for the subtractor and output.
		Tensor sub, out;

	public:
		// Constructor, stores all the data of the operation.
		ScaSubOp(const Tensor& _sub, const Tensor& _out) : TensorOp{ "Scalar Subtraction" },
			sub{ _sub }, out{ _out }
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
	TENSOR_CHECK(ten._data,
		"Trying to do scalar subtraction with an empty tensor."
	);

	// Create output with the same shape as tensor.
	Tensor out(ten.shape(), ten.device(), ten.has_grad());

	// Now we actually subtract the tensors.
	if (out._data->is_gpu)
	{
		TENSOR_ERROR("CUDA backend not implemented yet.");
	}
	else
	{
		// Extract the data.
		float* out_data = out._data->array.data();
		float* ten_data = ten._data->array.data();
		// Exptract the shapes.
		Shape out_shape = out.shape();
		// Find the last non-unitary dimension to iterate through.
		unsigned last_long_dim = 0;
		for (unsigned i = 0; i < out_shape.dim(); i++)
			if (out_shape[i] > 1)
				last_long_dim = i;
		// Find the vector length to iterate.
		const unsigned vector_len = out_shape[last_long_dim];
		// Create a running shape to count.
		Shape counting_shape(last_long_dim, (int*)nullptr);
		// Iterate through vectors.
		while (true)
		{
			unsigned out_idx = 0, ten_idx = 0;
			for (unsigned d = 0; d < counting_shape.dim(); d++)
			{
				out_idx += counting_shape[d] * out._offset[d];
				ten_idx += counting_shape[d] * ten._offset[d];
			}

			unsigned count = 0u;
			while (count++ < vector_len)
				out_data[out_idx++] = val - ten_data[ten_idx++];

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

	// If it was a gradient operation store a ScaSubOp instance.
	if (out.has_grad())
		out._data->op = new ScaSubOp(ten, out);

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
	class ScaDivOp : public Tensor::TensorOp
	{
		// Tensor copy storage for the factor and output.
		Tensor div, out;

	public:
		// Constructor, stores all the data of the operation.
		ScaDivOp(const Tensor& _div, const Tensor& _out) : TensorOp{ "Scalar Division" },
			div{ _div }, out{ _out }
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
	TENSOR_CHECK(ten._data,
		"Trying to do scalar division with an empty tensor."
	);

	// Create output with the same shape as tensor.
	Tensor out(ten.shape(), ten.device(), ten.has_grad());

	// Now we actually divide the tensors.
	if (out._data->is_gpu)
	{
		TENSOR_ERROR("CUDA backend not implemented yet.");
	}
	else
	{
		// Extract the data.
		float* out_data = out._data->array.data();
		float* ten_data = ten._data->array.data();
		// Exptract the shapes.
		Shape out_shape = out.shape();
		// Find the last non-unitary dimension to iterate through.
		unsigned last_long_dim = 0;
		for (unsigned i = 0; i < out_shape.dim(); i++)
			if (out_shape[i] > 1)
				last_long_dim = i;
		// Find the vector length to iterate.
		const unsigned vector_len = out_shape[last_long_dim];
		// Create a running shape to count.
		Shape counting_shape(last_long_dim, (int*)nullptr);
		// Iterate through vectors.
		while (true)
		{
			unsigned out_idx = 0, ten_idx = 0;
			for (unsigned d = 0; d < counting_shape.dim(); d++)
			{
				out_idx += counting_shape[d] * out._offset[d];
				ten_idx += counting_shape[d] * ten._offset[d];
			}

			unsigned count = 0u;
			while (count++ < vector_len)
				out_data[out_idx++] = val / ten_data[ten_idx++];

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

	// If it was a gradient operation store a ScaDivOp instance.
	if (out.has_grad())
		out._data->op = new ScaDivOp(ten, out);

	// Return out.
	return out;
}

Tensor Tensor::operator-() const
{
	// Negation tensor operator for backpropagation.
	class NegOp : public Tensor::TensorOp
	{
		// Tensor copy storage for the negator and output.
		Tensor neg, out;

	public:
		// Constructor, stores all the data of the operation.
		NegOp(const Tensor& _neg, const Tensor& _out) : TensorOp{ "Negation" },
			neg{ _neg }, out{ _out }
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
	TENSOR_CHECK(_data,
		"Trying to negate an empty tensor."
	);

	// Create output with the same shape as tensor.
	Tensor out(shape(), device(), has_grad());

	// Now we actually negate the tensor.
	if (out._data->is_gpu)
	{
		TENSOR_ERROR("CUDA backend not implemented yet.");
	}
	else
	{
		// Extract the data.
		float* out_data = out._data->array.data();
		float* ten_data = _data->array.data();
		// Exptract the shapes.
		Shape out_shape = out.shape();
		// Find the last non-unitary dimension to iterate through.
		unsigned last_long_dim = 0;
		for (unsigned i = 0; i < out_shape.dim(); i++)
			if (out_shape[i] > 1)
				last_long_dim = i;
		// Find the vector length to iterate.
		const unsigned vector_len = out_shape[last_long_dim];
		// Create a running shape to count.
		Shape counting_shape(last_long_dim, (int*)nullptr);
		// Iterate through vectors.
		while (true)
		{
			unsigned out_idx = 0, ten_idx = 0;
			for (unsigned d = 0; d < counting_shape.dim(); d++)
			{
				out_idx += counting_shape[d] * out._offset[d];
				ten_idx += counting_shape[d] * _offset[d];
			}

			unsigned count = 0u;
			while (count++ < vector_len)
				out_data[out_idx++] = -ten_data[ten_idx++];

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

	// If it was a gradient operation store a NegOp instance.
	if (out.has_grad())
		out._data->op = new NegOp(*this, out);

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
	// If there is gradient the middle man must exist.
	if (has_grad() || (_data && _data->instances > 1))
	{
		Tensor out = *this + other;
		if (has_grad())
			out.internal_gradient() = gradient().copy(device(), false);
		return *this = out;
	}

	// --- Sanity checks ---
	
	// Both tensors must be initialized.
	TENSOR_CHECK(_data,
		"Trying to add two tensors while the first tensor is empty."
	);
	TENSOR_CHECK(other._data,
		"Trying to add two tensors while the second tensor is empty."
	);
	// Both tensors must be on the same device
	TENSOR_CHECK(_data->is_gpu == other._data->is_gpu,
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
	if (_data->is_gpu)
	{
		TENSOR_ERROR("CUDA backend not implemented yet.");
	}
	else
	{
		// Extract the data.
		float* out_data = _data->array.data();
		float* oth_data = other_summed._data->array.data();
		// Exptract the shapes.
		Shape out_shape = shape();
		Shape oth_shape = other_summed.shape();
		// Find the last non-unitary dimension to iterate through.
		unsigned last_long_dim = 0;
		for (unsigned i = 0; i < out_shape.dim(); i++)
			if (out_shape[i] > 1)
				last_long_dim = i;
		// Find the offset, will be 1 except for ten1 if broadcasting.
		const unsigned dout = _offset[last_long_dim];
		const unsigned doth = other_summed._offset[last_long_dim];
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
				out_idx += counting_shape[d] * _offset[d];
				oth_idx += counting_shape[d] * other_summed._offset[d];
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
	// If there is gradient the middle man must exist.
	if (has_grad() || (_data && _data->instances > 1))
	{
		Tensor out = *this - other;
		if (has_grad())
			out.internal_gradient() = gradient().copy(device(), false);
		return *this = out;
	}

	// --- Sanity checks ---

	// Both tensors must be initialized.
	TENSOR_CHECK(_data,
		"Trying to subtract two tensors while the first tensor is empty."
	);
	TENSOR_CHECK(other._data,
		"Trying to subtract two tensors while the second tensor is empty."
	);
	// Both tensors must be on the same device
	TENSOR_CHECK(_data->is_gpu == other._data->is_gpu,
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
	if (_data->is_gpu)
	{
		TENSOR_ERROR("CUDA backend not implemented yet.");
	}
	else
	{
		// Extract the data.
		float* out_data = _data->array.data();
		float* oth_data = other_summed._data->array.data();
		// Exptract the shapes.
		Shape out_shape = shape();
		Shape oth_shape = other_summed.shape();
		// Find the last non-unitary dimension to iterate through.
		unsigned last_long_dim = 0;
		for (unsigned i = 0; i < out_shape.dim(); i++)
			if (out_shape[i] > 1)
				last_long_dim = i;
		// Find the offset, will be 1 except for ten1 if broadcasting.
		const unsigned dout = _offset[last_long_dim];
		const unsigned doth = other_summed._offset[last_long_dim];
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
				out_idx += counting_shape[d] * _offset[d];
				oth_idx += counting_shape[d] * other_summed._offset[d];
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
	// If there is gradient the middle man must exist.
	if (has_grad() || (_data && _data->instances > 1))
	{
		Tensor out = *this * other;
		if (has_grad())
			out.internal_gradient() = gradient().copy(device(), false);
		return *this = out;
	}

	// --- Sanity checks ---

	// Both tensors must be initialized.
	TENSOR_CHECK(_data,
		"Trying to multiply two tensors while the first tensor is empty."
	);
	TENSOR_CHECK(other._data,
		"Trying to multiply two tensors while the second tensor is empty."
	);
	// Both tensors must be on the same device
	TENSOR_CHECK(_data->is_gpu == other._data->is_gpu,
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
	if (_data->is_gpu)
	{
		TENSOR_ERROR("CUDA backend not implemented yet.");
	}
	else
	{
		// Extract the data.
		float* out_data = _data->array.data();
		float* oth_data = other_summed._data->array.data();
		// Exptract the shapes.
		Shape out_shape = shape();
		Shape oth_shape = other_summed.shape();
		// Find the last non-unitary dimension to iterate through.
		unsigned last_long_dim = 0;
		for (unsigned i = 0; i < out_shape.dim(); i++)
			if (out_shape[i] > 1)
				last_long_dim = i;
		// Find the offset, will be 1 except for ten1 if broadcasting.
		const unsigned dout = _offset[last_long_dim];
		const unsigned doth = other_summed._offset[last_long_dim];
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
				out_idx += counting_shape[d] * _offset[d];
				oth_idx += counting_shape[d] * other_summed._offset[d];
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
	// If there is gradient the middle man must exist.
	if (has_grad() || (_data && _data->instances > 1))
	{
		Tensor out = *this / other;
		if (has_grad())
			out.internal_gradient() = gradient().copy(device(), false);
		return *this = out;
	}

	// --- Sanity checks ---

// Both tensors must be initialized.
	TENSOR_CHECK(_data,
		"Trying to divide two tensors while the first tensor is empty."
	);
	TENSOR_CHECK(other._data,
		"Trying to divide two tensors while the second tensor is empty."
	);
	// Both tensors must be on the same device
	TENSOR_CHECK(_data->is_gpu == other._data->is_gpu,
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
	if (_data->is_gpu)
	{
		TENSOR_ERROR("CUDA backend not implemented yet.");
	}
	else
	{
		// Extract the data.
		float* out_data = _data->array.data();
		float* oth_data = other_summed._data->array.data();
		// Exptract the shapes.
		Shape out_shape = shape();
		Shape oth_shape = other_summed.shape();
		// Find the last non-unitary dimension to iterate through.
		unsigned last_long_dim = 0;
		for (unsigned i = 0; i < out_shape.dim(); i++)
			if (out_shape[i] > 1)
				last_long_dim = i;
		// Find the offset, will be 1 except for ten1 if broadcasting.
		const unsigned dout = _offset[last_long_dim];
		const unsigned doth = other_summed._offset[last_long_dim];
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
				out_idx += counting_shape[d] * _offset[d];
				oth_idx += counting_shape[d] * other_summed._offset[d];
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
	if (_data && _data->instances > 1)
	{
		Tensor out = *this + val;
		if (has_grad())
			out.internal_gradient() = gradient().copy(device(), false);
		return *this = out;
	}

	// Tensor must be initialized.
	TENSOR_CHECK(_data,
		"Trying to add two tensors while the first tensor is empty."
	);

	// Now we actually sum the tensors.
	if (_data->is_gpu)
	{
		TENSOR_ERROR("CUDA backend not implemented yet.");
	}
	else
	{
		// Extract the data.
		float* out_data = _data->array.data();
		// Exptract the shapes.
		Shape out_shape = shape();
		// Find the last non-unitary dimension to iterate through.
		unsigned last_long_dim = 0;
		for (unsigned i = 0; i < out_shape.dim(); i++)
			if (out_shape[i] > 1)
				last_long_dim = i;
		// Find the vector length to iterate.
		const unsigned vector_len = out_shape[last_long_dim];
		// Create a running shape to count.
		Shape counting_shape(last_long_dim, (int*)nullptr);
		// Iterate through vectors.
		while (true)
		{
			unsigned out_idx = 0;
			for (unsigned d = 0; d < counting_shape.dim(); d++)
				out_idx += counting_shape[d] * _offset[d];

			unsigned count = 0u;
			while (count++ < vector_len)
				out_data[out_idx++] += val;

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

Tensor& Tensor::operator-=(float val)
{
	return *this += -val;
}

Tensor& Tensor::operator*=(float val)
{
	// If there is gradient the middle man must exist.
	if (has_grad() || (_data && _data->instances > 1))
	{
		Tensor out = *this * val;
		if (has_grad())
			out.internal_gradient() = gradient().copy(device(), false);
		return *this = out;
	}

	// Tensor must be initialized.
	TENSOR_CHECK(_data,
		"Trying to multiply by a scalar on an empty tensor."
	);

	// Now we actually multiply the tensors.
	if (_data->is_gpu)
	{
		TENSOR_ERROR("CUDA backend not implemented yet.");
	}
	else
	{
		// Extract the data.
		float* out_data = _data->array.data();
		// Exptract the shapes.
		Shape out_shape = shape();
		// Find the last non-unitary dimension to iterate through.
		unsigned last_long_dim = 0;
		for (unsigned i = 0; i < out_shape.dim(); i++)
			if (out_shape[i] > 1)
				last_long_dim = i;
		// Find the vector length to iterate.
		const unsigned vector_len = out_shape[last_long_dim];
		// Create a running shape to count.
		Shape counting_shape(last_long_dim, (int*)nullptr);
		// Iterate through vectors.
		while (true)
		{
			unsigned out_idx = 0;
			for (unsigned d = 0; d < counting_shape.dim(); d++)
				out_idx += counting_shape[d] * _offset[d];

			unsigned count = 0u;
			while (count++ < vector_len)
				out_data[out_idx++] *= val;

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

Tensor& Tensor::operator/=(float val)
{
	return *this *= 1.f / val;
}

/*
--------------------------------------------------------------------------------------------------------------------------
 Functional Namespace Operators
--------------------------------------------------------------------------------------------------------------------------
*/

Tensor Functional::matmul(const Tensor& ten0, const Tensor& ten1)
{
	return Tensor();
}

Tensor Functional::matmul(const Tensor& ten0, const Tensor& ten1, const Tensor& bias)
{
	return Tensor();
}

Tensor Functional::cat(const Tensor& ten0, const Tensor& ten1, int dim)
{
	return Tensor();
}

Tensor Functional::mean_squared_error(const Tensor& out, const Tensor& y)
{
	return Tensor();
}

Tensor Functional::cross_entropy_loss(const Tensor& out, unsigned* labels)
{
	return Tensor();
}

Tensor Functional::causal_mask(unsigned L)
{
	return Tensor();
}
