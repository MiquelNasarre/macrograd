#include "macrograd.h"
#include "macrograd_error.h"
#include "cuda_backend.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/*
--------------------------------------------------------------------------------------------------------------------------
 Constructor / Destructor
--------------------------------------------------------------------------------------------------------------------------
*/

Tensor::Tensor(const Tensor& other)
{
	if (other._data)
	{
		_data = other._data;
		_data->instances++;

		_view = other._view;
		_offset = other._offset;
		_is_no_grad = other._is_no_grad;
	}
}

Tensor::Tensor(const Array& array, const char* device, bool requires_grad)
{
	_data = new TensorInternals;
	_data->instances++;

	snprintf(_data->device, sizeof(_data->device), "%s", device);

	if (!strcmp(device, "cpu"))
	{
		_data->is_gpu = false;
		_data->array = array;

		_view = array._shape;
		_offset = array._offset;
	}
	else if (!strcmp(device, "cuda"))
	{
		_data->is_gpu = true;

		TENSOR_ERROR("CUDA backend is not implemented yet.");
	}
	else TENSOR_ERROR(
		"Unknown device string found \"%s\".\n"
		"Supported devices are \"cpu\" and \"cuda\".",
		device
	);

	if (requires_grad)
		_data->gradient = new Tensor(array.shape(), device, false);
}

Tensor::Tensor(const Shape& shape, const char* device, bool requires_grad)
{
	TENSOR_CHECK(shape._dim, 
		"If the tensor created will have zero dimensions please use the default constructor."
	);

	_data = new TensorInternals;
	_data->instances++;

	snprintf(_data->device, sizeof(_data->device), "%s", device);

	if (!strcmp(device, "cpu"))
	{
		_data->is_gpu = false;
		_data->array.create(shape);

		_view = shape;
		_offset = _data->array._offset;
	}
	else if (!strcmp(device, "cuda"))
	{
		_data->is_gpu = true;

		TENSOR_ERROR("CUDA backend is not implemented yet.");
	}
	else TENSOR_ERROR(
		"Unknown device string found \"%s\".\n"
		"Supported devices are \"cpu\" and \"cuda\".",
		device
	);

	if (requires_grad)
		_data->gradient = new Tensor(shape, device, false);
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

const Array& Tensor::array() const
{
	TENSOR_CHECK(_data,
		"Trying to get the array on an empty tensor is not allowed."
	);

	// Return internal array.
	return _data->array;
}

const char* Tensor::str() const
{
	TENSOR_CHECK(_data,
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
		has_grad() ? gradient().array().str() : "None",
		array().str()
	);

	return buffer[(next++) % 8];
}

const char* Tensor::device() const
{
	TENSOR_CHECK(_data,
		"Trying to get the device on an empty tensor is not allowed."
	);

	// Return internal device string.
	return _data->device;
}

bool Tensor::has_grad() const
{
	return _data && !_is_no_grad && _data->gradient != nullptr;
}

void Tensor::backward()
{
	// Sanity checks.
	TENSOR_CHECK(_data,
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
	if (_data->is_gpu)
	{
		TENSOR_ERROR("CUDA backend is not implemented yet.");
	}
	else
		((float*)_data->gradient->_data->array._data)[0] = 1.f;

	// Create the topological graph.
	Tensor** list = nullptr;
	unsigned count = 0u;
	add_to_backward_list(&list, &count);

	// Backprop and reset list.
	for (unsigned i = 0; i < count; i++)
	{
		list[i]->_data->added_to_backward = false;
		list[i]->_data->op->_backward();
	}
}

void Tensor::zero_grad()
{
	// Sanity check.
	if (!has_grad())
		return;

	// Do changes on GPU tensors.
	if (_data->is_gpu)
	{
		TENSOR_ERROR("CUDA backend is not implemented yet.");
	}
	// Zero the gradient on the CPU.
	else
		memset(_data->gradient->_data->array._data, 0, _data->array._data_size);
}

Tensor& Tensor::internal_gradient()
{
	// Sanity checks.
	TENSOR_CHECK(has_grad(),
		"Trying to get the gradient on an tensor with no gradient is not allowed."
	);

	// Set the gradient view and offsets to your own.
	_data->gradient->_view = _view;
	_data->gradient->_offset = _offset;

	// Return reference to its gradient tensor.
	return *_data->gradient;
}

const char* Tensor::get_operator() const
{
	if (_data && _data->op) 
		return _data->op->_type;

	return "None";
}

const Tensor& Tensor::gradient() const
{
	// Sanity checks.
	TENSOR_CHECK(has_grad(),
		"Trying to get the gradient on an tensor with no gradient is not allowed."
	);

	// Set the gradient view and offsets to your own.
	_data->gradient->_view = _view;
	_data->gradient->_offset = _offset;

	// Return reference to its gradient tensor.
	return *_data->gradient;
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
	if (!_data) return;

	// Reduce the count.
	_data->instances--;

	// If no more instances delete everything.
	if (!_data->instances)
	{
		// If GPU I'll deal with it when implementig backend.
		if (_data->is_gpu)
		{
			TENSOR_ERROR("CUDA backend not implemented yet.");
		}
		// If CPU tensor simply delete all memory stored in data.
		else
		{
			if (_data->op)
				delete _data->op;

			if (_data->gradient)
				delete _data->gradient;
		}
		// Delete the struct itself.
		delete _data;
	}
}

// Function that to add itself and its relatives to the backward pass.

void Tensor::add_to_backward_list(Tensor*** p_list, unsigned* count)
{
	// Only add to the list if the tensor
	// can backpropagate, this ensures grad too.
	if (!_data->op || _data->added_to_backward)
		return;

	// Add relatives to the list first.
	for (Tensor* t : _data->op->_relatives)
		if (t) t->add_to_backward_list(p_list, count);

	// Add yourself at the front of the list and increase count.
	Tensor** new_list = new Tensor*[*count + 1];
	new_list[0] = this;
	for (unsigned i = 0; i < *count; i++)
		new_list[i + 1] = (*p_list)[i];

	if (*p_list)
		delete* p_list;
	*p_list = new_list;
	(*count)++;
	_data->added_to_backward = true;
}
