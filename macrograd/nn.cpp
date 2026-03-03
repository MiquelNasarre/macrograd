#include "macrograd_nn.h"
#include "macrograd_error.h"

void Module::add_module(const Module& module)
{
	// Add the parameters to your list.
	for (unsigned i = 0; i < module._parameter_count; i++)
		add_parameter(module._parameters[i]);
}

void Module::add_parameter(Tensor* tensor)
{
	// The tensor must be initialized.
	TENSOR_CHECK(tensor && tensor->_internals,
		"An empty tensor can not be a parameter.\n"
		"Make sure your tensors are properly initialized before adding them as parameters."
	);
	// Force gradient into the tensor.
	tensor->_is_no_grad = false;
	if (!tensor->_internals->gradient)
		tensor->_internals->gradient = new Tensor(tensor->shape(), tensor->device(), false);

	// Create new tensor list.
	Tensor** new_parameters = new Tensor*[_parameter_count + 1];

	// Copy all tensors to new list, including new tensor.
	for (unsigned i = 0; i < _parameter_count; i++)
		new_parameters[i] = _parameters[i];
	new_parameters[_parameter_count++] = tensor;

	// Delete all list.
	if (_parameters)
		delete[] _parameters;
	_parameters = new_parameters;
}

void Module::no_grad()
{
	for (unsigned i = 0; i < _parameter_count; i++)
		_parameters[i]->_is_no_grad = true;
}

void Module::with_grad()
{
	for (unsigned i = 0; i < _parameter_count; i++)
		_parameters[i]->_is_no_grad = false;
}

void Module::zero_grad()
{
	for (unsigned i = 0; i < _parameter_count; i++)
		_parameters[i]->zero_grad();
}

void Module::to(const char* device)
{
	for (unsigned i = 0; i < _parameter_count; i++)
		*_parameters[i] = _parameters[i]->copy(device, true);
}