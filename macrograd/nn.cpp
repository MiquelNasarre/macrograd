#include "macrograd_nn.h"
#include "macrograd_error.h"

#include <math.h>

/*
--------------------------------------------------------------------------------------------------------------------------
 Module class functions
--------------------------------------------------------------------------------------------------------------------------
*/

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
		*_parameters[i] = _parameters[i]->to(device, true);
}

/*
--------------------------------------------------------------------------------------------------------------------------
 LinearLR Scheduler
--------------------------------------------------------------------------------------------------------------------------
*/

Scheduler::LinearLR::LinearLR(Optim& optimizer, float initial_lr, float final_lr, unsigned epoch) :
	Sched(optimizer), _lr0{ initial_lr }, _lr1{ final_lr }, total_epoch{ epoch }
{
	step();
}

void Scheduler::LinearLR::step()
{
	ref_learning_rate() = (_lr1 - _lr0) * (float(_epoch++) / total_epoch) + _lr0;
}

/*
--------------------------------------------------------------------------------------------------------------------------
 CosineLR Scheduler
--------------------------------------------------------------------------------------------------------------------------
*/

Scheduler::CosineLR::CosineLR(Optim& optimizer, float initial_lr, float final_lr, unsigned epoch) :
	Sched(optimizer), _lr0{ initial_lr }, _lr1{ final_lr }, total_epoch{ epoch } 
{
	step();
}

void Scheduler::CosineLR::step()
{
	float cos_epoch = cosf((3.141592f * _epoch++) / total_epoch);
	ref_learning_rate() = (_lr0 - _lr1) * (1.f + cos_epoch) / 2.f + _lr1;
}

/*
--------------------------------------------------------------------------------------------------------------------------
 SGD Optimizer
--------------------------------------------------------------------------------------------------------------------------
*/

Optimizer::SGD::SGD(Module& module, float momentum, float learning_rate, float weight_decay) :
	Optim(module), _momentum{ momentum }
{
	// Set hyperparameters.
	_learning_rate = learning_rate;
	_weight_decay = weight_decay;

	// Get the parameter count.
	unsigned count = _module.get_parameter_count();
	// Get the parameters.
	Tensor** parameters = _module.get_parameters();

	// Save a gradient accumulator for all tensors for momentum.
	_gradient_storage = new Tensor[count];
	for (unsigned i = 0; i < count; i++)
		_gradient_storage[i] = Tensor(Shape(parameters[i]->numel()), parameters[i]->device(), false);
}

Optimizer::SGD::~SGD() 
{ 
	delete[] _gradient_storage; 
}

void Optimizer::SGD::step()
{
	// Get the parameter count.
	unsigned count = _module.get_parameter_count();
	// Get the parameters.
	Tensor** parameters = _module.get_parameters();

	// Do SGD with momentum step.
	for (unsigned i = 0; i < count; i++)
	{
		_gradient_storage[i].internal_multiply(&_momentum);
		_gradient_storage[i].internal_add(parameters[i]->internal_gradient());

		float decay_factor = 1.f - _learning_rate * _weight_decay;
		parameters[i]->internal_multiply(&decay_factor);
		parameters[i]->internal_add_prod(-_learning_rate, _gradient_storage[i]);
	}
}

/*
--------------------------------------------------------------------------------------------------------------------------
 AdamW Optimizer
--------------------------------------------------------------------------------------------------------------------------
*/

Optimizer::AdamW::AdamW(Module& module, float learning_rate, float weight_decay, float beta1, float beta2, float eps) :
	Optim(module), _beta1{ beta1 }, _beta2{ beta2 }, _eps{ eps }
{
	// Set hyperparameters.
	_learning_rate = learning_rate;
	_weight_decay = weight_decay;

	// Get the parameter count.
	unsigned count = _module.get_parameter_count();
	// Get the parameters.
	Tensor** parameters = _module.get_parameters();

	_moment1 = new Tensor[count];
	_moment2 = new Tensor[count];
	for (unsigned i = 0; i < count; i++)
	{
		_moment1[i] = Tensor(Shape(parameters[i]->numel()), parameters[i]->device(), false);
		_moment2[i] = Tensor(Shape(parameters[i]->numel()), parameters[i]->device(), false);
	}
}

Optimizer::AdamW::~AdamW()
{
	delete[] _moment1;
	delete[] _moment2;
}

void Optimizer::AdamW::step()
{
	// Get the parameter count.
	unsigned count = _module.get_parameter_count();
	// Get the parameters.
	Tensor** parameters = _module.get_parameters();

	// Compute step corrections.
	float bc1 = 1.f - powf(float(_beta1), float(++t));
	float bc2 = 1.f - powf(float(_beta2), float(  t));

	// Do AdamW optimization step.
	for (unsigned i = 0; i < count; i++)
	{
		Tensor& grad = parameters[i]->internal_gradient();

		// Update first momentum.
		_moment1[i].internal_multiply(&_beta1);
		_moment1[i].internal_add_prod(1.f - _beta1, grad);

		// Update second momentum.
		_moment2[i].internal_multiply(&_beta2);
		_moment2[i].internal_add_prod(1.f - _beta2, grad.square());

		// Step corrections.
		Tensor m_hat = _moment1[i] / bc1;
		Tensor v_hat = _moment2[i] / bc2;

		// Update parameters. 
		float decay_factor = 1.f - _learning_rate * _weight_decay;
		parameters[i]->internal_multiply(&decay_factor);
		parameters[i]->internal_add_prod(-_learning_rate, m_hat / (v_hat.sqrt() + _eps));
	}
}