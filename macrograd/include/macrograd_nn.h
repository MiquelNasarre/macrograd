#pragma once
#include "macrograd.h"

class Module
{
private:
	unsigned _parameter_count = 0;
	Tensor** _parameters = nullptr;

protected:
	Module() = default;

	void add_module(const Module& module);
	void add_parameter(Tensor* tensor);

public:
	virtual ~Module() { if (_parameters) delete[] _parameters; }
	Module(const Module& other) = delete;
	virtual Module& operator=(const Module& other) = delete;

	void no_grad();
	void with_grad();

	void zero_grad();
	void to(const char* device);

	virtual Tensor forward(const Tensor& in) const	{ return Tensor();		}
	Tensor operator()(const Tensor& in) const		{ return forward(in);	}

	Tensor** get_parameters()					{ return _parameters;		}
	const Tensor* const* get_parameters() const	{ return _parameters;		}
	unsigned get_parameter_count() const		{ return _parameter_count;	}
};

class Optimizer
{
private:
	Module& _module;

	Tensor* _gradient_storage = nullptr;

	float _momentum = 0.f;
	float _learning_rate = 0.f;
	float _weight_decay = 0.f;

public:
	Optimizer(Module& module, float momentum, float weight_decay, float learning_rate) :
		_module{ module }, _momentum{ momentum }, _learning_rate{ learning_rate }, _weight_decay{ weight_decay }
	{
		// Get the parameter count.
		unsigned count = _module.get_parameter_count();
		// Get the parameters.
		Tensor** parameters = _module.get_parameters();

		_gradient_storage = new Tensor[count];
		for (unsigned i = 0; i < count; i++)
			_gradient_storage[i] = Tensor(Shape(parameters[i]->numel()), parameters[i]->device(), false);
	}

	~Optimizer() { delete[] _gradient_storage; }

	void step()
	{
		// Get the parameter count.
		unsigned count = _module.get_parameter_count();
		// Get the parameters.
		Tensor** parameters = _module.get_parameters();

		for (unsigned i = 0; i < count; i++)
		{
			_gradient_storage[i].internal_multiply(_momentum);
			_gradient_storage[i].internal_add_prod(1.f - _momentum, parameters[i]->internal_gradient());
			
			parameters[i]->internal_multiply(1.f - _learning_rate * _weight_decay);
			parameters[i]->internal_add_prod(-_learning_rate, _gradient_storage[i]);
		}
	}
};

