#pragma once

/* LICENSE AND COPYRIGHT
--------------------------------------------------------------------------------------------------------------------------
 * Macrograd - a CUDA/C++ Autograd Tensor Library
 * Copyright (c) 2026 Miguel Nasarre Budiño
 * Licensed under the MIT License. See LICENSE file.
--------------------------------------------------------------------------------------------------------------------------
*/

/* MACROGRAD NEURAL NETWORK EXTENSION HEADER
--------------------------------------------------------------------------------------------------------------------------
--------------------------------------------------------------------------------------------------------------------------





--------------------------------------------------------------------------------------------------------------------------
--------------------------------------------------------------------------------------------------------------------------
*/

// Include main library header.
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

public:
	void save_weights(const char* path = "model.mg") const;
	void load_weights(const char* path = "model.mg");
};

class Optim
{
	friend class Sched;
protected:
	float _learning_rate = 0.f;
	float _weight_decay  = 0.f;

	Module& _module;

	Optim(Module& module) : _module{ module } {}
	virtual ~Optim() = default;

public:
	float learning_rate() const { return _learning_rate; }
	float weight_decay()  const { return _weight_decay;  }

	virtual void zero_grad() { _module.zero_grad(); }
	virtual void step() = 0;
private:
	Optim(const Optim&) = delete;
	Optim& operator=(const Optim&) = delete;
};

class Sched
{
protected:
	Optim& _optimizer;

	Sched(Optim& optim) : _optimizer{ optim } {}
	virtual ~Sched() = default;

	float& ref_learning_rate() { return _optimizer._learning_rate; }
	float& ref_weight_decay()  { return _optimizer._weight_decay;  }
public:
	virtual void step() = 0;
private:
	Sched(const Sched&) = delete;
	Sched& operator=(const Sched&) = delete;
};

namespace Optimizer
{
	class SGD : public Optim
	{
	private:
		// Storage for the momentum tensors.
		Tensor* _gradient_storage = nullptr;

		// Momentum value.
		const float _momentum;

	public:
		SGD(Module& module, float momentum, float learning_rate = 0.1f, float weight_decay = 0.f);
		~SGD();

		void step() override;
	};

	class AdamW : public Optim
	{
	private:
		// Optimizer hyperparameters.
		const float _beta1, _beta2, _eps;
		unsigned t = 0u;

		// Storage for the two momentum tensors.
		Tensor *_moment1 = nullptr, *_moment2 = nullptr;

	public:
		AdamW(Module& module, float learning_rate, float weight_decay = 0.f, float beta1 = 0.9f, float beta2 = 0.999f, float eps = 1e-8);
		~AdamW();

		void step() override;
	};
}

namespace Scheduler
{
	class FunctionalLR : public Sched
	{
	private:
		// Storage for the lr function.
		float(*_function)() = nullptr;

	public:
		FunctionalLR(Optim& optimizer, float(*lr_function)()) : 
			Sched(optimizer), _function{ lr_function } { step(); }

		void step() override { ref_learning_rate() = _function(); }
	};

	class FunctionalWD : public Sched
	{
	private:
		// Storage for the lr function.
		float(*_function)() = nullptr;

	public:
		FunctionalWD(Optim& optimizer, float(*wd_function)()) : 
			Sched(optimizer), _function{ wd_function } { step(); }

		void step() override { ref_weight_decay() = _function(); }
	};

	class LinearLR : public Sched
	{
	private:
		// Storage for initial and final LR.
		const float _lr0, _lr1;

		// Storage for epoch count.
		const unsigned total_epoch;
		unsigned _epoch = 0u;

	public:
		LinearLR(Optim& optimizer, float initial_lr, float final_lr, unsigned epoch);

		void step() override;
	};

	class CosineLR : public Sched
	{
	private:
		// Storage for initial and final LR.
		const float _lr0, _lr1;

		// Storage for epoch count.
		const unsigned total_epoch;
		unsigned _epoch = 0u;

	public:
		CosineLR(Optim& optimizer, float initial_lr, float final_lr, unsigned epoch);
		
		void step() override;
	};
}