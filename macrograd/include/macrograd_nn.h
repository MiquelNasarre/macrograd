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
 Inside ML environments tensors are not usually used on their own to create neural networks, instead
 they are wrapped inside Modules. These are just a list of tensors with gradients that define a forward
 operation through them.

 Importantly, modules can be stacked and combined, or contained inside other modules. This allows a 
 single module to define any arbitrarily long function with any combination of tensors, improving
 interpretability, abstraction, and centralizing gradient descent.

 This centralization gives birth to classes that automatically perform gradient descent in different
 ways for any module. These are called optimizers. They hold a reference to a module and when stepped
 apply gradient descent in specialized ways, using the module's tensors and gradients.

 For gradient descent there are two hyperparameters that reappear in all optimizers: learning rate and
 weight decay. The first one controls the intensity at which gradient descent is performed at a given
 step, and the second one is a regularization parameter to avoid the weights from getting too large.

 If optimizers update the parameters of the modules, who updates the hyperparameters of the optimizers?
 These parameters are usually not constant and change over time, especially learning rate, which is
 usually decreased during the training run. That is done through a class called scheduler, which holds
 a reference to an optimizer. When the step function is called inside a scheduler this may change the
 learning rate and weight decay of the optimizer.

 This header contains the module, optimizer and scheduler base classes. It also contains common optimizers
 and schedulers used for machine learning. Feel free to use these base classes to create your own modules.
--------------------------------------------------------------------------------------------------------------------------
--------------------------------------------------------------------------------------------------------------------------
*/

// Include main library header.
#include "macrograd.h"

// Macrograd base Module class. During construction, all tensors and submodules 
// used by the module should be created and registered using add_parameter() 
// and add_module(). 
//
// Parameters registered this way are converted into tensors with gradients
// enabled and are collected into an internal parameter list. This list also
// includes the parameters of any contained submodules.
//
// The internal parameter list is used by the module's methods and by any 
// optimizer operating on the module. 
class Module
{
private:
	unsigned _parameter_count = 0;	// Stores the module's paramenter count.
	Tensor** _parameters = nullptr;	// Stores a pointer to the module's parameters.

	unsigned _submodule_count = 0;	// Stores the module's submodule count.
	Module** _submodules = nullptr; // Stores a pointer to the module's submodules.

	// Whether the parameters require gradient.
	bool _requires_grad = true;

	// Whether the module is on evaluation mode.
	bool _eval_mode = false;

	// Device where the parameters are stored.
	char _device[16] = "cpu";

	// No module copies are allowed.
	Module(const Module& other) = delete;
	virtual Module& operator=(const Module& other) = delete;
protected:
	// Default protected constructor, creates an empty list.
	Module() = default;

	// Default protected destructor, deletes parameter and submodule lists.
	virtual ~Module();

	// Registers a submodule inside this module. Any modules contained inside 
	// the class must be registered using this function. The parameters of 
	// the submodule are appended to this module's internal parameter list.
	void add_module(Module& module);

	// Registers a tensor as a parameter of the module. Only tensors
	// registered through this function are included in the internal
	// parameter list. Gradients are automatically enabled for the tensor.
	void add_parameter(Tensor& tensor);

public:
	// Whether the module's parameters have gradient enabled.
	virtual bool has_grad() const { return _requires_grad; }

	// Whether the module is on evaluation mode. Can also be used by 
	// module internals to adapt their methods to the modules mode.
	bool is_eval() const { return _eval_mode; }

	// Disables gradients for all parameters in the module. Forward
	// passes executed afterwards will not create a gradient tree.
	virtual void no_grad();

	// Enables gradients for all parameters in the module. Forward
	// passes executed afterwards will create a gradient tree.
	virtual void with_grad();

	// Sets the module to evaluation mode. This flag is propagated across
	// submodules and can be used to cause changes in model behavior.
	void eval();

	// Sets the module to training mode. This flag is propagated across
	// submodules and can be used to cause changes in model behavior.
	void train();

	// Sets all parameter gradients to zero.
	void zero_grad();

	// Sends all the parameters to the device specified.
	void to(const char* device);

	// Virtual forward method meant to be overridden in derived classes. This 
	// is common in neural network libraries, usually used to define computation 
	// performed by the module.
	virtual Tensor forward(const Tensor& in) const { return Tensor(); }

	// Parenthesis operator, calls the module's forward pass.
	virtual Tensor operator()(const Tensor& in) const { return forward(in); }

	// Returns the internal parameter list.
	Tensor** get_parameters()					{ return _parameters; }
	const Tensor* const* get_parameters() const	{ return _parameters; }

	// Returns the number of parameters in the internal parameter list.
	unsigned get_parameter_count() const { return _parameter_count;	}

	// Returns device where the parameters are stored.
	const char* device() const { return _device; }

public:
	// Save/load weight functions. These allow the user to store the model weights at 
	// any time in binary files, for later reloading. They can be used from any device. 
	// The module architecture when loading must match the one used during saving.
	// I suggest the *.mg file extension because it looks cool :)
	void save_weights(const char* path = "model.mg") const;
	void load_weights(const char* path = "model.mg");
};

// Macrograd base optimizer virtual class. This class defines the basic functionality 
// required for an optimizer to update the weights of a module. In particular it 
// provides a gradient zeroing function and a virtual stepping function.
//
// During construction a reference to the module must be provided and will be stored as 
// a protected member. To avoid parameter registration issues, make sure the module is 
// fully constructed before creating its optimizer. 
//
// Learning rate and weight decay are also stored as protected variables so that any 
// optimizer can easily interoperate with any scheduler defined afterwards.
class Optim
{
	// Needs internal access to update 
	// learning rate and weight decay.
	friend class Sched;
protected:
	float _learning_rate = 0.f;	  // Optimizer's current learning rate.
	float _weight_decay  = 0.f;	  // Optimizer's current weight decay.

	// Module referenced by the optimizer.
	Module& _module;

	// Protected constructor. Saves the provided module's reference.
	Optim(Module& module) : _module{ module } {}

	// Virtual default destructor.
	virtual ~Optim() = default;

public:
	// Returns the optimizer's current learning rate.
	float learning_rate() const { return _learning_rate; }

	// Returns the optimizer's current weight decay.
	float weight_decay()  const { return _weight_decay; }

	// Sets all module's parameter gradients to zero.
	virtual void zero_grad() { _module.zero_grad(); }

	// Virtual stepping function, must be overridden in derived classes.
	// Defines how each optimizer updates the module parameters.
	virtual void step() = 0;
private:

	// Optimizer copies are not allowed.
	Optim(const Optim&) = delete;
	Optim& operator=(const Optim&) = delete;
};

// Macrograd base scheduler virtual class. This class defines the basic
// functionality required for schedulers to update optimizer hyperparameters.
// In particular it provides a virtual stepping function and protected access
// to the optimizer hyperparameters.
//
// During construction a reference to the optimizer must be provided and will
// be stored as a protected member. 
//
// Learning rate and weight decay updates should be implemented through the
// virtual step() function.
class Sched
{
protected:
	// Optimizer referenced by the scheduler.
	Optim& _optimizer;

	// Protected constructor. Saves the provided optimizer's reference.
	Sched(Optim& optim) : _optimizer{ optim } {}

	// Virtual default destructor.
	virtual ~Sched() = default;

	// Returns a reference to the optimizer's learning rate.
	float& ref_learning_rate() { return _optimizer._learning_rate; }

	// Returns a reference to the optimizer's weight decay.
	float& ref_weight_decay()  { return _optimizer._weight_decay;  }
public:

	// Virtual stepping function, must be overridden in derived classes.
	// Defines how each scheduler updates the optimizer hyperparameters.
	virtual void step() = 0;
private:

	// Scheduler copies are not allowed.
	Sched(const Sched&) = delete;
	Sched& operator=(const Sched&) = delete;
};

// Optimizer namespace. This namespace contains two optimizer classes widely
// used in machine learning: stochastic gradient descent with momentum (SGD),
// and Adam with decoupled weight decay (AdamW).
namespace Optimizer
{
	// Stochastic Gradient Descent with Momentum optimizer class. It keeps an internally 
	// stored momentum tensor for each module parameter. These tensors combine past updates 
	// with the current parameter gradients according to the momentum hyperparameter. The 
	// parameters themselves are then updated using these accumulated momentum values.
	class SGD : public Optim
	{
	private:
		// Storage for the momentum tensors.
		Tensor* _gradient_storage = nullptr;

		// Momentum value.
		const float _momentum;

	public:
		// Constructor. Stores the module and creates the momentum tensor for each
		// one of its parameters. Make sure the module is fully initialized.
		SGD(Module& module, float momentum, float learning_rate = 0.1f, float weight_decay = 0.f);

		// Destructor, deletes the momentum storage.
		~SGD();

		// During an optimizer step, each momentum tensor is updated from the current
		// parameter gradient and the momentum value. The model parameters are then
		// updated using these accumulated momentum terms, and weight decay is applied.
		void step() override;
	};

	// AdamW optimizer class. It keeps two moment tensors for each module parameter. 
	// One accumulates the gradient and another accumulates the squared gradient. 
	// During optimization both are used to compute adaptive updates for the model 
	// parameters, while weight decay is applied separately.
	class AdamW : public Optim
	{
	private:
		// Optimizer hyperparameters.
		const float _beta1, _beta2, _eps;
		unsigned t = 0u;

		// Storage for the two momentum tensors of each parameter.
		Tensor *_moment1 = nullptr, *_moment2 = nullptr;

	public:
		// Constructor. Stores the module and creates the two moment tensors for 
		// each of its parameters. Make sure the module is fully initialized before 
		// creating its optimizer.
		AdamW(Module& module, float learning_rate, float weight_decay = 0.f, 
			float beta1 = 0.9f, float beta2 = 0.999f, float eps = 1e-8);

		// Destructor, deletes the moment tensors.
		~AdamW();

		// During an AdamW optimizer step, the first and second moments of each
		// parameter are updated and used to compute adaptive parameter updates.
		// Weight decay is applied separately from the gradient-based update.
		void step() override;
	};
}

// Scheduler namespace. This namespace contains a few useful scheduler classes
// that can serve as a baseline for training models, including CosineLR, which
// is widely used in many machine learning applications.
namespace Scheduler
{
	// FunctionalLR scheduler. This scheduler takes a user-defined function and
	// calls it on each step() to update the optimizer's learning rate.
	class FunctionalLR : public Sched
	{
	private:
		// Storage for the lr function.
		float(*_function)() = nullptr;

	public:
		// Constructor, takes as input the target optimizer and the update function
		// and stores them. Performs one step() call to set the initial learning rate.
		FunctionalLR(Optim& optimizer, float(*lr_function)()) : 
			Sched(optimizer), _function{ lr_function } { step(); }

		// Scheduler step call, sets the learning rate to the user's function output.
		void step() override { ref_learning_rate() = _function(); }
	};

	// FunctionalWD scheduler. This scheduler takes a user-defined function and
	// calls it on each step() to update the optimizer's weight decay.
	class FunctionalWD : public Sched
	{
	private:
		// Storage for the wd function.
		float(*_function)() = nullptr;

	public:
		// Constructor, takes as input the target optimizer and the update function
		// and stores them. Performs one step() call to set the initial weight decay.
		FunctionalWD(Optim& optimizer, float(*wd_function)()) : 
			Sched(optimizer), _function{ wd_function } { step(); }

		// Scheduler step call, sets the weight decay to the user's function output.
		void step() override { ref_weight_decay() = _function(); }
	};

	// LinearLR scheduler. This scheduler takes initial and final learning rates
	// as input, and on each step() linearly changes the learning rate from the
	// initial value to the final value over the specified number of epochs.
	class LinearLR : public Sched
	{
	private:
		// Storage for initial and final LR.
		const float _lr0, _lr1;

		// Storage for epoch count.
		const unsigned total_epoch;
		unsigned _epoch = 0u;

	public:
		// Constructor, takes as input the target optimizer, the initial and final
		// learning rate values, and the number of epochs. Performs one step() call
		// to set the initial learning rate.
		LinearLR(Optim& optimizer, float initial_lr, float final_lr, unsigned epoch);

		// Scheduler step call, linearly interpolates between the initial and final
		// learning rate values over the specified number of epochs.
		void step() override;
	};

	// CosineLR scheduler. This scheduler is widely used in machine learning due
	// to its natural progression from an aggressive learning rate to a long flat
	// tail for fine-tuning. It follows a cosine curve descending from the initial
	// learning rate to the final learning rate over the specified number of epochs.
	class CosineLR : public Sched
	{
	private:
		// Storage for initial and final LR.
		const float _lr0, _lr1;

		// Storage for epoch count.
		const unsigned total_epoch;
		unsigned _epoch = 0u;

	public:
		// Constructor, takes as input the target optimizer, the initial and final
		// learning rate values, and the number of epochs. Performs one step() call
		// to set the initial learning rate.
		CosineLR(Optim& optimizer, float initial_lr, float final_lr, unsigned epoch);
		
		// Scheduler step call, follows a cosine curve from the initial learning
		// rate to the final learning rate over the specified number of epochs.
		void step() override;
	};
}
