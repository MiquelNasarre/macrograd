#include "macrograd_nn.h"
#include "macrograd_error.h"
#include "cuda_backend.h"

#include <stdint.h>
#include <math.h>
#include <new>

/*
--------------------------------------------------------------------------------------------------------------------------
 Module class functions
--------------------------------------------------------------------------------------------------------------------------
*/

// Registers a submodule inside this module. Any modules contained inside 
// the class must be registered using this function. The parameters of 
// the submodule are appended to this module's internal parameter list.

void Module::add_module(Module& module)
{
	// Add the parameters to your list.
	for (unsigned i = 0; i < module._parameter_count; i++)
		add_parameter(*module._parameters[i]);
}

// Registers a tensor as a parameter of the module. Only tensors
// registered through this function are included in the internal
// parameter list. Gradients are automatically enabled for the tensor.

void Module::add_parameter(Tensor& tensor)
{
	// The tensor must be initialized.
	MACROGRAD_CHECK(tensor.is_init(),
		"An empty tensor can not be a parameter.\n"
		"Make sure your tensors are properly initialized before adding them as parameters."
	);
	// Force gradient into the tensor.
	tensor = tensor.with_grad();

	// Create new tensor list.
	Tensor** new_parameters = new Tensor*[_parameter_count + 1];

	// Copy all tensors to new list, including new tensor.
	for (unsigned i = 0; i < _parameter_count; i++)
		new_parameters[i] = _parameters[i];
	new_parameters[_parameter_count++] = &tensor;

	// Delete old list, store new one.
	if (_parameters)
		delete[] _parameters;
	_parameters = new_parameters;
}

// Disables gradients for all parameters in the module. Forward
// passes executed afterwards will not create a gradient tree.

void Module::no_grad()
{
	for (unsigned i = 0; i < _parameter_count; i++)
		*_parameters[i] = _parameters[i]->no_grad();
}

// Enables gradients for all parameters in the module. Forward
// passes executed afterwards will create a gradient tree.

void Module::with_grad()
{
	for (unsigned i = 0; i < _parameter_count; i++)
		*_parameters[i] = _parameters[i]->with_grad();
}

// Sets all parameter gradients to zero.

void Module::zero_grad()
{
	for (unsigned i = 0; i < _parameter_count; i++)
		_parameters[i]->zero_grad();
}

// Sends all the parameters to the device specified.

void Module::to(const char* device)
{
	for (unsigned i = 0; i < _parameter_count; i++)
		*_parameters[i] = _parameters[i]->to(device, true);
}

/*
--------------------------------------------------------------------------------------------------------------------------
 File handling Functions
--------------------------------------------------------------------------------------------------------------------------
*/

// Header of a weights file.
// Magic spells "MACROGD" :)
struct FileHeader 
{
	uint64_t magic	 = 0x0044474F5243414Dull;
	uint64_t version = 0x00000000000A3156ull;
	uint64_t flags	 = 0x0000000000000000ull;
	uint64_t count;
};

// Tensor descriptor struct.
// Guess what magic spells?
struct TensorHeader
{
	uint64_t magic = 0x0A524F534E45540Aull;
	uint64_t idx;
	uint64_t hash;
	uint64_t byte_size;
};

// Inline helper to create a deterministice key based on the 
// shape dimensions. Collisions are extremely unlikely.
static inline uint64_t hash_tensor(const Shape& shape)
{
	auto splitmix = [](uint64_t _seed)
	{
		_seed += 0x9E3779B97F4A7C15ull;
		_seed = (_seed ^ (_seed >> 30)) * 0xBF58476D1CE4E5B9ull;
		_seed = (_seed ^ (_seed >> 27)) * 0x94D049BB133111EBull;
		_seed ^= (_seed >> 31);
		return _seed;
	};

	uint64_t key = 123ull;
	for (unsigned i = 0; i < shape.dim(); i++)
		key ^= splitmix(i) * shape[i];

	return key;
}

// Save/load weight functions. These allow the user to store the model weights at 
// any time in binary files, for later reloading. They can be used from any device. 
// The module architecture when loading must match the one used during saving.
// I suggest the *.mg file extension because it looks cool :)

void Module::save_weights(const char* path) const
{
	// First open the raw bytes file.
	FILE* file = nullptr;
	fopen_s(&file, path, "wb");
	MACROGRAD_CHECK(file,
		"Unable to create the file \"%s\" during a save weights call.", path
	);

	// Simple helper to reduce bloat.
	auto write = [&](const void* src, size_t size)
	{
		if (fwrite(src, 1, size, file) != size)
			MACROGRAD_ERROR("Unexpected error ocurred when trying to write to file during a save weights call.");
	};

	// Prepare the file header and write.
	FileHeader fhead = {};
	fhead.count = _parameter_count;
	write(&fhead, sizeof(fhead));

	// For every tensor write its header.
	for (unsigned i = 0; i < _parameter_count; i++)
	{
		TensorHeader thead = {};
		thead.idx = i;
		thead.hash = hash_tensor(_parameters[i]->shape());
		thead.byte_size = _parameters[i]->data_size();
		write(&thead, sizeof(thead));
	}

	// For every tensor dump all the data.
	for (unsigned i = 0; i < _parameter_count; i++)
	{
		Tensor& ten = *_parameters[i];
		
		// If the tensor is in the GPU we will need to copy the memory to an intermediate buffer.
		if (ten.is_gpu())
		{
			void* buffer = ::operator new[](ten.data_size(), std::align_val_t(64));

			// Copy memory to buffer.
			cuda::copy_gpu_to_cpu(buffer, ten.internal_data(), ten.data_size());

			// Dump it to file.
			write(buffer, ten.data_size());

			// Free the buffer.
			::operator delete[](buffer, std::align_val_t(64));
		}
		// Else simply dump the internal data.
		else write(ten.internal_data(), ten.data_size());
	}

	// Close file and return.
	if (fclose(file) != 0)
		MACROGRAD_ERROR("File close failure during save weights call.");
	return;
}

// Save/load weight functions. These allow the user to store the model weights at 
// any time in binary files, for later reloading. They can be used from any device. 
// The module architecture when loading must match the one used during saving.
// I suggest the *.mg file extension because it looks cool :)

void Module::load_weights(const char* path)
{
	// First open the raw bytes file.
	FILE* file = nullptr;
	fopen_s(&file, path, "rb");
	MACROGRAD_CHECK(file,
		"Unable to open the file \"%s\" during a load weights call.", path
	);

	// Simple helper to reduce bloat.
	auto read = [&](void* dst, size_t size)
	{
		if (fread(dst, 1, size, file) != size)
			MACROGRAD_ERROR("Unexpected error ocurred when trying to read from file during a load weights call.");
	};

	// Read file header and compare.
	FileHeader fhead = {};
	read(&fhead, sizeof(fhead));

	// Perform header checks.
	FileHeader fdef = {};
	MACROGRAD_CHECK(fhead.magic == fdef.magic,
		"Invalid file magic found during a load weights call.\n" 
		"Make sure the file path is correct and it was created with this software."
	);
	MACROGRAD_CHECK(fhead.version == fdef.version,
		"Different formatting version found during a load weights call.\n"
		"Make sure the file was created with the same version of this software."
	);
	MACROGRAD_CHECK(fhead.count == _parameter_count,
		"Missmatch in the parameter count found during a load weights call.\n"
		"Make sure the Module architecture matches exactly the one corresponding to this save file."
	);

	// For every tensor perform tensor header checks.
	TensorHeader tdef = {};
	for (unsigned i = 0; i < _parameter_count; i++)
	{
		TensorHeader thead = {};
		read(&thead, sizeof(thead));

		// Perform header checks.
		MACROGRAD_CHECK(thead.magic == tdef.magic,
			"Invalid tensor magic found during a load weights call. Possible file corruption. \n"
			"Make sure the file was saved successfully with the same software version."
		);
		MACROGRAD_CHECK(thead.idx == i,
			"Invalid tensor indexing found during a load weights call. Possible file corruption. \n"
			"Make sure the file was saved successfully with the same software version."
		);
		MACROGRAD_CHECK(thead.hash == hash_tensor(_parameters[i]->shape()),
			"Invalid tensor shape hash found during a load weights call. Make sure the shape of the model\n"
			"parameters and the model architecture are the same to the ones used to generate the save file."
		);
		MACROGRAD_CHECK(thead.byte_size == _parameters[i]->data_size(),
			"Invalid tensor total byte size found during a load weights call. Make sure the shape of the model\n"
			"parameters and the model architecture are the same to the ones used to generate the save file."
		);
	}

	// For every tensor read all the data.
	for (unsigned i = 0; i < _parameter_count; i++)
	{
		Tensor& ten = *_parameters[i];

		// If the tensor is in the GPU we will need to read the file to an intermediate buffer.
		if (ten.is_gpu())
		{
			void* buffer = ::operator new[](ten.data_size(), std::align_val_t(64));

			// Read file to buffer.
			read(buffer, ten.data_size());

			// Copy buffer to CUDA tensor memory.
			cuda::copy_cpu_to_gpu(ten.internal_data(), buffer, ten.data_size());

			// Free the buffer.
			::operator delete[](buffer, std::align_val_t(64));
		}
		// Else simply read to the internal data.
		else read(ten.internal_data(), ten.data_size());
	}

	// Just to see if you wrote something in my save file.
	char extra;
	if (fread(&extra, 1, 1, file) != 0)
		MACROGRAD_ERROR("Unexpected trailing data found during load weights call.");
	
	// Close file and return.
	if (fclose(file) != 0)
		MACROGRAD_ERROR("File close failure during load weights call.");
	return;
}

/*
--------------------------------------------------------------------------------------------------------------------------
 SGD Optimizer
--------------------------------------------------------------------------------------------------------------------------
*/

// Constructor. Stores the module and creates the momentum tensor for each
// one of its parameters. Make sure the module is fully initialized.

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

// Destructor, deletes the momentum storage.

Optimizer::SGD::~SGD() 
{ 
	delete[] _gradient_storage; 
}

// During an optimizer step, each momentum tensor is updated from the current
// parameter gradient and the momentum value. The model parameters are then
// updated using these accumulated momentum terms, and weight decay is applied.

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

// Constructor. Stores the module and creates the two moment tensors for 
// each of its parameters. Make sure the module is fully initialized before 
// creating its optimizer.

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

// Destructor, deletes the moment tensors.

Optimizer::AdamW::~AdamW()
{
	delete[] _moment1;
	delete[] _moment2;
}

// During an AdamW optimizer step, the first and second moments of each
// parameter are updated and used to compute adaptive parameter updates.
// Weight decay is applied separately from the gradient-based update.

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

/*
--------------------------------------------------------------------------------------------------------------------------
 LinearLR Scheduler
--------------------------------------------------------------------------------------------------------------------------
*/

// Constructor, takes as input the target optimizer, the initial and final
// learning rate values, and the number of epochs. Performs one step() call
// to set the initial learning rate.

Scheduler::LinearLR::LinearLR(Optim& optimizer, float initial_lr, float final_lr, unsigned epoch) :
	Sched(optimizer), _lr0{ initial_lr }, _lr1{ final_lr }, total_epoch{ epoch }
{
	step();
}

// Scheduler step call, linearly interpolates between the initial and final
// learning rate values over the specified number of epochs.

void Scheduler::LinearLR::step()
{
	ref_learning_rate() = (_lr1 - _lr0) * (float(_epoch++) / total_epoch) + _lr0;
}

/*
--------------------------------------------------------------------------------------------------------------------------
 CosineLR Scheduler
--------------------------------------------------------------------------------------------------------------------------
*/

// Constructor, takes as input the target optimizer, the initial and final
// learning rate values, and the number of epochs. Performs one step() call
// to set the initial learning rate.

Scheduler::CosineLR::CosineLR(Optim& optimizer, float initial_lr, float final_lr, unsigned epoch) :
	Sched(optimizer), _lr0{ initial_lr }, _lr1{ final_lr }, total_epoch{ epoch }
{
	step();
}

// Scheduler step call, follows a cosine curve from the initial learning
// rate to the final learning rate over the specified number of epochs.

void Scheduler::CosineLR::step()
{
	float cos_epoch = cosf((3.141592f * _epoch++) / total_epoch);
	ref_learning_rate() = (_lr0 - _lr1) * (1.f + cos_epoch) / 2.f + _lr1;
}
