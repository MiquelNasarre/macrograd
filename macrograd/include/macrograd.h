#pragma once

/* LICENSE AND COPYRIGHT
--------------------------------------------------------------------------------------------------------------------------
 * Macrograd - a CUDA/C++ Autograd Tensor Library
 * Copyright (c) 2026 Miguel Nasarre Budiño
 * Licensed under the MIT License. See LICENSE file.
--------------------------------------------------------------------------------------------------------------------------
*/

/* MACROGRAD LIBRARY HEADER
--------------------------------------------------------------------------------------------------------------------------
--------------------------------------------------------------------------------------------------------------------------
 Inspired by Andrej Karpathy's micrograd library, I decided to start this amazing coding project. Macrograd
 is an autograd library built on the same principles, but scaled to actually train neural networks,
 including transformer architectures.
 
 Macrograd is a fully functional tensor library that runs on both CPU and CUDA, supports many kinds of
 operations, and is designed for a wide variety of ML architectures. While the simplicity of this library
 cannot rival that of micrograd, it was still one of the core ideas behind the project, so I tried to keep
 everything as intuitive and compact as possible.
 
 This header contains the three main classes:
 
 - Shape: Stores tensor shapes and provides simple quality-of-life operators that make working with
   tensor dimensions easy and intuitive.
 
 - VectorInt: Synchronization is the worst enemy of performance when working with CUDA kernels, which
   is why this class exists. It is essentially a list of integers that can be stored on both the CPU
   and GPU, allowing fast indexing and integer-based operations (such as cross-entropy loss) without
   unnecessary CPU-to-GPU data transfers.
 
 - Tensor: This is the main class of the library. It stores elements in contiguous memory and defines
   standard tensor operations while also keeping track of gradients for backpropagation. Tensor data
   can live either on the CPU or CUDA, allowing for much faster parallel computations.
 
 This header also contains several useful functions organized in the following namespaces:
 
 - Functional: Defines tensor operations that didn't quite fit inside the Tensor class, most notably
   matmul with bias, cat, cross-entropy loss, and others.
 
 - Random: Provides useful random number generator functions, including a seeding function for the CPU
   and another one for CUDA, allowing deterministic training runs.
 
 - Initialization: Contains tensor initialization functions, currently supporting normal and uniform
   distributions.
 
 Feel free to explore the header and read the comments to better understand how tensors are meant to be
 used and how to build neural networks with them. The code is well commented and explains how the
 different functions are intended to work.
 
 For an ML implementation, you can check the 'macrograd_nn.h' header, where the Module class is defined
 along with some common optimizers and schedulers. Feel free to expand it and create your own modules
 for your machine learning implementations.
--------------------------------------------------------------------------------------------------------------------------
--------------------------------------------------------------------------------------------------------------------------
*/

// Shape class: This class is used to store arbitrary tensor shapes, being able to easily access 
// the sizes, and with some quality-of-life functions to make for easy interaction. Mainly it 
// supports negative indexing and templated initialization. For example 'Shape sh{256,4,32,64}' 
// is a valid initializer, and sh[-1] would return 64.
class Shape
{
private:
	unsigned _dim = 0u;		// Stores the number of dimensions.
	int* _sizes = nullptr;	// Stores the size of the dimensions.

	// Inline helper function, modulo for negative and positive values.
	static inline int mod(int a, int b) { return ((a % b) + b) % b; }

public:
	// Default constructor, creates an empty shape.
	Shape() = default;

	// Copy constructor, copies the sizes of the other shape.
	Shape(const Shape& other) { *this = other; }

	// Pointer initializer. Creates a shape with the given number of 
	// dimensions and copies them from the pointer if it is not null.
	// Else the dimensions are zero-initialized.
	explicit Shape(unsigned dim, int* sizes);

	// Templated constructor, allows for arbitrary shape initializations.
	template<class... args>
	Shape(args... sizes)
	{
		int processed_sizes[sizeof...(args)] = { int(sizes)... };
		*this = Shape(unsigned(sizeof...(args)), processed_sizes);
	}

	// Copy operator, copies the sizes of the other shape.
	Shape& operator=(const Shape& other);

	// Destructor, deletes the sizes data if initialized.
	~Shape() { if (_sizes) delete[] _sizes; }

	// Access operator, returns a reference to the corresponding inner 
	// element of the shape. Supports negative and circular indexing.
	int&		operator[](int dim)			{ return _sizes[mod(dim, _dim)]; }
	const int&	operator[](int dim) const	{ return _sizes[mod(dim, _dim)]; }

	// Checks if all individual dimensions match.
	bool operator==(const Shape& other) const;

	// Removes the specified dimension. Supports negative indexing.
	void remove(int dim);

	// Adds a new dimension on the specified spot with the given value.
	// Supports negative indexing (modulo dim() + 1).
	void add(int dim, int size);

	// Returns the dimension of the shape.
	unsigned dim() const { return _dim; }

	// Returns a string representation of the shape.
	const char* str(const char* fmt = "%i") const;
};

// VectorInt class: This class is used to store vectors of integers both on the CPU and on CUDA,
// allowing for fast indexing and integer/tensor interaction without the need for CPU-GPU data 
// transfers or synchronization. It supports negative indexing, subset views and different kinds
// of initialization.
class VectorInt
{
private:
	unsigned _length = 0u;	// Stores the length of the vector.
	unsigned _offset = 0u;	// Stores the offset of the current view to the internal data.

	// To support views and avoid additional allocation different VectorInt instances
	// can hold a pointer to the same internal data, this is efficient for subsets 
	// and avoids unnecessary bloat, but means that modifying a VectorInt will also 
	// change any other view of the data.
	struct VecInternals
	{
		void* _data = nullptr;		// CPU or CUDA pointer to the integer data.
		char _device[16] = "cpu";	// Device string of the vector.
		bool _is_gpu = false;		// Whether the data is on the GPU.
		unsigned _instances = 0u;	// The number of instances holding a view on this data.
	}
	*_internals = nullptr;	// Pointer to the internal VectorInt data.

	// Inline helper function, modulo for negative and positive values.
	static inline int mod(int a, int b) { return ((a % b) + b) % b; }

	// Reduces the instance count by one. If none are left, it deletes the data.
	void reduce_instance_count();

public:
	// Default constructor, creates an empty vector.
	explicit VectorInt() = default;
	// Constructor, creates a zero-initialized vector with the given length on the specified device.
	explicit VectorInt(unsigned length, const char* device = "cpu");
	// Arange constructor, creates a vector with values in the range [a,b) with the specified stride. 
	// The distance between a and b must be divisible by the stride and must have the same sign.
	VectorInt(int a, int b, int stride = 1, const char* device = "cpu");
	// Copy constructor, gets a view on the other vector's data.
	VectorInt(const VectorInt& other) { *this = other; }
	// Destructor, reduces the instance count.
	~VectorInt() { reduce_instance_count(); }
	// Copy operator, gets a view on the other vector's data.
	VectorInt& operator=(const VectorInt& other);

	// Returns a view of the current vector with its elements in the range [a,b).
	VectorInt subset(int a, int b) const;

	// Access operator, returns a reference to the corresponding element of 
	// the vector. Supports negative and circular indexing. These operators 
	// are only allowed on CPU vectors, for CUDA use get()/set() instead.
	int& operator[](int i);
	const int& operator[](int i) const;

	// Permutation operator. Returns a new vector containing the data 
	// of the current vector, reordered as specified by the indices.
	VectorInt operator[](const VectorInt& idxs) const;

	// Returns the integer at the i-th position in the vector. 
	// Supports negative and circular indexing.
	int get(int i) const;
	// Sets the integer at the i-th position in the vector to the 
	// given value. Supports negative and circular indexing.
	void set(int i, int val);
	// Copies the indices from the pointer to the [a,b) range
	// of the vector. This being a total of (b-a) elements.
	void set(int a, int b, int* values);

	// Returns an element-wise copy of the vector in the specified device.
	VectorInt to(const char* device) const;

	// Returns an element-wise copy of the vector.
	VectorInt copy() const;

	// Returns a pointer to the internal data of the vector. This also includes CUDA pointers.
	int* data()			 	{ return _internals ? (int*)_internals->_data + _offset : nullptr; }
	const int* data() const { return _internals ? (int*)_internals->_data + _offset : nullptr; }

	const char* device() const { return _internals ? _internals->_device : nullptr; } // Returns the device string of the vector data.
	bool is_gpu()		 const { return _internals ? _internals->_is_gpu : false; }   // Returns whether the vector data is stored on the GPU.

	unsigned len()		 const { return _length; }	// Returns the length of the vector.
	const char* str(const char* fmt = "%i") const;	// Returns a string representation of the vector.
};

// Tensor class declaration.
class Tensor;

// Functional namespace: This namespace contains some additional tensor operations, including 
// matrix multiplication, concatenation, loss functions and masks. All of them return a tensor.
namespace Functional
{
	// Matrix multiplication function. Multiplies the matrices along the last two dimensions, meaning these must 
	// match the layout (..., M, K) @ (..., K, N) = (..., M, N). If the number of dimensions does not match the 
	// matrices will be unsqueezed from the leading dimension. Broadcasting is allowed along any given dimension 
	// as long as one of the matrices has size 1. Input transposition is also possible via the boolean toggles. 
	// Though arbitrary broadcasting is allowed, standard broadcasting is preferred for efficiency.
	Tensor matmul(const Tensor& mat0, const Tensor& mat1, bool transA = false, bool transB = false);

	// Matrix multiplication with fused bias. Follows the same broadcasting logic as the regular matrix 
	// multiplication for the matrices. The bias must broadcast to the final output shape of the multiplication.
	// If running on CUDA, a regular column bias is preferred since it allows fusion with the operation itself.
	Tensor matmul(const Tensor& mat0, const Tensor& mat1, const Tensor& bias, bool transA = false, bool transB = false);

	// Concatenation function, allows for the concatenation of two tensors along a given dimension.
	// Tensor shapes must match in all dimensions except for the concatenated one. Supports negative indexing.
	Tensor cat(const Tensor& ten0, const Tensor& ten1, int dim);

	// Mean-squared error computation. Expects two flattened tensors as inputs with the same number of elements. 
	// Computes the average element-wise mean-squared error of the two tensors, returning a single-element tensor.
	Tensor mean_squared_error(const Tensor& ten0, const Tensor& ten1);

	// Cross-entropy loss computation. Expects a 2-dimensional tensor with shape (n_cases, n_classes) containing 
	// the logits and a vector of length n_cases and elements in the range [0, n_classes). It computes the average 
	// cross-entropy loss of the logits with respect to the provided labels. Returns a single-element tensor.
	Tensor cross_entropy_loss(const Tensor& logits, const VectorInt& labels);

	// Negative log-likelihood computation. Expects a 2-dimensional tensor with shape (n_cases, n_classes) 
	// containing the probabilities and a vector of length n_cases and elements in the range [0, n_classes). 
	// It computes the average negative log-likelihood loss of the probabilities with respect to the provided 
	// labels. Returns a single-element tensor.
	Tensor negative_log_likelihood(const Tensor& probs, const VectorInt& labels);

	// One-hot encoder. Given a certain number of classes and a vector of labels of length n_cases, it returns a 
	// tensor with shape (n_cases, n_classes), one-hot encoding the labels into the tensor. Label values must be 
	// in the range [0, n_classes). The output tensor has no gradient and its device matches the vector's device.
	Tensor one_hot(const VectorInt& labels, unsigned num_classes);

	// Returns a 2-dimensional causal mask tensor of shape (L, L). The layout matches the common causal mask 
	// layout for transformers. The tensor has no gradient and its device matches the one specified.
	Tensor causal_mask(unsigned L, const char* device = "cpu");
}

// Random namespace: This namespace contains functions regarding random number generation for the library. This 
// includes seeding functions for both the CPU and CUDA to ensure reproducibility, random number generators, and 
// a shuffling function for vectors.
namespace Random
{
	// This function sets the seed for CPU random number generation. The seed is tied to the splitmix function, 
	// which is used to generate deterministic random numbers for all other CPU functions that require RNG.
	void set_seed(unsigned long long seed);

	// This function sets the seed for CUDA random number generation. The seed is tied to a cuRAND instance, 
	// which is used by all CUDA functions in this library that require RNG.
	void set_cuda_seed(unsigned long long seed);

	// Shuffling algorithm. Takes a vector by reference as input and shuffles all its elements, perfect for 
	// generating random permutations. On the CPU, it uses Fisher-Yates. On CUDA, it arranges random keys to 
	// generate a permutation.
	void shuffle(VectorInt& values);

	// Returns a random float following a normal distribution with the specified mean and standard deviation.
	float rand_normal(float mean, float std);

	// Returns a random float following a uniform distribution in the specified range.
	float rand_uniform(float min, float max);

	// Returns a random integer in the range [min, max]. Both bounds are inclusive.
	int rand_int(int min, int max);
}

// Initialization namespace: This namespace contains two functions for tensor initialization, these being
// normally distributed and uniformly distributed initialization.
namespace Initialization
{
	// Normal initialization. Takes a tensor by reference, and initializes all its elements following 
	// a normal distribution with the specified mean and standard deviation. Defaults to N(0, 1).
	void normal(Tensor& tensor, float mean = 0.f, float std = 1.f);

	// Uniform initialization. Takes a tensor by reference, and initializes all its elements 
	// following a uniform distribution over the specified range. Defaults to (0, 1).
	void uniform(Tensor& tensor, float min = 0.f, float max = 1.f);
}

/* TENSOR CLASS
--------------------------------------------------------------------------------------------------------------------------
--------------------------------------------------------------------------------------------------------------------------
 The most important concept for machine learning is backpropagation, which allows us to compute the derivative
 of a given loss function with respect to every single element involved in the operation by using the chain rule.

 Once we have the derivative we can use it to update every single parameter, with the idea that if we move the 
 parameters in the opposite direction of their derivative we can reduce slightly the loss function output. This 
 idea holds, and helps us train massive neural networks with simple derivatives anyone can analytically solve.
 
 Our main limitation, however, is compute. If we want to train a computer for what we consider simple tasks, like 
 number recognition for example, we will need to define a function with hundreds of thousands of parameters, and 
 differentiate with respect to each one of them individually. And even that is considered a trivial task in ML.

 So how are we able to compute these long chains of derivatives to solve our desired weird optimization problems?
 Tensors and GPUs.

 A tensor is a list of values stored in contiguous space in memory. It can be represented in any arbitrary shape. 
 This includes elements, vectors, matrices, and any arbitrary number of dimensions. Several operations are defined 
 for tensors including addition, multiplication, matrix multiplication and all kinds of element-wise and row-wise 
 operations. A tensor can also have gradient, which is another contiguous space in memory of the same size, used 
 to store the derivatives of each value.
 
 All operations generate a new tensor, and if this has gradient it stores an internal TensorOp class, which 
 contains references to the tensors that created it and information about the operation. These references hold 
 on to the same memory as the original tensor, because they need it in order to propagate the gradient, that is 
 why even if you destroy the original, its data will still persist in the backpropagation tree, and will not be 
 released until the last element of the forward pass is destroyed.
 
 This graph allows us to propagate the derivative (called gradient) of the entire forward pass function, from 
 the final loss output all the way to the first parameters of our network. Once we have all the gradients we 
 can update the parameters as previously described.

 The main caveat of this, though, is that we cannot really modify tensors if these are part of an operation 
 tree. The original values are used for backpropagation and modifying them would mean messing up the entire 
 backward pass. That is why there are not really any in-place operations for tensors, all of them create a 
 new tensor instead while the old one's values stay alive and unchanged as long as someone keeps a reference 
 to that data somewhere.

 With this in mind, the following class is my attempt to implement an efficient autograd tensor and define 
 its operations and backpropagation.
--------------------------------------------------------------------------------------------------------------------------
--------------------------------------------------------------------------------------------------------------------------
*/

// Tensor class: This class is the core class of the library and defines what a tensor is. Every tensor 
// has a private shape/view and a pointer to the internal data, this pointer may be shared by other tensor 
// instances. Most operations return a new tensor. If the operation has a gradient, the new tensor will 
// hold references to the tensors that created it.
// 
// The backward function can be called on single-element tensors that have gradient. A tensor will have a 
// gradient if any of the tensors used to create it did, or if specified by the user during its construction. 
// Specific functions that do not support gradients are flagged as such in the comments.
// 
// Tensors can be stored on the "cpu" or on "cuda". Tensors on different devices cannot interact.
// Some internal functions that modify tensor data are made public for the user's convenience.
// I recommend reading the class function comments to understand its capabilities and limitations.
class Tensor
{
private:
	// --- Internal Tensor Data ---
	// The following is the only private section of the class, it holds its view, stride, and 
	// no_grad flag, as well as the TensorInternals struct definition and pointer.

	Shape _view = {};	// Tensor view shape, it is the one considered for operations and unique to each instance.
	Shape _stride = {}; // Strides according to the view. All tensors are contiguous, so they only depend on the view.

	bool _is_no_grad = false; // Whether the tensor says it has grad or not (no_grad() method).

	// Holds all the internal data of a tensor, shared across instances.
	struct TensorInternals
	{
		// Number of tensor instances referencing this data. This can either be a real variable that 
		// is holding the tensor or a virtual instance created for backpropagation. If a destructor 
		// is called and no instances remain, the internal data is properly destroyed.
		unsigned instances = 0u;

		void* _data = nullptr;		// Pointer to the internal tensor values.
		unsigned _numel = 0u;		// Total number of elements in the array.
		unsigned _data_size = 0u;	// Total byte size of the array data.

		bool is_gpu = false;		// Indicates whether the data is stored on CUDA.
		char device[16] = "cpu";	// Stores the data device string.

		Tensor* gradient = nullptr;	// Gradient tensor if gradients are required.
		bool already_added = false; // Whether this tensor has been added to the backward list.

		// For backpropagation it is important to keep track of the operations used to generate 
		// or modify tensors. This is why each function defines its own inheritance of this 
		// class and stores it in the new tensor data.
		class TensorOp
		{
		public:
			// Constructor/Destructor. We want to store the output tensor because it is used for all 
			// _backward calls, but we don't want it to count to instances, so shenanigans are necessary.
			explicit TensorOp(const char* type, const Tensor& _out) : _type{ type }, out{ *new Tensor(_out) } { out._internals->instances--; }
			virtual ~TensorOp() { out._internals = nullptr; delete& out; }

			// Output tensor data.
			Tensor& out;

			// List of tensors related to this operation.
			Tensor* _relatives[3] = { nullptr, nullptr, nullptr };

			// Operation type string. To be initialized by inheritance.
			const char* _type;

			// Internal backpropagation function to distribute gradients to its relatives.
			virtual void _backward() = 0;
		}
		// Operation used to generate this tensor if exists.
		*op = nullptr;	
	}
	// Pointer to the shared internal data of the tensor.
	*_internals = nullptr;

	// When the tensor is transformed or destroyed this method is called to reduce the count 
	// of instances holding a specific tensor data. Deletes the data if the count reaches zero.
	void reduce_instances_count();

	// Function to add itself and its relatives to the backward pass.
	void add_to_backward_list(Tensor*** p_list, unsigned* count);

public:
	// --- Constructors / Destructor ---

	// Default constructor creates an empty tensor.
	Tensor() = default;

	// Copy constructor, creates a tensor with the same view 
	// that shares the pointer to internal data.
	Tensor(const Tensor& other);

	// Shape constructor, creates a new tensor with the specified shape 
	// on the provided device and with gradient if specified.
	Tensor(const Shape& shape, const char* device = "cpu", bool requires_grad = false);
	
	// Reduces the instance count of the internal data by one.
	~Tensor() { reduce_instances_count(); }

	// --- Gradients ---
	// These are the gradient related functions inside the class. 
	// Used to interact with the gradient. Including zero_grad and backward.
	
	// Returns whether the tensor has a gradient and is flagged as such.
	bool has_grad() const { return !_is_no_grad && _internals && _internals->gradient; }

	// If the tensor has gradient it sets all the gradient values to zero.
	void zero_grad();

	// Returns the internal tensor operation string if exists, else "None".
	const char* get_operator() const;

	// If it has gradient, returns a constant reference to the gradient tensor.
	const Tensor& gradient() const;

	// The backward pass can only be called on single element tensors that have gradient. 
	// It first sets its gradient to one, then generates the topological graph of the 
	// backward pass, and finally calls the internal _backward() functions of each tensor 
	// operation in topological order.
	void backward();

	// --- Quality-of-life ---
	// These functions do not represent any tensor operations but return important information 
	// about the tensor such as its size, its shape, its device, string representations, etc.

	// Returns whether the tensor has internal data.
	bool is_init()			const { return _internals; }

	// Returns the dimensionality of the tensor shape.
	unsigned dim()			const { return _view.dim();	}	

	// Returns the size of the specified dimension.
	unsigned size(int dim)	const { return _view[dim]; }	

	// Returns a constant reference to the tensor shape.
	const Shape& shape()	const { return _view; }	

	// Returns a constant reference to the tensor strides.
	const Shape& stride()	const { return _stride;	}	

	// Returns whether the tensor data is stored on CUDA.
	bool is_gpu()			const { return is_init() && _internals->is_gpu; }

	// Returns the total byte size of the tensor data.
	unsigned byte_size()	const { return is_init() ? _internals->_data_size : 0; } 

	// Returns the total number of elements of the tensor.
	unsigned numel()		const { return is_init() ? _internals->_numel : 0; } 

	// If the tensor is initialized it returns the device string.
	const char* device() const;

	// If the tensor has a single element it returns the value of that element.
	// If the tensor is on CUDA this will force a synchronization.
	float item() const;

	// Returns a string representation of the entire tensor. This includes shape,
	// device, operation, gradient and internal data. If the tensor is on CUDA
	// this will force a synchronization.
	const char* str() const;

	// Returns a string representation of the tensor data, separated by dimensions
	// according to the tensor view. If the tensor is on CUDA this will force a 
	// synchronization.
	const char* array_str(const char* fmt = "%+.4f") const;

public:
	// --- Internal Operators ---
	// Internal helpers to modify data without affecting anything else. When using tensors, numel must match. 
	// Functions are public for convenience, but they cannot be used inside operations that require gradients.

	// Returns a pointer to the internal tensor data. This includes CUDA pointers if the tensor is on CUDA.
	float* internal_data()				{ return _internals ? (float*)_internals->_data : nullptr; }
	const float* internal_data() const	{ return _internals ? (float*)_internals->_data : nullptr; }

	// If has gradient, returns a reference to the gradient tensor.
	Tensor& internal_gradient();

	// The tensor values are incremented by a floating value stored in a pointer times a factor.
	// The pointer can be a CUDA pointer to avoid the need for synchronization or data transfers.
	void internal_add(const float* val, bool gpu = false, float factor = 1.f);
	// The tensor values get multiplied by a floating value stored in a pointer times a factor. 
	// The pointer can be a CUDA pointer to avoid the need for synchronization or data transfers.
	void internal_multiply(const float* val, bool gpu = false, float factor = 1.f);
	// The tensor values are set using a floating value stored in a pointer times a factor.
	// The pointer can be a CUDA pointer to avoid the need for synchronization or data transfers.
	void internal_set(const float* val, bool gpu = false, float factor = 1.f);

	// Adds the other tensor's values to this, numel must match, does not support gradient.
	void internal_add(const Tensor& other);
	// Adds the other tensor's values multiplied by a factor to this, numel must match, does not support gradient.
	void internal_add_prod(float val, const Tensor& other);
	// Subtracts the other tensor's values from this, numel must match, does not support gradient.
	void internal_subtract(const Tensor& other);
	// Multiplies this tensor's values by the other tensor's values, numel must match, does not support gradient.
	void internal_multiply(const Tensor& other);

	// Sets the single element specified by the route to the one provided.
	// If the tensor is on CUDA this will force a synchronization.
	void internal_set_value(const Shape& route, float value);
	// Returns the single element value specified by the route.
	// If the tensor is on CUDA this will force a synchronization.
	float internal_get_value(const Shape& route) const;

	// Sets an entire vector of tensor data to the values provided by the pointer. The vector can 
	// have an arbitrary shape, meaning that if you have a tensor of shape (32,16,24,4) and you 
	// call this function with the route (4,5), the function will try to write the entire last two 
	// dimensions corresponding to this path, reading 24*4 = 96 elements from the pointer. If the 
	// tensor is on CUDA this will force a synchronization.
	void internal_set_vector(const Shape& route, const float* values);
	// Returns a pointer to the internal tensor data at the specified route.
	// This includes CUDA pointers if the tensor is on CUDA.
	float* internal_get_vector(const Shape& route);

	// Returns a tensor containing the leading dimensions with the indices specified.
	// It is useful for generating training set permutations but it does not support gradient.
	Tensor operator[](const VectorInt& idxs) const;

	// --- Copy functions ---
	// These functions create new tensor data by copying the current tensor. They do not 
	// define a gradient operation so they should not be used inside backpropagation.

	// Returns a copy of the tensor with the specified configuration.
	Tensor internal_copy(bool with_grad, bool copy_grad) const;

	// Returns a copy of the tensor in the specified device. If both tensor have gradients it also 
	// copies the gradient. Backpropagation data is lost, so it must not be used inside a forward pass.
	Tensor to(const char* device, bool with_grad = false) const;

	// --- Non allocation operators ---
	// The following functions return tensors that contain the same internal pointer, so they do not
	// allocate new data in the process, instead they change the private view or no_grad flag. They 
	// can safely be used for backpropagation since they preserve internal data, including operations.
	
	// Returns a tensor with the same data that is marked as having no gradient, this allows 
	// to do operations without using gradient with tensors that do have gradient stored.
	Tensor no_grad() const;

	// Equality operator. Takes the same data pointer as the other tensor and increases the 
	// instances by one. If it was initialized it reduces the instances count on the old data.
	Tensor& operator=(const Tensor& other);

	// Returns a tensor with the same data but different view. The view must be compatible with the 
	// data, meaning that it results in the same number of elements. It also supports formatted view, 
	// for example a tensor with shape (64, 96) can be called with (64, -1, 32) and will return a 
	// tensor with shape (64, 3, 32). This will work as long as the shape is compatible.
	Tensor view(const Shape& shape) const;	

	// Returns a tensor with the same data and the view reduced to a single dimension of shape (numel,).
	Tensor flatten() const;

	// Returns a tensor with the specified dimension removed from view, must be 1. 
	// Supports negative indexing, for example (32, 16, 1).squeeze(-1) -> (32, 16).
	Tensor squeeze(int dim) const;

	// Returns a tensor with an added dimension 1 in view, in the specified spot.
	// Supports negative indexing, for example (32, 16).unsqueeze(-1) -> (32, 16, 1).
	Tensor unsqueeze(int dim) const;

	// --- Shape operators ---
	// These operators do create new tensor data while they are not performing any operation, they 
	// are just moving the indices around. Despite that operations like transpose() can become very
	// computationally demanding, so avoid their use inside forward passes if possible.

	// Returns a new tensor with the specified dimensions transposed. This is not a view operation.
	// Supports negative indexing, for example (64, 16, 8).transpose(0, -2) -> (8, 16, 64).
	Tensor transpose(int dim0, int dim1) const;

	// Returns a subset of the tensor with the specified shape starting from the specified indices.
	// If the indices are truncated 0 is assumed, if the shape is truncated full shape is assumed.
	// For example given a tensor with shape (12, 36, 64) calling subset({3, 11}, {2}) would return
	// a tensor of shape (3, 11, 64) containing dimensions (2:5, :11, :) of the original one. 
	Tensor subset(const Shape& shape, const Shape& start_indices) const;

	// Returns a tensor with the same shape but with a subset substituted by the specified tensor.
	// The start indices mark where the subtitution begins, if they are truncated 0 is assumed.
	// For example given a tensor with shape (12, 36, 64) and a modifier of shape (3, 11, 64), 
	// calling modify(other, {2,1}) would write the other's data to the indices (2:5, 1:12, :).
	Tensor modify(const Tensor& other, const Shape& start_indices) const;

	// Returns a tensor with the specified dimension repeated this many times. The initial size
	// of the dimension must be one. Supports negative indexing, for example (32, 1).repeat(-1, 6)
	// would return (32, 6).
	Tensor repeat(int dim, unsigned repetitions) const;

	// Returns a tensor corresponding to the i-th row of the leading dimension. 
	// The leading dimension is squeezed out. Supports negative indexing.
	Tensor operator[](int i) const { return subset({1}, {i}).squeeze(0); }	

	// --- Element/Row-wise Functions ---
	// The following functions are all operators that return an operation on only their data,
	// these being element-wise operations like ReLU or row-wise operators like mean. All of
	// them allow for backpropagation except for argmax, argmin and sign.

	Tensor sign() const;
	Tensor exp() const;
	Tensor log() const;
	Tensor relu() const;
	Tensor silu() const;
	Tensor gelu() const;
	Tensor sigmoid() const;
	Tensor tanh() const;
	Tensor sqrt() const;
	Tensor square() const;
	Tensor pow(float exp) const;
	Tensor sum(int dim, bool keepdim = false) const;
	Tensor mean(int dim, bool keepdim = false) const;
	Tensor var(int dim, bool keepdim = false) const;
	Tensor std(int dim, bool keepdim = false) const;
	Tensor softmax(int dim) const;

	Tensor max(int dim, bool keepdim = false) const;
	Tensor min(int dim, bool keepdim = false) const;
	VectorInt argmax(bool last_dim = true) const;
	VectorInt argmin(bool last_dim = true) const;

	// --- Regular operators ---

	friend Tensor operator+(const Tensor& ten0, const Tensor& ten1);
	friend Tensor operator-(const Tensor& ten0, const Tensor& ten1);
	friend Tensor operator*(const Tensor& ten0, const Tensor& ten1);
	friend Tensor operator/(const Tensor& ten0, const Tensor& ten1);

	friend Tensor operator+(const Tensor& ten, float val);
	friend Tensor operator-(const Tensor& ten, float val);
	friend Tensor operator*(const Tensor& ten, float val);
	friend Tensor operator/(const Tensor& ten, float val);

	friend Tensor operator+(float val, const Tensor& ten);
	friend Tensor operator-(float val, const Tensor& ten);
	friend Tensor operator*(float val, const Tensor& ten);
	friend Tensor operator/(float val, const Tensor& ten);

	Tensor operator-() const;

	// --- Comparissons ---
	// These operators will return all boolean tensors, meaning tensors with zeros or 
	// ones with the same shape as the first tensor, the second tensor must either have 
	// the same shape or cleanly broadcast to the first one. The individual values depend 
	// on individual comparissons between elements.

	friend Tensor operator< (const Tensor& ten0, const Tensor& ten1);
	friend Tensor operator> (const Tensor& ten0, const Tensor& ten1);
	friend Tensor operator<=(const Tensor& ten0, const Tensor& ten1);
	friend Tensor operator>=(const Tensor& ten0, const Tensor& ten1);
	friend Tensor operator==(const Tensor& ten0, const Tensor& ten1);
	friend Tensor operator!=(const Tensor& ten0, const Tensor& ten1);

	friend Tensor operator< (const Tensor& ten, float val);
	friend Tensor operator> (const Tensor& ten, float val);
	friend Tensor operator<=(const Tensor& ten, float val);
	friend Tensor operator>=(const Tensor& ten, float val);
	friend Tensor operator==(const Tensor& ten, float val);
	friend Tensor operator!=(const Tensor& ten, float val);

	friend Tensor operator< (float val, const Tensor& ten);
	friend Tensor operator> (float val, const Tensor& ten);
	friend Tensor operator<=(float val, const Tensor& ten);
	friend Tensor operator>=(float val, const Tensor& ten);
	friend Tensor operator==(float val, const Tensor& ten);
	friend Tensor operator!=(float val, const Tensor& ten);

	// --- "In-place" operators ---
	// In-place operators do not modify the data inside of them, instead they create a new
	// tensor data and assign it to themselves, while the old data remains unchanged. That 
	// being said, they do act as in-place operators if the tensor does not have gradient 
	// and there is only one instance, making operations more efficient.

	Tensor& operator+=(const Tensor& other);
	Tensor& operator-=(const Tensor& other);
	Tensor& operator*=(const Tensor& other);
	Tensor& operator/=(const Tensor& other);

	Tensor& operator+=(float val);
	Tensor& operator-=(float val);
	Tensor& operator*=(float val);
	Tensor& operator/=(float val);

	// --- Friendzone ---
	// These is the section for all the functions and classes that require access to the internal 
	// data of Tensor. Either to create TensorOps for backpropagation or to modify tensor internals.

	// Needs access to internal data. To modify input tensor properties.
	friend class Module;

	// All these functions define TensorOp classes, so they need internal access.
	friend Tensor Functional::matmul(const Tensor& mat0, const Tensor& mat1, bool transA, bool transB);
	friend Tensor Functional::matmul(const Tensor& mat0, const Tensor& mat1, const Tensor& bias, bool transA, bool transB);
	friend Tensor Functional::cat(const Tensor& ten0, const Tensor& ten1, int dim);
	friend Tensor Functional::mean_squared_error(const Tensor& ten0, const Tensor& ten1);
	friend Tensor Functional::cross_entropy_loss(const Tensor& logits, const VectorInt& labels);
	friend Tensor Functional::negative_log_likelihood(const Tensor& probs, const VectorInt& labels);
};
