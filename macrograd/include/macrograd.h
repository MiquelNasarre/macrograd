#pragma once

class Shape
{
private:
	unsigned _dim = 0u;		// Stores the number of dimensions.
	int* _sizes = nullptr;	// Stores the size of the dimensions.

	static inline int mod(int a, int b) { return ((a % b) + b) % b; }
public:
	Shape() = default;
	Shape(const Shape& other) { *this = other; }
	Shape(unsigned dim, int* sizes);

	template<class... args>
	Shape(args... sizes)
	{
		int processed_sizes[sizeof...(args)] = { int(sizes)... };
		*this = Shape(unsigned(sizeof...(args)), processed_sizes);
	}

	Shape& operator=(const Shape& other);
	~Shape() { if (_sizes) delete[] _sizes; }

	int&		operator[](int dim)			{ return _sizes[mod(dim, _dim)]; }
	const int&	operator[](int dim) const	{ return _sizes[mod(dim, _dim)]; }

	void remove(int dim);
	void add(int dim, int size);

	unsigned dim() const { return _dim; }
	const char* str(const char* fmt = "%i") const;
};

class Tensor;
namespace Functional
{
	Tensor matmul(const Tensor& mat0, const Tensor& mat1);
	Tensor matmul(const Tensor& mat0, const Tensor& mat1, const Tensor& bias);
	Tensor cat(const Tensor& ten0, const Tensor& ten1, int dim);
	Tensor mean_squared_error(const Tensor& ten0, const Tensor& ten1);
	Tensor cross_entropy_loss(const Tensor& logits, unsigned* labels);
	Tensor negative_log_likelihood(const Tensor& probs, unsigned* labels);
	Tensor one_hot(const Shape& size_x_labels, unsigned* labels, const char* device = "cpu");
	Tensor causal_mask(unsigned L, const char* device = "cpu");
}
namespace Random
{
	void set_seed(unsigned long long seed);
	void set_cuda_seed(unsigned long long seed);
	void shuffle(unsigned size, int* data);
	float rand_normal(float mean, float std);
	float rand_uniform(float min, float max);
	int rand_int(int min, int max);
}
namespace Initialization
{
	const Tensor& normal(const Tensor& tensor, float mean = 0.f, float std = 1.f);
	const Tensor& uniform(const Tensor& tensor, float min = 0.f, float max = 1.f);
}

// No operators are in-place except for the equality operator. All others create a new tensor which can be 
// instanciated via an equality and will exist as long as there is at least one instance of it. If the tensor
// is only created as part of a multiple tensor operation, as long as there is gradient involved the tensor
// will persist in memory until backpropagation.
class Tensor
{
	friend class Module;
public:
	// --- Constructors / Destructor ---

	Tensor() = default;
	Tensor(const Tensor& other);
	Tensor(const Shape& shape, const char* device = "cpu", bool requires_grad = false);
	
	// Reduces the instance count by one.
	~Tensor();

	// --- Gradients ---

	bool has_grad() const;
	void backward();
	void zero_grad();
	const Tensor& gradient() const;
	const char* get_operator() const;

	// --- Quality of life ---

	// Returns a copy of the tensor in the specified device. If the tensor had gradient it also 
	// copies the gradient. Backpropagation data is lost, so you must not use inside a forward pass.
	Tensor to(const char* device, bool with_grad = false) const;

	bool is_gpu()			const { return _internals && _internals->is_gpu; }
	bool is_init()			const { return _internals;			}
	unsigned numel()		const { return _internals->_numel;	}
	unsigned dim()			const { return _view.dim();			}
	unsigned size(int dim)	const { return _view[dim];			}
	const Shape& shape()	const { return _view;				}
	const Shape& stride()	const { return _stride;				}
	float item()			const;
	const char* str()		const;
	const char* device()	const;
	const char* array_str(const char* fmt = "%+.4f") const;
private:
	// --- Internal Tensor Data ---

	Shape _view = {};	// Viewed shape, it is the one considered for operations and unique to each instance.
	Shape _stride = {}; // Offsets according to the view, all tensors are contiguous so it only depends on view.

	bool _is_no_grad = false; // Whether the tensor says it has grad or not (no_grad() operator).

	// Holds all the internal data of a tensor, to be shared across instances.
	struct TensorInternals
	{
		// How many times a tensor exists in the program. 
		// This can either be a real variable that is holding the tensor
		// or can be a virtual instance being held for backpropagation.
		// If a desctructor is called and there are no instances left the 
		// Tensor is properly destroyed.
		unsigned instances = 0u;

		void* _data = nullptr;		// Pointer to the internal tensor values.
		unsigned _numel = 0u;		// Total number of element in the array.
		unsigned _data_size = 0u;	// Total byte size of the array data.

		bool is_gpu = false;		// Stores whether the data is stored on the GPU or not.
		char device[16] = "cpu";	// STores the exact device string.

		// For backpropagation it is important to keep track of the operations
		// used to generate or modify given tensors. This is why each operator 
		// defines its own inheritance of this class and stores it in the created 
		// tensor data.
		class TensorOp
		{
		public:
			// Constructor/Destructor. We want to store the output tensor beacause it is used for all 
			// _backward calls, but we don't want it to count to instances, so shinnanigans are necessary.
			explicit TensorOp(const char* type, const Tensor& _out) : _type{ type }, out{ *new Tensor(_out) } { out._internals->instances--; }
			virtual ~TensorOp() { out._internals = nullptr; delete &out; }

			// Output tensor data.
			Tensor& out;

			// List to store operator relatives. Last element is nullptr.
			Tensor* _relatives[3] = { nullptr, nullptr, nullptr };

			// To be initialized by inheritance.
			const char* _type;

			// Internal backpropagation function to distribute gradients to its relatives.
			virtual void _backward() = 0;
		}
		*op = nullptr;					// Operation used to generate this tensor if exists.
		Tensor* gradient = nullptr;		// Gradient tensor if this is required.
		bool added_to_backward = false; // Whether this tensor has already been added to the backward list.
	}
	// This is a pointer so that multiple instances of the tensor can share 
	// the same internal data, this also means in-place operations effectively
	// can not exist since it would modify all instances.
	*_internals = nullptr;

	// When the tensor is transformed or destroyed this method is called to reduce the count 
	// of instances to a specific tensor data. Deletes the data if that count is zero.
	void reduce_instances_count();

	// Function that to add itself and its relatives to the backward pass.
	void add_to_backward_list(Tensor*** p_list, unsigned* count);

public:
	// --- Internal Operators ---
	// Internal helpers to modify data without affecting anything else. If using tensors numel must match. 
	// Functions are public for convenience but they cannot be used inside neural networks logic.

	float* internal_data() const { return _internals ? (float*)_internals->_data : nullptr; }

	Tensor& internal_gradient();
	Tensor internal_copy(bool with_grad, bool copy_grad) const;

	void internal_add(float val);
	void internal_multiply(float val);
	void internal_set(float val);

	void internal_add(const Tensor& other);
	void internal_add_prod(float val, const Tensor& other);
	void internal_subtract(const Tensor& other);
	void internal_multiply(const Tensor& other);

	void internal_set_value(const Shape& route, float value);
	float internal_get_value(const Shape& route) const;
	void internal_set_vector(const Shape& route, const float* values);
	float* internal_get_vector(const Shape& route);

	// --- Non allocation operators ---
	
	// Returns a tensor with the same data but that says it has no gradient, this helps to 
	// do operations with no gradient with almost no extra overhead.
	Tensor no_grad() const;

	// Equality operator. Takes the same data pointer as the other tensor and increases the instances by one. 
	// If it was holding data, after takeing the new pointer, it decreases the count on the old one.
	Tensor& operator=(const Tensor& other);

	Tensor view(const Shape& shape) const;	// Creates a new tensor with the same data but different view.
	Tensor flatten() const;					// Returns a tensor with the same data reduced to a single vector.
	Tensor squeeze(int dim) const;			// Returns a tensor with the specified dimension removed from view, must be 1.
	Tensor unsqueeze(int dim) const;		// Returns a tensor with an added dimension 1 in view, in the specified spot.

	// --- Shape operators ---

	Tensor transpose(int dim0, int dim1) const;								// Returns a tensor with the specified dimensions transposed.
	Tensor subset(const Shape& shape, const Shape& start_indices) const;	// Returns a subset of the tensor with the specified shape starting from the specified indices.
	Tensor modify(const Tensor& other, const Shape& start_indices) const;	// Returns a tensor with the same shape but with a subset substituted by the specified tensor.
	Tensor repeat(int dim, unsigned repetitions) const;						// Returns a tensor with repeated dimensions of out_shape = shape * repetitions.

	// --- Functions ---

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
	Tensor mean(int dim, bool keepdim = false) const;
	Tensor var(int dim, bool keepdim = false) const;
	Tensor std(int dim, bool keepdim = false) const;
	Tensor sum(int dim, bool keepdim = false) const;
	Tensor softmax(int dim) const;

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

	friend Tensor Functional::matmul(const Tensor& mat0, const Tensor& mat1);
	friend Tensor Functional::matmul(const Tensor& mat0, const Tensor& mat1, const Tensor& bias);
	friend Tensor Functional::cat(const Tensor& ten0, const Tensor& ten1, int dim);
	friend Tensor Functional::mean_squared_error(const Tensor& ten0, const Tensor& ten1);
	friend Tensor Functional::cross_entropy_loss(const Tensor& logits, unsigned* labels);
	friend Tensor Functional::negative_log_likelihood(const Tensor& probs, unsigned* labels);
};
