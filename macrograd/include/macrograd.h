#pragma once

class Shape
{
	friend class Array;
	friend class Tensor;
private:
	unsigned _dim = 0u;			// Stores the number of dimensions.
	int* _sizes = nullptr;		// Stores the size of the dimensions.

public:
	Shape() = default;
	Shape(const Shape& other) { *this = other; }
	Shape(unsigned dim, int* sizes);

	template<class... args>
	Shape(args... sizes)
	{
		constexpr unsigned count = sizeof...(args);
		int processed_sizes[count] = { int(sizes)... };

		*this = Shape(count, processed_sizes);
	}

	Shape& operator=(const Shape& other);

	~Shape() { if (_sizes) delete[] _sizes; }

	int& operator[](int dim)				{ return _sizes[unsigned(dim + _dim * (2 - dim / int(_dim))) % _dim]; }
	const int& operator[](int dim) const	{ return _sizes[unsigned(dim + _dim * (2 - dim / int(_dim))) % _dim]; }

	void remove(int dim);
	void add(int dim, int size);

	unsigned dim() const { return _dim; }
	const char* str(const char* fmt = "%i") const;
};

// Array class, for easy creation and modification of arrays. Since tensors carry a lot of 
// baggage for taking care of all its operations, arrays are made to provide easy manipulation 
// of data without worrying about gradients or in-place modifications. They can also be used 
// to initialize tensors.
class Array
{
	friend class Tensor;
public:
	// Performs a full copy of the other array contents.
	Array& operator=(const Array& other);

	// Stores data for an array with the specified dimensions.
	void create(const Shape& shape);

	// Default constructor for empty arrays.
	Array() = default;

	// Creates an array with the specified shape.
	Array(const Shape& shape) { create(shape); }

	// Copies the data of the other array.
	Array(const Array& other) { *this = other; }

	// Deletes the array data.
	~Array();

	// --- User functions ---

	const char* str(const char* fmt = "%+.4f") const;

	void set_value(const Shape& idxs, float value);
	float get_value(const Shape& idxs) const;

	void set_vector(unsigned* route, int idx0, int idx1, float* values);
	const float* get_vector(unsigned* route, int idx0, int idx1) const;
	float* get_vector(unsigned* route, int idx0, int idx1);

	float*			data()				{ return _data; }
	const float*	data() const		{ return _data; }
	unsigned		dim() const			{ return _shape.dim(); }
	unsigned		size(int dim) const	{ return _shape[dim]; }
	unsigned		data_size() const	{ return sizeof(float) * _total_size; }
	const Shape&	shape() const		{ return _shape; }
	const Shape&	offset() const		{ return _offset; }

private:
	Shape _shape;	// Stores the array's shape.

	float* _data = nullptr;		// Pointer to the internal array values.

	Shape _offset;				// Vector holding offset information of the data.
	unsigned _total_size = 0u;	// Total size of the data pointer.
};

class Tensor
{
public:
	Tensor() = default;
	Tensor(const Tensor& other);
	Tensor(const Array& array, const char* device = "cpu", bool requires_grad = false);
	Tensor(const Shape& shape, const char* device = "cpu", bool requires_grad = false);
	
	// Reduces the instance count by one.
	~Tensor();

	// --- Data functions ---

	const Array& array() const;

	// --- Gradients ---

	bool has_grad() const;
	void backward();
	void zero_grad();
	const Tensor& gradient() const;
private:
	Tensor& internal_gradient();
public:
	const char* get_operator() const;

	// --- Quality of life ---

	unsigned dim()			const { return _view.dim(); }
	unsigned size(int dim)	const { return _view[dim]; }
	const Shape& shape()	const { return _view; }
	const char* str()		const;
	const char* device()	const;
private:
	// For backpropagation it is important to keep track of the operations
	// used to generate or modify given tensors. This is why each operator 
	// defines its own inheritance of this class and stores it in the created 
	// tensor data.
	class TensorOp
	{
	public:
		// default Constructor/Destructor.
		explicit TensorOp(const char* type) : _type{ type }{}
		virtual ~TensorOp() = default;
		
		// List to store operator relatives. Last element is nullptr.
		Tensor* _relatives[2] = { nullptr, nullptr };

		// To be initialized by inheritance.
		const char* _type;

		// Internal backpropagation function to distribute gradients to its relatives.
		virtual void _backward() = 0;
	};

	// Viewed shape, it is the one considered for operations and unique to each instance.
	Shape _view = {};

	// Offsets according to the view, if not 64 byte aligned the tensor will not be contiguous.
	Shape _offset = {};

	// Whether the tensor says it has grad or not (no_grad() operator).
	bool _is_no_grad = false;

	// Holds all the internal data of a tensor, to be shared across instances.
	struct TensorInternals
	{
		// How many times a tensor exists in the program. 
		// This can either be a real variable that is holding the tensor
		// or can be a virtual instance being held for backpropagation.
		// If a desctructor is called and there are no instances left the 
		// Tensor is properly destroyed.
		unsigned instances = 0u;

		// Array holding the tensor values on CPU.
		Array array = {};

		bool is_gpu = false;		// Stores whether the data is stored on the GPU or not.
		char device[16] = "cpu";	// STores the exact device string.

		TensorOp* op = nullptr;			// Operation used to generate this tensor if exists.
		Tensor* gradient = nullptr;		// Gradient tensor if this is required.
		bool added_to_backward = false; // Whether this tensor has already been added to the backward list.
	}
	// This is a pointer so that multiple instances of the tensor can share 
	// the same exact data, this also means in-place operations effectively
	// can not exist since it would modify all instances.
	*_data = nullptr;

	// When the tensor is transformed or destroyed this method is called to reduce the count 
	// of instances to a specific tensor data. Deletes the data if that count is zero.
	void reduce_instances_count();

	// Function that to add itself and its relatives to the backward pass.
	void add_to_backward_list(Tensor*** p_list, unsigned* count);

public:
	// --- Operators ---
	// No operators are in-place except for the equality operator. All others create a new tensor which can be 
	// instanciated via an equality and will exist as long as there is at least one instance of it. If the tensor
	// is only created as part of a multiple tensor operation, as long as there is gradient involved the tensor
	// will persist in memory until backpropagation.
	
	// --- Non allocation operators ---

	// Returns a tensor with the same data but that says it has no gradient, this helps to 
	// do operations with no gradient with almost no extra overhead.
	Tensor no_grad() const;

	// Equality operator. Takes the same data pointer as the other tensor and increases the instances by one. 
	// If it was holding data, after takeing the new pointer, it decreases the count on the old one.
	Tensor& operator=(const Tensor& other);

	Tensor view(const Shape& shape) const;		// Creates a new tensor with the same data but different view.
	Tensor squeeze(int dim) const;				// Returns a tensor with the specified dimension removed from view, must be 1.
	Tensor unsqueeze(int dim) const;			// Returns a tensor with an added dimension 1 in view, in the specified spot.

	// --- Shape operators ---

	Tensor flatten() const;												// Returns a tensor with the same data reduced to a single vector.
	Tensor contiguous() const;											// If offsets are not properly aligned it creates a new tensor reshaped, else returns itself.
	Tensor transpose(int dim0, int dim1) const;							// Returns a tensor with the specified dimensions transposed.
	Tensor reshape(const Shape& shape) const;							// Returns a tensor reshaped from to the specified dimensions if possible.
	Tensor subset(const Shape& shape, int* start_indices) const;		// Returns a subset of the tensor with the specified shape starting from the specified indices.
	Tensor modified(int* start_indices, const Tensor& other) const;		// Returns a tensor with the same shape but with a subset substituted by the specified tensor.
	Tensor repeat(int dim, unsigned repetitions) const;					// Returns a tensor with repeated dimensions of out_shape = shape * repetitions.
	Tensor copy(const char* device = "cpu", bool grad = false) const;	// Returns an exact copy of the tensor. This includes operator data and gradient if exist.

	// --- Functions ---

	Tensor sign() const;
	Tensor exp() const;
	Tensor log() const;
	Tensor relu() const;
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
};

namespace Functional
{
	static Tensor matmul(const Tensor& ten0, const Tensor& ten1);
	static Tensor matmul(const Tensor& ten0, const Tensor& ten1, const Tensor& bias);
	static Tensor cat(const Tensor& ten0, const Tensor& ten1, int dim);
	static Tensor mean_squared_error(const Tensor& out, const Tensor& y);
	static Tensor cross_entropy_loss(const Tensor& out, unsigned* labels);
	static Tensor causal_mask(unsigned L);

	static Tensor sign(const Tensor& tensor)								{ return tensor.sign();				}
	static Tensor exp(const Tensor& tensor)									{ return tensor.exp();				}
	static Tensor log(const Tensor& tensor)									{ return tensor.log();				}
	static Tensor relu(const Tensor& tensor)								{ return tensor.relu();				}
	static Tensor sigmoid(const Tensor& tensor)								{ return tensor.sigmoid();			}
	static Tensor tanh(const Tensor& tensor)								{ return tensor.tanh();				}
	static Tensor sqrt(const Tensor& tensor)								{ return tensor.sqrt();				}
	static Tensor square(const Tensor& tensor)								{ return tensor.square();			}
	static Tensor pow(const Tensor& tensor, float exp)						{ return tensor.pow(exp);			}
	static Tensor mean(const Tensor& tensor, int dim, bool keepdim = false) { return tensor.mean(dim, keepdim); }
	static Tensor var(const Tensor& tensor, int dim, bool keepdim = false)  { return tensor.var(dim, keepdim);  }
	static Tensor std(const Tensor& tensor, int dim, bool keepdim = false)	{ return tensor.std(dim, keepdim);	}
	static Tensor sum(const Tensor& tensor, int dim, bool keepdim = false)  { return tensor.sum(dim, keepdim);	}
	static Tensor softmax(const Tensor& tensor, int dim)					{ return tensor.softmax(dim);		}
}
