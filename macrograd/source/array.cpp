#include "macrograd.h"
#include "macrograd_error.h"

#include <new>

/*
--------------------------------------------------------------------------------------------------------------------------
 Shape functions
--------------------------------------------------------------------------------------------------------------------------
*/

Shape::Shape(unsigned dim, int* sizes) : _dim{ dim }
{
	if (!dim)
		return;

	_sizes = new int[dim];

	if (sizes)
		for (unsigned i = 0; i < dim; i++)
			_sizes[i] = sizes[i];
	else 
		for (unsigned i = 0; i < dim; i++)
			_sizes[i] = 0;
}

Shape& Shape::operator=(const Shape& other)
{
	if (_dim == other._dim)
	{
		for (unsigned i = 0; i < _dim; i++)
			_sizes[i] = other._sizes[i];
		return *this;
	}

	if (_sizes)
	{
		delete[] _sizes;
		_sizes = nullptr;
	}

	_dim = other._dim;
	if (!_dim)
		return *this;

	_sizes = new int[other._dim];
	for (unsigned i = 0; i < other._dim; i++)
		_sizes[i] = other._sizes[i];

	return *this;
}

void Shape::remove(int dim)
{
	if (!_dim)
		return;

	if (_dim == 1)
	{
		*this = Shape();
		return;
	}

	// Modulo dim.
	dim = unsigned(dim + _dim * (2 - dim / int(_dim))) % _dim;

	// Write new sizes.
	int* new_sizes = new int[_dim - 1];

	for (int i = 0; i < dim; i++)
		new_sizes[i] = _sizes[i];
	for (unsigned i = dim + 1; i < _dim; i++)
		new_sizes[i - 1] = _sizes[i];

	// Adopt new sizes.
	_dim--;
	delete[] _sizes;
	_sizes = new_sizes;
}

void Shape::add(int dim, int size)
{
	if (!_dim)
	{
		_dim = 1;
		_sizes = new int[1];
		_sizes[0] = size;
		return;
	}

	// Modulo dim.
	dim = unsigned(dim + (_dim + 1) * (2 - dim / int(_dim + 1))) % (_dim + 1);

	// Write new sizes.
	int* new_sizes = new int[_dim + 1];

	for (int i = 0; i < dim; i++)
		new_sizes[i] = _sizes[i];
	new_sizes[dim] = size;
	for (unsigned i = dim; i < _dim; i++)
		new_sizes[i + 1] = _sizes[i];

	// Adopt new sizes.
	_dim++;
	delete[] _sizes;
	_sizes = new_sizes;
}

const char* Shape::str(const char* fmt) const
{
	thread_local static char buffer[16][256] = {};
	thread_local static int next = 0;

	char* buf = buffer[(next++) % 16];
	int left = 64;

	*(buf++) = '('; left--;

	for (unsigned i = 0; i < _dim && left > 0; i++)
	{
		int added = snprintf(buf, left, fmt, _sizes[i]);
		buf += added, left -= added;

		if (i < _dim - 1 && left > 2)
		{
			*(buf++) = ','; left--;
			*(buf++) = ' '; left--;
		}
	}
	if (left > 1)
	{
		*(buf++) = ')'; left--;
		*(buf++) = '\0'; left--;
	}

	return buffer[(next - 1) % 16];
}

/*
--------------------------------------------------------------------------------------------------------------------------
 Constructor / Destructor
--------------------------------------------------------------------------------------------------------------------------
*/

Array& Array::operator=(const Array& other)
{
	if (dim())
		::operator delete[](_data, std::align_val_t(64));

	_shape = other.shape();
	if (!dim())
	{
		_data = nullptr;
		_total_size = 0u;
		return *this;
	}

	_offset = other._offset;
	_total_size = other._total_size;
	_data = (float*)::operator new[](_total_size * sizeof(float), std::align_val_t(64));
	memcpy(_data, other._data, _total_size * sizeof(float));

	return *this;
}

void Array::create(const Shape& shape)
{
	TENSOR_CHECK(shape.dim(),
		"If the array created will have zero dimensions please use the default constructor."
	);
	for (unsigned i = 0; i < dim(); i++)
		TENSOR_CHECK(shape[i] >= 0,
			"A shape with negative values is not allowed for initialization."
		);

	if (dim())
		::operator delete[](_data, std::align_val_t(64));

	_shape = shape;
	_offset = shape;

	// Set offsets with 64 bit alignment.
	_offset[dim() - 1] = 1u;
	for (int i = dim() - 2; i >= 0; i--)
	{
		_offset[i] = shape[i + 1] * _offset[i + 1];
		if (_offset[i] % 16 != 0 && shape[i + 1] != 1u)
			_offset[i] += 16 - _offset[i] % 16;
	}

	// Get the total size needed given the alignment.
	_total_size = _offset[0] * shape[0];
	if (_total_size % 16 != 0)
		_total_size += 16 - _total_size % 16;

	// Set offsets to 0 for unitary dimensions.
	for (unsigned i = 0; i < dim(); i++)
		if (_shape[i] == 1) _offset[i] = 0;

	// Allocate the data.
	_data = (float*)::operator new[](_total_size * sizeof(float), std::align_val_t(64));

	// Be clean and zero it out.
	memset(_data, 0, _total_size * sizeof(float));
}

Array::~Array()
{
	if (dim())
		::operator delete[](_data, std::align_val_t(64));
}

/*
--------------------------------------------------------------------------------------------------------------------------
 User Functions
--------------------------------------------------------------------------------------------------------------------------
*/

const char* Array::str(const char* fmt) const
{
	constexpr unsigned truncation_size = 7;
	constexpr unsigned truncation_count = 3;

	thread_local static char buffer[8][4096] = {};
	thread_local static int next = 0;

	char* buf = buffer[(next++) % 8];
	int left = 4096;

	auto s_print = [&](const char* fmt, ...)
		{
			va_list ap;
			va_start(ap, fmt);
			int added = vsnprintf(buf, left, fmt, ap);
			va_end(ap);
			buf += added, left -= added;
		};

	Shape counting_shape(_shape.dim(), (int*)nullptr);

	while (left > 0)
	{
		for (unsigned d = 0; d < dim(); d++)
		{
			bool opening = true;
			for (unsigned i = d; i < dim(); i++)
				if (counting_shape[i]) opening = false;

			if (opening)
			{
				if (_shape[d] <= 1 || d == 0)
					s_print("(");
				else
					s_print("\n(");
			}
		}

		unsigned idx = 0;
		for (unsigned d = 0; d < dim(); d++)
			idx += counting_shape[d] * _offset[d];

		s_print(fmt, _data[idx]);

		if (++counting_shape[-1] < _shape[-1])
		{
			s_print(", ");
			if (_shape[-1] > truncation_size && counting_shape[-1] == truncation_count)
			{
				counting_shape[-1] = _shape[-1] - truncation_count;
				s_print("..., ");
			}
		}


		for (int d = dim() - 1; d > 0; d--)
		{
			if (counting_shape[d] >= _shape[d])
			{
				counting_shape[d] -= _shape[d];

				if (++counting_shape[d - 1] < _shape[d - 1])
				{
					s_print("), ");
					if (_shape[d - 1] > truncation_size && counting_shape[d - 1] == truncation_count)
					{
						counting_shape[d - 1] = _shape[d - 1] - truncation_count;

						if (_shape[d] <= 1)
							s_print("..., ");
						else
							s_print("\n ... ");
					}
				}
				else
				{
					if (_shape[d] <= 1)
						s_print(")");
					else
						s_print(")\n");
				}

			}
		}

		if (counting_shape[0] >= _shape[0])
		{
			s_print(")");
			break;
		}
	}

	return buffer[(next - 1) % 8];
}

void Array::set_value(const Shape& idxs, float value)
{
	TENSOR_CHECK(_data,
		"Trying to set a value on an empty array."
	);
	TENSOR_CHECK(_total_size,
		"Trying to set a value on an array with no values.\n"
		"The array shape is %s", _shape.str()
	);

	float* ptr = _data;

	for (unsigned d = 0; d < dim(); d++)
	{
		unsigned idx = (idxs[d] + _shape[d] * (2 - idxs[d] / _shape[d])) % _shape[d];
		ptr += idx * _offset[d];
	}


	*ptr = value;
}

float Array::get_value(const Shape& idxs) const
{
	TENSOR_CHECK(_data,
		"Trying to get a value on an empty array."
	);
	TENSOR_CHECK(_total_size,
		"Trying to get a value on an array with no values.\n"
		"The array shape is %s", _shape.str()
	);

	float* ptr = _data;

	for (unsigned d = 0; d < dim(); d++)
	{
		unsigned idx = (idxs[d] + _shape[d] * (2 - idxs[d] / _shape[d])) % _shape[d];
		ptr += idx * _offset[d];
	}

	return *ptr;
}

void Array::set_vector(unsigned* route, int idx0, int idx1, float* values)
{
}

const float* Array::get_vector(unsigned* route, int idx0, int idx1) const
{
	return nullptr;
}

float* Array::get_vector(unsigned* route, int idx0, int idx1)
{
	return nullptr;
}
