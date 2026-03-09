#pragma once

/* LICENSE AND COPYRIGHT
--------------------------------------------------------------------------------------------------------------------------
 * Macrograd - a CUDA/C++ Autograd Tensor Library
 * Copyright (c) 2026 Miguel Nasarre Budiño
 * Licensed under the MIT License. See LICENSE file.
--------------------------------------------------------------------------------------------------------------------------
*/

/* MACROGRAD ERROR CLASS HEADER
--------------------------------------------------------------------------------------------------------------------------
--------------------------------------------------------------------------------------------------------------------------





--------------------------------------------------------------------------------------------------------------------------
--------------------------------------------------------------------------------------------------------------------------
*/

// Include dependencies.
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>

/*
--------------------------------------------------------------------------------------------------------------------------
 Error Class
--------------------------------------------------------------------------------------------------------------------------
*/

// Library error class. To be used by library functions to arise errors on the console
// and shut down the program. Stores the line and file where the error occurred and a 
// message string, to be created using the macros.
class MacrogradError
{
public:
	// Stores the error data and formats the string.
	MacrogradError(unsigned line, const char* file, const char* fmt, ...)
		: _line{ line }
	{
		snprintf(_file, sizeof(_file), "%s", file);

		va_list ap;
		va_start(ap, fmt);
		vsnprintf(_msg, sizeof(_msg), fmt, ap);
		va_end(ap);
	}

	// Prints the error information on console and calls abort.
	[[noreturn]] void PrintAbort() const
	{
		fprintf(stderr,
			"Macrograd Error Occurred:\n"
			"Line: %u\n"
			"File: %s\n"
			"Message: %s\n",
			_line, _file, _msg
		);
		abort();
	}

private:
	unsigned _line = 0u;	// Stores the line where the error ocurred.
	char _file[512] = {};	// Stores the file where the error ocurred.
	char _msg[512] = {};	// Stores the message string of the error.
};

/*
--------------------------------------------------------------------------------------------------------------------------
 Error Macros
--------------------------------------------------------------------------------------------------------------------------
*/

// Creates and prints an error.
#define MACROGRAD_ERROR(...) MacrogradError(__LINE__,__FILE__,__VA_ARGS__).PrintAbort()

// Checks expression, if false raises an error.
#define MACROGRAD_CHECK(expr, ...) do { if (!(expr)) { MACROGRAD_ERROR(__VA_ARGS__); } } while(0)
