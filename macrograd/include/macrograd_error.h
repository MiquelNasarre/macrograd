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
 As part of a tensor library, it is important to provide clear error messages to easily pinpoint why a
 program aborted. This is why this header contains a simple but powerful error class and a set of macros, 
 which centralize error handling across Macrograd.

 All functions along the library perform various validation checks like shape compatibility, device
 compatibility, etc. All these checks trigger an error and abort if they fail. This prints a formatted
 string to the console explaining the reason for the error, while also reporting the file and line where 
 the error was detected.

 The checks are always active regardless of the configuration. This is because diagnostics are always
 necessary in tensor libraries, and they introduce almost insignificant overhead compared to the functions
 that they guard.

 The macros used for all checks and errors can be found at the end of this file. Feel free to use them in
 your custom modules to funnel your own error detection through the same pipeline.
--------------------------------------------------------------------------------------------------------------------------
--------------------------------------------------------------------------------------------------------------------------
*/

// Include dependencies.
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/*
--------------------------------------------------------------------------------------------------------------------------
 Error Class
--------------------------------------------------------------------------------------------------------------------------
*/

// Library error class. To be used by library functions to raise errors on the console
// and shut down the program. Stores the line and file where the error occurred and a 
// message string. To be created using the error macros.
class MacrogradError
{
public:
	// Stores the error data and formats the string.
	MacrogradError(unsigned line, const char* file, const char* fmt, ...)
		: _line{ line }, _file{ file }
	{
		va_list ap;
		va_start(ap, fmt);
		vsnprintf(_msg, sizeof(_msg), fmt, ap);
		va_end(ap);
	}

	// Prints the error information to the console and calls abort.
	[[noreturn]] void PrintAbort() const
	{
		// Find the last dir slash inside file path.
		const char* slash = strrchr(_file, '/');
		const char* backslash = strrchr(_file, '\\');
		const char* last = (slash > backslash) ? slash : backslash;

		// Print error to console.
		fprintf(stderr,
			"Macrograd Error Occurred:\n"
			"Line: %u\n"
			"File: %s\n"
			"Message: %s\n",
			_line, last ? last + 1 : _file, _msg
		);

		// Terminate process.
		abort();
	}

private:
	unsigned _line = 0u;	// Stores the line where the error occurred.
	const char* _file;		// Stores the file where the error occurred.
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
