#pragma once
#include "macrograd.h"

class Tokenizer
{
public:
	static constexpr unsigned num_tokens = 0x40;
	static constexpr int INVALID = 0xFF;
private:
	static inline int  tokens[256] = {};
	static inline char reversed[256] = {};

	static inline void init_tokenizer()
	{
		// If already initialized return.
		if (tokens['\\'] == INVALID)
			return;

		// Make all tokens invalid for bookkeeping.
		for (int i = 0; i < 256; i++)
			tokens[i] = INVALID;

		// Special end of line token.
		tokens['\n'] = 0x00;

		// Space.
		tokens[' '] = 0x01;

		// Other non-letter characters.
		tokens['!'] = 0x02; tokens['\''] = 0x03; tokens[','] = 0x04; tokens['-'] = 0x05;
		tokens['.'] = 0x06; tokens[':'] = 0x07; tokens[';'] = 0x08; tokens['?'] = 0x09;
		tokens['3'] = 0x0A; tokens['&'] = 0x0B;

		// Lowecase letters.
		tokens['a'] = 0x0C; tokens['b'] = 0x0D; tokens['c'] = 0x0E; tokens['d'] = 0x0F;
		tokens['e'] = 0x10; tokens['f'] = 0x11; tokens['g'] = 0x12; tokens['h'] = 0x13;
		tokens['i'] = 0x14; tokens['j'] = 0x15; tokens['k'] = 0x16; tokens['l'] = 0x17;
		tokens['m'] = 0x18; tokens['n'] = 0x19; tokens['o'] = 0x1A; tokens['p'] = 0x1B;
		tokens['q'] = 0x1C; tokens['r'] = 0x1D; tokens['s'] = 0x1E; tokens['t'] = 0x1F;
		tokens['u'] = 0x20; tokens['v'] = 0x21; tokens['w'] = 0x22; tokens['x'] = 0x23;
		tokens['y'] = 0x24; tokens['z'] = 0x25;

		// Uppercase letters.
		tokens['A'] = 0x26; tokens['B'] = 0x27; tokens['C'] = 0x28; tokens['D'] = 0x29;
		tokens['E'] = 0x2A; tokens['F'] = 0x2B; tokens['G'] = 0x2C; tokens['H'] = 0x2D;
		tokens['I'] = 0x2E; tokens['J'] = 0x2F; tokens['K'] = 0x30; tokens['L'] = 0x31;
		tokens['M'] = 0x32; tokens['N'] = 0x33; tokens['O'] = 0x34; tokens['P'] = 0x35;
		tokens['Q'] = 0x36; tokens['R'] = 0x37; tokens['S'] = 0x38; tokens['T'] = 0x39;
		tokens['U'] = 0x3A; tokens['V'] = 0x3B; tokens['W'] = 0x3C; tokens['X'] = 0x3D;
		tokens['Y'] = 0x3E; tokens['Z'] = 0x3F;

		// Create the reversed tokenizer.
		for (int i = 0; i < 256; i++)
			if (tokens[i] != INVALID)
				reversed[tokens[i]] = (char)i;

		// Explicitly set the reversed invalid to zero.
		reversed[INVALID] = '\0';
	}
public:
	static inline int tokenize(char c)
	{
		init_tokenizer();
		return tokens[(unsigned char)c];
	}

	static inline char revert(int i)
	{
		init_tokenizer();
		if (i < 0 || i > 255)
			return '\0';
		return reversed[i];
	}

	static inline VectorInt encode(const char* text)
	{
		unsigned length = 0;
		while (text[length]) length++;

		if (!length)
		{
			VectorInt single_token(1);
			single_token[0] = tokenize('\n');
			return single_token;
		}

		VectorInt tokens(length);

		for (unsigned i = 0; i < length; i++)
			tokens[i] = tokenize(text[i]);

		return tokens;
	}

	static inline void decode(const VectorInt& tokens, char* dst)
	{
		VectorInt tok = tokens;
		if (tok.is_gpu())
			tok = tok.to("cpu");

		for (unsigned i = 0; i < tokens.len(); i++)
			dst[i] = revert(tok[i]);

		dst[tokens.len()] = '\0';
	}
};

