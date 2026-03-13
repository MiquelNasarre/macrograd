#pragma once

/* LICENSE AND COPYRIGHT
--------------------------------------------------------------------------------------------------------------------------
 * Macrograd - a CUDA/C++ Autograd Tensor Library
 * Copyright (c) 2026 Miguel Nasarre Budiño
 * Licensed under the MIT License. See LICENSE file.
--------------------------------------------------------------------------------------------------------------------------
*/

/* MINI SHAKESPEARE MODEL HEADER
--------------------------------------------------------------------------------------------------------------------------
--------------------------------------------------------------------------------------------------------------------------
 What better demo for a library named Macrograd than following the steps of Nano-GPT and training it 
 on the tiny Shakespeare dataset. Therefore this is what this header is about. This model I call my 
 Tiny Shakespeare, and it represents my best effort with my 2GB VRAM GPU to write some Shakespeare.

 The class itself consists of a transformer stack, a token embedding, and a positional embedding. 
 The dimensions of the trained model are the ones specified in the global parameters.

 The modules used are defined inside the 'transformer.h' header file, and it uses the 'tokenizer.h' 
 dependencies for tokenization.

 This model, coupled with the training header, represents the official demo of Macrograd, having trained
 this small model to achieve a validation loss of (...working on it...), which I am quite proud of.

 If you want to play with the model yourself, use the native function Module::load_weights() with the
 'my_shakespeare.mg' weights file provided. Then call the class function add_one_character() with a
 string of your own. The main file has a basic writing function defined. I hope you have fun with it :)
--------------------------------------------------------------------------------------------------------------------------
--------------------------------------------------------------------------------------------------------------------------
*/

// Include library headers.
#include "macrograd_nn.h"
#include "transformer.h"
#include "tokenizer.h"

// My Mini Shakespeare is a small Generative Pre-trained Transformer that has been trained on the 
// tiny_shakespeare dataset. If you let him write, it will produce endless theater plays with no 
// meaning, which is quite distracting to watch.
//
// You can use the native function Module::load_weights() to load pre-trained models from the weights
// file, and use the functions add_one_character() and set_temperature() to allow him to write.
//
// For details on the model training check the 'training.h' header file.
class MiniShakespeare : public Module
{
private:
	// Storage for the class submodules.
	Transformer transformer;
	PositionalEmbedding pos;
	Embedding embedding;

	// Global constant model dimensions.
	static constexpr unsigned n_layers = 6;
	static constexpr unsigned n_heads  = 4;
	static constexpr unsigned emb_dim  = 128;
	static constexpr unsigned ff_dim   = 512;
	static constexpr unsigned context  = 256;

	// Variable temperature for generation.
	float temperature = 1.0f;
public:

	// My Mini Shakespeare constructor. Creates a new Mini Shakespeare model.
	// This one cannot write yet; it will need some training or a load file.
	MiniShakespeare(float training_dropout = 0.0f) :
		transformer{ n_layers, n_heads, emb_dim, ff_dim, training_dropout },
		embedding{ Tokenizer::num_tokens, emb_dim },
		pos{ context, emb_dim }
	{
		// Add modules to the parameter list.
		add_module(transformer);
		add_module(embedding);
		add_module(pos);
	}

	// Sets the model temperature for token generation.
	void set_temperature(float temp) { temperature = temp; }

	// Model's forward pass, used for training the model. Takes a list of token
	// vectors, runs them through its internal modules, and outputs the logits.
	Tensor internal_forward(const VectorInt* tokens, unsigned batch_size)
	{
		// Get tokens into its embeddings.
		Tensor token_emb = embedding.embed(tokens, batch_size);

		// Add positional embedding.
		Tensor trans_in = pos(token_emb);

		// Generate causal mask and set it.
		Tensor mask = Functional::causal_mask(trans_in.size(1), device());
		transformer.set_attention_mask(mask);

		// Run through the transformer.
		Tensor trans_out = transformer(trans_in);

		// Unembed to get logits.
		Tensor logits = embedding.unembed(trans_out);

		// Return logits.
		return logits;
	}

	// Given a token vector as input, it runs the forward pass on the tokens, copies the 
	// list, and appends a sampled token from its probability distribution to the list.
	VectorInt add_one_token(const VectorInt& tokens)
	{
		bool had_grad = has_grad();

		// Disable gradients.
		no_grad();

		// Make sure it does not exceed context length.
		VectorInt context_tokens = (tokens.len() > context) ? tokens.subset(-(int)context, 0) : tokens;

		// Get tokens into its embeddings.
		Tensor token_emb = embedding.embed(&context_tokens, 1);

		// Add positional embedding.
		Tensor trans_in = pos(token_emb);

		// Generate causal mask and set it.
		Tensor mask = Functional::causal_mask(trans_in.size(1), device());
		transformer.set_attention_mask(mask);

		// Run single batch through the transformer.
		Tensor trans_out = transformer(trans_in);

		// Retrieve last token from the single batch.
		Tensor last_token = trans_out.squeeze(0)[-1];

		// Get logits by unembedding.
		Tensor logits = embedding.unembed(last_token);

		// Apply temperature.
		logits /= temperature;

		// Softmax to get probabilities.
		Tensor probs = logits.softmax(-1);

		// Sample random token.
		VectorInt new_token = Random::sample(probs);

		// Create new vector.
		VectorInt out(tokens.len() + 1, tokens.device());

		// Assign token values.
		out.set(0, (int)tokens.len(), tokens.data(), tokens.is_gpu());
		out.set(-1, 0, new_token.data(), new_token.is_gpu());

		// Reset gradients if they were previously enabled.
		if (had_grad) with_grad();

		// Return output.
		return out;
	}

	// Enter your string buffer here and it will generate the next character and add it 
	// to the string. Do it enough times and you'll have your own Shakespeare novel :)
	void add_one_character(char* str)
	{
		// Encode string via the tokenizer.
		VectorInt tokens = Tokenizer::encode(str).to(device());

		// Run the add token forward pass.
		VectorInt new_tokens = add_one_token(tokens);

		// Decode into string.
		Tokenizer::decode(new_tokens, str);
	}
};
