#pragma once
#include "macrograd_nn.h"
#include "transformer.h"
#include "tokenizer.h"

class MiniShakespeare : public Module
{
private:
	Transformer transformer;
	PositionalEmbedding pos;
	Embedding embedding;

	static constexpr unsigned n_layers = 6;
	static constexpr unsigned n_heads = 2;
	static constexpr unsigned emb_dim = 64;
	static constexpr unsigned ff_dim = 256;
	static constexpr unsigned context = 256;

	float temperature = 1.0f;
public:
	MiniShakespeare() :
		transformer{ n_layers, n_heads, emb_dim, ff_dim },
		embedding{ Tokenizer::num_tokens, emb_dim },
		pos{ context, emb_dim }
	{
		add_module(transformer);
		add_module(embedding);
		add_module(pos);
	}

	void set_temperature(float temp) { temperature = temp; }

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

		// Assing token values.
		out.set(0, (int)tokens.len(), tokens.data(), tokens.is_gpu());
		out.set(-1, 0, new_token.data(), new_token.is_gpu());

		// Reset gradient if it had.
		if (had_grad) with_grad();

		// Return output-
		return out;
	}

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