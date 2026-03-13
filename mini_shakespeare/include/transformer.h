#pragma once

/* LICENSE AND COPYRIGHT
--------------------------------------------------------------------------------------------------------------------------
 * Macrograd - a CUDA/C++ Autograd Tensor Library
 * Copyright (c) 2026 Miguel Nasarre Budiño
 * Licensed under the MIT License. See LICENSE file.
--------------------------------------------------------------------------------------------------------------------------
*/

/* MACROGRAD TRANSFORMER MODULE HEADER
--------------------------------------------------------------------------------------------------------------------------
--------------------------------------------------------------------------------------------------------------------------
 One of the main goals of Macrograd was to be able to create a trainable transformer with it
 that was reasonably fast enough to be trained using my 2GB VRAM GPU. This file represents
 the realization of that objective.

 All modules needed to create the MiniShakespeare model are defined inside this header, going
 from a linear layer implementation all the way to the full transformer architecture, which
 uses multi-head self-attention. It also contains the embeddings used by the model.

 For examples on how to implement your own modules I suggest checking the ones found in this
 header. All of them have comments explaining the implementation in detail. Hope you like it :)
--------------------------------------------------------------------------------------------------------------------------
--------------------------------------------------------------------------------------------------------------------------
*/

// Include main library headers.
#include "macrograd_nn.h"
#include "macrograd_error.h"

// Math dependencies.
#include <math.h>

// Linear module class. Consists of a single linear layer with or without bias. Defines 
// the projection matrix and optionally the bias vector. During the forward pass it calls 
// the function matmul, fused with the bias accordingly.
class Linear : public Module
{
private:
    // Class parameter storage.
	Tensor matrix;
	Tensor bias;

    // Bias internal flag.
	bool _has_bias;

public:
    // Linear module constructor. Initializes the matrix using Xavier uniform, and 
    // optionally creates the bias vector. Adds the parameters to the internal list.
	Linear(unsigned fan_in, unsigned fan_out, bool has_bias = true) : _has_bias{ has_bias }
	{
        // Create matrix of shape (fan_in, fan_out).
		matrix = Tensor(Shape{ fan_in, fan_out });
        // Initialize with Xavier uniform.
        Initialization::uniform(matrix, -sqrtf(6.f / (fan_in + fan_out)), sqrtf(6.f / (fan_in + fan_out)));
        // Add matrix to the parameter list.
		add_parameter(matrix);

        // If has bias zero initialize it and add it to the list.
		if (has_bias)
		{
			bias = Tensor(Shape{ fan_out, });
			add_parameter(bias);
		}
	}

    // Linear forward pass, applies a matrix multiplication, with fused bias 
    // depending on class construction. Returns the resulting tensor.
	Tensor forward(const Tensor& in) const override
	{
        // If has bias return matmul with fused bias.
		if (_has_bias)
			return Functional::matmul(in, matrix, bias); // (..., fan_in) -> (..., fan_out)
        // Else return simple matrix multiplication.
		return Functional::matmul(in, matrix); // (..., fan_in) -> (..., fan_out)
	}
};

// Dropout module class. Consists of a module that applies dropout depending on the its
// current mode. Does not contain any internal parameters. It applies it by generating
// a random mask for each element of the tensor every call.
class Dropout : public Module
{
private:
    // Internal storage for the dropout rate.
    float _rate;

public:
    // Dropout module constructor. Initializes the dropout rate to the specified value.
    Dropout(float rate) { set_rate(rate); }

    // Updates the internal dropout rate.
    void set_rate(float rate) 
    { 
        // Sanity check.
        MACROGRAD_CHECK(rate >= 0.f && rate < 1.f,
            "Invalid dropout rate received inside a Dropout module.\n"
            "Expected a value between 0 and 1 but got: %.4f", rate
        );
        // Set rate.
        _rate = rate; 
    }

    // Dropout module forward pass. If th module is on training mode it creates a random
    // mask with the specified droppout proportion and applies it to the input tensor.
    Tensor forward(const Tensor& in) const override
    {
        // If on evaluation mode return input.
        if (is_eval() || _rate == 0.f)
            return in;

        // Prepare the dropout mask with same shape.
        Tensor mask(in.shape(), in.device(), false);

        // Initialize it with random uniform distribution from 0 to 1.
        Initialization::uniform(mask);

        // Apply the mask to the input tensor and return.
        return in * ((mask >= _rate) / (1.0f - _rate));
    }
};

// Feed Forward module class. Consists of two linear layers with a GELU operation as the 
// non-linearity. Used after MHSA inside the transformer layer. Applies a simple MLP 
// transformation to each token individually.
class FeedForward : public Module
{
private:
    // Class submodule storage.
	Linear linear0; 
	Linear linear1;

public:
    // Feed Forward forward pass. Given an input tensor, it applies the first 
    // linear layer, then GELU, then the second linear layer and returns.
	FeedForward(unsigned emb_dim, unsigned hid_dim, bool has_bias = true):
		linear0{ emb_dim, hid_dim, has_bias }, // (..., emb) -> (..., hid)
		linear1{ hid_dim, emb_dim, has_bias }  // (..., hid) -> (..., emb)
    { add_module(linear0); add_module(linear1); /* Add both layer to the list */ }

    // Feed Forward forward pass, given an input tensor it passes it applies the 
    // first linear layer, then GELU then the second linear layer and returns.
	Tensor forward(const Tensor& in) const override
	{
		Tensor out0 = linear0(in).gelu(); // (..., emb) -> (..., hid)
        Tensor out1 = linear1(out0);      // (..., hid) -> (..., emb)
		return out1; // (..., emb)
	}
};

// Multi-Head Self-Attention module class. Applies the basic layout of MHSA with an additive 
// attention mask. Since this is the main compute block of the class, it is optimized to be 
// decently fast. Despite that, it is not at the efficiency level of other libraries. 
// Proper debugging tools and optimizations could further increase its performance.
class MultiHeadSelfAttention : public Module
{
private:
    // Class submodule storage.
    Linear q_linear;
    Linear k_linear;
    Linear v_linear;
    Linear o_linear;

    // Attention mask storage.
    Tensor mask;

    // Class dimension storage for reference.
    unsigned n_heads;
    unsigned emb_dim;
    unsigned head_dim;

public:
    // Multi-Head Self-Attention module constructor. Initializes the linear layers and adds
    // them to the parameter list. Checks and stores the head dimension and the other sizes.
    MultiHeadSelfAttention(unsigned _n_heads, unsigned _emb_dim) :
        n_heads{ _n_heads },
        emb_dim{ _emb_dim },
        q_linear{ _emb_dim, _emb_dim },
        k_linear{ _emb_dim, _emb_dim },
        v_linear{ _emb_dim, _emb_dim },
        o_linear{ _emb_dim, _emb_dim }
    {
        // Sanity check.
        MACROGRAD_CHECK(_emb_dim % _n_heads == 0 && _emb_dim,
            "Invalid parameters found for MHSA initialization.\n"
            "Embeding dimension must be divisible by the number of heads."
        );
        // Add all modules to the parameter list.
        add_module(q_linear);
        add_module(k_linear);
        add_module(v_linear);
        add_module(o_linear);
        // Store head dimension.
        head_dim = emb_dim / n_heads;
    }

    // Stores attention mask to be used during the next forward passes.
    // To disable the attention mask you can send in an empty tensor.
    void set_attetion_mask(const Tensor& _mask) { mask = _mask; }
	
    // Multi-Head Self-Attention forward pass. First computes the Q, K, V matrices, separates
    // them by heads and transposes them. Then computes the attention matrices, applies the 
    // mask and softmax. Finally obtains the attention weights, reshapes back and applies the 
    // output transformation.
    Tensor forward(const Tensor& attn_in) const override
    {
        // Sanity checks.
        MACROGRAD_CHECK(attn_in.dim() == 3 && attn_in.size(2) == emb_dim,
            "Invalid tensor shape received insize a MHSA forward pass. \n"
            "The module expects a three dimensional tensor of shape (B, L, emb_dim).\n"
            "But got: %s", attn_in.shape().str()
        );

        // First store the dimensions.
        int B = attn_in.size(0); // batch size
        int L = attn_in.size(1); // num tokens

        // Obtain keys, queries and values.
        Tensor Q = q_linear(attn_in); // (B, L, E)
        Tensor K = k_linear(attn_in); // (B, L, E)
        Tensor V = v_linear(attn_in); // (B, L, E)

        // Separate by heads.
        Q = Q.view({ B, L, n_heads, -1 }).transpose(1, 2); // (B, h, L, dH)
        K = K.view({ B, L, n_heads, -1 }).transpose(1, 2); // (B, h, L, dH)
        V = V.view({ B, L, n_heads, -1 }).transpose(1, 2); // (B, h, L, dH)

        // Generate attention scores. Fuse with mask if exists.
        Tensor scores;
        if (mask.is_init())
            scores = Functional::matmul(Q / sqrtf((float)head_dim), K, mask, false, true); // (B, h, L, L)
        else
            scores = Functional::matmul(Q / sqrtf((float)head_dim), K, false, true); // (B, h, L, L)

        // Get attention weights
        Tensor weights = scores.softmax(-1);

        // Apply self-attention from values.
        Tensor O = Functional::matmul(weights, V); // (B, h, L, dH)

        // Reshape back.
        O = O.transpose(1, 2).view({ B, L, emb_dim }); // (B, L, E)

        // Apply last linear transformation.
        Tensor attn_out = o_linear(O); // (B, L, E)

        // Return output tensor.
        return attn_out; // (B, L, E)
    }
};

// Layer Normalization module class. Normalizes the last layer and applies gamma/beta scaling
// from learned parameters. Used by the transformer layer before or after MHSA and FF.
class LayerNorm : public Module
{
private:
    // Class parameter storage.
    Tensor gamma;
    Tensor beta;

    // Class constants storage.
    unsigned layer_dim;
    float eps;
public:

    // Layer Normalization module constructor. Stores the provided layer dimension. 
    // Initializes the gamma and beta tensors to 1 and 0 respectively, and adds them 
    // to the parameter list.
    LayerNorm(unsigned _layer_dim, float _eps = 1e-5f)
    {
        // Store constants.
        layer_dim = _layer_dim;
        eps = _eps;

        // Create vectors.
        gamma = Tensor(Shape{ layer_dim });
        beta = Tensor(Shape{ layer_dim });

        // Initialize vectors.
        float beta_init = 0.f;
        float gamma_init = 1.f;
        beta.internal_fill(&beta_init);
        gamma.internal_fill(&gamma_init);

        // Add vectors to parameter list.
        add_parameter(gamma);
        add_parameter(beta);
    }

    // Layer Normalization forward pass. Expects a tensor with the last dimension size set at 
    // construction. Normalizes along the last layer and applies the learned transformation.
    Tensor forward(const Tensor& in) const override
    {
        // Sanity check.
        MACROGRAD_CHECK(in.size(-1) == layer_dim,
            "Invalid tensor shape received inside a LayerNorm forward pass.\n"
            "Expected last dimension %i but got %s",
            layer_dim, in.shape().str()
        );

        // Get normalization tensors.
        Tensor mean = in.mean(-1, true);   // (..., 1)
        Tensor var = in.var(-1, true);     // (..., 1)

        // Normalize input.
        Tensor norm_in = (in - mean) / (var + eps).sqrt(); // (..., E)

        // Apply gamma and beta.
        Tensor out = norm_in * gamma + beta; // (..., E)

        // Return output.
        return out;
    }
};

// Transformer Layer module class. During the forward pass applies MHSA followed by layer 
// normalization, then FF and normalization. Used as a single layer of a transformer.
class Layer : public Module
{
private:
    // Class submodule storage.
    MultiHeadSelfAttention attention;
    FeedForward feed_forward;
    LayerNorm attn_norm;
    LayerNorm ff_norm;
    Dropout dropout;

public:
    // Transformer Layer module constructor. Takes as input the dimensions and initializes
    // its modules accordingly. Adds all modules to the internal parameter list.
    Layer(unsigned n_heads, unsigned emb_dim, unsigned ff_dim, float dropout_rate) :
        attention{ n_heads, emb_dim },
        feed_forward{ emb_dim, ff_dim },
        attn_norm{ emb_dim },
        ff_norm{ emb_dim },
        dropout{ dropout_rate }
    {
        // Add all modules to the list.
        add_module(attention);
        add_module(attn_norm);
        add_module(feed_forward);
        add_module(ff_norm);
        add_module(dropout);
    }

    // Sets mask for its attention block. Used during upcoming forward 
    // passes. To disable the attention mask you can send in an empty tensor.
    void set_attetion_mask(const Tensor& mask) { attention.set_attetion_mask(mask); }

    // Transformer Layer forward pass. Expects an input of shape (B, L, E), applies MHSA, 
    // followed by residual addition and layer-norm, then applies FF, followed by residual 
    // addition and layer-norm. Returns an output tensor with the same shape as the input.
    Tensor forward(const Tensor& layer_in) const override
    {
        // First do attention.
        Tensor attn_out = attention(layer_in); // (B, L, E)

        // Add residual and layer normalize.
        Tensor ff_in = attn_norm(dropout(attn_out) + layer_in); // (B, L, E)

        // feed forward.
        Tensor ff_out = feed_forward(ff_in); // (B, L, E)

        // Add residual and layer normalize.
        Tensor out = ff_norm(dropout(ff_out) + ff_in); // (B, L, E)

        // Return output.
        return out; // (B, L, E)
    }
};

// Transformer Stack module class. This class combines all the components defined above.
// Consists of a stack of layers, and iterates through them during the forward pass.
class Transformer : public Module
{
private:
    // Class submodule storage.
    Layer** layer;

    // Layer count.
    unsigned n_layers;

public:
    // Transformer Stack module constructor. Takes the transformer dimensions as input 
    // and initializes the layers accordingly. Adds all layers to the parameter list.
    Transformer(unsigned num_layers, unsigned n_heads, unsigned emb_dim, unsigned ff_dim, float dropout_rate)
    {
        // Allocate space for the layers.
        n_layers = num_layers;
        layer = new Layer*[num_layers];

        // Initialize layers and add to parameter list.
        for (unsigned i = 0; i < n_layers; i++)
        {
            layer[i] = new Layer(n_heads, emb_dim, ff_dim, dropout_rate);
            add_module(*layer[i]);
        }
    }

    // Transformer Stack module destructor. Deletes all the layers.
    ~Transformer()
    {
        for (unsigned i = 0; i < n_layers; i++)
            delete layer[i];

        delete[] layer;
    }

    // Sets the attention mask for all the layers.
    // Set to empty tensor to disable masking.
    void set_attention_mask(const Tensor& mask)
    {
        for (unsigned i = 0; i < n_layers; i++)
            layer[i]->set_attetion_mask(mask);
    }

    // Transformer Stack forward pass. Expects an input of shape (B, L, E).
    // Runs the input sequentially through all the layers and returns output.
    Tensor forward(const Tensor& trans_in) const override
    {
        // Store input in a tensor.
        Tensor ten = trans_in;

        // Iterate through all the layers.
        for (unsigned i = 0; i < n_layers; i++)
            ten = (*layer[i])(ten);

        // Return final tensor.
        return ten;
    }
};

// Token Embedding module class. The embed function takes as input a list of token vectors 
// and returns an embedded batched tensor of shape (B, L, E). The unembed function takes as 
// input the transformer output and returns logits using the same matrix.
class Embedding : public Module
{
private:
    // Class parameter storage.
    Tensor lookup_table;

    // Module dimensions for reference.
    unsigned emb_dim;
    unsigned num_tokens;

public:
    // Token Embedding module constructor. Stores the dimensions, initializes 
    // the lookup table matrix, and adds it to the internal parameter list.
    Embedding(unsigned _num_tokens, unsigned _emb_dim) :
        num_tokens{ _num_tokens },
        emb_dim{ _emb_dim }
    {
        // Initialize lookup table.
        lookup_table = Tensor({ num_tokens, emb_dim });
        // Standard initialization N(0, 0.02).
        Initialization::normal(lookup_table, 0.0f, 0.02f);
        // Add it to parameter list.
        add_parameter(lookup_table);
    }

    // Embed function. Takes as input a list of token vectors of equal length. Using 
    // the lookup table it generates the embedded batch tensor of shape (B, L, E).
    Tensor embed(const VectorInt* tokens, unsigned batch_size) const
    {
        // Sanity check.
        MACROGRAD_CHECK(batch_size != 0,
            "Batch size 0 is not allowed in the embedding."
        );
        // Embed first token list via lookup table vector permutation.
        Tensor embedded = lookup_table[tokens[0]].unsqueeze(0);
        // Concatenate following token lists to embedded tensor.
        for (unsigned i = 1; i < batch_size; i++)
            embedded = Functional::cat(embedded, lookup_table[tokens[i]].unsqueeze(0), 0);
        // Return embedded tensor.
        return embedded;
    }

    // Token unembedding table. Takes as input a transformer output tensor of shape
    // (B, L, E), applies matmul with the transposed lookup table to obtain the logits.
    Tensor unembed(const Tensor& trans_out)
    {
        // Multiply by transposed embeddings to obtain logits.
        Tensor logits = Functional::matmul(trans_out, lookup_table, false, true);

        // Return logits.
        return logits;
    }
};

// Positional Embedding module class. Takes as input the token embeddings of shape (B, L, E)
// and adds the first L rows of its learned positional matrix. L cannot exceed context length.
class PositionalEmbedding : public Module
{
private:
    // Class parameter storage.
    Tensor embedding;

    // Max context length for reference.
    unsigned max_length;
public:

    // Positional Embeddings module constructor. Takes as input the maximum context length and 
    // embedding dimension, initializes the embeddings and adds them to the parameters list.
    PositionalEmbedding(unsigned max_context_length, unsigned emb_dim):
        max_length{ max_context_length }
    {
        // Create positional embeddings matrix.
        embedding = Tensor({ max_context_length, emb_dim });
        // Standard initialization N(0, 0.02).
        Initialization::normal(embedding, 0.0f, 0.02f);
        // Add matrix to parameter list.
        add_parameter(embedding);
    }

    // Positional Embeddings forward pass. Given the token embeddings of shape (B, L, E), it 
    // adds the first L rows of its learned positional matrix. L cannot exceed context length.
    Tensor forward(const Tensor& token_emb) const override
    {
        // Check that the context length is below maximum.
        MACROGRAD_CHECK(token_emb.size(1) <= max_length,
            "Incorrect tensor received for a positional embedding addition.\n" 
            "Maximum context length is %u but found tensor with shape %s", max_length, token_emb.shape().str()
        );

        // Prepare embedding.
        Tensor embedding_subset = embedding.subset({ token_emb.size(1) }, {});

        // Add embedding.
        Tensor out = token_emb + embedding_subset;

        // Return output.
        return out;
    }
};
