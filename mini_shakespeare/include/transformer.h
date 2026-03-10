#pragma once
#include "macrograd_nn.h"
#include "macrograd_error.h"

#include <math.h>

class Linear : public Module
{
private:
	Tensor matrix;
	Tensor bias;

	bool _has_bias;

public:
	Linear(unsigned fan_in, unsigned fan_out, bool has_bias = true) : _has_bias{ has_bias }
	{
		matrix = Tensor(Shape{ fan_in, fan_out });
        Initialization::uniform(matrix, -sqrtf(6.f / (fan_in + fan_out)), sqrtf(6.f / (fan_in + fan_out)));
		add_parameter(matrix);

		if (has_bias)
		{
			bias = Tensor(Shape{ fan_out, });
			add_parameter(bias);
		}
	}

	Tensor forward(const Tensor& in) const override
	{
		if (_has_bias)
			return Functional::matmul(in, matrix, bias);
		return Functional::matmul(in, matrix);
	}
};

class FeedForward : public Module
{
private:
	Linear linear0;
	Linear linear1;

public:
	FeedForward(unsigned emb_dim, unsigned hid_dim, bool has_bias = true):
		linear0{ emb_dim, hid_dim, has_bias },
		linear1{ hid_dim, emb_dim, has_bias } 
    { add_module(linear0); add_module(linear1); }

	Tensor forward(const Tensor& in) const override
	{
		Tensor out0 = linear0(in).gelu();
        Tensor out1 = linear1(out0);
		return out1;
	}
};

class MultiHeadSelfAttention : public Module
{
private:
    Linear q_linear;
    Linear k_linear;
    Linear v_linear;
    Linear o_linear;

    Tensor mask;

    unsigned n_heads;
    unsigned emb_dim;
    unsigned head_dim;

public:
    MultiHeadSelfAttention(unsigned _n_heads, unsigned _emb_dim) :
        n_heads{ _n_heads },
        emb_dim{ _emb_dim },
        q_linear{ _emb_dim, _emb_dim },
        k_linear{ _emb_dim, _emb_dim },
        v_linear{ _emb_dim, _emb_dim },
        o_linear{ _emb_dim, _emb_dim }
    {
        MACROGRAD_CHECK(_emb_dim % _n_heads == 0 && _emb_dim,
            "Invalid parameters found for MHSA initialization.\n"
            "Embeding dimension must be divisible by the number of heads."
        );
        add_module(q_linear);
        add_module(k_linear);
        add_module(v_linear);
        add_module(o_linear);

        head_dim = emb_dim / n_heads;
    }

    void set_attetion_mask(const Tensor& _mask) { mask = _mask; }
	
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

        // Generate attention scores.
        Tensor scores = Functional::matmul(Q, K, false, true) / sqrtf((float)head_dim); // (B, h, L, L)

        // Apply mask.
        if (mask.is_init())
            scores += mask; // (B, h, L, L)

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

class LayerNorm : public Module
{
private:
    Tensor gamma;
    Tensor beta;

    unsigned layer_dim;
    float eps;
public:
    LayerNorm(unsigned _layer_dim, float _eps = 1e-5f)
    {
        layer_dim = _layer_dim;
        eps = _eps;
        gamma = Tensor(Shape{ layer_dim });
        beta = Tensor(Shape{ layer_dim });

        float beta_init = 0.f;
        float gamma_init = 1.f;
        beta.internal_set(&beta_init);
        gamma.internal_set(&gamma_init);

        add_parameter(gamma);
        add_parameter(beta);
    }

    Tensor forward(const Tensor& in) const override
    {
        MACROGRAD_CHECK(in.size(-1) == layer_dim,
            "Invalid tensor shape received inside a LayerNorm forward pass.\n"
            "Expected last dimension %i but got %s",
            layer_dim, in.shape().str()
        );

        // Get safe normalization tensors.
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

class Layer : public Module
{
private:
    MultiHeadSelfAttention attention;
    FeedForward feed_forward;
    LayerNorm attn_norm;
    LayerNorm ff_norm;

public:
    Layer(unsigned emb_dim, unsigned n_heads, unsigned ff_dim) :
        attention{ n_heads, emb_dim },
        attn_norm{ emb_dim },
        feed_forward{ emb_dim, ff_dim },
        ff_norm{ emb_dim }
    {
        add_module(attention);
        add_module(attn_norm);
        add_module(feed_forward);
        add_module(ff_norm);
    }

    void set_attetion_mask(const Tensor& mask) { attention.set_attetion_mask(mask); }

    Tensor forward(const Tensor& layer_in) const override
    {
        // First do attention.
        Tensor attn_out = attention(layer_in); // (B, L, E)

        // Add residual and layer normalize.
        Tensor ff_in = attn_norm(attn_out + layer_in); // (B, L, E)

        // feed forward.
        Tensor ff_out = feed_forward(ff_in); // (B, L, E)

        // Add residual and layer normalize.
        Tensor out = ff_norm(ff_out + ff_in); // (B, L, E)

        // Return output.
        return out; // (B, L, E)
    }
};

class Transformer : public Module
{
private:
    Layer** layer;

    unsigned n_layers;

public:
    Transformer(unsigned num_layers, unsigned emb_dim, unsigned n_heads, unsigned ff_dim)
    {
        n_layers = num_layers;
        layer = new Layer*[num_layers];

        for (unsigned i = 0; i < n_layers; i++)
        {
            layer[i] = new Layer(emb_dim, n_heads, ff_dim);
            add_module(*layer[i]);
        }
    }

    ~Transformer()
    {
        for (unsigned i = 0; i < n_layers; i++)
            delete layer[i];

        delete[] layer;
    }

    void set_attention_mask(const Tensor& mask)
    {
        for (unsigned i = 0; i < n_layers; i++)
            layer[i]->set_attetion_mask(mask);
    }

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