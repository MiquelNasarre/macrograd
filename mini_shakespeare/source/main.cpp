#include "macrograd_nn.h"
#include <stdio.h>
#include <math.h>

class Linear : public Module
{
private:
	Tensor matrix;
	Tensor bias;

	bool _has_bias;

public:
	Linear(unsigned in_dim, unsigned out_dim, bool has_bias = true, const char* device = "cpu") : _has_bias { has_bias }
	{
		matrix = Initialization::uniform(Tensor(Shape{ in_dim, out_dim }, device), -sqrtf(6.f / in_dim), sqrtf(6.f / in_dim));
		add_parameter(&matrix);

		if (has_bias)
		{
			bias = Initialization::uniform(Tensor(Shape{ out_dim, }, device), -sqrtf(1.f / in_dim), sqrtf(1.f / in_dim));
			add_parameter(&bias);
		}
	}

	Tensor forward(const Tensor& in) const override 
	{ 
		if (_has_bias)
			return Functional::matmul(in, matrix, bias); 
		return Functional::matmul(in, matrix);
	}
};

class MLP : public Module
{
	Linear** lins;
	unsigned count;
public:
	template<class... args>
	MLP(args... fans)
	{
		static_assert(sizeof...(args) > 1);		
		count = sizeof...(args) - 1;
		unsigned dims[sizeof...(args)] = { unsigned(fans)... };
		
		lins = new Linear*[count];

		for (unsigned i = 0; i < count; i++)
		{
			lins[i] = new Linear(dims[i], dims[i + 1]);
			add_module(*lins[i]);
		}
	}

	Tensor forward(const Tensor& in) const override
	{
		Tensor out = in;
		for (unsigned i = 0; i < count; i++)
		{
			out = lins[i]->forward(out);
			if (i < count - 1) out = out.relu();
		}
		return out;
	}
};

#include "MNIST.h"
int main()
{
	float** training_images = NumberRecognition::getImages(TRAINING, 0, 50000);
	unsigned* training_labels = NumberRecognition::getLabels(TRAINING, 0, 50000);
	float** testing_images = NumberRecognition::getImages(TESTING, 0, 10000);
	unsigned* testing_labels = NumberRecognition::getLabels(TESTING, 0, 10000);
	printf("images loaded successfully!\n\n");

	unsigned train_size = 50000;
	unsigned test_size = 10000;

	unsigned epochs = 1000;
	unsigned batch_size = 256;
	float lr = 0.02f;
	float momentum = 0.9f;
	float weight_decay = 0.001f;

	MLP mlp(IMAGE_DIM, 64, 64, 10);
	Optimizer opt(mlp, momentum, weight_decay, lr);
	Tensor a({batch_size, IMAGE_DIM}, "cpu");
	unsigned* labels = new unsigned[batch_size];

	int* randperm = new int[train_size];
	for (unsigned i = 0; i < train_size; i++)
		randperm[i] = i;

	for (unsigned epoch = 1; epoch < epochs + 1; epoch++)
	{
		// Training set.
		mlp.with_grad();
		{
			Random::shuffle(train_size, randperm);
			unsigned idx = 0u;
			while (idx < train_size) 
			{
				// Generate input tensor.
				bool shortened = train_size < idx + batch_size;
				unsigned start =  idx;
				unsigned end =  shortened ? train_size : idx + batch_size;
				idx = end;

				for (unsigned v = start; v < end; v++)
				{
					a.internal_set_vector({ v - start }, training_images[randperm[v]]);
					labels[v - start] = training_labels[randperm[v]];
				}
				Tensor in = shortened ? a.subset({ end - start, IMAGE_DIM }, { 0, 0 }) : a;

				// Forward pass.
				Tensor out = mlp(in);
				Tensor loss = Functional::cross_entropy_loss(out, labels);
				// Backward pass.
				mlp.zero_grad();
				loss.backward();
				opt.step();
			}
		}
		// Test set evalueation.
		mlp.no_grad();
		{
			float accum_loss = 0.f;
			unsigned accum_count = 0u;
			unsigned correct_count = 0u;
			unsigned idx = 0u;
			while (idx < test_size)
			{
				// Generate input tensor.
				bool shortened = test_size < idx + batch_size;
				unsigned start = idx;
				unsigned end = shortened ? test_size : idx + batch_size;
				idx = end;

				for (unsigned v = start; v < end; v++)
				{
					a.internal_set_vector({ v - start }, testing_images[v]);
					labels[v - start] = testing_labels[v];
				}
				Tensor in = shortened ? a.subset({ end - start, IMAGE_DIM }, { 0, 0 }) : a;

				// Forward pass.
				Tensor out = mlp(in);
				Tensor loss = Functional::cross_entropy_loss(out, labels);
				accum_loss += loss.item();
				accum_count++;

				// Count corrects.
				for (unsigned v = 0; v < out.size(0); v++)
				{
					float* logits = out.internal_get_vector({ v });
					unsigned argmax = 0;
					float max = logits[0];
					for (unsigned i = 1; i < out.size(-1); i++)
						if (logits[i] > max)
						{
							max = logits[i];
							argmax = i;
						}

					if (argmax == labels[v])
						correct_count++;
				}
			}
			printf("Epoch %04u finished | Loss: %.4f | Accuracy: %.2f%%\n", 
				epoch, accum_loss / accum_count, (100.f * correct_count) / test_size
			);
		}
	}

	delete[] randperm;
	delete[] labels;
	return 0;
}