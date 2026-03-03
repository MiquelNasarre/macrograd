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

	~MLP()
	{
		for (unsigned i = 0; i < count; i++)
			delete lins[i];
		
		delete[] lins;
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
	float** training_images   = NumberRecognition::getImages(TRAINING, 0, 50000);
	unsigned* training_labels = NumberRecognition::getLabels(TRAINING, 0, 50000);
	float** testing_images    = NumberRecognition::getImages( TESTING, 0, 10000);
	unsigned* testing_labels  = NumberRecognition::getLabels( TESTING, 0, 10000);
	printf("Images loaded successfully!\n\n");

	{
		unsigned train_size = 50000;
		unsigned test_size = 10000;

		unsigned epochs = 5;
		unsigned batch_size = 256;
		float initial_lr = 0.020f;
		float final_lr = 0.002f;
		float momentum = 0.900f;
		float weight_decay = 0.005f;

		MLP mlp(IMAGE_DIM, 128, 64, 10);
		Optimizer::SGD optimizer(mlp, momentum, initial_lr, weight_decay);
		Scheduler::CosineLR scheduler(optimizer, initial_lr, final_lr, epochs);

		unsigned* labels = new unsigned[batch_size];
		int* randperm = new int[train_size];
		for (unsigned i = 0; i < train_size; i++)
			randperm[i] = i;

		auto get_batch = [&](unsigned end, unsigned start, bool train)
			{
				Tensor in({ end - start, IMAGE_DIM });
				for (unsigned v = start; v < end; v++)
				{
					in.internal_set_vector({ v - start }, train ? training_images[randperm[v]] : testing_images[v]);
					labels[v - start] = train ? training_labels[randperm[v]] : testing_labels[v];
				}
				return in;
			};

		for (unsigned epoch = 1; epoch < epochs + 1; epoch++)
		{
			// Training set.
			mlp.with_grad();
			{
				Random::shuffle(train_size, randperm);
				unsigned start = 0u, end = 0u;
				while (end < train_size)
				{
					// Generate input tensor.
					start = end;
					end = (train_size < start + batch_size) ? train_size : start + batch_size;

					// Get input tensor.
					Tensor in = get_batch(end, start, true);

					// Forward pass.
					Tensor out = mlp(in);
					Tensor loss = Functional::cross_entropy_loss(out, labels);
					// Backward pass.
					optimizer.zero_grad();
					loss.backward();
					optimizer.step();
				}
				// Step the scheduler.
				scheduler.step();
			}
			// Test set evaluation.
			mlp.no_grad();
			{
				float accum_loss = 0.f;
				unsigned correct_count = 0u;
				unsigned start = 0u, end = 0u;
				while (end < test_size)
				{
					// Generate input tensor.
					start = end;
					end = (test_size < start + batch_size) ? test_size : start + batch_size;

					// Get input tensor.
					Tensor in = get_batch(end, start, false);

					// Forward pass.
					Tensor out = mlp(in);
					Tensor loss = Functional::cross_entropy_loss(out, labels);
					accum_loss += loss.item() * (end - start);

					// Count corrects.
					for (unsigned v = 0; v < out.size(0); v++)
					{
						float* logits = out.internal_get_vector({ v });
						unsigned argmax = 0;
						for (unsigned i = 1; i < out.size(-1); i++)
							if (logits[i] > logits[argmax]) argmax = i;

						if (argmax == labels[v]) correct_count++;
					}
				}
				printf("Epoch %04u finished | Learning Rate: %.4f | Loss: %.4f | Accuracy: %.2f%%\n",
					epoch, optimizer.learning_rate(), accum_loss / test_size, (100.f * correct_count) / test_size
				);
			}
		}

		delete[] randperm;
		delete[] labels;
	}

	return 0;
}