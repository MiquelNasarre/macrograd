#pragma once
#pragma once
#include <cstdint>
#include <math.h>
#include "macrograd_nn.h"
#include "mini_shakespeare.h"

/* TINY SHAKESPEARE MACROGRAD TRAINING HEADER
--------------------------------------------------------------------------------------------------------------------------
--------------------------------------------------------------------------------------------------------------------------




--------------------------------------------------------------------------------------------------------------------------
--------------------------------------------------------------------------------------------------------------------------
*/

// Path for the tiny_shakespeare dataset.
#define TINY_SHAKESPEARE_PATH DATA_PATH "tiny_shakespeare/tiny_shakespeare.txt"

// Simple static class to handle the loading of the tiny_shakespeare.
class ShakespeareDataset
{
private:
	static inline char* dataset = nullptr;
	static inline unsigned _length = 0;

	static void initialize()
	{
		if (dataset) return;

		FILE* f = nullptr; 
		fopen_s(&f, TINY_SHAKESPEARE_PATH, "rb");
		MACROGRAD_CHECK(f != nullptr,
			"Unable to load file \"%s\"", TINY_SHAKESPEARE_PATH
		);

		fseek(f, 0, SEEK_END);
		_length = ftell(f);
		rewind(f);

		dataset = new char[_length + 1];
		fread(dataset, 1, _length, f);
		fclose(f);

		dataset[_length] = '\0';
	}

public:
	static const char* data() { initialize(); return dataset; }
	static unsigned length() { initialize(); return _length; }
};

struct ShakespeareTrainingDesc
{
	char device[16]       = "cuda";
	char load_path[128]   = "";
	char save_path[128]	  = "my_shakespeare.mg";
	int warmup_steps      = 300;
	int total_steps       = 10000;
	int start_step		  = 0;
	int log_every         = 100;
	int batch_size        = 32;
	int context_lentgh    = 256;
	float initial_lr      = 0.002f;
	float final_lr        = 0.0002f;
	float weight_decay    = 0.001f;
};

static void generateBatch(const ShakespeareTrainingDesc& desc, const VectorInt& dataset, VectorInt& input, VectorInt& output)
{
	// Iterate through batches.
	for (int i = 0; i < desc.batch_size; i++)
	{
		// Generate random number.
		int rand = Random::rand_int(0, dataset.len() - desc.context_lentgh - 1);

		// Append the tokens selected to the input and output lists.
		input.set(i * desc.context_lentgh, (i + 1) * desc.context_lentgh, dataset.data() + rand, dataset.is_gpu());
		output.set(i * desc.context_lentgh, (i + 1) * desc.context_lentgh, dataset.data() + rand + 1, dataset.is_gpu());
	}
}

static void train_shakespeare(ShakespeareTrainingDesc desc = {})
{
	const char* dataset       = ShakespeareDataset::data();
	const unsigned train_size = ShakespeareDataset::length();

	const VectorInt training_data = Tokenizer::encode(dataset).to(desc.device);

	printf("Tiny Shakespeare loaded successfully!\n");
	printf("Expected size: %u | Got size: %u\n\n", train_size, training_data.len());

	MiniShakespeare shakespeare; shakespeare.to(desc.device);
	if (desc.load_path[0] != '\0')
	{
		shakespeare.load_weights(desc.load_path);
		printf("[ Weights correctly loaded from \"%s\" ]\n\n", desc.load_path);
	}

	Optimizer::AdamW optimizer(shakespeare, desc.initial_lr, desc.weight_decay);
	Scheduler::LinearLR warmup_sched(optimizer, 0.0f, desc.initial_lr, desc.warmup_steps);
	Scheduler::CosineLR cosine_sched(optimizer, desc.initial_lr, desc.final_lr, desc.total_steps - desc.warmup_steps);

	Tensor accum_loss({ 1 }, desc.device);
	VectorInt* model_input = new VectorInt[desc.batch_size];
	VectorInt input_tokens(desc.context_lentgh * desc.batch_size, desc.device);
	VectorInt output_tokens(desc.context_lentgh * desc.batch_size, desc.device);
	Shape flattened(desc.batch_size * desc.context_lentgh, -1);

	for (int step = 1; step < desc.total_steps + 1; step++)
	{
		// Run scheduler until the starting step is reached.
		if (step <= desc.start_step)
		{
			(step <= desc.warmup_steps) ?
				warmup_sched.step() :
				cosine_sched.step();
			continue;
		}

		// Generate random batch.
		generateBatch(desc, training_data, input_tokens, output_tokens);

		// Splice training batches.
		for (int i = 0; i < desc.batch_size; i++)
			model_input[i] = input_tokens.subset(
				i * desc.context_lentgh, 
				(i + 1) * desc.context_lentgh
			);

		// Forward pass.
		Tensor logits = shakespeare.internal_forward(model_input, desc.batch_size);

		// Compute loss.
		Tensor loss = Functional::cross_entropy_loss(logits.view(flattened), output_tokens);

		// Backpropagate.
		optimizer.zero_grad();
		loss.backward();
		optimizer.step();

		// Accumulate loss.
		accum_loss += loss.no_grad();

		// Step scheduler.
		(step <= desc.warmup_steps) ? 
			warmup_sched.step() : 
			cosine_sched.step();

		// Log and save wieghts.
		if (step % desc.log_every == 0)
		{
			printf("Finished Step %04i | Learning Rate: %.6f | Accumulated Loss: %.6f\n", step, optimizer.learning_rate(), accum_loss.item() / desc.log_every);
			if (desc.save_path[0] != '\0') shakespeare.save_weights(desc.save_path);
			accum_loss = Tensor({ 1 }, desc.device);
		}
	}

	// End of training.
	delete[] model_input;
	printf("\nFinished training Shakespeare for %i steps.", desc.total_steps);
}
