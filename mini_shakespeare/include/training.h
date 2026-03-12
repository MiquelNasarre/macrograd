#pragma once

/* LICENSE AND COPYRIGHT
--------------------------------------------------------------------------------------------------------------------------
 * Macrograd - a CUDA/C++ Autograd Tensor Library
 * Copyright (c) 2026 Miguel Nasarre Budiño
 * Licensed under the MIT License. See LICENSE file.
--------------------------------------------------------------------------------------------------------------------------
*/

/* TINY SHAKESPEARE MACROGRAD TRAINING HEADER
--------------------------------------------------------------------------------------------------------------------------
--------------------------------------------------------------------------------------------------------------------------
 My Mini Shakespeare's first ever words were:
 "Ar,i :bHmvoF!qIA?J,&spsRJDTJoTojv-HrgDGUdIWhVDDYPwTA! hAmQb;-iERMKjqf,vBvmgWbsee"

 Although those words looked beautiful to me, no one else really seemed to like them, so training
 time it is. But how are you supposed to train a transformer to write Shakespeare?

 First step: you need to have all of Shakespeare's work accessible in a single file, preferably.
 Which is nice because Andrej Karpathy kindly did it for us. The dataset, called tiny Shakespeare, 
 can be found in the 'data/tiny_shakespeare' folder. The class TinyDataset in this file loads the 
 entire dataset into a single string buffer.

 Second step: you need to transform that string into something our little model can understand.
 The Tokenizer class found in 'tokenizer.h' already takes care of that, if we feed it our string
 it will give us a VectorInt of the same length with all the characters tokenized.

 Third step: you need to decide on some hyperparameters and other variables. That is done by the
 ShakespeareTrainingDesc struct, which defines how a training run will be performed. That includes 
 learning rate routines, batch size, validation splits, etc.

 Finally you need to put that together in a training routine. That is what the training function,
 helped by the batch generation function, they run multiple training steps. They also compute 
 validation loss and log to the console from time to time, saving the model when a better
 validation loss is reached.

 Once you have done all those steps and run the training for a bit, your Tiny Shakespeare
 will be able to write coherent words, and people will stop calling you crazy for being proud
 of it still generating garbage :)
--------------------------------------------------------------------------------------------------------------------------
--------------------------------------------------------------------------------------------------------------------------
*/

// Library headers.
#include "macrograd_nn.h"
#include "mini_shakespeare.h"

// Other file dependencies.
#include <stdint.h>
#include <string.h>
#include <math.h>

// Path for the tiny_shakespeare dataset.
#define TINY_SHAKESPEARE_PATH DATA_PATH "tiny_shakespeare/tiny_shakespeare.txt"

// Simple static class to handle the loading of tiny_shakespeare.
class TinyDataset
{
private:
	// Static variables to store the dataset.
	static inline char* dataset = nullptr;
	static inline unsigned _length = 0;

	// Downloads the data into a single string.
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
	// Static helpers to access the dataset.
	static const char* data() { initialize(); return dataset; }
	static unsigned length() { initialize(); return _length; }
};

// Training descriptor for the tiny_shakespeare dataset.
// Contains all training hyperparameters that can be modified.
struct ShakespeareTrainingDesc
{
	char device[16]       = "cuda";
	char load_path[128]   = "";
	char save_path[128]	  = "my_shakespeare.mg";
	int warmup_steps      = 300;
	int total_steps       = 10000;
	int log_every         = 50;
	int batch_size        = 128;
	int micro_batch_size  = 32;
	int context_lentgh    = 256;
	float train_split	  = 0.9f;
	int eval_micro_batch  = 8;
	float initial_lr      = 0.001f;
	float final_lr        = 0.0001f;
	float weight_decay    = 0.01f;
};

// Helper function to generate batches for the training run. It generates random numbers for
// every line that defines the start of the data. Each one of those streams is copied into the
// model input vectors, and in the output vector with a shift of 1 token for loss computation.
static void generateBatch(const ShakespeareTrainingDesc& desc, const VectorInt& dataset, VectorInt* model_input, VectorInt& output)
{
	// Iterate through batches.
	for (int i = 0; i < desc.micro_batch_size; i++)
	{
		// Generate random number.
		int rand = Random::rand_int(0, dataset.len() - desc.context_lentgh - 1);

		// Append the tokens selected to the input and output lists.
		model_input[i] = dataset.subset(rand, rand + desc.context_lentgh);
		output.set(i * desc.context_lentgh, (i + 1) * desc.context_lentgh, dataset.data() + rand + 1, dataset.is_gpu());
	}
}

// My Mini Shakespeare training routine. It takes a training descriptor as input. 
// First it loads the dataset into a Vector and separates training and test splits.
// 
// The training performs gradient accumulation until the batch size is reached and
// then steps the optimizer and scheduler and goes to the next step. It uses AdamW 
// optimizer and CosineLR scheduler with a warmup phase.
// 
// It routinely computes validation loss and prints it to the console. When the 
// validation loss hits a new minimum the model weights are saved.
static void train_shakespeare(ShakespeareTrainingDesc desc = {})
{
	// Sanity check
	MACROGRAD_CHECK(desc.micro_batch_size && desc.batch_size % desc.micro_batch_size == 0,
		"Invalid batch sizes received for a tiny_shakespeare training run. Make sure micro batch\n"
		"size diviced batch size. To avoid gradient accumulation set the values to the same size."
	);
	int accum_steps = desc.batch_size / desc.micro_batch_size;

	// Load and tokenize entire dataset.
	const VectorInt tokenized_dataset = Tokenizer::encode(TinyDataset::data()).to(desc.device);
	// Log upon success.
	printf("Tiny Shakespeare loaded successfully!\n");
	printf("Expected size: %u | Got size: %u\n\n", TinyDataset::length(), tokenized_dataset.len());

	// Split training and evaluation.
	const VectorInt training_data = tokenized_dataset.subset(0, int(desc.train_split * tokenized_dataset.len()));
	const VectorInt validation_data = tokenized_dataset.subset(int(desc.train_split * tokenized_dataset.len()), 0);

	// Initialize the model, send it to device and load its weights.
	MiniShakespeare shakespeare; shakespeare.to(desc.device);
	if (desc.load_path[0] != '\0')
	{
		shakespeare.load_weights(desc.load_path);
		printf("[ Weights correctly loaded from \"%s\" ]\n\n", desc.load_path);
	}

	// Log device data.
	if (!strcmp(desc.device, "cuda"))
		printf("[ CUDA: %s | VRAM: %.2fGB ]\n\n",Cuda::device_name(),float(Cuda::device_memory()) / (1024 * 1024 * 1024));

	// Create optimizer and schedulers.
	Optimizer::AdamW optimizer(shakespeare, desc.initial_lr, desc.weight_decay);
	Scheduler::LinearLR warmup_sched(optimizer, 0.0f, desc.initial_lr, desc.warmup_steps);
	Scheduler::CosineLR cosine_sched(optimizer, desc.initial_lr, desc.final_lr, desc.total_steps - desc.warmup_steps);

	// Prepare vectors and tensors for the training run.
	Tensor accum_loss({ 1 }, desc.device, false);
	VectorInt* model_input = new VectorInt[desc.micro_batch_size];
	VectorInt output_tokens(desc.context_lentgh * desc.micro_batch_size, desc.device);
	float best_validation = INFINITY;

	// Iterate through all the steps.
	for (int step = 1; step < desc.total_steps + 1; step++)
	{
		// Training step use gradient.
		shakespeare.with_grad();
		{
			// Zero gradients.
			optimizer.zero_grad();

			// Accumulate gradient for accum_steps.
			for (int micro_step = 0; micro_step < accum_steps; micro_step++)
			{
				// Generate random batch.
				generateBatch(desc, training_data, model_input, output_tokens);

				// Forward pass.
				Tensor logits = shakespeare.internal_forward(model_input, desc.micro_batch_size);

				// Compute loss.
				Tensor loss = Functional::cross_entropy_loss(logits.view({ -1, Tokenizer::num_tokens }), output_tokens) / (float)accum_steps;

				// Backpropagate.
				loss.backward();

				// Accumulate loss.
				accum_loss += loss.no_grad();
			}

			// Step optimizer.
			optimizer.step();

			// Step scheduler.
			(step <= desc.warmup_steps) ? 
				warmup_sched.step() : 
				cosine_sched.step();
		}

		// Log validation and save weights.
		if (step % desc.log_every == 0)
		{
			// Tensor for validation loss.
			Tensor val_loss({ 1 }, desc.device, false);

			// Make sure we do not use gradient for validation.
			shakespeare.no_grad();
			for (int micro_step = 0; micro_step < desc.eval_micro_batch; micro_step++)
			{
				// Generate random eval batch.
				generateBatch(desc, validation_data, model_input, output_tokens);

				// Forward pass.
				Tensor logits = shakespeare.internal_forward(model_input, desc.micro_batch_size);

				// Compute loss.
				val_loss += Functional::cross_entropy_loss(logits.view({ -1, Tokenizer::num_tokens }), output_tokens) / (float)desc.eval_micro_batch;
			}

			// Log to console.
			float validation_loss = val_loss.item();
			printf("Finished Step %04i | Learning Rate: %.6f | Train Loss: %.4f | Validation Loss: %.4f\n", 
				step, optimizer.learning_rate(), accum_loss.item() / desc.log_every, validation_loss);

			// If new best validation loss save weights.
			if (desc.save_path[0] != '\0' && validation_loss < best_validation)
			{
				shakespeare.save_weights(desc.save_path);
				best_validation = validation_loss;
			}

			// Reset training loss.
			accum_loss.internal_set_value({}, 0.f);
		}
	}

	// End of training.
	delete[] model_input;
	printf("\nFinished training Shakespeare for %i steps.\n", desc.total_steps);
	printf("Best Validation Loss: %.4f\n", best_validation);
	if (desc.save_path[0] != '\0')
		printf("Model weights saved to \"%s\"\n", desc.save_path);
}
