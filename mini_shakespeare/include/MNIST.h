#pragma once
#include <cstdint>

/* MNIST DATASET HELPER CLASS HEADER
-------------------------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------------------------
This class is a set of handy static functions to deal with the MNIST dataset.
This is stored in files in the data/ folder of the project and contains a big
set of training data for handwritten digits.

This class is meant for easy manipulation of the data in such set. First,
loadDataSet() should be called to load the files, and then you can easily 
obtain a pointer to the images and labels, and training values for subsets.
-------------------------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------------------------
*/

// Expected rows and columns to read from the dataset

#define IMG_ROWS 28u
#define IMG_COLS 28u
#define IMG_PIXELS (IMG_ROWS * IMG_COLS)

// Simple static class to handle the loading of the MNIST dataset for handwritten digits.
class MNIST
{
private:

	static inline uint8_t** trainingSetImages = nullptr;	// MNIST pointer to the training images
	static inline uint8_t** testingSetImages = nullptr;		// MNIST pointer to the testing images

	static inline uint8_t* trainingSetLabels = nullptr;		// MNIST pointer to the training labels
	static inline uint8_t* testingSetLabels = nullptr;		// MNIST pointer to the testing labels

	static inline size_t n_training_images = 0;				// Total images loaded from the training set
	static inline size_t n_testing_images = 0;				// Total images loaded from the testing set

	static MNIST helper;	// Helper to call the destructor at the end of the program

	// When the program ends the helper will call the destructor,
	// freeing the dataset if this was loaded.
	~MNIST();
	MNIST() = default;

public:

	// Loads the files for the dataset and stores the images in RAM.
	static void loadDataSet();

	// Returns the pointer to the training images as a uint8_t**.
	static uint8_t** getTrainingSetImages();

	// Returns the pointer to the training lables as an uint8_t*.
	static uint8_t* getTrainingSetLabels();

	// Returns the pointer to the testing images as a uint8_t**.
	static uint8_t** getTestingSetImages();

	// Returns the pointer to the testing lables as an uint8_t*.
	static uint8_t* getTestingSetLabels();

	// Returns the numpber of images stored in the training dataset.
	static size_t get_n_training_images();

	// Returns the numpber of images stored in the testing dataset.
	static size_t get_n_testing_images();

#ifdef _CONSOLE
	// Simple function to print the images in the console.
	static void consolePrint(uint8_t* image);
#endif
};


#define IMAGE_DIM 784

enum Set
{
	TESTING,
	TRAINING
};

class NumberRecognition
{
private:

	static inline unsigned* trainingLabels;
	static inline unsigned* testingLabels;

	static inline float** trainingImages;
	static inline float** testingImages;

	static inline float** trainingValues;
	static inline float** testingValues;

	static inline unsigned n_training;
	static inline unsigned n_testing;

	static NumberRecognition loader;

	NumberRecognition();
	~NumberRecognition();
public:

	static float** getImages(Set test_train, size_t start_idx, size_t end_idx);

	static unsigned* getLabels(Set test_train, size_t start_idx, size_t end_idx);

	static void printImage(Set test_train, size_t idx);

	static unsigned getSize(Set test_train);
};

#include "macrograd_nn.h"
#include <math.h>
#include <stdio.h>

class Linear : public Module
{
private:
	Tensor matrix;
	Tensor bias;

	bool _has_bias;

public:
	Linear(unsigned in_dim, unsigned out_dim, bool has_bias = true, const char* device = "cpu") : _has_bias{ has_bias }
	{
		matrix = Tensor(Shape{ in_dim, out_dim }, device);
		Initialization::uniform(matrix, -sqrtf(6.f / in_dim), sqrtf(6.f / in_dim));
		add_parameter(&matrix);

		if (has_bias)
		{
			bias = Tensor(Shape{ out_dim, }, device);
			Initialization::uniform(bias, -sqrtf(1.f / in_dim), sqrtf(1.f / in_dim));
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

		lins = new Linear * [count];

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

static void run_mlp_training_run()
{
	constexpr int train_size = 50000;
	constexpr int test_size  = 10000;

	float** training_images   = NumberRecognition::getImages(TRAINING, 0, train_size);
	unsigned* training_labels = NumberRecognition::getLabels(TRAINING, 0, train_size);
	float** testing_images    = NumberRecognition::getImages( TESTING, 0,  test_size);
	unsigned* testing_labels  = NumberRecognition::getLabels( TESTING, 0,  test_size);

	Tensor train_data({ train_size, IMAGE_DIM });
	Tensor test_data({ test_size, IMAGE_DIM });
	VectorInt train_labels(train_size); 
	VectorInt test_labels(test_size);

	for (int i = 0; i < train_size; i++) train_data.internal_set_vector({ i }, training_images[i]);
	for (int i = 0; i <  test_size; i++)  test_data.internal_set_vector({ i },  testing_images[i]);
	for (int i = 0; i < train_size; i++) train_labels[i] = training_labels[i];
	for (int i = 0; i <  test_size; i++)  test_labels[i] =  testing_labels[i];

	train_data -= train_data.mean(-1, true);
	train_data /= train_data.std(-1, true);
	test_data -= test_data.mean(-1, true);
	test_data /= test_data.std(-1, true);

	printf("MNIST loaded successfully!\n\n");

	int epochs			= 1000;
	int batch_size		= 256;
	float initial_lr    = 0.020f;
	float final_lr      = 0.002f;
	float momentum      = 0.900f;
	float weight_decay  = 0.005f;

	MLP mlp(IMAGE_DIM, 128, 64, 10);
	Optimizer::SGD optimizer(mlp, momentum, initial_lr, weight_decay);
	Scheduler::CosineLR scheduler(optimizer, initial_lr, final_lr, epochs);

	VectorInt randperm(0, train_size);

	for (int epoch = 1; epoch < epochs + 1; epoch++)
	{
		// Training set.
		mlp.with_grad();
		{
			Random::shuffle(randperm);
			Tensor perm_train_data = train_data[randperm];
			VectorInt perm_train_labels = train_labels[randperm];

			int start = 0u, end = 0u;
			while (end < train_size)
			{
				// Generate input tensor.
				start = end;
				end = (train_size < start + batch_size) ? train_size : start + batch_size;

				// Get input tensor.
				Tensor in = perm_train_data[{start, end}];
				VectorInt labels = perm_train_labels[{start, end}];

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
			int correct_count = 0u;
			int start = 0u, end = 0u;
			while (end < test_size)
			{
				// Generate input tensor.
				start = end;
				end = (test_size < start + batch_size) ? test_size : start + batch_size;

				// Get input tensor.
				Tensor in = test_data[{start, end}];
				VectorInt labels = test_labels[{start, end}];

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
					// If it is a match increase correct count.
					if (argmax == labels[v]) correct_count++;
				}
			}
			printf("Epoch %04u finished | Learning Rate: %.4f | Loss: %.4f | Accuracy: %.2f%%\n",
				epoch, optimizer.learning_rate(), accum_loss / test_size, (100.f * correct_count) / test_size
			);
		}
	}
}
