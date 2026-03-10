#pragma once
#include <cstdint>
#include <math.h>
#include "macrograd_nn.h"
#include "macrograd_error.h"

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

// Paths for the MNIST files

#define MNIST_TRAINING_IMAGES_PATH MNIST_PATH "train-images.idx3-ubyte"
#define MNIST_TRAINING_LABELS_PATH MNIST_PATH "train-labels.idx1-ubyte"
#define MNIST_TESTING_IMAGES_PATH MNIST_PATH "t10k-images.idx3-ubyte"
#define MNIST_TESTING_LABELS_PATH MNIST_PATH "t10k-labels.idx1-ubyte"

// Simple static class to handle the loading of the MNIST dataset for handwritten digits.
class MNIST
{
private:
	static inline Tensor trainingImages;
	static inline Tensor testingImages;

	static inline VectorInt trainingLabels;
	static inline VectorInt testingLabels;

	// Static helper to read 32bits from a file.
	static inline uint32_t read_be_u32(FILE* f)
	{
		uint8_t b[4];
		if (fread(b, 1, 4, f) != 4)
			return 0xFFFFFFFFu;

		return ((uint32_t)b[0] << 24) | ((uint32_t)b[1] << 16) | ((uint32_t)b[2] << 8) | ((uint32_t)b[3]);
	}

	// Reads the file data and creates the dataset tensors and label vectors.
	static void init_tensors()
	{
		if (trainingImages.is_init())
			return;

		FILE* training_images_file = nullptr;
		FILE* testing_images_file = nullptr;
		FILE* training_labels_file = nullptr;
		FILE* testing_labels_file = nullptr;

		// Open files
		fopen_s(&training_images_file, MNIST_TRAINING_IMAGES_PATH, "rb");
		fopen_s(&testing_images_file, MNIST_TESTING_IMAGES_PATH, "rb");
		fopen_s(&training_labels_file, MNIST_TRAINING_LABELS_PATH, "rb");
		fopen_s(&testing_labels_file, MNIST_TESTING_LABELS_PATH, "rb");

		MACROGRAD_CHECK(training_images_file, "[MNIST] Failed to open '%s'\n", MNIST_TRAINING_IMAGES_PATH);
		MACROGRAD_CHECK(testing_images_file, "[MNIST] Failed to open '%s'\n", MNIST_TESTING_IMAGES_PATH);
		MACROGRAD_CHECK(training_labels_file, "[MNIST] Failed to open '%s'\n", MNIST_TRAINING_LABELS_PATH);
		MACROGRAD_CHECK(testing_labels_file, "[MNIST] Failed to open '%s'\n", MNIST_TESTING_LABELS_PATH);

		// Read Headers
		// Training
		const uint32_t magic_i0 = read_be_u32(training_images_file);
		const uint32_t count_i0 = read_be_u32(training_images_file);
		const uint32_t rows0 = read_be_u32(training_images_file);
		const uint32_t cols0 = read_be_u32(training_images_file);
		const uint32_t magic_l0 = read_be_u32(training_labels_file);
		const uint32_t count_l0 = read_be_u32(training_labels_file);

		// Testing
		const uint32_t magic_i1 = read_be_u32(testing_images_file);
		const uint32_t count_i1 = read_be_u32(testing_images_file);
		const uint32_t rows1 = read_be_u32(testing_images_file);
		const uint32_t cols1 = read_be_u32(testing_images_file);
		const uint32_t magic_l1 = read_be_u32(testing_labels_file);
		const uint32_t count_l1 = read_be_u32(testing_labels_file);

		// Make sure there is nothing weird going on here

		MACROGRAD_CHECK(magic_i0 == 0x00000803u, "[MNIST] Bad image magic: 0x%08X (expected 0x00000803)\n", magic_i0);
		MACROGRAD_CHECK(magic_i1 == 0x00000803u, "[MNIST] Bad image magic: 0x%08X (expected 0x00000803)\n", magic_i1);
		MACROGRAD_CHECK(magic_l0 == 0x00000801u, "[MNIST] Bad label magic: 0x%08X (expected 0x00000801)\n", magic_l0);
		MACROGRAD_CHECK(magic_l1 == 0x00000801u, "[MNIST] Bad label magic: 0x%08X (expected 0x00000801)\n", magic_l1);
		MACROGRAD_CHECK(rows0 == IMG_ROWS && cols0 == IMG_COLS, "[MNIST] Unexpected image dims: %u x %u (expected 28 x 28)\n", rows0, cols0);
		MACROGRAD_CHECK(rows1 == IMG_ROWS && cols1 == IMG_COLS, "[MNIST] Unexpected image dims: %u x %u (expected 28 x 28)\n", rows1, cols1);
		MACROGRAD_CHECK(count_l0 == count_i0, "[MNIST] Count mismatch (images=%u, labels=%u)\n", count_i0, count_l0);
		MACROGRAD_CHECK(count_l1 == count_i1, "[MNIST] Count mismatch (images=%u, labels=%u)\n", count_i1, count_l1);
		MACROGRAD_CHECK(count_i0 != 0, "[MNIST] No images in file.\n");
		MACROGRAD_CHECK(count_i1 != 0, "[MNIST] No images in file.\n");

		uint32_t n_training = count_i0;
		uint32_t n_testing = count_i1;

		uint8_t* training_labels = new uint8_t[n_training];
		uint8_t* testing_labels = new uint8_t[n_testing];

		uint8_t* raw_training_images = new uint8_t[n_training * IMG_PIXELS];
		uint8_t* raw_testing_images = new uint8_t[n_testing * IMG_PIXELS];

		// Read labels
		if (fread(training_labels, 1, n_training, training_labels_file) != n_training)
			MACROGRAD_ERROR("[MNIST] Unexpected EOF while reading training labels.\n");
		if (fread(testing_labels, 1, n_testing, testing_labels_file) != n_testing)
			MACROGRAD_ERROR("[MNIST] Unexpected EOF while reading testing labels.\n");

		// Read images
		if (fread(raw_training_images, 1, IMG_PIXELS * n_training, training_images_file) != IMG_PIXELS * n_training)
			MACROGRAD_ERROR("[MNIST] Unexpected EOF while reading training images.\n");
		if (fread(raw_testing_images, 1, IMG_PIXELS * n_testing, testing_images_file) != IMG_PIXELS * n_testing)
			MACROGRAD_ERROR("[MNIST] Unexpected EOF while reading testing images.\n");

		fclose(training_images_file);
		fclose(training_labels_file);
		fclose(testing_images_file);
		fclose(testing_labels_file);

		trainingImages = Tensor({ n_training, IMG_PIXELS });
		testingImages = Tensor({ n_testing, IMG_PIXELS });

		trainingLabels = VectorInt((unsigned)n_training);
		testingLabels = VectorInt((unsigned)n_testing);

		for (unsigned i = 0; i < n_training; i++)
			trainingLabels[i] = (unsigned)training_labels[i];

		for (unsigned i = 0; i < n_testing; i++)
			testingLabels[i] = (unsigned)testing_labels[i];

		float* training_data = trainingImages.internal_data();
		float* testing_data = testingImages.internal_data();

		for (unsigned i = 0; i < n_training; i++)
			for (unsigned p = 0; p < IMG_PIXELS; p++)
				training_data[i * IMG_PIXELS + p] = float(raw_training_images[i * IMG_PIXELS + p]) / 256.f;

		for (unsigned i = 0; i < n_testing; i++)
			for (unsigned p = 0; p < IMG_PIXELS; p++)
				testing_data[i * IMG_PIXELS + p] = float(raw_testing_images[i * IMG_PIXELS + p]) / 256.f;

		// Delete temporary image storage.
		delete[] training_labels;
		delete[] testing_labels;
		delete[] raw_training_images;
		delete[] raw_testing_images;

		// Normalize the data.
		trainingImages -= trainingImages.mean(-1, true);
		trainingImages /= trainingImages.std(-1, true);
		testingImages -= testingImages.mean(-1, true);
		testingImages /= testingImages.std(-1, true);
	}

public:
	static const Tensor getTrainingImages() { init_tensors(); return trainingImages; }
	static const Tensor  getTestingImages() { init_tensors(); return  testingImages; }

	static const VectorInt getTrainingLabels() { init_tensors(); return trainingLabels; }
	static const VectorInt  getTestingLabels() { init_tensors(); return  testingLabels; }

	// To release tensor memory from cache.
	static void resetTensors()
	{
		trainingImages = Tensor();
		testingImages  = Tensor();
		trainingLabels = VectorInt();
		testingLabels  = VectorInt();
	}

	// Simple function to print the images in the console.
	static void consolePrint(float* image)
	{
		system("color");
		printf("\033[0;34m");
		for (unsigned int r = 0; r < IMG_ROWS; r++)
		{
			for (unsigned int c = 0; c < IMG_COLS; c++)
			{
				if (image[c + r * IMG_COLS] > 0.f)
					printf("\033[0;34m");
				else
					printf("\033[0;31m");

				printf("%c%c", 219, 219);
			}
			printf("\n");
		}
		printf("\033[0m");
	}
};

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
		add_parameter(matrix);

		if (has_bias)
		{
			bias = Tensor(Shape{ out_dim, }, device);
			Initialization::uniform(bias, -sqrtf(1.f / in_dim), sqrtf(1.f / in_dim));
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

static void run_mlp_training_run(int epochs = 10000, const char* device = "cuda", const char* load_path = nullptr)
{
	constexpr int train_size = 50000;
	constexpr int test_size  = 10000;

	Tensor train_data = MNIST::getTrainingImages().to(device);
	Tensor  test_data = MNIST::getTestingImages().to(device);
	VectorInt train_labels = MNIST::getTrainingLabels().to(device);
	VectorInt  test_labels = MNIST::getTestingLabels().to(device);
	// Clear CPU cache.
	MNIST::resetTensors();

	printf("MNIST loaded successfully!\n\n");

	int log_every		= 10;
	int batch_size		= 2048;
	float initial_lr    = 0.100f;
	float final_lr      = 0.010f;
	float momentum      = 0.900f;
	float weight_decay  = 0.005f;

	MLP mlp(IMG_PIXELS, 128, 64, 10); mlp.to(device);
	if (load_path)
	{
		mlp.load_weights(load_path);
		printf("[ Weights correctly loaded from \"%s\" ]\n\n", load_path);
	}

	Optimizer::SGD optimizer(mlp, momentum, initial_lr, weight_decay);
	Scheduler::CosineLR scheduler(optimizer, initial_lr, final_lr, epochs);

	VectorInt randperm(0, train_size, 1, device);

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
				Tensor in = perm_train_data.subset({ end - start }, { start });
				VectorInt labels = perm_train_labels.subset(start, end);

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
		if (epoch % log_every == 0)
		{
			Tensor accum_loss({1}, device);
			int correct_count = 0u;
			int start = 0u, end = 0u;
			while (end < test_size)
			{
				// Generate input tensor.
				start = end;
				end = (test_size < start + batch_size) ? test_size : start + batch_size;

				// Get input tensor.
				Tensor in = test_data.subset({ end - start }, { start });
				VectorInt labels = test_labels.subset(start, end);

				// Forward pass.
				Tensor logits = mlp(in);
				Tensor loss = Functional::cross_entropy_loss(logits, labels);
				accum_loss += loss * float(end - start);

				// Count corrects.
				VectorInt preds = logits.argmax().to("cpu");
				labels = labels.to("cpu");
				for (int v = 0; v < end - start; v++)
					if (preds[v] == labels[v]) correct_count++;
			}
			printf("Epoch %04u finished | Learning Rate: %.4f | Loss: %.4f | Accuracy: %.2f%%\n",
				epoch, optimizer.learning_rate(), accum_loss.item() / test_size, (100.f * correct_count) / test_size
			);
		}
	}
	printf("\nFinished training MLP on MNIST for %i epoch.", epochs);

	
	size_t seed; char c;
	// --- Model showcase ---
image_loop:
	printf("\nDo you want to see a test set image? (y/n/s) ");
	scanf_s(" %c", &c, 1);
	if (c == 'y' || c == 'Y')
	{
		mlp.no_grad();
		{
			unsigned idx = Random::rand_int(0, test_size - 1);
			Tensor image = test_data[idx];
			Tensor preds = mlp(image).softmax(-1).to("cpu");
			image = image.to("cpu");
			printf("\n");
			for (int _ = 0; _ < 56; _++)printf("-"); printf("\n");
			MNIST::consolePrint(image.internal_data());
			for (int _ = 0; _ < 56; _++)printf("-"); printf("\n");
			float* preds_data = preds.internal_data();
			printf("Predicted probabilities:\n");
			for (unsigned i = 0; i < preds.numel(); i++)
				printf("[ %u ]  %.2f%%\n", i, preds_data[i] * 100);
		}
		goto image_loop;
	}
	// --- Random seeding ---
	if (c == 's' || c == 'S')
	{
		printf("Set a different random seed: ");
		scanf_s(" %llu", &seed);
		Random::set_seed(seed);
		goto image_loop;
	}
	// --- Model weights saving ---
	printf("\nDo you want to save the model weights? (y/n) ");
	scanf_s(" %c", &c, 1);
	if (c == 'y')
	{
		char save_path[128];
		printf("Specify a valid save path for the file: ");
		scanf_s(" %127s", &save_path, 128);

		mlp.save_weights(save_path);
		printf("\n[ Weights successfully saved to \"%s\" ]\n", save_path);
	}
}
