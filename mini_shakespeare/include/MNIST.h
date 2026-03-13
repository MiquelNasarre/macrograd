#pragma once
#include <cstdint>
#include <math.h>
#include "macrograd_nn.h"
#include "macrograd_error.h"

/* MNIST MACROGRAD TRAINING HEADER
--------------------------------------------------------------------------------------------------------------------------
--------------------------------------------------------------------------------------------------------------------------
 During the creation of Macrograd, since the moment it started being able to train I was
 performing basic tests with this dataset. And since I didn't want to get rid of the file
 here it is :)
 
 The MNIST is a very well knonw dataset for basic ML applications, it consists of 60k 28x28
 images of handritten digits, which serve as a fun dataset to play with and learn about MLPs
 and convolutional networks. 
  
 This implementation uses a simple MLP of hidden layer sizes 128 and 64, and achieves test
 set performances of +98% almost instantly, even in my tiny GPU which is nice. 
 
 Feel free to read the implemetation and play with the parameters. If you want to see the 
 actual images at the end of the training run you can see the model guesses on certain 
 training set images, which are painted on console by the function MNIST::consolePrint.
--------------------------------------------------------------------------------------------------------------------------
--------------------------------------------------------------------------------------------------------------------------
*/

// Expected rows and columns to read from the dataset.

#define IMG_ROWS 28u
#define IMG_COLS 28u
#define IMG_PIXELS (IMG_ROWS * IMG_COLS)

// Paths for the MNIST files.

#define MNIST_TRAINING_IMAGES_PATH "data/MNIST/train-images.idx3-ubyte"
#define MNIST_TRAINING_LABELS_PATH "data/MNIST/train-labels.idx1-ubyte"
#define MNIST_TESTING_IMAGES_PATH "data/MNIST/t10k-images.idx3-ubyte"
#define MNIST_TESTING_LABELS_PATH "data/MNIST/t10k-labels.idx1-ubyte"

// Simple static class to handle the loading of the MNIST dataset. It stores the datset 
// directly as tensors of size (50k, IMAGE_DIM) and (10k, IMAGE_DIM). And stores the 
// labels as VectorInts of the same lengths. 
class MNIST
{
private:
	// Static tensors to store images.
	static inline Tensor trainingImages;
	static inline Tensor testingImages;
	// Static vectors to store labels.
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
	// Public functions to load the dataset into the tensors before returning them.
	static const Tensor getTrainingImages() { init_tensors(); return trainingImages; }
	static const Tensor  getTestingImages() { init_tensors(); return  testingImages; }
	// Public functions to load the labels into the vectors before returning them.
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

// Simple linear class module implemetation. Defines a matrix and a bias. 
// During forward pass does matrix multiplication with bias and returns.
class Linear : public Module
{
private:
	// Parameter storage for the module.
	Tensor matrix;
	Tensor bias;

	// Bias boolean.
	bool _has_bias;

public:
	// Linear module constructor. Creates the tensors, initializes them 
	// with kaiming uniform, and adds them to the parameter list.
	Linear(unsigned in_dim, unsigned out_dim, bool has_bias = true) : _has_bias{ has_bias }
	{
		matrix = Tensor(Shape{ in_dim, out_dim });
		Initialization::uniform(matrix, -sqrtf(6.f / in_dim), sqrtf(6.f / in_dim));
		add_parameter(matrix);
		// If requires bias create it.
		if (has_bias)
		{
			bias = Tensor(Shape{ out_dim, });
			Initialization::uniform(bias, -sqrtf(1.f / in_dim), sqrtf(1.f / in_dim));
			add_parameter(bias);
		}
	}

	// Linear layer forward pass. Performs matrix multiplication.
	// Bias addition is fused if specified during creation.
	Tensor forward(const Tensor& in) const override
	{
		if (_has_bias)
			return Functional::matmul(in, matrix, bias);
		return Functional::matmul(in, matrix);
	}
};

// Multi Layer Perceptron module class. Defines a stack of linear layers
// of abitrary sizes with bias. During the forward pass performs the linear
// passes followes by ReLU as the non-liniarity.
class MLP : public Module
{
	// Linear module storage.
	Linear** lins;

	// Count of linear layers.
	unsigned count;
public:
	// Templated constructor. Just for fun you can enter any arbitrary number of 
	// layers and sizes and it will create that MLP, which is actually pretty 
	// convenient. Initializes the layers and adds them to the parameter list. 
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

	// Destructor. Deletes the layers.
	~MLP()
	{
		for (unsigned i = 0; i < count; i++)
			delete lins[i];

		delete[] lins;
	}

	// Runs the stack of linear layers, applying ReLU for the hidden layers.
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

// Descriptor for the MLP training. Feel free to play with 
// the parameters and try different training settings.
struct MNISTTrainingDesc
{
	char device[16]     = "cuda";
	char load_path[128] = "";
	char save_path[128] = "my_mlp.mg";
	int hidden0			= 128;
	int hidden1			= 64;
	int epochs          = 1000;
	int log_every       = 10;
	int batch_size      = 2048;
	float initial_lr    = 0.100f;
	float final_lr      = 0.010f;
	float momentum      = 0.900f;
	float weight_decay  = 0.005f;
};

// Trains an MLP on the MNIST dataset with the specified training parameters.
// Uses a SGD optimizer and a CosineLR scheduler. Every epoch applies a random
// permutation to the training data and trains on the entire set. During eval 
// epochs it runs evaluation on the test set and prints to console.
// At the end you can see some of the results of the trained model.
static void run_mlp_training_run(MNISTTrainingDesc desc = {})
{
	// Expected train and test sizes.
	constexpr int train_size = 50000;
	constexpr int test_size  = 10000;

	// Load datasets from files.
	Tensor train_data = MNIST::getTrainingImages().to(desc.device);
	Tensor  test_data = MNIST::getTestingImages().to(desc.device);
	VectorInt train_labels = MNIST::getTrainingLabels().to(desc.device);
	VectorInt  test_labels = MNIST::getTestingLabels().to(desc.device);
	// Clear CPU cache.
	MNIST::resetTensors();
	printf("MNIST loaded successfully!\n\n");

	// Initialize MLP and send to device.
	MLP mlp(IMG_PIXELS, desc.hidden0, desc.hidden1, 10); 
		mlp.to(desc.device);

	// Load weights from file.
	if (desc.load_path[0] != '\0')
	{
		mlp.load_weights(desc.load_path);
		printf("[ Weights correctly loaded from \"%s\" ]\n\n", desc.load_path);
	}

	// Create SGD otimizer and Cosine decay scheduler.
	Optimizer::SGD optimizer(mlp, desc.momentum, desc.initial_lr, desc.weight_decay);
	Scheduler::CosineLR scheduler(optimizer, desc.initial_lr, desc.final_lr, desc.epochs);

	// Create an aranged vector of training size for permutations.
	VectorInt randperm(0, train_size, 1, desc.device);

	// Loop through epoch.
	for (int epoch = 1; epoch < desc.epochs + 1; epoch++)
	{
		// Training set.
		mlp.with_grad();
		{
			// Randomly permute images for this epoch.
			Random::shuffle(randperm);
			Tensor perm_train_data = train_data[randperm];
			VectorInt perm_train_labels = train_labels[randperm];

			int start = 0u, end = 0u;
			while (end < train_size)
			{
				// Generate input tensor.
				start = end;
				end = (train_size < start + desc.batch_size) ? train_size : start + desc.batch_size;

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
		if (epoch % desc.log_every == 0)
		{
			Tensor accum_loss({1}, desc.device);
			int correct_count = 0u;
			int start = 0u, end = 0u;
			while (end < test_size)
			{
				// Generate input tensor.
				start = end;
				end = (test_size < start + desc.batch_size) ? test_size : start + desc.batch_size;

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
			// Report to console.
			printf("Epoch %04u finished | Learning Rate: %.4f | Loss: %.4f | Accuracy: %.2f%%\n",
				epoch, optimizer.learning_rate(), accum_loss.item() / test_size, (100.f * correct_count) / test_size
			);
		}
	}
	// Finished training :)
	printf("\nFinished training MLP on MNIST for %i epoch.", desc.epochs);

	
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
	if (desc.save_path[0] != '\0')
	{
		mlp.save_weights(desc.save_path);
		printf("\n\n[ Weights successfully saved to \"%s\" ]\n", desc.save_path);
	}
}
