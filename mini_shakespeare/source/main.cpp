#include "transformer.h"
#include "MNIST.h"

int main()
{
	const char* device = "cuda";
	constexpr int epochs = 10000;
	constexpr int train_size = 50000;
	constexpr int test_size  = 10000;

	Tensor train_data = MNIST::getTrainingImages().to(device).view({ -1, 49, 16 });
	Tensor  test_data = MNIST::getTestingImages().to(device).view({ -1, 49, 16 });
	VectorInt train_labels = MNIST::getTrainingLabels().to(device);
	VectorInt  test_labels = MNIST::getTestingLabels().to(device);
	// Clear CPU cache.
	MNIST::resetTensors();

	printf("MNIST loaded successfully!\n\n");

	int log_every		= 1;
	int batch_size		= 1024;
	float initial_lr    = 0.100f;
	float final_lr      = 0.010f;
	float momentum      = 0.900f;
	float weight_decay  = 0.005f;

	//MLP mlp(IMG_PIXELS, 128, 64, 10); mlp.to(device);
	Tensor mask = Functional::causal_mask(49, device);
	Transformer mlp(2, 16, 1, 64); mlp.to(device); mlp.set_attention_mask(mask);

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
				Tensor logits = out.subset({ -1, 1, 10 }, {0, -1, 0}).squeeze(1);
				Tensor loss = Functional::cross_entropy_loss(logits, labels);
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
				Tensor out = mlp(in);
				Tensor logits = out.subset({ -1, 1, 10 }, {0, -1, 0}).squeeze(1);
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