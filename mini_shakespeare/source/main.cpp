#include "macrograd_nn.h"
#include "stdio.h"
#include "MNIST.h"

int main()
{
	constexpr int train_size = 50000;
	constexpr int test_size  = 10000;
	const char* device = "cuda";

	float** training_images   = NumberRecognition::getImages(TRAINING, 0, train_size);
	unsigned* training_labels = NumberRecognition::getLabels(TRAINING, 0, train_size);
	float** testing_images    = NumberRecognition::getImages( TESTING, 0,  test_size);
	unsigned* testing_labels  = NumberRecognition::getLabels( TESTING, 0,  test_size);

	Tensor train_data({ train_size, IMAGE_DIM }, device);
	Tensor test_data({ test_size, IMAGE_DIM }, device);
	VectorInt train_labels(train_size, device);
	VectorInt test_labels(test_size, device);

	for (int i = 0; i < train_size; i++) train_data.internal_set_vector({ i }, training_images[i]);
	for (int i = 0; i <  test_size; i++)  test_data.internal_set_vector({ i },  testing_images[i]);
	for (int i = 0; i < train_size; i++) train_labels.set(i, training_labels[i]);
	for (int i = 0; i <  test_size; i++)  test_labels.set(i,  testing_labels[i]);

	train_data -= train_data.mean(-1, true);
	train_data /= train_data.std (-1, true);
	test_data  -=  test_data.mean(-1, true);
	test_data  /=  test_data.std (-1, true);

	printf("MNIST loaded successfully!\n\n");

	int epochs			= 1000;
	int batch_size		= 2048;
	float initial_lr    = 0.100f;
	float final_lr      = 0.010f;
	float momentum      = 0.900f;
	float weight_decay  = 0.005f;

	MLP mlp(IMAGE_DIM, 128, 64, 10);
	mlp.to(device);
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
				Tensor in = test_data.subset({ end - start }, { start });
				VectorInt labels = test_labels.subset(start, end);

				// Forward pass.
				Tensor out = mlp(in);
				Tensor loss = Functional::cross_entropy_loss(out, labels);
				accum_loss += loss.item() * (end - start);

				// Count corrects.
				for (unsigned v = 0; v < out.size(0); v++)
				{
					out = out.to("cpu");
					float* logits = out.internal_get_vector({ v });
					unsigned argmax = 0;
					for (unsigned i = 1; i < out.size(-1); i++)
						if (logits[i] > logits[argmax]) argmax = i;
					// If it is a match increase correct count.
					if (argmax == labels.get(v)) correct_count++;
				}
			}
			printf("Epoch %04u finished | Learning Rate: %.4f | Loss: %.4f | Accuracy: %.2f%%\n",
				epoch, optimizer.learning_rate(), accum_loss / test_size, (100.f * correct_count) / test_size
			);
		}
	}
}