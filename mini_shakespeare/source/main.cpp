#include "mini_shakespeare.h"
#include "training.h"

// Give your literary indications
// to mini Shakespeare here!
struct EditorialGuidance
{
	size_t seed         = 200994;
	float temperature   = 0.75f;
	unsigned length     = 16384;
	char load_file[128] = "weights/my_shakespeare.mg";
	char device[16]     = "cuda";
};

// Basic writing function, takes a descriptor 
// as input and lets mini Shakespare write!
void write(EditorialGuidance desc = {})
{
	// Set random seed.
	Random::set_seed(desc.seed);
	Random::set_cuda_seed(desc.seed);

	// Initialize the model and send to device.
	MiniShakespeare shakespeare;
	shakespeare.to(desc.device);
	shakespeare.set_temperature(desc.temperature);

	// Load weights file if exists.
	if (desc.load_file[0] != '\0')
		shakespeare.load_weights(desc.load_file);

	// Allocate some space for his words.
	char* his_words = new char[desc.length];
	his_words[0] = '\n'; his_words[1] = '\0';

	// Let him generate.
	for (unsigned i = 0; i < desc.length - 2; i++)
	{
		shakespeare.add_one_character(his_words);
		printf("%c", his_words[i + 1]);
	}
	printf("\n\n\n");

	// Clean up his words.
	delete[] his_words;
}

// Simple function for deterministic 
// seeded writing on the console.
void random_writing_default()
{
	// Mixer function for deterministic seed.
	auto splitmix = [](size_t _seed)
	{
		_seed += 0x9E3779B97F4A7C15ull;
		_seed = (_seed ^ (_seed >> 30)) * 0xBF58476D1CE4E5B9ull;
		_seed = (_seed ^ (_seed >> 27)) * 0x94D049BB133111EBull;
		_seed ^= (_seed >> 31);
		return _seed;
	};

	// Provide a starting seed.
	printf("This shall provide our start: ");
	char buffer[256] = {};
	scanf_s(" %255s", &buffer, 256);
	printf("\n");

	// Generate random seed from input.
	size_t seed = 0;
	for (unsigned i = 0; i < 256; i++)
		seed = splitmix(seed + (size_t)buffer[i]);

	// Let Shakespeare write.
	EditorialGuidance desc = { seed };
	write(desc);
}

// Main function.
int main()
{
	//random_writing_default();
	train_shakespeare();
	return 0;
}
