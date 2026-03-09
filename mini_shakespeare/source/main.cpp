#include "macrograd_nn.h"
#include "MNIST.h"

int main()
{
	run_mlp_training_run(100, "cuda");
}