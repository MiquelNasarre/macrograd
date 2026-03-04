#include "macrograd_nn.h"
#include "MNIST.h"

int main()
{
	Random::set_cuda_seed(4453);
	Tensor A({ 500, 600 }, "cuda");
	Initialization::uniform(A, 10.f, 50.f);

	A = A.flatten().to("cpu");
	
	printf("%.4f     %.4f", A.mean(0).to("cuda").item(), A.std(0).to("cuda").item());
	
	return 0;
}