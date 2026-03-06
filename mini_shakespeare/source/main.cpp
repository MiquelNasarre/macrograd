#include "macrograd_nn.h"
#include "MNIST.h"

int main()
{
	//run_mlp_training_run();
	//return 0;

	Random::set_seed(141);

	Tensor A({ 2, 3, 2 });
	Tensor I({ 1, 2, 4 });
	Tensor J({ 1, 1, 4 });

	Initialization::normal(A, 1.f, 0.25f);
	Initialization::uniform(I, 1.f, 2.f);
	Initialization::uniform(J, -2.f, -1.f);

	Tensor B = Functional::matmul(A, I, J);
	Tensor C = Functional::matmul(A.to("cuda"), I.to("cuda"), J.to("cuda"));
	
	printf("%s\n%s\n%s\n%s", A.str(), I.str(), B.str(), C.str());
	return 0;
}