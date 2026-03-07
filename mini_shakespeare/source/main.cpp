#include "macrograd_nn.h"
#include "stdio.h"
#include "MNIST.h"

int main()
{
	run_mlp_training_run();
	return 0;

	Random::set_seed(14451);

	Tensor A({ 1, 2,3,4,5,1,7,1 });
	Tensor I({ 2, 1,3,4,5,6,1,3 });
	Tensor J({ 2, 1,3,4,5,6,1,3 });
	Tensor K({ 2, 1,3,4,5,6,1,3 });

	Initialization::normal(A, 2.f, 0.25f);
	Initialization::uniform(I, 1.f, 2.f);
	Initialization::uniform(J, 1.f, 2.f);
	Initialization::uniform(K, 1.f, 2.f);

	Tensor B = (A + I) / (A * J) - (J - K);
	B += I;
	B *= I;
	B /= J;
	B /= A;
	B *= I.square();
	B /= J.sqrt();

	A = A.to("cuda");
	I = I.to("cuda");
	J = J.to("cuda");
	K = K.to("cuda");

	Tensor C = (A + I) / (A * J) - (J - K);
	C += I;
	C *= I;
	C /= J;
	C /= A;
	C *= I.square();
	C /= J.sqrt();

	C = (C - B.to("cuda")).flatten().sum(0);
	
	printf("%s\n%s\n%s\n%s", A.str(), I.str(), B.str(), C.str());
	return 0;
}