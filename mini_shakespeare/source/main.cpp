#include "macrograd.h"
#include <stdio.h>
#include <random>

int main()
{
	Array s0({ 4,6 });
	Array s1({ 1,5 });
	Array s2({ 4,1 });

	for (unsigned i = 0; i < 4; i++)
	{
		s2.set_value({ i,0 }, float(i + 1));
		for (unsigned j = 0; j < 5; j++)
		{
			s0.set_value({ i,j }, rand() / 32000.f);
			s1.set_value({ 0,j }, float(j));
		}
	}

	Tensor a(s0, "cpu", true);
	Tensor b, c, d;

	//float lr = 0.05f;
	//for (unsigned i = 0; i < 200; i++)
	//{
	//	a -= lr * a.gradient();
	//	b = a.softmax(-1).sum(-2, true);
	//	c = b * a - 0.5f;
	//	d = c.mean(1).square();

	//	a.zero_grad();
	//	d.backward();
	//}

	b = a.reshape({2,2,2,3}).reshape({ 2,3,2,2 }).reshape({ 2,4,3 }).reshape({ 4,6 }).unsqueeze(0).repeat(0, 4);
	c = b.subset({ 2,1,4 }, { 0,-1,2 }).modify(Tensor(s2).squeeze(1).unsqueeze(0).unsqueeze(0), {1,0,0}).squeeze(1);
	d = (c + 1.f).square().sum(0).sum(0, true);
	d.backward();


	printf("%s\n\n%s\n\n%s\n\n%s", a.str(), b.str(), c.str(), d.str());
	return 0;
}