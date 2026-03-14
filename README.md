# Macrograd

Macrograd is a fully functional autograd library, build in C++ and accelerated with CUDA. Inspired by the video
[The spelled-out intro to neural networks and backpropagation](https://www.youtube.com/watch?v=VMj-3S1tku0&t=7109s),
where Andrej Karpathy presents his library [micrograd](https://github.com/karpathy/micrograd).

---

> *Micrograd is all you need to train neural networks, and everything else is just efficiency.*
> *So you'd think that micrograd would be a very complex piece of code, and that turns out to not be the case...*

---

When I first watched the micrograd introduction video I found it like a breath of fresh air, Andrej's explanations 
were clear and simple, and allowed anyone to get a much cleaner understanding of what autograd libraries actually are,
with just basic mathematical concepts.

Since that video, though, I always wanted to build Macrograd, an autograd library with the same overall simplicity 
of micrograd but that actually handled the efficiency, being able to train production neural networks. It turns out that 
his quote up top is true, but the efficiency part brings a lot more complexity and technical challenges into the table, 
which are also essential to be able to create and optimize today's neural networds.

## How to use it



## Tensors

The main object of the library is the `Tensor` class, it is an arbitrarily shaped array with multiple operations defined
between them and optionally the capability to track which operations have been performed, automatically creating a gredient 
tree and backpropagating it when the `backward()` function is called. 

The following code shows a simple implementation of the tensor class to solve a linear algebra problem. If we consider a 
data matrix $A$, a target vector $y$, and a parameter vector $x$, we solve $Ax=y$ for the parameters using gradient descent.

``` cpp
// Define an input dataset, in this case a 4x4 matrix.
float data[4][4] =
{
	{ +0.2f, +1.0f, +0.2f, -2.0f },
	{ +0.5f, -1.1f, +1.4f, +2.0f },
	{ +0.2f, -1.4f, +0.7f, -2.0f },
	{ +3.0f, +1.0f, -2.1f, -1.2f },
};
// Define a target output.
float target[4] = { +1.3f, -2.5f, +0.2f, -0.1f };

// Create a tensor with the data.
Tensor data_tensor(Shape{ 4, 4 }, "cpu");
data_tensor.internal_set_vector({ 0 }, data[0]);
data_tensor.internal_set_vector({ 1 }, data[1]);
data_tensor.internal_set_vector({ 2 }, data[2]);
data_tensor.internal_set_vector({ 3 }, data[3]);
// Create a tensor with the target.
Tensor target_tensor(Shape{ 4 }, "cpu");
target_tensor.internal_set_vector({}, target);

// Create the tensor you want to train. Zero initializes.
Tensor parameters(Shape{ 4 }, "cpu", true /*requires grad*/);

// Repeat 10 epoch.
for (int epoch = 0; epoch < 10; epoch++)
{
	// Compute forward pass.
	Tensor preds = Functional::matmul(data_tensor, parameters);
	// Compute loss.
	Tensor loss = Functional::mean_squared_error(preds, target_tensor);

	// Backpropagate.
	parameters.zero_grad();
	loss.backward();
	// Gradient descent.
	const float learning_rate = 0.1f;
	parameters.internal_add(-learning_rate * parameters.gradient());

	// Log loss.
	printf("Epoch %i Finished | Loss: %.4f\n", epoch, loss.item());
}
// Print final output.
printf("\nFinal Parameters:\n%s\n", parameters.str());
```

From this code we obtain the following console output:

``` bat
Epoch 0 Finished | Loss: 1.9975
Epoch 1 Finished | Loss: 0.6437
Epoch 2 Finished | Loss: 0.3287
Epoch 3 Finished | Loss: 0.1824
Epoch 4 Finished | Loss: 0.1077
Epoch 5 Finished | Loss: 0.0668
Epoch 6 Finished | Loss: 0.0432
Epoch 7 Finished | Loss: 0.0292
Epoch 8 Finished | Loss: 0.0207
Epoch 9 Finished | Loss: 0.0153

Final Parameters:
Shape:    (4)
Device:   cpu
Operator: None
Grad:
(+0.1446, -0.0588, +0.1176, -0.0043)
Data:
(-0.5934, +0.4394, -0.2778, -0.5795)
```

Which shows a fast loss decrease, cleanly converging to the solution of the equation.

## Neural Networks Extension

The previous example is just a very simple implementation of the class, but since one of the objectives of
the library was to support modern architecture it needed some wrappers to support better model building. 
This is where the `macrograd_nn.h` classes in handy.

There you will find support for model building and training through familiar classes. Starting by the `Module`
class, this class allows you to create simple modules and combinations of them. For example this is the `Linear`
module as defined inslde the `transformer.h` header:

``` cpp
// Linear module class. Consists of a single linear layer with or without bias. Defines 
// the projection matrix and optionally the bias vector. During the forward pass it calls 
// the function matmul, fused with the bias accordingly.
class Linear : public Module
{
private:
    // Class parameter storage.
	Tensor matrix;
	Tensor bias;

    // Bias internal flag.
	bool _has_bias;

public:
    // Linear module constructor. Initializes the matrix using Xavier uniform, and 
    // optionally creates the bias vector. Adds the parameters to the internal list.
	Linear(unsigned fan_in, unsigned fan_out, bool has_bias = true) : _has_bias{ has_bias }
	{
        // Create matrix of shape (fan_in, fan_out).
		matrix = Tensor(Shape{ fan_in, fan_out });
        // Initialize with Xavier uniform.
        float xavier = sqrtf(6.f / (fan_in + fan_out));
        Initialization::uniform(matrix, -xavier, xavier);
        // Add matrix to the parameter list.
		add_parameter(matrix);

        // If has bias zero initialize it and add it to the list.
		if (has_bias)
		{
			bias = Tensor(Shape{ fan_out, });
			add_parameter(bias);
		}
	}

    // Linear forward pass, applies a matrix multiplication, with fused bias 
    // depending on class construction. Returns the resulting tensor.
	Tensor forward(const Tensor& in) const override
	{
        // If has bias return matmul with fused bias.
		if (_has_bias)
			return Functional::matmul(in, matrix, bias); // (..., fan_in) -> (..., fan_out)
        // Else return simple matrix multiplication.
		return Functional::matmul(in, matrix); // (..., fan_in) -> (..., fan_out)
	}
};
```

In the `macrograd_nn.h` header you can also find `Optim`, the base optimizer class, and `Sched`, the base 
scheduler class. These include implementations on optimizers and schedulers used in modern training, like
`Optimizer::SGD`, `Optimizer::AdamW`, `Scheduler::CosineLR`, and other schedulers.

Then a standard training routine becomes similar to the following code:

``` cpp
// Initialize your model and send to device.
MyModule model(...); model.to(device);

// Initialize optimizer and scheduler.
Optimizer::AdamW optimizer(model, initial_lr, weight_decay);
Scheduler::CosineLR scheduler(optimizer, initial_lr, final_lr, total_epoch);

// Create index vector for random permutations.
VectorInt randperm(0, train_size, device);

// Iterate through epoch.
for (int epoch = 0; eposh < total_epoch; epoch++)
{
  // Randomly permute data.
  Random::shuffle(randperm);
  Tensor permuted_data = data[randperm];
  Tensor permuted_labels = lables[randperm];

  // Set model for training.
  model.train(); model.with_grad();

  // Iterate through batches.
  for (int batch = 0; batch < num_batches; batch++)
  {
    // Create batch.
    generate_batch(permuted_data, permuted_labels, batch, &batch_data, &batch_labels);

    // Compute forward pass and loss.
    Tensor logits = model(batch_data);
    Tensor loss = Functional::cross_entropy_loss(logits, batch_labels);

    // Bacpropagate and step optimizer.
    optimizer.zero_grad();
    loss.backward();
    optimizer.step();
  }

  // Step scheduler.
  scheduler.step();

  // Log epoch + validation.
  model.eval(); model.no_grad();
  ...
}
// Finish training.
...
```

The following section covers the demo with concrete implementation examples.

## My Mini Shakespeare

What better demo for a library named Macrograd than following the steps of [Nano-GPT](https://github.com/karpathy/nanoGPT) 
and training it on the tiny Shakespeare dataset. This led to the creation of `MiniShakespeare`, the demo model for 
this library, who produces literature like the following passage:

```
QUEEN ELIZABETH:
So this speak of wars in your life,
Who do the please of your swords and discover
Strike the king and all this counterfeit
Than I know my content for seems are not hers.

KING RICHARD II:
He is our note the treasure of the common is the news,
And honour a care a pale, for I would seem
I cannot good for your care, I have power
To the love of the wings, that are for me.

CLARENCE:
Here is no known shall be much protest,
Great, Lord Angelo
```

lol `¯\_(ツ)_/¯`. The model can produce similar dialogue for hours and is quite mesmorizing even though it makes 
almost no sense. It consists on a transformer stack, with $6$ layers, $128$ embedded dimension, $4$ heads and 
$512$ feed-forward hidden dimension. During training the model has achieved a validation loss of $1.4954$.
Rivalring the Nano-GPT loss of $1.4697$ despite being much smaller in size.

The files for the demo are organized as follows:

- `transformer.h`: This file contains all the definitions of the `MiniShakespeare` submodules. I recommend
  taking a look at it for examples on how to build your modules with Macrograd. In particular it defines:
  `Linear`, `Dropout`, `FeedForward`, `MultiHeadSelfAttention`, `LayerNorm`, `Layer`, `Transformer`,
  `Embedding` and `PositionalEmbedding`.
- `tokenizer.h`: The model uses character tokenization, which is defined inside of the `Tokenizer` class.
  The class holds two arrays fo point between tokens and characters in both directions. It provides some
  basic functions for converting characters/strings into token integers/vectors and vice-versa. You can
  check the file itself for details.
- `mini_shakespeare.h`: This file contains the `MiniShakespeare` module itself, combining the `Transformer`,
  `Embedding` and `PositionalEmbedding` modules. It defines the forward pass as well as some generative methods
  for easy user interaction, these being `add_one_token()` and `add_one_character`, which are used by the main
  file for the generative demo.
- `training.h`: This file contains the training routine for the model. This includes a basic class for dataset
  loading and some training functions governed by the following parameters:

``` cpp
// Training descriptor for the tiny_shakespeare dataset.
// Contains all training hyperparameters that can be modified.
struct ShakespeareTrainingDesc
{
	char device[16]       = "cuda";
	char load_path[128]   = "";
	char save_path[128]	  = "aprentice.mg";
	char log_path[128]	  = "training_log.txt";
	int warmup_steps      = 300;
	int total_steps       = 10000;
	int log_every         = 50;
	int batch_size        = 128;
	int micro_batch_size  = 16;
	int context_lentgh    = 256;
	float train_split	  = 0.9f;
	int eval_micro_batch  = 16;
	float initial_lr      = 0.001f;
	float final_lr        = 0.0002f;
	float weight_decay    = 0.05f;
	float dropout_rate    = 0.20f;
};
```
- The values you see inside the descriptor are the ones used to train the final model, and if run will 
  reproduce the same training run.
- `main.cpp`: Main file, defines some basic model writing functions and allows to switch between training
  and generation.

Last but not least we also have the `MNIST.h` file. This I was using for library debugging but then I did not
want to delete it, so it stayed as part of the library. It loads the MNIST dataset and trains a small MLP on it
achieving accuracies above $98%$ almost instantly even on my computer. Feel free to explore it for a more basic 
training implementation.

## CUDA Backend

## Error Handling

## Limitations

This library is just an educational project for myself and my own understanding, and in its current 
form it should not be used for production training of large models. These are some of the limitations
of Macrograd:

- Only supports `float32` tensors, with no integration of different precisions.
- It is not fully optimized for CUDA or CPU computations, mostly being outperformed by major libraries.
- It does not support multiple device settings, although it allows some customization for CUDA.
- All dimensions are stored in 4 Byte integers, meaning that large scale models could surpass the integer capacity and error.
- The amount of functionalities is severely limited compared to major libraries.

Despite this just being an educational project I still encourage people to use it and have fun with it.
I genuenly think it provides a simple interface to play with tensors in C++. But for real large scale model 
training, [PyTorch](https://github.com/pytorch/pytorch) or other major autograd libraries should be used instead.

## License

Macrograd is released under the MIT License. See the LICENSE file for details.

## Contact Information

You can contact me with questions, offers and suggestions at my email:

[miguel.nasarre.budino@gmail.com](mailto:miguel.nasarre.budino@gmail.com)
