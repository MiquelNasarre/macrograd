# Macrograd

Macrograd is an educational autograd library, built in C++ with CUDA acceleration. It was inspired by Andrej Karpathy's video
[The spelled-out intro to neural networks and backpropagation](https://www.youtube.com/watch?v=VMj-3S1tku0&t=7109s),
where he presents his library, [micrograd](https://github.com/karpathy/micrograd).

> *Micrograd is all you need to train neural networks, and everything else is just efficiency.*
> *So you'd think that micrograd would be a very complex piece of code, and that turns out to not be the case...*

\
When I first watched the micrograd introduction video, it felt like a breath of fresh air. Andrej's explanations 
were simple and approachable, and allowed anyone to get a clearer understanding of what autograd libraries actually are,
using only basic mathematical concepts.

Ever since watching that video, I wanted to build Macrograd, an autograd library with the same overall simplicity, 
but with enough performance to train real-world neural networks. His quote up top turned out to be true, but the 
“efficiency” part also introduces much more complexity and many more technical challenges, and understanding them is 
essential for building and optimizing today's neural networks.

## Build & Run

The project uses [CMake](https://cmake.org/) and is split into two parts: the library itself in `macrograd/`, and 
the demo in `mini_shakespeare/`.

To get the most out of the library, you will need a CUDA-capable GPU. The CMake file should handle 
selecting the correct architecture based on your device, but you can also change it manually.

To get started, build the solution and run the demo. In the main file, you can switch between the generation function, which 
produces text, and the training function, which reproduces the final training run. Later we will discuss the demo in more detail.

In Windows, you can use the following command line:

```
git clone https://github.com/MiquelNasarre/macrograd
cd macrograd
cmake --preset ninja-multi
cmake --build --preset release
cd mini_shakespeare
bin\release\mini_shakespeare
```

To create your own projects, just link to the library binaries `macrograd.lib`/`macrograd_d.lib` and include the headers needed
for your specific implementation.

## Library Capabilities & Structure

Some of the features Macrograd offers are:

- `Tensor` objects with arbitrary shapes, stored on either the CPU or CUDA devices.
- Several tensor operations that support gradient propagation and CUDA acceleration.
- Automatic differentiation of arbitrary expressions via the `backward()` function.
- Neural network abstraction classes, including `Module`, optimizers and schedulers.
- Auxiliary `Shape` class, with a templated constructor that simplifies usage, for example: `Shape{ 128, 32, 4, 64 }`.
- Auxiliary `VectorInt` class, for CPU and CUDA integer vectors, mainly used for permutations and labeling.
- Small language-model demo, using the `Transformer` module from `transformer.h` and character tokenization.

The library code is organized as follows:

- `macrograd.h`: Main library header. It defines the `Shape`, `VectorInt`, and `Tensor` classes, and some other functions
  within the `Functional`, `Random`, `Initialization` and `Cuda` namespaces. Its source code can be found in `tensor.cpp`
  and `tensor_ops.cpp`.
- `macrograd_nn.h`: Neural network extension header, contains the `Module`, `Optim` and `Sched` base classes, and some
  specific class implementations within the `Optimizer` and `Scheduler` namespaces. Its source code can be found in `nn.cpp`.
- `macrograd_error.h`: Centralized error-handling utilities for the library. Defines the `MacrogradError` class, and the
  `MACROGRAD_ERROR` and `MACROGRAD_CHECK` macros.
- `cuda_backend.h`: Internal CUDA operations used to perform the library operations on the GPU. Contains the class `MemPool`,
  and the namespaces `cuda_methods` and `kernel_ops`. Its source code can be found in `cuda_backend.cu`.

The headers also contain detailed comments for readers who want to explore the implementation more deeply.
The following sections explore all of its functionalities in a bit more detail.

## Tensors

The main object of the library is the `Tensor` class. Similar to a [PyTorch](https://github.com/pytorch/pytorch) tensor, it is 
an arbitrarily shaped collection of numbers with multiple operations defined between them, which can live on either the CPU or the GPU.

![Tensor illustration image from PyTorch repository](https://raw.githubusercontent.com/pytorch/pytorch/main/docs/source/_static/img/tensor_illustration.png)

### How they work

In this implementation, tensor data is always stored in a contiguous array of floating-point values. The pointer holding this list 
can either be a CPU or a CUDA pointer. This is found inside `TensorInternals`, which can be shared between multiple tensor instances.
A `TensorInternals` struct is not deleted until all instances referencing that data are deleted.

Each individual tensor instance contains a `_view` to that data, meaning a way of structuring it, for example a pointer containing $36$
different values can be viewed in shapes like: $(36,)$, $(6, 6)$, $(2, 3, 6)$, $(9, 4)$, etc. This view belongs to each tensor instance, so 
creating a tensor with a different view on the same data does not require copying the data or any new allocations.

Individual tensors also have a flag called `_requires_grad`, when that flag is active, any operation where that tensor is involved that 
supports gradient will store in the output internals a `TensorOp`. This describes the operation used to create it and its derivative
and references the tensors involved in its creation. These references count toward the instance count of the tensors, meaning that 
the outputs of intermediate operations will persist until the gradient tree is destroyed, since they are needed for backpropagation.

When `backward()` is called on a single element tensor, a gradient tensor is generated for all the tensors involved in the operation that
had gradient enabled, and the gradient is propagated through their `TensorOp`. A simple example looks as follows:

``` cpp
// Create single element tensors.
Tensor a(Shape{ 1 }, "cpu", true /*requires_grad*/); a.internal_fill(2.0f);
Tensor b(Shape{ 1 }); b.internal_fill(-1.5f);

// Define an operation.
Tensor c = a * (a + b);

// Compute derivative with respect to a.
c.backward();

// Print results.
printf("Tensor a:\n%s\n\nTensor b:\n%s\n\nTensor c:\n%s\n", a.str(), b.str(), c.str());
```

We obtain the following output:

```
Tensor a:
Shape:    (1)
Device:   cpu
Operator: None
Grad:
(+2.5000)
Data:
(+2.0000)

Tensor b:
Shape:    (1)
Device:   cpu
Operator: None
Grad:     None
Data:
(-1.5000)

Tensor c:
Shape:    (1)
Device:   cpu
Operator: Multiplication
Grad:
(+1.0000)
Data:
(+1.0000)
```

If we compute the derivative analytically, we obtain $\frac{dc}{da} = 2a + b = 2.5$, the same result.
As you can see the operator stored in $c$ is the last product, while the intermediate sum is also 
stored inside the gradient graph, even though no explicit variable is assigned to it.

### What can you do with them

These are all the tensor operations supported by Macrograd:

- Standard operators with broadcasting and reduction rules: `+`, `-`, `*`, `/`, `+=`, `-=`, `*=`, `/=`.
- View operations (do not allocate data): `=`, `view`, `squeeze`, `unsqueeze`, `flatten`.
- Reshaping operations (do allocate data): `transpose`, `subset`, `modify`, `repeat`, `operator[]` (permutation).
- Element-wise operations: `sign` (no grad), `exp`, `log`, `relu`, `silu`, `gelu`, `sigmoid`, `tanh`, `sqrt`, `square`, `pow`.
- Dimension-wise operations: `sum`, `mean`, `var`, `std`, `softmax`, `max`, `min`, `argmax` (no grad), `argmin` (no grad).
- Element-wise comparisons (output with same shape as input, no gradient): `>`, `<`, `>=`, `<=`, `==`, `!=`.
- Functional namespace: `matmul`, `cat`, `mean_squared_error`, `cross_entropy_loss`, `negative_log_likelihood`, `one_hot` (no grad), `causal_mask` (no grad).

These functions, and others, can be found in `macrograd.h`, feel free to look through the comments to understand how to 
use each one of these functions.

### Simple example

The following code shows a use case of the `Tensor` class to solve a linear algebra problem. If we consider a data 
matrix $A$, a target vector $y$, and a parameter vector $x$, we solve $Ax=y$ for the parameters using gradient descent.

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

// Repeat for 10 epochs.
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

```
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

This shows a rapid decrease in loss, cleanly converging to the solution of the equation.

## Neural Networks Extension

The previous example is just a very simple usage case of the class, but since one of the objectives of
the library was to support modern architectures, it needed some wrappers to allow for better model building. 
This is where the classes in `macrograd_nn.h` come in handy.

There you will find support for model building and training through familiar classes. Starting with the `Module`
class, this class allows you to create neural network modules and combinations of them. For example this is the `Linear`
module as declared in the `transformer.h` demo header:

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
scheduler class. These include implementations of optimizers and schedulers used in modern training workflows, 
like `Optimizer::SGD`, `Optimizer::AdamW`, `Scheduler::CosineLR`, and other schedulers.

Then a standard training routine becomes similar to the following code:

``` cpp
// Initialize your model and send to device.
MyModule model(...); model.to(device);

// Initialize optimizer and scheduler.
Optimizer::AdamW optimizer(model, initial_lr, weight_decay);
Scheduler::CosineLR scheduler(optimizer, initial_lr, final_lr, total_epoch);

// Create index vector for random permutations.
VectorInt randperm(0, train_size, device);

// Iterate through epochs.
for (int epoch = 0; epoch < total_epoch; epoch++)
{
  // Randomly permute data.
  Random::shuffle(randperm);
  Tensor permuted_data = train_data[randperm];
  Tensor permuted_labels = train_labels[randperm];

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

    // Backpropagate and step optimizer.
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

What better demo for a library named Macrograd than following the steps of [nanoGPT](https://github.com/karpathy/nanoGPT) 
and training it on the tiny Shakespeare dataset. This led to the creation of `MiniShakespeare`, the demo model for 
this library, which produces literature like the following passage:

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

lol `¯\_(ツ)_/¯`. The model can produce similar dialogue for hours and is quite mesmerizing even though it makes 
almost no sense. It consists of a transformer stack, with $6$ layers, $128$ embedding dimension, $4$ heads and 
$512$ feed-forward hidden dimension. During training, the model achieved a validation loss of $1.4954$,
rivaling nanoGPT's $1.4697$ despite being much smaller in size.

The files for the demo are organized as follows:

- `transformer.h`: This file contains all the definitions of the `MiniShakespeare` submodules. I recommend
  taking a look at it for examples on how to build your modules with Macrograd. In particular it defines:
  `Linear`, `Dropout`, `FeedForward`, `MultiHeadSelfAttention`, `LayerNorm`, `Layer`, `Transformer`,
  `Embedding` and `PositionalEmbedding`.
- `tokenizer.h`: The model uses character tokenization, which is provided by the `Tokenizer` class.
  The class holds two arrays to point between tokens and characters in both directions. It provides some
  basic functions for converting characters/strings into token integers/vectors and vice-versa. You can
  check the file itself for details.
- `mini_shakespeare.h`: This file contains the `MiniShakespeare` module itself, combining the `Transformer`,
  `Embedding` and `PositionalEmbedding` modules. It defines the forward pass as well as some generative methods
  for easy user interaction, namely `add_one_token()` and `add_one_character()`, which are used in the main
  file for the generation demo.
- `training.h`: Describes the training routine for the model. This includes a basic class for dataset
  loading and some training functions governed by the following parameters:

``` cpp
// Training descriptor for the tiny_shakespeare dataset.
// Contains all training hyperparameters that can be modified.
struct ShakespeareTrainingDesc
{
	char device[16]       = "cuda";
	char load_path[128]   = "";
	char save_path[128]	  = "apprentice.mg";
	char log_path[128]	  = "training_log.txt";
	int warmup_steps      = 300;
	int total_steps       = 10000;
	int log_every         = 50;
	int batch_size        = 128;
	int micro_batch_size  = 16;
	int context_length    = 256;
	float train_split	  = 0.9f;
	int eval_micro_batch  = 16;
	float initial_lr      = 0.001f;
	float final_lr        = 0.0002f;
	float weight_decay    = 0.05f;
	float dropout_rate    = 0.20f;
};
```
- The values you see inside the descriptor are the ones used to train the final model, and if used as-is,
  will reproduce the same final training run.
- `main.cpp`: Main file, defines some basic text generation functions, and allows to switch between training
  and generation.

Last but not least we also have the `MNIST.h` file. I originally used it for library debugging, but then I did not
want to delete it, so it stayed as part of the library. It loads the MNIST dataset and trains a small MLP on it
achieving accuracies above $98$% almost instantly, even on my computer. Feel free to explore it for a more basic 
training implementation.

## CUDA Backend

It would be impossible to train transformer architectures without GPU acceleration. This is why the library 
supports storing its tensors on the GPU and performing operations with CUDA. Those functions are declared
inside the `cuda_backend.h` header, and are organized as follows:

- `MemPool` class: Handles all CUDA allocations and frees inside the library. Uses the functions `cudaMallocAsync`
  and `cudaFreeAsync`. Some details of the pool can be controlled by the user, most importantly the allocation threshold,
  which determines when the memory pool attempts to return memory to CUDA, by default it is set to $80$% of the device
  memory.
- `cuda_methods` namespace: Contains some general CUDA functions that do not require kernel launching. These include memory
  transfers, zeroing memory, stream synchronization and a function to set a single value to `1.0f` for the `backward()` call.
- `kernel_ops` namespace: Contains all other implementations of tensor operations that require CUDA, mostly requiring kernels.
  These include all `Tensor` operations, RNG functions, and some `VectorInt` functions.

All CUDA operations run on a single stream per device, meaning they run asynchronously, which improves performance. Some 
operations do force synchronization though, these are all operations that require data transfers between the two devices.

The library uses `cuRAND` for random number generation, all CUDA functions that need it are linked to the same 
`curandGenerator_t` instance. The seed for the generation can be set via the function `Random::set_cuda_seed()`. 
Therefore all outcomes of the library are purely deterministic.

All tensor operations run on kernels implemented by me except for matrix multiplication. Since this operation represents
most of the compute used in common neural network architectures, I felt the need to use the best tools to ensure efficiency.
For that reason `cuBLAS` is used, with cached operation information to ensure maximum throughput for matrix multiplications
with the same shapes, common during training. For details on its implementation you can check the definitions inside 
`cuda_backend.cu` for the functions `kernel_ops::matmul()` and `kernel_ops::matmul_bias()`.

## Error Handling

When operating with tensors, defining a proper error detection pipeline is essential. For example, many times you will
accidentally do operations with incompatible shapes, and in those cases, you would expect the operation to fail and display 
an error in the console explaining the exact reason for the crash.

This is what the `MacrogradError` class does. All operations inside the library perform checks using the `MACROGRAD_CHECK` 
and `MACROGRAD_ERROR` macros. These checks are always active regardless of configuration, since the overhead they introduce is 
negligible compared to the operations they guard, and their information is really valuable.

The following code shows a basic error message example:

``` cpp
Tensor A(Shape{ 6, 4 });
Tensor B = A.view({ -1, -1, 4 });
```

This produces the following console output:

```
Macrograd Error Occurred:
Line: 360
File: tensor_ops.cpp
Message: Ambiguous shape found inside a view call.
Make sure you only have one unknown dimension marked as -1 to avoid ambiguity.
Old Shape: (6, 4) | View Input Shape: (-1, -1, 4).
```

These macros can and should also be used inside your own implementations. For example, this is the forward pass of the 
`LayerNorm` class inside `transformer.h`:

``` cpp
    // Layer Normalization forward pass. Expects a tensor with the last dimension size set at 
    // construction. Normalizes along the last layer and applies the learned transformation.
    Tensor forward(const Tensor& in) const override
    {
        // Sanity check.
        MACROGRAD_CHECK(in.size(-1) == layer_dim,
            "Invalid tensor shape received inside a LayerNorm forward pass.\n"
            "Expected last dimension of size %i, but got input shape %s.",
            layer_dim, in.shape().str()
        );

        // Get normalization tensors.
        Tensor mean = in.mean(-1, true);   // (..., 1)
        Tensor var = in.var(-1, true);     // (..., 1)

        // Normalize input.
        Tensor norm_in = (in - mean) / (var + eps).sqrt(); // (..., E)

        // Apply gamma and beta.
        Tensor out = norm_in * gamma + beta; // (..., E)

        // Return output.
        return out;
    }
```

And if used incorrectly you can get errors with rich information, like the following:

```
Macrograd Error Occurred:
Line: 301
File: transformer.h
Message: Invalid tensor shape received inside a LayerNorm forward pass.
Expected last dimension of size 128, but got input shape (16, 48, 127).
```

## Limitations

This library is just an educational project for myself and my own understanding. In its current 
form, it should not be used for production training of large models. These are some of the limitations
of Macrograd:

- Only supports `float32` tensors, with no integration of different precisions.
- Does not support tensors with zero-sized dimensions. Some operations will fail when they are encountered.
- It is not fully optimized for CUDA or CPU computations, mostly being outperformed by major libraries.
- It does not support multi-device execution, although it allows some customization for CUDA.
- All dimensions are stored in 4-byte integers, meaning that large-scale models could exceed integer capacity and cause errors.
- The number of features is severely limited compared to major libraries.

Despite this being just an educational project, I still encourage people to use it and have fun with it.
I genuinely think it provides a simple interface to play with tensors in C++. But for real large-scale model 
training, [PyTorch](https://github.com/pytorch/pytorch) or other major autograd libraries should be used instead.

## License

Macrograd is released under the MIT License. See the LICENSE file for details.

## Contact Information

You can contact me with questions, offers, and suggestions at my email:

[miguel.nasarre.budino@gmail.com](mailto:miguel.nasarre.budino@gmail.com)
