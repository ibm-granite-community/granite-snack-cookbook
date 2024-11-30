# Post training quantization of Granite-3.0-8B-Instruct in Python with watsonx
**Author**: Joshua Noble

## What is post training quantization? 

Quantization of large language models (LLMs) is a model optimization technique that reduces memory space and latency by sacrificing some model accuracy. Large transformer-based models such as LLMs often require significant GPU resources to run. In turn, a quantized model can allow you to run machine learning inference on limited GPUs or even on a CPU. Frameworks such as TensorFlow Lite (tflite) can run quantized TensorFlow models on edge devices including phones or microcontrollers. In the era of larger and larger LLMs, quantization is an essential technique during the training, fine tuning and inference stages of modeling. Quantization is especially helpful for users who want to run models locally on limited hardware machines. Low-resource hardware that has a hardware machine learning accelerator can also run quantized models very efficiently.

## Quantization methods

Quantization is the process of mapping input values from a large set of continuous elements to a smaller set with a finite number of elements. Quantization methods have evolved rapidly and been an area of active research. For instance, a simple algorithm might be integer quantization, which is simply scaling 32-bit floating point (f32) numbers to 8-bit integers (int8). This technique is often called zero-point quantization. More sophisticated techniques use FP8, an 8-bit floating point with a dynamic range that can be set by the user. In this tutorial, we'll use k-means quantization to create very small models. That saves us from needing to do model calibration or the time-intensive step of creating an importance matrix that defines the importance of each activation in the neural network.

We'll focus on post training quantization (PTQ) which focuses on decreasing the precision (and thus resource demands) after the model is trained. Quantization-Aware Training (QAT) is a common quantization technique for mitigating model accuracy and perplexity degradation that arises from quantization but is a more advanced technique with more limited use cases. In particular, we'll use k-means quantization via llama.cpp, an open source library that quantizes PyTorch models.

When working with LLMs, model quantization allows us to convert high-precision floating-point numbers in the neural network layers to low-precision numbers that consume much less space. We'll be converting models to GPT-Generated Unified Format (GGUF) to run them efficiently in constrained resource scenarios. GGUF is a binary format optimized for quick loading and saving of models that makes it efficient for inference purposes. It achieves this efficiency by combining the model parameters (weights and biases) with more metadata for effective execution. Because it’s compatible with various programming languages such as Python and R and supports fine tuning so users can adapt LLMs to specialized applications, it has become a popular format.

In this tutorial, we’ll quantize the IBM® Granite-3.0-8B-Instruct model in a few different ways to show the size of the models and compare how they perform on a task. To view more Granite tutorials, check out the [IBM Granite Community](https://github.com/ibm-granite-community). This tutorial is also available on [Github](https://github.com/IBM/ibmdotcom-tutorials/tree/main/generative-ai/post-training-quantization.md).

## Step 1. Set up your enviroment

This can be done in either a terminal on OSX or Linux or in VS Code on Windows.

First, we need to create a virtual environment in Python that we can use to save all of our libraries:

```
python3 -m venv .
source ./bin/activate
```

Next, we'll want to install the HuggingFace Hub library so that we can use to download the Granite Model files

```
./bin/pip install huggingface_hub
```

Next, either save the following script to a file and run it, or simply start a Python3 session and run it there:

```
from huggingface_hub import snapshot_download

snapshot_download("ibm-granite/granite-3.0-8b-instruct", local_dir="granite-3.0-8b-instruct")
```

Now we can copy the files into our local directory for easier access

Next up, we need to install llama.cpp at a system level. The instructions here to build from source are rather complex but very well documented [here](https://github.com/ggerganov/llama.cpp/blob/master/docs/build.md)

Alternatively on OSX you can use homebrew:

```
brew install llama.cpp
```

On Mac and Linux, the Nix package manager can be used:

```
nix profile install nixpkgs#llama-cpp
```

You can also look for prebuilt binaries at the [Github Releases page](https://github.com/ggerganov/llama.cpp/releases)

Once we have llama.cpp installed, we can install the libraries to run the llama-cpp scripts in the virtual environment:

```
./bin/pip install 'llama-cpp-python[server]'
```

## Step 2. Prepare to quantize

We need to get the entire repository for llama.cpp in order to convert models to GGUF format.

```
git clone https://github.com/ggerganov/llama.cpp
```

Now we install libraries that we'll need in order to run the GGUF conversion:

```
./bin/pip install -r llama.cpp/requirements.txt
```

Now we're ready to convert the model to gguf using a script inside the repository:

```
./bin/python3 llama.cpp/convert_hf_to_gguf.py granite-3.0-8b-instruct
```

This gives us a new GGUF file based on our original model files.

Now we're ready to quantize. Common quantization schemes that are supported by GGUF include:

2-bit quantization: This offers the highest compression, significantly reducing model size and inference speed, though with a potential impact on accuracy.
4-bit quantization: This balances compression and accuracy, making it suitable for many practical applications.
6-bit quantization: This quantization setting provides higher accuracy than 4 or 2, with some reduction in memory requirements that can help in running it locally.

For a the smallest possible model we can use 2 bit quantization:

```
/opt/homebrew/bin/llama-quantize \
    granite-3.0-8b-instruct/granite-3.0-8b-instruct-F16.gguf \
    granite-3.0-8b-instruct/granite-8B-instruct-Q2_K.gguf Q2_K
```

For a medium sized quanitized model, we can use 4 bit quantization:

```
/opt/homebrew/bin/llama-quantize \
    granite-3.0-8b-instruct/granite-3.0-8b-instruct-F16.gguf \
    granite-3.0-8b-instruct/granite-8B-instruct-Q4_K_M.gguf Q4_K_M
```

For a larger quanitized model on a machine with more resources, we can use 6 bit quantization:

```
/opt/homebrew/bin/llama-quantize \
    granite-3.0-8b-instruct/granite-3.0-8b-instruct-F16.gguf \
    granite-3.0-8b-instruct/granite-8B-instruct-Q6.gguf Q6_K
```

Here's a size comparison

| Param  | Size    |
|--------|---------|
| Q2_K   | 3.17 GB |
| Q4_K_M | 5.06 GB |
| Q6_K   | 6.87 GB |


Each of these steps may take up to 15 minutes but when they're done we have multiple versions of the model that we can use to compare.

## Step 3. Running the model

We could run the model in llamacpp with the following command:

```
llama-server -m granite-3.0-8b-instruct/granite-8B-instruct-Q4_K_M.gguf --port 8080
```

This allows you to open a webpage at `localhost:8080` which hosts an interactive session which runs Granite Instruct as a helpful assistant. 

We could also add the file to ollama. To do this, first we need to create a modelfile:

```
# Modelfile
FROM "<PATH_TO_MODEL>/granite-8B-instruct-Q4_K_M.gguf"

PARAMETER stop "<|im_start|>"
PARAMETER stop "<|im_end|>"

TEMPLATE """
<|im_start|>system
{{ .System }}<|im_end|>
<|im_start|>user
{{ .Prompt }}<|im_end|>
<|im_start|>assistant
"""
```

Save this file as `GCI_8b_modelfile_Q4`. Start ollama:

```
ollama serve
```

Then load the model into ollama:

```
ollama create Granite_Instruct:8b_Q4 -f GCI_8b_modelfile_Q4
```

Now you're ready to run it:

```
ollama run Granite_Instruct:8b_Q4
```

We repeat the process with two other model files, one for the 2 bit and one for the 6 bit quantization:

```
# Modelfile
FROM "<PATH_TO_MODEL>/granite-8B-instruct-Q2.gguf"

PARAMETER stop "<|im_start|>"
PARAMETER stop "<|im_end|>"

TEMPLATE """
<|im_start|>system
{{ .System }}<|im_end|>
<|im_start|>user
{{ .Prompt }}<|im_end|>
<|im_start|>assistant
"""
```

Save this as `GCI_8b_modelfile_Q2` and load it into ollama with the following command:

```
ollama create Granite_Instruct:8b_Q2 -f GCI_8b_modelfile_Q2
```

One last modelfile for the Q6 version:

```
# Modelfile
FROM "<PATH_TO_MODEL>/granite-8B-instruct-Q6.gguf"

PARAMETER stop "<|im_start|>"
PARAMETER stop "<|im_end|>"

TEMPLATE """
<|im_start|>system
{{ .System }}<|im_end|>
<|im_start|>user
{{ .Prompt }}<|im_end|>
<|im_start|>assistant
"""
```

Save this as `GCI_8b_modelfile_Q6` and load into ollama with the following command:

```
ollama create Granite_Instruct:8b_Q6 -f GCI_8b_modelfile_Q6
```

Now we're ready to compare our models.

## Step 4. Comparing the models

Let's compare how the 2 bit version does to the 4 bit version with a prompt asking the model to fix bugs in code. First, we'll load the 2 bit version:

```
ollama run Granite_Instruct:8b_Q2
```

Once that's running, we can paste our prompt:

```
>>>Fix any errors in the following Python function and format it correctly:

import math

class Circle:
    def __init__(self, radius):
        self.radius = radius
    
    def area(self):
        return math.pi* self.radius** math.pi
    
    def perimeter(self):
        return math.pi* math.pi * self.radius

def print_circle_properties(circles):
    for circle in circles:
        print(f"Circle with radius {circle.radius}:")
        print(f"Area: {circle.area()}")
        print(f"Perimeter: {circle.perimeter()}\n")

circles = [Circle(3), Circle(5), Circle(7)]
print_circle_properties(circle)
```

With the 2 bit quantization, the model doesn't perform quite as well as we might hope. It misses that the equation for the area of a circle and the perimeters are both wrong. It also doesn't explain any of the errors.


	I have fixed the errors in your Python code and formatted it as follows:
	import math

	class Circle:
	    def __init__(self, radius):
	        self.radius = radius
	    
	    def area(self):
	        return math.pi* self.radius** math.pi
	    
	    def perimeter(self):
	        return math.pi * self.radius
	    
	def print_circle_properties(circles):
	    for circle in circles:
	        print(f"Circle with radius {circle.radius}:")
	        print(f"Area: {circle.area()}")
	        print(f"Perimeter: {circle.perimeter()}\n")
	    
	circles = [Circle(3), Circle(5), Circle(7)]
	print_circle_properties(circles)


Now let's load the Q4:

```
ollama run Granite_Instruct:8b_Q4
```

Copy/pasting in the previous prompt, we get a much better response, although the 3rd item doesn't quite make sense:


	There are a few errors in the provided Python code:
	1. In the `area()` method of the `Circle` class, the `math.pi* self.radius** math.pi` should be `math.pi * (self.radius ** 2)`. This is because the area of a circle is given by pi times the 
	square of its radius.
	2. The `perimeter()` method in the `Circle` class should return `2 * math.pi * self.radius`, since the perimeter of a circle is given by 2 * pi * radius.
	3. In the `print_circle_properties(circles)` function, the parameter `circle` should be `circle`.
	4. The last line of code should be `print_circle_properties(circles)`, not `print_circle_properties(circle)`.
	Here is the corrected code:
	```python
	import math

	class Circle:
	    def __init__(self, radius):
	        self.radius = radius

	    def area(self):
	        return math.pi * (self.radius ** 2)

	    def perimeter(self):
	        return 2 * math.pi * self.radius

	def print_circle_properties(circles):
	    for circle in circles:
	        print(f"Circle with radius {circle.radius}:")
	        print(f"Area: {circle.area()}")
	        print(f"Perimeter: {circle.perimeter()}\n")

	circles = [Circle(3), Circle(5), Circle(7)]
	print_circle_properties(circles)
	```


Finally, we can load the 6 bit quantized model:

```
ollama run Granite_Instruct:8b_Q6
```
This answer is much better than either of the two previous versions:

	The provided Python function contains two errors:

	1. In the `area` method, there is an extra `math.pi` before `self.radius**`. It should be `return math.pi * self.radius ** 2`.
	2. In the `perimeter` method, there are two `math.pi` multiplications. The correct formula for perimeter of a circle is `2 * math.pi * radius`. Therefore, it should be `return 2 * math.pi * 
	self.radius`.
	3. At the end of the code, the variable name is misspelled as `circle` instead of `circles`.

	Here's the corrected version of the code:
	```python
	import math

	class Circle:
	    def __init__(self, radius):
	        self.radius = radius

	    def area(self):
	        return math.pi * self.radius ** 2

	    def perimeter(self):
	        return 2 * math.pi * self.radius

	def print_circle_properties(circles):
	    for circle in circles:
	        print(f"Circle with radius {circle.radius}:")
	        print(f"Area: {circle.area()}")
	        print(f"Perimeter: {circle.perimeter()}\n")

	circles = [Circle(3), Circle(5), Circle(7)]
	print_circle_properties(circles)
	```


We can see that the 2 bit quantized model does save space but also is less adept at picking out errors in our code as well as fixing them and explaining them. The 4 bit model corrects all of the code errors but doesn't fully explain its instructions. The 6 bit model corrects all of the errors and explains those errors correctly and in greater detail.