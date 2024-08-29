import torch
import numpy as np
from torch import Tensor

""" PyTorch Documentation :
https://pytorch.org/docs/stable/index.html
"""

# print(torch.__version__)
print()


""" POSSIBLE ISSUES :

# 1. Tensors not right datatypes
# 2. Tensors not right shape
# 3. Tensors not on the right device

"""


"""
Introduction to Tensors
"""


""" SCALARS """

print("-- SCALAR --")
scalar = torch.tensor(7)
print("scalar =",scalar)

# get scalar dimension
print("scalar.ndim =",scalar.ndim)

# get tensor back as python int
print("scalar.item() =",scalar.item())
print()


""" VECTORS """

print("-- VECTOR --")
vector = torch.tensor([7, 7])
print("vector =",vector)

# get vector dimensions --> number of []
print("vector.ndim =",vector.ndim)

# get vector shape --> number of elements
print("vector.shape =",vector.shape)
print()


""" MATRICES """

print("-- MATRIX --")
MATRIX = torch.tensor([[7, 8],
                       [9, 10]])
print("MATRIX =",MATRIX)

# get MATRIX dimensions
print("MATRIX.ndim =",MATRIX.ndim)

# indexing MATRIX
print("MATRIX[0] =",MATRIX[0])
print("MATRIX[1] =",MATRIX[1])

# get MATRIX shape
print("MATRIX.shape =",MATRIX.shape)
print()


""" TENSORS """

print("-- TENSOR --")
TENSOR = torch.tensor([[[1, 2, 3],
                        [4, 5, 6],
                        [7, 8, 9]]])
print(TENSOR)

# get TENSOR dimension
print("ndim =",TENSOR.ndim)

# get TENSOR shape
print("shape =",TENSOR.shape)

# indexing TENSOR
print("TENSOR[0] =",MATRIX[0])
print("TENSOR[1] =",MATRIX[1])
print()


""" RANDOM TENSORS """

print("-- RANDOM TENSOR --")
random_tensor = torch.rand(3, 4)
print(random_tensor)

# get random tensor dimension
print("ndim =",random_tensor.ndim)
print()

print("-- RANDOM IMAGE TENSOR --")
# create a random tensor with similar shape to an image tensor
random_image_size_tensor = torch.rand(size=(224, 224, 3))    # height, width, color channels (RGB)
print("shape =",random_image_size_tensor.shape)
print("ndim =",random_image_size_tensor.ndim)
print()


""" ZEROS """

print("-- ZEROS --")
zeros = torch.zeros(size=(3, 4))
print(zeros)
print()


""" ONES """

print("-- Ones --")
ones = torch.ones(size=(3, 4))
print(ones)
print()


""" RANGE """

print("-- RANGE --")
one_to_ten = torch.arange(start=0, end=1000, step=77)
print(one_to_ten)
print()


""" TENSORS-LIKE """

print("-- TENSORS-LIKE --")
ten_zeros = torch.zeros_like(input=one_to_ten)
print(ten_zeros)
print()


""" DTYPE TENSORS """

print("-- DTYPE TENSOR --")
float_32_tensor = torch.tensor([13.0, 6.0, 9.0],
                               dtype=None,             # what datatype is the tensor (float16 or float32)
                               device=None,            # what device is your tensor on
                               requires_grad=False)    # track gradients or not with this tensor's operations
print(float_32_tensor)

# get float_32_tensor type
print(float_32_tensor.dtype)    # default is type 32

float_16_tensor = float_32_tensor.type(torch.float16)
print(float_16_tensor)

print("m =",float_16_tensor * float_32_tensor)
print()


""" GETTING INFORMATION ABOUT TENSORS """

print("-- ABOUT TENSORS --")
some_tensor = torch.rand(size=(3, 4))
print(some_tensor)

print("CUDA GPU:", torch.cuda.is_available())
some_tensor = some_tensor.cuda()

# get information about some_tensor
print(f"Datatype of some_tensor: {some_tensor.dtype}")
print(f"Shape of some_tensor: {some_tensor.shape}")    # équivalent to some_tensor.size()
print(f"Device tensor is on: {some_tensor.device}")
print()


""" MANIPULATING TENSORS """

print("-- MANIPULATING TENSORS --")
tensor = torch.tensor([1, 2, 3])

tensor_1: Tensor = tensor + 10
tensor_2: Tensor = torch.add(tensor, 10)
print(tensor_1)
print(tensor_2)

tensor_3 = tensor * 10
tensor_4 = torch.mul(tensor, 10)
print(tensor_3)
print(tensor_4)
print()


""" MATRIX MULTIPLICATION """

print("-- MATRIX MULTIPLICATION --")
tensor = torch.tensor([1, 2, 3])

# element-wise multiplication
tensor_1 = tensor * tensor
print(tensor_1)

# matrix multiplication
tensor_2 = torch.matmul(tensor, tensor)
print(tensor_2)

tensor_3 = tensor @ tensor    # @ est équivalent à torch.matmul
print(tensor_3)

print(torch.mm(torch.rand(5, 2), torch.rand(2, 2)))    # torch.mm is équivalent to torch.matmul

# get a matrix transpose
tensor_a = torch.tensor([[1, 8],
                         [2, 9],
                         [3, 10]])
tensor_a_t = tensor_a.T
print(tensor_a)
print(tensor_a_t)
print()


""" TENSOR AGGREGATION """

print("-- MIN - MAX - MEAN - SUM --")
x = torch.arange(0, 100, 10)
print(x)

# get the minimum value
min_1, min_2 = torch.min(x), x.min()
print(min_1, min_2)

# get the maximum value
max_1, max_2 = torch.max(x), x.max()
print(max_1, max_2)

# get the mean value
# tensor needs to be either floating point or complex dtypes
mean_1, mean_2 = torch.mean(x.type(torch.float32)), x.type(torch.float32).mean()
print(mean_1, mean_2)

# get the sum
sum_1, sum_2 = torch.sum(x), x.sum()
print(sum_1, sum_2)

# get the min/max indices
min_i, max_i = x.argmin(), x.argmax()
print(min_i, max_i)
print()


""" RESHAPING, VIEWING, STACKING """

print("-- RESHAPING - VIEWING - STACKING --")
x = torch.arange(1, 10)
print(x)

# reshape the tensor
x_reshaped = x.reshape(3, 3)
print(f"{x_reshaped}\n{x_reshaped.shape}")

# change the view
z = x.view(1, 9)
print(f"{z}\n{z.shape}")

# changing z changes x because a view of a tensor shares the same memory as the original
z[:, 0] = 5
print(f"{z}\n{x}")

# stack tensors on top of each other
x_stacked = torch.stack([x, x, x, x], dim=0)
print(x_stacked)
print()


""" SQUEEZING, PERMUTING """

print("-- SQUEEZING - PERMUTING --")
x = torch.arange(1, 10)
print(x)

x_reshaped = x.reshape(1, 9)
print(x_reshaped)
print(x_reshaped.shape)    # --> torch.Size([1, 9])

# squeeze the reshaped tensor
x_squeezed = x_reshaped.squeeze()    # removes all single dimensions
print(x_squeezed)
print(x_squeezed.shape)    # --> torch.Size([9])

# unsqueeze the tensor
x_unsqueezed = x_squeezed.unsqueeze(dim=1)
print(x_unsqueezed)
print(x_unsqueezed.shape)    # --> torch.Size([1, 9])

# permute the tensor
x_original = torch.rand(size=(224, 224, 3))    # height, width, color channels

# permute the tensor to rearrange the axis (or dim) order
x_permuted = x_original.permute(2, 0, 1)
print(x_original.shape)
print(x_permuted.shape)

# x_original and x_permuted share the same memory --> changing one changes both
print(x_original[0, 0, 0])
print(x_permuted[0, 0, 0])
print()


""" INDEXING """

print("-- INDEXING --")
x = torch.arange(1, 10).reshape(1, 3, 3)
print(f"x = {x}")

print(x[0])
print(x[0, 0])       # same as x[0][0]
print(x[0, 2, 2])    # same as x[0][2][2]
print(x[:, 0, 2])

print(x[:, 0])
print(x[:, 1, 1])
print(x[0, 1, 1])

print(x[0, 0, :])
print()


""" NUMPY -> TENSOR """

print("-- NUMPY -> TENSOR --")
array = np.arange(1.0, 8.0)         # numpy default dtype is float64
tensor = torch.from_numpy(array)    # changing the array doesn't change the tensor (no shared memory)
print(f"{array}\n{tensor}")

tensor = tensor.type(torch.float32)
print(tensor.dtype)
print()


""" TENSOR -> NUMPY """

print("-- TENSOR -> NUMPY --")
tensor = torch.ones(7)
array = tensor.numpy()         # tensor default dtype is float32
print(f"{tensor}\n{array}")    # changing the tensor doesn't change the array (no shared memory)

print(array.dtype)
print()


""" REPRODUCIBILITY"""

print("-- REPRODUCIBILITY --")
random_tensor_A = torch.rand(3, 4)
random_tensor_B = torch.rand(3, 4)

# create random but reproducible tensors
RANDOM_SEED = 42

torch.manual_seed(RANDOM_SEED)
random_tensor_C = torch.rand(3, 4)

torch.manual_seed(RANDOM_SEED)
random_tensor_D = torch.rand(3, 4)    # these 2 tensors are equal

print(random_tensor_C)
print(random_tensor_D)

print(random_tensor_C == random_tensor_D)
print()


""" AGNOSTIC CODE """

print("-- AGNOSTIC CODE --")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

# count devices
print(torch.cuda.device_count())    # 1
print()


""" PUTTING THINGS ON GPU """

print("-- PUTTING THINGS ON GPU --")
tensor = torch.tensor([1, 2, 3])

# tensor on CPU
print(f"{tensor}, {tensor.device}")

# move tensor to GPU
tensor_on_gpu = tensor.to(device)
print(tensor_on_gpu)                 # prints device='cuda:0', 0 being the index of the GPU

# move tensor back to CPU
tensor_back_on_cpu = tensor_on_gpu.cpu().numpy()
print(tensor_back_on_cpu)
