import torch
from torch import Tensor, FloatTensor, IntTensor
from typing import Optional, Dict
import math

device = torch.device('cpu')
gen = torch.Generator(device=device)

seed = 42
# std = 2**4
# m0 = torch.randn([2, 2], generator=gen, device=device) * std
# m1 = torch.randn([2, 2], generator=gen, device=device) * std
m0 = torch.tensor([[0.,   1.],
                   [15.5, 0.4]], device=device)
m1 = torch.tensor([[0.,    1.],
                   [-15.5, 0.23]], device=device)
s = m0 @ m1

def get_exp(input: FloatTensor) -> IntTensor:
  # TODO: provide a way for user to pass in an already-allocated buffer (frexp out parameter)
  _, exp = input.frexp()
  exp.sub_(1)
  return exp

exp0 = get_exp(m0)
exp1 = get_exp(m1)
exps = get_exp(s)

# out = torch.zeros()

# def edotv(input: Tensor, tensor: Tensor, *, out: Optional[Tensor] = None) -> Tensor:
#   assert not input.is_floating_point()
#   assert not tensor.is_floating_point()
#   if out is None:
#     out_shape: torch.Size = torch.broadcast_shapes(input.shape, tensor.shape)
#     out = input.new_zeros(out_shape)
#   else:
#     assert not out.is_floating_point()
#     out_shape: torch.Size = torch.broadcast_shapes(input.shape, tensor.shape, out.shape)
#     out = out.expand(out_shape)
#   sum = torch.add(input, tensor, out=out)

emaxs: Dict[torch.dtype, int] = {
  torch.float16: 15,
  torch.bfloat16: 127,
  torch.float32: 127,
  # torch.fp8e5m2: 15,
  # torch.fp8e4m3: 7,
}
emins: Dict[torch.dtype, int] = {
  k: -(v-1) for k,v in emaxs.items()
}
# float16 exponent range goes from 2^-14 to 2^15, encoded as 00001–11110 (1 to 30) with an offset of 15

def edotv(
  input: FloatTensor,
  tensor: FloatTensor,
  *,
  out: Optional[IntTensor] = None,
  acc_dtype=torch.int16,
  acc: Optional[IntTensor] = None,
) -> IntTensor:
  assert input.is_floating_point()
  assert tensor.is_floating_point()
  assert input.ndim == 1
  assert tensor.ndim == 1
  assert input.shape == tensor.shape
  assert input.dtype in emaxs
  emin: int = emins[input.dtype]
  emax: int = emaxs[input.dtype]
  elems = input.size(-1)
  # worst-case scenario of multiplying a pair is that you square the worst-case element, which in exponent-space means doubling
  # math.log2((2**8) ** 2) == 16
  min_product = emin*2
  max_product = emax*2
  # worst-case scenario of adding a product is that you double the worst-case element, which in exponent-space means adding 1
  # this can happen per product accumulated, so n-1 times
  max_accs = elems-1
  min_dot_exp = min_product-max_accs
  max_dot_exp = max_product+max_accs
  assert min_dot_exp >= torch.iinfo(acc_dtype).min
  assert max_dot_exp <= torch.iinfo(acc_dtype).max

  # 1e8 * 1e8
  # 2e8
  # if two worst-case 

  if out is None:
    out = torch.zeros_like(input, dtype=acc_dtype)
  else:
    assert not out.is_floating_point()
    assert out.is_signed()
    assert out.ndim == 1
    assert out.shape == input.shape
  
  if acc is None:
    acc = input.new_zeros((2, torch.iinfo(acc_dtype).bits,), dtype=acc_dtype)
  else:
    assert acc.dtype == acc_dtype
    assert acc.ndim == 2
    assert acc.size(-1) == torch.iinfo(acc_dtype).bits
    # one batch into which to accumulate positive products, another for negative
    assert acc.size(-2) == 2
    assert not acc.requires_grad
    acc.zero_()
  
  # TODO: provide a way for user to pass in an already-allocated buffer for each of these
  exp_i = get_exp(input)
  exp_t = get_exp(tensor)

  # TODO: provide a way for user to pass in an already-allocated buffer for each of these
  sign_i = input.sign()
  sign_t = tensor.sign()
  # TODO: is there a bitwise way to do this that's cheaper? would be nice to be able to say no multiplication was used!
  #       and beware: sign 0 exists too, but for our purposes we could simplify that to 1 if it helps at all
  sign_prods = sign_i * sign_t

  # TODO: this could probably use a smaller dtype. only needs to be able to fit min_product ≤ x ≤ max_product
  dp_dtype = acc_dtype
  # TODO: provide a way for user to pass in an already-allocated buffer
  # e_prod = input.new_zeros((1,), dtype=dp_dtype)

  ix = input.size(-1) - 1
  while ix >= 0:
    # operands (for multiplication)
    o_i, o_t = input[ix], tensor[ix]
    # exponents
    e_i, e_t = exp_i[ix], exp_t[ix]
    # exponent-space elementwise product
    # TODO: sum exponents into re-usable buffer instead of re-allocating e_prod
    e_prod = e_i + e_t
    sign_prod = sign_prods[ix]
    # TODO: add to the appropriate counter for this e_prod,sign_prod acc
    ix -= 1



# def edot(input: Tensor, tensor: Tensor, *, out: Optional[Tensor] = None) -> Tensor:
#   assert input.ndim >= 2
#   assert tensor.ndim >= 2
#   if out is None:
#     out = torch.zeros()
#   pass

# bdot = torch.vmap(torch.dot)
# bedot = torch.vmap(torch.edot)

# map(torch.dot, m0, m1.mT[0])

s_e = edotv(m0[0], m1.mT[0])
print(s_e)
# map(lambda m0_, m1_T: torch.vmap(torch.dot)(m0_, m1_T), m0, m1.mT)
pass

