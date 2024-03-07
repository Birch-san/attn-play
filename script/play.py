import torch
from torch import Tensor, FloatTensor, IntTensor
from typing import Optional, Dict
import math

device = torch.device('cpu')
gen = torch.Generator(device=device)

seed = 42
std = 2**4
fp_dtype = torch.float16
m0 = torch.randn([2, 2], generator=gen, device=device, dtype=fp_dtype) * std
m1 = torch.randn([2, 2], generator=gen, device=device, dtype=fp_dtype) * std
# m0 = torch.tensor([[0.,   1.],
#                    [15.5, 0.4]], device=device)
# m1 = torch.tensor([[0.,    1.],
#                    [-15.5, 0.23]], device=device)
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
  min_product_exp = emin*2
  max_product_exp = emax*2
  # worst-case scenario of adding a product is that you double the worst-case element, which in exponent-space means adding 1
  # this can happen per product accumulated, so n-1 times
  max_accs = elems-1
  min_dot_exp = min_product_exp-max_accs
  max_dot_exp = max_product_exp+max_accs
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
  
  # (product_sign, exp_sign, exp_magnitude)
  # we could use exp_magnitude-1 I think but may as well use power-of-2 buffer
  # acc_shape = (2, 2, torch.iinfo(acc_dtype).bits//2)
  acc_shape = (2, 2, max_product_exp)
  if acc is None:
    acc = input.new_zeros(acc_shape, dtype=acc_dtype, requires_grad=False)
  else:
    assert acc.dtype == acc_dtype
    assert acc.shape == acc_shape
    assert not acc.requires_grad
    acc.zero_()
  
  # TODO: provide a way for user to pass in an already-allocated buffer for each of these
  exp_i = get_exp(input)
  exp_t = get_exp(tensor)

  # TODO: provide a way for user to pass in an already-allocated buffer for each of these
  sign_i = input.signbit()
  sign_t = tensor.signbit()
  # I am trying to do sign_i * sign_t, is this right?
  sign_prods = sign_i != sign_t
  prod_iszero = (input == 0) | (tensor == 0)
  prod_isnan = input.isnan() | tensor.isnan()
  prod_isinf = (input.isinf() | tensor.isinf()) & (prod_iszero == 0)

  # TODO: provide a way for user to pass in an already-allocated buffer
  dp_isnan = input.new_zeros((1,), dtype=torch.bool)
  # TODO: provide a way for user to pass in an already-allocated buffer
  dp_isinf = input.new_zeros((1,), dtype=torch.bool)

  # TODO: this could probably use a smaller dtype. only needs to be able to fit min_product ≤ x ≤ max_product
  # dp_dtype = acc_dtype
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
    e_prod_sign = e_prod.signbit()
    iszero = prod_iszero[ix]
    isnan = prod_isnan[ix]
    isinf = prod_isinf[ix]
    dp_isnan |= isnan
    dp_isinf |= isinf

    sign_prod = sign_prods[ix]
    acc[sign_prod.int(), e_prod_sign.int(), e_prod] += (~iszero).int()
    ix -= 1
  # TODO: now read through all the per-exp counters in acc, find pairs, carry those up to larger exponents, until we find
  # largest exponent among +ve products and largest exponent among -ve products, then pick whichever of those is larger
  # (or return 0 if equal)
  pass



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

