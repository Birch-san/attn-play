from torch import FloatTensor
def softmax(x: FloatTensor, dim=-1) -> FloatTensor:
  """A normal softmax"""
  maxes = x.max(dim, keepdim=True).values
  diffs = x-maxes
  x_exp = diffs.exp()
  x_exp_sum = x_exp.sum(dim, keepdim=True)
  quotient = x_exp/x_exp_sum
  return quotient