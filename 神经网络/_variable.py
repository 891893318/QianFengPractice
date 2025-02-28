import torch
from torch.autograd import Variable




tensor = torch.FloatTensor([[1,2],[3,4]])
variable = Variable(tensor, requires_grad=True)

print(tensor)
print(variable)

t_out = torch.mean(tensor*tensor)
v_out = torch.mean(variable*variable)
print(t_out)
print(v_out)    # 7.5

v_out.backward()

print(variable.grad)


print(variable)


print(variable.data)


print(variable.data.numpy())
