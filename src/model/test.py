import os

a = os.path.abspath('./checkpoints/model.pt')
b = os.path.abspath(f'{os.path.dirname(os.path.dirname(__file__))}/checkpoints/model.pt')

print(a)
print(b)
print(a == b)