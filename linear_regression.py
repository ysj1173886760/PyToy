import pytoy as pt
import numpy as cp

BATCH_SIZE = 10
x = pt.core.Variable(dims=(BATCH_SIZE, 6), init=False, trainable=False, name='x')
w = pt.core.Variable(dims=(6, 1), init=True, trainable=True, name='w')
b = pt.core.Variable(dims=(1, 1), init=True, trainable=True, name='b', bias=True)
boardcast_b = pt.ops.Boardcast(b, to_shape=(BATCH_SIZE, 1))

mat = pt.ops.MatMul(x, w)
output = pt.ops.Add(mat, boardcast_b)
label = pt.core.Variable(dims=(BATCH_SIZE, 1), init=False, trainable=False, name='label')
loss = pt.loss.L2Loss(output, label)

real_w = cp.array([[2, 5, 7, 16, -10, -3]], dtype=cp.float32).T
real_b = cp.array([[3]], dtype=cp.float32)

lr = 0.01

for epoch in range(1):
    input = cp.random.randint(-5, 5, (BATCH_SIZE, 6))
    x.set_value(input)
    label.set_value(cp.add(cp.matmul(input, real_w), real_b))
    loss.forward()
    w.backward(loss)
    b.backward(loss)

    pt.default_graph.draw()

    w.set_value(w.value - lr * w.graident)
    b.set_value(b.value - lr * b.graident)
    pt.default_graph.clear_graident()

print(w.value, b.value)
        
