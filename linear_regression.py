import pytoy as pt
import numpy as cp

x = pt.core.Variable(dims=(3, 1), init=False, trainable=False)
w = pt.core.Variable(dims=(1, 3), init=True, trainable=True)
b = pt.core.Variable(dims=(1, 1), init=True, trainable=True)

output = pt.ops.Add(pt.ops.MatMul(w, x), b)
label = pt.core.Variable(dims=(1, 1), init=False, trainable=False)
loss = pt.loss.L2Loss(output, label)

real_w = cp.array([[2, 5, 7]], dtype=cp.float32)
real_b = cp.array([[3]], dtype=cp.float32)

lr = 0.01

for epoch in range(10):
    input = cp.random.randint(0, 10, (3, 1))
    x.set_value(input)
    label.set_value(cp.add(cp.matmul(real_w, input), real_b))
    loss.forward()
    w.backward(loss)
    b.backward(loss)

    print(w.value)
    print(b.value)
    # print(x.value) 
    # print(output.value)
    # print(loss.value)
    # print(w.graident, b.graident, loss.graident, output.graident)
    w.set_value(w.value - lr * w.graident)
    b.set_value(b.value - lr * b.graident)
    pt.default_graph.clear_graident()

print(w.value, b.value)