import warp as wp
import numpy as np

wp.init()

@wp.kernel
def check_inc_index(index: wp.array(dtype=int), source: wp.array(dtype=float), limit: int, manipulated: wp.array(dtype=float)):
    tid = wp.tid()
    idx = wp.inc_index(index, tid, limit)
    # print(idx)
    if idx >= 0:
        manipulated[idx] = wp.sqrt(source[idx])

@wp.kernel
def check_inc_index_before(index: wp.array(dtype=int), source: wp.array(dtype=float), limit: int, manipulated: wp.array(dtype=float)):
    tid = wp.tid()
    idx = wp.atomic_add(index, 0, 1)
    # print(idx)
    if idx < limit:
        manipulated[idx] = wp.sqrt(source[idx])


def main():
    # wp.set_device("cpu")
    np.random.seed(123)
    dim = 128
    index = wp.zeros(1 + dim, dtype=wp.int32)
    limit = 10
    # source = wp.array(np.random.randn(limit), dtype=wp.float32, requires_grad=True)
    source = wp.array(np.arange(1, limit+1), dtype=wp.float32, requires_grad=True)
    manipulated = wp.zeros(limit, dtype=wp.float32, requires_grad=True)
    tape = wp.Tape()
    with tape:
        # wp.launch(check_inc_index, dim, inputs=[index, source, limit], outputs=[manipulated])
        wp.launch(check_inc_index_before, dim, inputs=[index, source, limit], outputs=[manipulated])

    print("index: ", index.numpy())
    print("source: ", source.numpy())
    print("manipulated: ", manipulated.numpy())

    try:
        tape.backward(grads={
            manipulated: wp.array(np.ones(limit), dtype=wp.float32)
        })
        print("source.grad: ", source.grad.numpy())
    except:
        # expected error when we use the previous atomic_add approach
        print("Access violation")
        raise

if __name__ == "__main__":
    main()