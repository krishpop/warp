import warp as wp

wp.init()

@wp.func
def sqr(x: float):
    return x*x

@wp.kernel
def kern(expect: float):
    wp.expect_eq(sqr(2.0), expect)


def run(expect, device):
    wp.launch(kern, dim=1, inputs=[expect], device=device)
  
