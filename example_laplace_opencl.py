from ocl import Device
from canvas import Canvas
import random
import numpy

n = 300
q = numpy.zeros((n,n), dtype=numpy.float32)
u = numpy.zeros((n,n), dtype=numpy.float32)
w = numpy.zeros((n,n), dtype=numpy.float32)

for k in range(n):
    q[random.randint(1,n-1),random.randint(1,n-1)] = random.choice((-1,+1))

device = Device()
q_buffer = device.buffer(source=q, mode=device.flags.READ_ONLY)
u_buffer = device.buffer(source=u)
w_buffer = device.buffer(source=w)

@device.compiler.define_kernel(
    w='global:ptr_float',
    u='global:const:ptr_float',
    q='global:const:ptr_float')
def solve(w,u,q):
    x = new_int(get_global_id(0))
    y = new_int(get_global_id(1))
    site = new_int(x*n+y)
    if y!=0 and y!=n-1 and x!=0 and x!=n-1:
        up = new_int(site-n)
        down = new_int(site+n)
        left = new_int(site-1)
        right = new_int(site+1)
        w[site] = 1.0/4*(u[up]+u[down]+u[left]+u[right] - q[site])

program = device.compile(constants=dict(n=n))

for k in range(3000):
    program.solve(device.queue, [n,n], None, w_buffer, u_buffer, q_buffer)
    (u_buffer, w_buffer) = (w_buffer, u_buffer)
    
u = device.retrieve(u_buffer,shape=(n,n))
Canvas(title='').imshow(u).save()
