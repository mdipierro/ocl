from ocl import Compiler

c99 = Compiler()

@c99.define(z='float')
def cdf(z):
    if z > 6.0:
        I = 1.0
    elif z < -6.0:
        I = 0.0
    else:
        a = new_float(fabs(z))
        b1 = 0.31938153
        b2 = -0.356563782
        b3 = 1.781477937
        b4 = -1.821255978
        b5 = 1.330274429
        p1 = 0.2316419
        c1 = 0.398923
        t = new_float(1.0/(1.0+a*p1))
        I = 1.0 - c1*exp(-z*z/2)*((((b5*t+b4)*t+b3)*t+b2)*t+b1)*t
        if z<0:
            I = 1.0-I
    return I

@c99.define(S='float',X='float',r='float',sigma='float',t='float')
def price_BS(S, X, r, sigma, t):
    sqrt_t = new_float(sqrt(t))
    d1 = new_float((log(S/X)+r*t)/(sigma*sqrt_t)+0.5*sigma*sqrt_t)
    d2 = new_float(d1 - sigma*sqrt_t)
    c = new_float(S*cdf(d1)-X*exp(-r*t)*cdf(d2))
    return c
        
compiled = c99.compile(includes=['#include "math.h"'])
for S in range(0,20):
    print compiled.price_BS(S,10.0,0.2,0.5,90.0/250)

