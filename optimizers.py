import theano
import theano.tensor as tensor
import numpy as np
from utils import numpy_floatX
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

def SGD(tparams, cost, inps, lr):
    """ default: lr=0.01 """
    
    grads = tensor.grad(cost, tparams.values())
    norm = tensor.sqrt(sum([tensor.sum(g**2) for g in grads]))
    if tensor.ge(norm, 5):
        grads = [g*5/norm for g in grads]
        
    gshared = [theano.shared(p.get_value() * 0., name='%s_grad'%k) 
                for k, p in tparams.iteritems()]
    gsup = [(gs, g) for gs, g in zip(gshared, grads)]
    f_grad_shared = theano.function(inps, cost, updates=gsup)
    
    updates = []

    for p, g in zip(tparams.values(), gshared):        
        updated_p = p - lr * g
        updates.append((p, updated_p))
    
    f_update = theano.function([lr], [], updates=updates)
    
    return f_grad_shared, f_update 

def RMSprop(tparams, cost, inps, lr, rho=0.9, epsilon=1e-6):
    """ default: lr=0.001 
        This is the implementation of the RMSprop algorithm used in
        http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf.
    """
    
    grads = tensor.grad(cost, tparams.values())
    norm = tensor.sqrt(sum([tensor.sum(g**2) for g in grads]))
    if tensor.ge(norm, 5):
        grads = [g*5/norm for g in grads]
        
    gshared = [theano.shared(p.get_value() * 0., name='%s_grad'%k) 
                for k, p in tparams.iteritems()]
    gsup = [(gs, g) for gs, g in zip(gshared, grads)]
    f_grad_shared = theano.function(inps, cost, updates=gsup)     
    
    updates = []

    for p, g in zip(tparams.values(), gshared):
        acc = theano.shared(p.get_value() * 0.)
        acc_new = rho * acc + (1 - rho) * g ** 2
        updates.append((acc, acc_new))
        
        updated_p = p - lr * (g / tensor.sqrt(acc_new + epsilon))
        updates.append((p, updated_p))
    
    f_update = theano.function([lr], [], updates=updates)
    
    return f_grad_shared, f_update

def Adam(tparams, cost, inps, lr, b1=0.1, b2=0.001, e=1e-8, clip_norm=5):
    
    grads = tensor.grad(cost, tparams.values())
    norm = tensor.sqrt(sum([tensor.sum(g**2) for g in grads]))
    if tensor.ge(norm, clip_norm):
        grads = [g*clip_norm/norm for g in grads]
    
    gshared = [theano.shared(p.get_value() * 0., name='%s_grad'%k) 
                for k, p in tparams.iteritems()]
    gsup = [(gs, g) for gs, g in zip(gshared, grads)]
    f_grad_shared = theano.function(inps, cost, updates=gsup)
    
    updates = []

    i = theano.shared(numpy_floatX(0.))    
    i_t = i + 1.
    fix1 = 1. - b1**(i_t)
    fix2 = 1. - b2**(i_t)
    lr_t = lr * (tensor.sqrt(fix2) / fix1)

    for p, g in zip(tparams.values(), gshared):
        m = theano.shared(p.get_value() * 0.)
        v = theano.shared(p.get_value() * 0.)
        m_t = (b1 * g) + ((1. - b1) * m)
        v_t = (b2 * tensor.sqr(g)) + ((1. - b2) * v)
        g_t = m_t / (tensor.sqrt(v_t) + e)
        p_t = p - (lr_t * g_t)
        updates.append((m, m_t))
        updates.append((v, v_t))
        updates.append((p, p_t))
    updates.append((i, i_t))
    
    f_update = theano.function([lr], [], updates=updates)
    
    return f_grad_shared, f_update  

def SGLD(tparams, cost, inps, ntrain, lr):
    """ default: lr=0.01 """

    trng = RandomStreams(123)

    grads = tensor.grad(cost, tparams.values())
    norm = tensor.sqrt(sum([tensor.sum(g**2) for g in grads]))
    if tensor.ge(norm, 5):
        grads = [g*5/norm for g in grads]

    gshared = [theano.shared(p.get_value() * 0., name='%s_grad'%k)
                for k, p in tparams.iteritems()]
    gsup = [(gs, g) for gs, g in zip(gshared, grads)]
    f_grad_shared = theano.function(inps, cost, updates=gsup)

    updates = []

    for p, g in zip(tparams.values(), gshared):
        eps = trng.normal(p.get_value().shape, avg = 0.0, std = 1.0, 
                          dtype=theano.config.floatX)

	updated_p = p - lr * (g-p/ntrain) + tensor.sqrt(lr*2./ntrain) * eps
        updates.append((p, updated_p))

    f_update = theano.function([lr,ntrain], [], updates=updates)

    return f_grad_shared, f_update

def SGLD_modified(tparams, cost, inps, ntrain, lr):
    """ default: lr=0.01 """

    trng = RandomStreams(123)

    grads = tensor.grad(cost, tparams.values())
    norm = tensor.sqrt(sum([tensor.sum(g**2) for g in grads]))
    if tensor.ge(norm, 5):
        grads = [g*5/norm for g in grads]

    gshared = [theano.shared(p.get_value() * 0., name='%s_grad'%k)
                for k, p in tparams.iteritems()]
    gsup = [(gs, g) for gs, g in zip(gshared, grads)]
    f_grad_shared = theano.function(inps, cost, updates=gsup)

    updates = []

    for p, g in zip(tparams.values(), gshared):
        eps = trng.normal(p.get_value().shape, avg = 0.0, std = 1.0, 
                          dtype=theano.config.floatX)

	updated_p = p - lr * (g-p/ntrain) + tensor.sqrt(lr)*2./ntrain * eps
        updates.append((p, updated_p))

    f_update = theano.function([lr,ntrain], [], updates=updates)

    return f_grad_shared, f_update
      
def pSGLD(tparams, cost, inps, ntrain, lr, rho=0.9, epsilon=1e-6, clip_norm=5):
    """ default: lr=0.001 """
    
    trng = RandomStreams(123)
    
    grads = tensor.grad(cost, tparams.values())
    norm = tensor.sqrt(sum([tensor.sum(g**2) for g in grads]))
    if tensor.ge(norm, clip_norm):
        grads = [g*clip_norm/norm for g in grads]
        
    gshared = [theano.shared(p.get_value() * 0., name='%s_grad'%k) 
                for k, p in tparams.iteritems()]
    gsup = [(gs, g) for gs, g in zip(gshared, grads)]
    f_grad_shared = theano.function(inps, cost, updates=gsup)     
    
    updates = []
    
    for p, g in zip(tparams.values(), gshared):
        acc = theano.shared(p.get_value() * 0.)
        acc_new = rho * acc + (1 - rho) * g ** 2
        updates.append((acc, acc_new))
        
        G = tensor.sqrt(acc_new + epsilon)
        
        eps = trng.normal(p.get_value().shape, avg = 0.0, std = 1.0, 
                          dtype=theano.config.floatX)
        
        updated_p = p - lr * (g-p/ntrain) / G + tensor.sqrt(lr/G)*2./ntrain * eps 
        updates.append((p, updated_p))
    
    f_update = theano.function([lr,ntrain], [], updates=updates)
    
    return f_grad_shared, f_update

def pSGLD_Adam(tparams, cost, inps, ntrain, lr, rho1=0.9, rho2=0.999, epsilon=1e-6, clip_norm=5):
    """ default: lr=0.000001 """
    
    trng = RandomStreams(123)
    
    grads = tensor.grad(cost, tparams.values())
    norm = tensor.sqrt(sum([tensor.sum(g**2) for g in grads]))
    if tensor.ge(norm, clip_norm):
        grads = [g*clip_norm/norm for g in grads]
        
    gshared = [theano.shared(p.get_value() * 0., name='%s_grad'%k) 
                for k, p in tparams.iteritems()]
    gsup = [(gs, g) for gs, g in zip(gshared, grads)]
    f_grad_shared = theano.function(inps, cost, updates=gsup)     
    
    updates = []
    
    for p, g in zip(tparams.values(), gshared):
        acc = theano.shared(p.get_value() * 0.)
        acc_m = theano.shared(p.get_value() * 0.)
        
        acc_new = rho1 * acc + (1. - rho1) * g ** 2
        acc_m_new = rho2 * acc_m + (1.-rho2) * g

        updates.append((acc, acc_new)) 
        updates.append((acc_m, acc_m_new)) 

        m_acc_new = acc_new / (1. - rho1)
        m_acc_m_new = acc_m_new / (1. - rho2)

        G = tensor.sqrt(m_acc_new + epsilon)
        # G = tensor.sqrt(m_acc_new + epsilon) * (g - p/ntrain) / (m_acc_m_new - p/ntrain)

        eps = trng.normal(p.get_value().shape, avg = 0.0, std = 1.0, 
                          dtype=theano.config.floatX)
        
        # updated_p = p - lr * (g - p/ntrain) / G + tensor.sqrt(lr/G)*2./ntrain * eps
        updated_p = p - lr * (m_acc_m_new - p/ntrain) / G + tensor.sqrt(lr / G)*2./ntrain * eps
        # updated_p = p - lr * (g - p/ntrain) / G + tensor.sqrt(lr * (g - p/ntrain) /(G * (m_acc_m_new - p/ntrain)))*2./ntrain * eps

        updates.append((p, updated_p))
    
    f_update = theano.function([lr,ntrain], [], updates=updates)
    
    return f_grad_shared, f_update
    
def pSGLD_AdaDelta(tparams, cost, inps, ntrain, lr, rho=0.9, epsilon=1e-6, clip_norm=5):
    
    trng = RandomStreams(123)
    
    grads = tensor.grad(cost, tparams.values())
    norm = tensor.sqrt(sum([tensor.sum(g**2) for g in grads]))
    if tensor.ge(norm, clip_norm):
        grads = [g*clip_norm/norm for g in grads]
        
    gshared = [theano.shared(p.get_value() * 0., name='%s_grad'%k) 
                for k, p in tparams.iteritems()]
    gsup = [(gs, g) for gs, g in zip(gshared, grads)]
    f_grad_shared = theano.function(inps, cost, updates=gsup)     
    
    updates = []
    
    for p, g in zip(tparams.values(), gshared):
        acc = theano.shared(p.get_value() * 0.)
        msdx = theano.shared(p.get_value() * 0.)
        delta_p = theano.shared(p.get_value() * 0.)

        acc_new = rho * acc + (1. - rho) * g ** 2.
        msdx_new = rho * msdx + (1.- rho) * delta_p ** 2.

        updates.append((acc, acc_new))
        
        G = tensor.sqrt(acc_new + epsilon) * lr / tensor.sqrt(msdx + epsilon)
        
        updates.append((msdx, msdx_new))

        eps = trng.normal(p.get_value().shape, avg = 0.0, std = 1.0, 
                          dtype=theano.config.floatX)
        
        update_delta_p = - lr * (g-p/ntrain) / G + tensor.sqrt(lr/G)*2./ntrain * eps 
        updated_p = p + update_delta_p

        updates.append((delta_p, update_delta_p)) 
        updates.append((p, updated_p))
    
    f_update = theano.function([lr,ntrain], [], updates=updates)
    
    return f_grad_shared, f_update

def pSGLD_RG(tparams, cost, inps, ntrain, lr, rho1=0.9, rho2=0.999, epsilon=1e-6, clip_norm=5, momentum=0.95):
    """ default: lr=0.00001 """
    """ default-no-momemnt: lr=0.001 """
    
    trng = RandomStreams(123)
    
    grads = tensor.grad(cost, tparams.values())
    norm = tensor.sqrt(sum([tensor.sum(g**2) for g in grads]))
    if tensor.ge(norm, clip_norm):
        grads = [g*clip_norm/norm for g in grads]
        
    gshared = [theano.shared(p.get_value() * 0., name='%s_grad'%k) 
                for k, p in tparams.iteritems()]
    gsup = [(gs, g) for gs, g in zip(gshared, grads)]
    f_grad_shared = theano.function(inps, cost, updates=gsup)     
    
    updates = []
    
    for p, g in zip(tparams.values(), gshared):
        acc = theano.shared(p.get_value() * 0.)
        acc_m = theano.shared(p.get_value() * 0.)
        delta_p = theano.shared(p.get_value() * 0.)
        
        acc_new = rho1 * acc + (1. - rho1) * g ** 2
        acc_m_new = rho2 * acc_m + (1.-rho2) * g

        updates.append((acc, acc_new)) 
        updates.append((acc_m, acc_m_new)) 

        G = tensor.sqrt(acc_new - acc_m_new ** 2 + epsilon)
        # G = 1. / ((momentum * delta_p / ((g - p/ntrain) * lr)) - (1 / tensor.sqrt(acc_new - acc_m_new ** 2 + epsilon)))
        eps = trng.normal(p.get_value().shape, avg = 0.0, std = 1.0, 
                          dtype=theano.config.floatX)
        
        # update_delta_p = - lr * (g - p/ntrain) / G + tensor.sqrt(lr/G)*2./ntrain * eps
        update_delta_p = momentum * delta_p - lr * (g - p/ntrain) / G + tensor.sqrt(lr/G)*2./ntrain * eps  # moment

        updated_p = p + update_delta_p

        updates.append((delta_p, update_delta_p)) 
        updates.append((p, updated_p))

    f_update = theano.function([lr,ntrain], [], updates=updates)
    
    return f_grad_shared, f_update