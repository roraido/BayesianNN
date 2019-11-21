function psgld_rmsprop(opfunc, x, config, state)
  -- (0) get/update state
  local config = config or {}
  local state = state or config
  local N = config.N
  local lr = config.learningRate or 1e-3
  local lrd = config.learningRateDecay or 0
  local wd = config.weightDecay or 0 
  
  -- local alpha = config.alpha or 0.99
  -- local beta1 = config.beta1 or 0.9
  -- local beta2 = config.beta2 or 0.999
  local rho = cofing.rho or 0.9
  local epsilon = config.epsilon or 1e-8
  local mfill = config.initialMean or 0

  state.noise =state.noise or torch.Tensor():typeAs(x):resizeAs(x):zero()
  state.evalCounter = state.evalCounter or 0
  local nevals = state.evalCounter
  assert(not nesterov or (mom > 0 and damp == 0), "Nesterov momentum requires a momentum and zero dampening")

  -- (1) evaluate f(x) and df/dx
  local _ ,dfdx = opfunc(1)

  -- (2) weight decay
  if wd~=0 then
    dfdx:add(wd, x)
  end

  -- (3) initialize mean square values and square gradient storage
  -- state.t = state.t or 0
  if not state.paramVariance then
    -- state.t = state.t or 0
    state.paramVariance = torch.Tensor():typeAs(x):resizeAs(dfdx):zero()
    state.paramStd = torch.Tensor():typeAs(x):resizeAs(dfdx):zero()
    state.delta = torch.Tensor():typeAs(x):resizeAs(dfdx):zero()
    state.accDelta = torch.Tensor():typeAs(x):resizeAs(dfdx):zero()
    stete.coeff = torch.Tensor():typeAs(x):resizeAs(dfdx):zero() 
  end
  -- (4) learning rate decay (annealing)
  local clr = lr / (1 + nevals*lrd)
    
  -- (5) calculate new (leaky) mean squared values
  -- Decay the first and second moment running average coefficient
  state.paramVariance:mul(rho):addcmul(1-rho,dfdx,dfdx)
  state.paramStd:resizeAs(state.paramVariance):copy(state.paramVariance):add(eps):sqrt()
  state.delta:resizeAs(state.paramVariance):copy(state.accDelta):add(eps):sqrt():cdiv(state.paramStd):cmul(dfdx)
  state.coeff:resizeAs(state.paramVariance):copy(state.accDelta):add(eps):sqrt():cdiv(state.paramStd)

  x:add(-1, state.delta)
  x:add(math.sqrt(2*state.coeff)/N, state.noise:normal())i
  state.accDelta:mul(rho):addcmul(1-rho, state.delta, state.delta)

  -- (6) update evaluation counter
  state.evalCounter = state.evalCounter + 1

  -- return x*, f(x) before optimization
  return x,{fx}
end
