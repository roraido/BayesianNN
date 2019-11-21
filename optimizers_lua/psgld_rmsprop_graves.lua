function psgld_rmsprop(opfunc, x, config, state)
  -- (0) get/update state
  local config = config or {}
  local state = state or config
  local N = config.N
  local lr = config.learningRate or 1e-3
  local lrd = config.learningRateDecay or 0
  local wd = config.weightDecay or 0 
  
  -- local alpha = config.alpha or 0.99
  local beta1 = config.beta1 or 0.9
  local beta2 = config.beta2 or 0.999
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
  -- Exponential moving average of gradient values
  state.m = state.m or x.new(dfdx:size()):zero()
  -- Exponential moving average of squared gradient values
  state.v = state.v or x.new(dfdx:size()):zero()
  -- A tmp tensor to hold the sqrt(v) + epsilon
  state.denom = state.denom or x.new(dfdx:size()):zero()
  
  -- (4) learning rate decay (annealing)
  local clr = lr / (1 + nevals*lrd)
    
  -- (5) calculate new (leaky) mean squared values
  -- Decay the first and second moment running average coefficient
  state.m:mul(beta1):add(1-beta1, dfdx)
  state.v:mul(beta2):addcmul(1-beta2, dfdx, dfdx)
  
  state.denom:copy(state.v):add(epsilon - state.m^2):sqrt()

  x:addcdiv(-clr, dfdx, state.denom)

  local coeff = clr * dfdx / state.dom
  x:add(math.sqrt(2*coeff)/N, state.noise:normal())

  -- (6) update evaluation counter
  state.evalCounter = state.evalCounter + 1

  -- return x*, f(x) before optimization
  return x,{fx}
end
