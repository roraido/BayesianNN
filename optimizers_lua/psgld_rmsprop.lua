function psgld_rmsprop(opfunc, x, config, state)
  -- (0) get/update state
  local config = config or {}
  local state = state or config
  local N = config.N
  local lr = config.learningRate or 1e-3
  local lrd = config.learningRateDecay or 0
  local wd = config.weightDecay or 0 
  
  local alpha = config.alpha or 0.99
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
  if not state.m then
    state.m = torch.Tensor():typeAs(x):resizeAs(dfdx):fill(mfill)
    state.tmp = torch.Tensor():typeAs(x):resizeAs(dfdx)
 end

 -- (4) calculate new (leaky) mean squared values
 state.m:mul(alpha)
 state.m:addcmul(1.0-alpha, dfdx, dfdx)
 state.tmp:sqrt(state.m + epsilon)

  -- (5) learning rate decay (annealing)
  local clr = lr / (1 + nevals*lrd)
  x:addcdiv(-clr, dfdx, state.tmp)
  local coeff = clr * dfdx / state.tmp

  x:add(math.sqrt(2*coeff)/N, state.noise:normal())

  -- (6) update evaluation counter
  state.evalCounter = state.evalCounter + 1

  -- return x*, f(x) before optimization
  return x,{fx}
end
