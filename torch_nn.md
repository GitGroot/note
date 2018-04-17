##note about torch nn
#module
module is an abstract class.
其维持两个变量 output, gradInput, 还有parameter和parameter的梯度，这些并不由module维持，由带参数的具体module来定义

- forward
- backward

#####forward
目的就是更新output的值，所以直接调用updateOutput函数，这个函数需要被重写，而forward不建议被重写
```lua
function Module:forward(input)
   return self:updateOutput(input)
end
```
#####backward
目的是更新grad，有两种grad
1.对input求梯度，这个可能是为了给前面的module传梯度对应updateGradInput
2.对parameter传梯度，用来更新parameter对应accGradParameters
```lua
function Module:backward(input, gradOutput, scale)
   scale = scale or 1
   self:updateGradInput(input, gradOutput)
   self:accGradParameters(input, gradOutput, scale)
   return self.gradInput
end
```
同样重写这两个函数updateGradInput, accGradParameters而不是backward
#####updateParameters
backward只是计算梯度，并没有更新parameter，updateParameters负责更新parameter
```lua
function Module:updateParameters(learningRate)
   local params, gradParams = self:parameters()
   if params then
      for i=1,#params do
         params[i]:add(-learningRate, gradParams[i])
      end
   end
end

```

#Container
Abstract Class，所有容器类的父类，例如Sequential，其和module的关系类似于caffe中的network类和layer类的关系，但不同的是Container可以作为一个module继续参与到其他的容器中，所以Container是Module的子类
一部分的Container函数是施加与其所有子module上，比如zeroGradParameter


