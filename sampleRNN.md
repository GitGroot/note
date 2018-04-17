---
typora-root-url: ../gitnote
---

##### 数据预处理

train.lua似乎把frame_level_rnn每个sample变换到[-2,2]

![sampleRNN](/sampleRNN.png)

train.lua中对sample_level_predictor的定义不同于之前的frame_level_rnn,其加入了对sample的embedding，原因在paper中提到过：

>		In addition, early on we noticed that the model can achieve better performance and generation quality when we embed the quantized input values before passing them through the sample-level MLP (see Table 4). The embedding steps maps each of the q discrete values to a real-valued vector embedding.However, real-valued raw samples are still used as input to the higher modules.

切片操作来提供数据

```lua
-- minibatch's shape [batchsize][audionframes]
local minibatch = make_minibatch(thread_pool, files, shuffled_files, start, stop)
-- minibatch_seqs' for tbptt and different indices
-- [1-batchsize][[f1,f2,...,f520],
--               [f513,...,f1032],
-- 				  ...
--			     [...]]
local minibatch_seqs = minibatch:unfold(2,seq_len+big_frame_size,seq_len)
-- big_input_sequences
-- [1-batchsize][[f1,f2,...,f512],
--               [f513,...,f1024],
-- 				  ...
--			     [...]]
local big_input_sequences = minibatch_seqs[{{},{},{1,-1-big_frame_size}}]
-- input_sequences
-- [1-batchsize][[f7,f8,...,f518],
--               [f519,...,f1030],
-- 				  ...
--			     [...]]
local input_sequences = minibatch_seqs[{{},{},{big_frame_size-frame_size+1,-1-frame_size}}]
-- target_sequences
-- [1-batchsize][[f9,f10,...,f520],
--               [f521,...,f1032],
-- 				  ...
--			     [...]]
local target_sequences = minibatch_seqs[{{},{},{big_frame_size+1,-1}}]
-- prev_sequences, notice this is for sample_level_predictor, and is not 
-- non-overlapping frames
-- [1-batchsize][[f7,f8,...,f519],
--               [f519,...,f1031],
-- 				  ...
--			     [...]]
local prev_samples = minibatch_seqs[{{},{},{big_frame_size-frame_size+1,-1-1}}]

-- just split the data into the needed dimension
-- big_frames [batchsize][tbptt_size][nframes][frame_size](bsxtsxnfx8)
-- [1-batchsize][[[f1,f2,...,f8],
--				  [f9,f10,...,f15],
--				  ...
--				  [f505,f506,...,f512]],
-- 				  ...
--			     [...]]
local big_frames = big_input_sequences:unfold(3,big_frame_size,big_frame_size)
-- frames [batchsize][tbptt_size][nframes][frame_size](bsxtsxnfx2)
-- [1-batchsize][[[f7,f8],
--				  [f9,f10],
--				  ...
--				  [f511,f512]],
-- 				  ...
--			     [...]]
local frames = input_sequences:unfold(3,frame_size,frame_size)
-- prev_samples [batchsize][tbptt_size][nframes][frame_size](bsxtsxnfx2)
-- [1-batchsize][[[f7,f8],
--				  [f8,f9],
--				  ...
--				  [f518,f519]],
-- 				  ...
--			     [...]]
prev_samples = prev_samples:unfold(3,frame_size,1)
```



#####Truncated BPTT

train.lua中训练实际是对每个batch训练时，每个batch又分了512个子序列上训练，实际paper中提到过使用的Truncated BPTT,rnn序列太长有高昂的计算代价（但实际上感觉是减小了内存代价，并且在子序列长度为512时效果较好

>		We enable efficient training of our recurrent model using truncated backpropagation through time, splitting each sequence into short subsequences and propagating gradients only to the beginning of each subsequence. We experiment with different subsequence lengths and demonstrate that we are able to train our networks.


​			
​		
​	