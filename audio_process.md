###利用ffmpeg来提取视频中的音频
提取的音频应为单通道，采样率16kHz，depth为8位
```
ffmpeg -i video_00000.mp4 -acodec pcm_u8 -ar 16000 -ac 1 test.wav
```
将video_00000.mp4作为输入 
-acodec pcm_u8 转换位宽为8bit
-ar 转换采样率
-ac 转换通道数
