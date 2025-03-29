Orpheus-TTS local speech synthesizer written entirely in C#

### Requirements

```
.NET 9
Cuda 12
```

### Usage

1. Clone the git repo:

 - `git clone https://github.com/TheAjaykrishnanR/TaraSharp`

2. Download the decoder weights from [here](https://github.com/TheAjaykrishnanR/TaraSharp/releases/download/files/pytorch_model_unnormed.bin) and place it in `snac\24khz` alongside `config.json`. If you want to use the original weights, see conversion method below.

3. `cd TaraSharp`

4. `dotnet run`

### Additional Info

The original snac model available [here](https://huggingface.co/hubertsiuzdak/snac_24khz) was created using PyTorch. For the most part models created in PyTorch can be loaded without much hassle in TorchSharp (not vice versa). However the snac decoder uses the `weighted_norm()` function to wrap it `Conv1d` layers which is not implemented in TorchSharp as of today. `weighted_norm()` decouples weights into their magnitude and direction. So every `Conv1d` layer's weight is now split into two : `parametrization.weight.original0` and  `parametrization.weight.original1`. So when this model is to be loaded into TorchSharp the weights have to be reconstructed in the state_dict and the above two keys deleted. `scripts\make_unnormed_torchsharp.py` does precisely that. 


