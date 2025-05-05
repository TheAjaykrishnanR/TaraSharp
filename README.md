# Orpheus-TTS local speech synthesizer written entirely in C#

![cs](https://github.com/TheAjaykrishnanR/TaraSharp/blob/master/imgs/cs.png)

written from : [isaiahbjork/orpheus-tts-local](https://github.com/isaiahbjork/orpheus-tts-local)

### Requirements

```
.NET 9
Cuda 12 (not required when running with colab)
```

## Usage

### Google Colab
1. Server: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1lu2Mocc3UU4RtSOgZCm66sf0DlIwQNmi?usp=sharing)
    1. Install .NET (press Enter in the colab output cell to confirm installation)
    2. Enter your NGROK auth key and copy the public url
    3. Download the GGUF file
    4. Run the server (Wait till you see the localhost link)

2. Client
    1. `git clone https://github.com/TheAjaykrishnanR/TaraSharp --branch colab`
    2. `cd TaraSharp/client`
    3. `dotnet run`
    4. Enter the NGROK public url that you have copied earlier
    5. Enter text

### Local 

1. Clone the git repo:

 - `git clone https://github.com/TheAjaykrishnanR/TaraSharp`

2. Download the the quantized model from [here](https://huggingface.co/isaiahbjork/orpheus-3b-0.1-ft-Q4_K_M-GGUF)

3. Download the decoder weights from [here](https://github.com/TheAjaykrishnanR/TaraSharp/releases/download/files/pytorch_model_unnormed.bin) and place it in `snac\24khz` alongside `config.json`. If you want to use the original weights, see conversion method below.

4. `cd TaraSharp`

5. Change `model_path` in `Program.cs` to the location where your gguf file is.

6. `dotnet run`

### Additional Info

The original snac model available [here](https://huggingface.co/hubertsiuzdak/snac_24khz) was created using PyTorch. For the most part models created in PyTorch can be loaded without much hassle in TorchSharp (not vice versa). However the snac decoder uses the `weighted_norm()` function to wrap `Conv1d` layers which is not implemented in TorchSharp as of today. `weighted_norm()` decouples weights into their magnitude and direction. So every `Conv1d` layer's weight is now split into two : `parametrization.weight.original0` and  `parametrization.weight.original1`. So when this model is to be loaded into TorchSharp the weights have to be reconstructed in the state_dict and the above two keys deleted. `scripts\make_unnormed_torchsharp.py` does precisely that. 


