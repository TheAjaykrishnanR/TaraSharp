using LLama;
using LLama.Common;
using TorchSharp;
using static TorchSharp.torch;
using NAudio.Wave;

public class Tara
{
	Snac model;
	Device snac_device;

	public Tara()
	{
		model = Snac.from_pretrained(@"snac\24khz\config.json", @"snac\24khz\pytorch_model_unnormed.bin");
		snac_device = device("cuda:0");
		model = model.to(snac_device);
	}

	static string makeOrpheusPrompt(string text)
	{
		string voice = "tara";
		string start = "<|audio|>";
		string end = "<|eot_id|>";
		string content = $"{voice}: {text}";

		return $"{start}{content}{end}";
	}

	byte[]? convert_to_audio(int[] multiframe)
	{
		int[] frames = [];
		if (multiframe.Length < 7) { Console.WriteLine("multiframe return"); return null; }

		Tensor codes_0 = tensor(Array.Empty<int>(), device: snac_device, dtype: int32);
		Tensor codes_1 = tensor(Array.Empty<int>(), device: snac_device, dtype: int32);
		Tensor codes_2 = tensor(Array.Empty<int>(), device: snac_device, dtype: int32);

		int num_frames = multiframe.Length / 7;
		frames = multiframe.Take(num_frames * 7).ToArray();

		for (int j = 0; j < num_frames; j++)
		{
			int i = 7 * j;
			if (codes_0.shape[0] == 0)
			{
				codes_0 = tensor([frames[i]], device: snac_device, dtype: int32);
			}
			else
			{
				codes_0 = cat([codes_0, tensor([frames[i]], device: snac_device, dtype: int32)]);
			}
			if (codes_1.shape[0] == 0)
			{
				codes_1 = tensor([frames[i + 1]], device: snac_device, dtype: int32);
				codes_1 = cat([codes_1, tensor([frames[i + 4]], device: snac_device, dtype: int32)]);
			}
			else
			{
				codes_1 = cat([codes_1, tensor([frames[i + 1]], device: snac_device, dtype: int32)]);
				codes_1 = cat([codes_1, tensor([frames[i + 4]], device: snac_device, dtype: int32)]);
			}

			if (codes_2.shape[0] == 0)
			{
				codes_2 = tensor([frames[i + 2]], device: snac_device, dtype: int32);
				codes_2 = cat([codes_2, tensor([frames[i + 3]], device: snac_device, dtype: int32)]);
				codes_2 = cat([codes_2, tensor([frames[i + 5]], device: snac_device, dtype: int32)]);
				codes_2 = cat([codes_2, tensor([frames[i + 6]], device: snac_device, dtype: int32)]);
			}
			else
			{
				codes_2 = cat([codes_2, tensor([frames[i + 2]], device: snac_device, dtype: int32)]);
				codes_2 = cat([codes_2, tensor([frames[i + 3]], device: snac_device, dtype: int32)]);
				codes_2 = cat([codes_2, tensor([frames[i + 5]], device: snac_device, dtype: int32)]);
				codes_2 = cat([codes_2, tensor([frames[i + 6]], device: snac_device, dtype: int32)]);
			}
		}
		List<Tensor> codes = [codes_0.unsqueeze(0), codes_1.unsqueeze(0), codes_2.unsqueeze(0)];
		/*	
		Console.WriteLine(codes[0].print());
		Console.WriteLine(codes[1].print());
		Console.WriteLine(codes[2].print());
		*/
		if (
			any(le(codes[0], 0)).ToBoolean() || any(ge(codes[0], 4096)).ToBoolean() ||
			any(le(codes[1], 0)).ToBoolean() || any(ge(codes[1], 4096)).ToBoolean() ||
			any(le(codes[2], 0)).ToBoolean() || any(ge(codes[2], 4096)).ToBoolean()
		) { Console.WriteLine("code over return"); return null; }

		Tensor audio_hat;
		using (_ = inference_mode())
		{
			audio_hat = model.decode(codes);
			//Console.WriteLine(audio_hat.print());
		}

		Tensor audio_slice = audio_hat.narrow(dim: -1, start: 2048, length: 2048);
		short[] audio_int16 = (audio_slice * 32767).to(int16).data<short>().ToArray();
		//Console.WriteLine($"min: {audio_int16.Min()}, max: {audio_int16.Max()}");
		byte[] audio_bytes = new byte[audio_int16.Length * sizeof(short)];
		Buffer.BlockCopy(audio_int16, 0, audio_bytes, 0, audio_bytes.Length);
		return audio_bytes;
	}
	int turn_token_into_ids(int token, int count)
	{
		/*
		token_string = token_string.Replace(" ", "");
		int last_token_start = token_string.LastIndexOf("<custom_token_");
		string last_token
        */
		return token - 10 - ((count % 7) * 4096);

	}

	public static async Task Main()
	{

		/*
		string modelPath = @"../isaiahbjork-orpheus-3b-0.1-ft-Q4_K_M-GGUF/orpheus-3b-0.1-ft-q4_k_m.gguf";
		ModelParams modelParams = new(modelPath) { ContextSize = 1024 };
		StatelessExecutor executor = new(LLamaWeights.LoadFromFile(modelParams), modelParams);
		IAsyncEnumerable<string> reply = executor.InferAsync(makeOrpheusPrompt("Hello, I am Tara"));
		await foreach (string piece in reply)
		{
			Console.Write($"{piece} ");
		}*/


		Tara tts = new();
		Console.WriteLine("Starting to write audio...");
		string[] token_strings = File.ReadAllLines(@"examples\lined_tokens.txt");
		token_strings = token_strings.Select(x => x.Replace("<custom_token_", "").Replace(">", "")).ToArray();
		int[] tokens = new int[token_strings.Length];
		for (int i = 0; i < tokens.Length; i++)
		{
			tokens[i] = Convert.ToInt32(token_strings[i]);
		}
		List<int> ids = new();
		int id = 0, count = 0;
		foreach (int token in tokens)
		{
			id = tts.turn_token_into_ids(token, count);
			//Console.WriteLine($"token: {token}, id: {id}, count: {count}");
			if (id > 0)
			{
				ids.Add(id);
				count++;
			}
		}

		List<byte> audio_bytes = new();
		for (int i = 28; i < ids.Count(); i++)
		{
			if (i % 7 == 0)
			{
				var bytes = tts.convert_to_audio(ids[(i - 28)..i].ToArray());
				audio_bytes.AddRange(bytes);
			}
		}
		File.Delete(@"outputs\new_tara.wav");
		Console.WriteLine($"min: {audio_bytes.Min()}, max: {audio_bytes.Max()}");
		using (FileStream wav_file = File.OpenWrite(@"outputs\cs_tara.wav"))
		{
			WaveFileWriter wav_writer = new(wav_file, new WaveFormat(24000, 1));
			wav_writer.Write(audio_bytes.ToArray(), 0, audio_bytes.Count());
		}
		Console.WriteLine("Finished writing audio !");
	}

}
