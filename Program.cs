using LLama;
using LLama.Common;
using TorchSharp;
using static TorchSharp.torch;
using NAudio.Wave;
using System.Diagnostics;

public class Tara
{
	StatelessExecutor executor;

	Snac model;
	Device snac_device;

	public Tara()
	{
		string modelPath = @"E:\ai\orpheus-tts\orpheus_gguf\orpheus-3b-0.1-ft-q4_k_m.gguf";
		ModelParams modelParams = new(modelPath)
		{
			ContextSize = 1024,
			GpuLayerCount = 28,
			Threads = 8,
			BatchSize = 512,
		};
		executor = new(LLamaWeights.LoadFromFile(modelParams), modelParams);

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
		int[] frames;
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

	public int parse_piece_to_token(string piece)
	{
		if (!piece.Contains(TOKEN_PREFIX)) { return 0; }
		string token_str = piece.Replace(TOKEN_PREFIX, "");
		int token = Convert.ToInt32(token_str.Replace(">", ""));
		return token;
	}

	string TOKEN_PREFIX = "<custom_token_";
	Stopwatch sw = new();
	public async Task text_to_wav_file(string text, string output_file = @"outputs\tara.wav")
	{
		if (File.Exists(output_file)) { File.Delete(output_file); }

		InferenceParams inferPrams = new()
		{
			MaxTokens = 600,
		};
		string prompt = makeOrpheusPrompt(text);
		Console.WriteLine($"prompt: {prompt}");

		List<int> ids = new();
		sw.Start();

		FileStream wav_file = File.OpenWrite(output_file);
		WaveFileWriter wav_writer = new(wav_file, new WaveFormat(24000, 1));
		IAsyncEnumerable<string> reply = executor.InferAsync(prompt, inferenceParams: inferPrams);
		await foreach (string piece in reply)
		{
			int id = turn_token_into_ids(parse_piece_to_token(piece), ids.Count);
			//Console.WriteLine($"piece: {piece}, id: {id}");
			if (id > 0)
			{
				ids.Add(id);
			}

			if (ids.Count % 7 == 0 && ids.Count > 27)
			{
				//Console.WriteLine($"{ids.Count}");
				var bytes = convert_to_audio(ids.TakeLast(28).ToArray());
				wav_writer.Write(bytes, 0, bytes.Length);
			}
		}
		wav_writer.Dispose();
		wav_file.Close();
		sw.Stop();
		long generation_time = sw.ElapsedMilliseconds;
		double audio_duration = (new MediaFoundationReader(output_file)).TotalTime.TotalMilliseconds;
		Console.WriteLine($"Finished, total: {generation_time}, rtf: {audio_duration / generation_time}");

		/*
		List<byte> audio_bytes = new();
		for (int i = 28; i < ids.Count(); i++)
		{
			if (i % 7 == 0)
			{
				var bytes = convert_to_audio(ids[(i - 28)..i].ToArray());
				audio_bytes.AddRange(bytes);
			}
		}
		Console.WriteLine($"min: {audio_bytes.Min()}, max: {audio_bytes.Max()}");
		if (File.Exists(output_file)) { File.Delete(output_file); }
		using (FileStream wav_file = File.OpenWrite(output_file))
		{
			WaveFileWriter wav_writer = new(wav_file, new WaveFormat(24000, 1));
			wav_writer.Write(audio_bytes.ToArray(), 0, audio_bytes.Count());
			wav_writer.Dispose();
		}

		double audio_duration = (new MediaFoundationReader(output_file)).TotalTime.TotalMilliseconds;

		Console.WriteLine($"Finished writing audio, rtf: {audio_duration / generation_time}");
		*/
	}

	public static async Task Main()
	{
		Tara tara = new();

		while (true)
		{
			Console.Write("Enter text: ");
			string? text = Console.ReadLine();
			await tara.text_to_wav_file(text);
		}

		/*
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

		*/
	}

}
