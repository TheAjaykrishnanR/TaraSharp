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
			ContextSize = 4096,
			GpuLayerCount = 30,
			Threads = 16,
			BatchSize = 1024,

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

		if (
			any(le(codes[0], 0)).ToBoolean() || any(ge(codes[0], 4096)).ToBoolean() ||
			any(le(codes[1], 0)).ToBoolean() || any(ge(codes[1], 4096)).ToBoolean() ||
			any(le(codes[2], 0)).ToBoolean() || any(ge(codes[2], 4096)).ToBoolean()
		) { Console.WriteLine("code over return"); return null; }

		short[] audio_int16;
		using(NewDisposeScope())
		using (_ = inference_mode())
		{
			Tensor audio_hat = model.decode(codes);
			//Console.WriteLine(audio_hat.print());
			Tensor audio_slice = audio_hat.narrow(dim: -1, start: 2048, length: 2048);
			audio_int16 = (audio_slice * 32767).to(int16).data<short>().ToArray();
		}
		//Console.WriteLine($"min: {audio_int16.Min()}, max: {audio_int16.Max()}");
		byte[] audio_bytes = new byte[audio_int16.Length * sizeof(short)];
		Buffer.BlockCopy(audio_int16, 0, audio_bytes, 0, audio_bytes.Length);
		return audio_bytes;
	}
	int turn_token_into_ids(int token, int count)
	{
		return token - 10 - ((count % 7) * 4096);
	}

	string TOKEN_PREFIX = "<custom_token_";
	public int parse_piece_to_token(string piece)
	{
		if (!piece.Contains(TOKEN_PREFIX)) { return 0; }
		string token_str = piece.Replace(TOKEN_PREFIX, "");
		int token = Convert.ToInt32(token_str.Replace(">", ""));
		return token;
	}

	MemoryStream audio_stream_buffer;
	WaveFileWriter wav_writer;
	public async Task speech_gen(string text)
	{
		InferenceParams inferPrams = new()
		{
			MaxTokens = 600,
		};
		string prompt = makeOrpheusPrompt(text);
		List<int> ids = new();

		audio_stream_buffer = new(8192);
		wav_writer = new(audio_stream_buffer, new WaveFormat(24000, 1));
		IAsyncEnumerable<string> reply = executor.InferAsync(prompt, inferenceParams: inferPrams);
		await foreach (string piece in reply)
		{
			int id = turn_token_into_ids(parse_piece_to_token(piece), ids.Count);
			if (id > 0)
			{
				ids.Add(id);
			}

			if (ids.Count % 7 == 0 && ids.Count > 27)
			{
				var bytes = convert_to_audio(ids.TakeLast(28).ToArray());
				wav_writer.Write(bytes, 0, bytes.Length);
			}
			//Console.WriteLine($"{piece}");
		}
		Console.WriteLine("wav_writing finished");
	}

	public async Task text_to_wav_file(string text, string output_file = @"outputs\tara.wav")
	{
		Stopwatch sw = new();
		using (FileStream wav_file = File.OpenWrite(output_file))
		{
			sw.Start();
			await speech_gen(text);
			sw.Stop();
			wav_writer.Flush(); // call Flush() before writing to file inorder to update the fmt chunk
			audio_stream_buffer.WriteTo(wav_file);
			wav_writer.Dispose();
		}
		long generation_time = sw.ElapsedMilliseconds;
		double audio_duration = new MediaFoundationReader(output_file).TotalTime.TotalMilliseconds;
		Console.WriteLine($"Finished, total: {generation_time}, rtf: {audio_duration / generation_time}");
	}

	public async Task stream_tts(string text) { }

	public static async Task Main()
	{
		Tara tara = new();

		while (true)
		{
			Console.Write("Enter text: ");
			string? text = Console.ReadLine();
			if (text != ":q")
			{
				await tara.text_to_wav_file(text);
			}
			else
			{
				break;
			}
		}
	}

}
