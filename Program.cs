using LLama;
using LLama.Common;
using TorchSharp;
using static TorchSharp.torch;
using NAudio.Wave;
using System.Diagnostics;
using Vosk;

public class Tara
{
	// llamasharp
	StatelessExecutor executor;

	// snac decoder
	Snac model;
	Device snac_device;

	// speech recognition
	Model vosk_model = new(@"vosk\vosk-model-small-en-us-0.15");
	VoskRecognizer vosk_rec;

	public Tara()
	{
		/*
		string modelPath = @"E:\ai\orpheus-tts\orpheus_gguf\orpheus-3b-0.1-ft-q4_k_m.gguf";
		ModelParams modelParams = new(modelPath)
		{
			ContextSize = 4096,
			GpuLayerCount = 30,
			Threads = 16,
			BatchSize = 1024,

		};
		executor = new(LLamaWeights.LoadFromFile(modelParams), modelParams);

		// snac decoder
		model = Snac.from_pretrained(@"snac\24khz\config.json", @"snac\24khz\pytorch_model_unnormed.bin");
		snac_device = device("cuda:0");
		model = model.to(snac_device);

		// naudio live audio streaming
		audio_buffered_stream = new(wave_format);
		player.Init(audio_buffered_stream);
		*/
		// vosk asr
		vosk_rec = new(vosk_model, 24000);
		vosk_rec.SetWords(true);
		vosk_rec.SetMaxAlternatives(0);
		audio_input_buffered = new(wave_format);
		audio_input_buffered.DiscardOnBufferOverflow = true;
		listener.WaveFormat = wave_format;
		listener.DataAvailable += listener_callback;
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
		using (NewDisposeScope())
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

	// <summary>
	// streaming infra
	// </summary>
	WaveFormat wave_format = new(24000, 1);
	BufferedWaveProvider audio_buffered_stream;
	MemoryStream memory_stream;
	WaveOutEvent player = new();
	WaveFileWriter wav_writer;
	public bool streaming = true;

	public async Task speech_gen(string text)
	{
		InferenceParams inferPrams = new()
		{
			MaxTokens = 600,
		};
		string prompt = makeOrpheusPrompt(text);
		List<int> ids = new();

		memory_stream = new(8192);
		wav_writer = new(memory_stream, wave_format);
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
				byte[] bytes = convert_to_audio(ids.TakeLast(28).ToArray());
				if (streaming)
				{
					audio_buffered_stream.AddSamples(bytes, 0, bytes.Length);
				}
				else
				{
					await wav_writer.WriteAsync(bytes, 0, bytes.Length);
				}
			}
		}
		wav_writer.Flush();
		Console.WriteLine("wav_writing finished");
	}

	void TimerCallback(object? sender, EventArgs e)
	{
		double buffered = audio_buffered_stream.BufferedDuration.TotalMilliseconds;
		if (buffered < 100)
		{
			player.Pause();
			Console.WriteLine($"Thread Sleeping");
			Thread.Sleep(200);
			player.Play();
		}
	}

	public async Task talk(string text, string output_file = @"outputs\tara.wav")
	{
		Stopwatch sw = new();
		sw.Start();
		System.Timers.Timer timer = new(100);
		if (streaming)
		{
			Task _t = Task.Run(() =>
			{
				timer.Elapsed += TimerCallback;
				timer.Start();
				player.Play();
			});
		}
		await speech_gen(text);
		timer.Stop();
		sw.Stop();
		if (!streaming)
		{
			FileStream wav_file = File.OpenWrite(output_file);
			memory_stream.WriteTo(wav_file);
			wav_file.Close();
			long generation_time = sw.ElapsedMilliseconds;
			double audio_duration = new MediaFoundationReader(output_file).TotalTime.TotalMilliseconds;
			Console.WriteLine($"Finished, generation_time: {generation_time} ms, rtf: {audio_duration / generation_time}");
		}
	}

	// <summary>
	// naudio input
	// </summary>
	public WaveInEvent listener = new();
	BufferedWaveProvider audio_input_buffered;
	void listener_callback(object? sender, WaveInEventArgs e)
	{
		Console.WriteLine("listener_callabck() fired");
		byte[] buffer = e.Buffer;
		int recorded = e.BytesRecorded;
		audio_input_buffered.AddSamples(buffer, 0, recorded);
		if (vosk_rec.AcceptWaveform(buffer, recorded))
		{
			Console.WriteLine(vosk_rec.Result());
		}
		else
		{
			Console.WriteLine(vosk_rec.PartialResult());
		}
	}

	public static async Task Main()
	{
		Tara tara = new();
		/*
		while (true)
		{
			Console.Write("Enter text: ");
			string? text = Console.ReadLine();
			if (text != ":q")
			{
				await tara.talk(text);
			}
			else
			{
				break;
			}
		}*/

		tara.listener.StartRecording();
		Console.ReadLine();
		/*
		WaveOutEvent _player = new();
		_player.Init(tara.audio_input_buffered);
		_player.Play();
		Console.ReadLine();
		*/

	}

}
