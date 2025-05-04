using TorchSharp;
using static TorchSharp.torch;
using NAudio.Wave;
using System.Diagnostics;
using Vosk;
using Microsoft.Extensions.Configuration;
using System.Text.Json;
using System.Net.Http.Headers;
using Newtonsoft.Json;
using OpenAI;
using OpenAI.Chat;
using System.ClientModel;

public class Client
{
	// snac decoder
	Snac model;
	Device snac_device;

	// speech recognition
	Model vosk_model = new(@"vosk\vosk-model-small-en-us-0.15");
	VoskRecognizer vosk_rec;

	string server_url;

	public Client(string server_url)
	{
		this.server_url = server_url;

		// snac decoder
		Console.WriteLine("[ INFO ] Loading the SNAC decoder...");
		model = Snac.from_pretrained(@"snac\24khz\config.json", @"snac\24khz\pytorch_model_unnormed.bin");
		snac_device = device("cuda:0");
		model = model.to(snac_device);

		// naudio live audio streaming
		audio_buffered_stream = new(wave_format);
		audio_buffered_stream.DiscardOnBufferOverflow = true;
		player.Init(audio_buffered_stream);

	}

	public async Task<IAsyncEnumerable<string>> send_to_server(string text)
	{
		HttpClient http = new();
		var response = await http.PostAsync(server_url, );

		char[] buffer = new char[100];
		int charsRead = 0;
		string token = "";
		using (StringReader sr = new(await response.Content.ReadAsStreamAsync()))
		{
			while ((charsRead = sr.Read(buffer, 0, buffer.Length)) > 0)
			{
				token = string.Join("", buffer.Take(charsRead));
				yield return token;
			}
		}
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

	public async Task decode_server_reply_and_fill_audio(IAsyncEnumerable<string> text_tokens)
	{
		List<int> ids = new();

		if (!streaming)
		{
			memory_stream = new(8192);
			wav_writer = new(memory_stream, wave_format);
		}

		await foreach (string piece in text_tokens)
		{
			int id = turn_token_into_ids(parse_piece_to_token(piece), ids.Count);
			if (id > 0)
			{
				ids.Add(id);
			}

			if (ids.Count % 7 == 0 && ids.Count > 27)
			{
				byte[]? bytes = convert_to_audio(ids.TakeLast(28).ToArray());
				if (bytes == null) { continue; }
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
		if (!streaming) wav_writer.Flush();
		Console.WriteLine("[ EVENT ] audio token generation finished");
	}

	int REQUIRED_BUFFER_DURATION = 300;
	int SLEEP_DURATION = 1000;
	void TimerCallback(object? sender, EventArgs e)
	{
		double buffered = audio_buffered_stream.BufferedDuration.TotalMilliseconds;
		if (buffered < REQUIRED_BUFFER_DURATION)
		{
			REQUIRED_BUFFER_DURATION = 100;
			player.Pause();
			Console.WriteLine($"[ EVENT ] [buf: {buffered}ms] Thread sleeping, waiting for audio to buffer...");
			Thread.Sleep(SLEEP_DURATION);
			player.Play();
		}
	}

	int SPEECH_START_DELAY = 500;
	public async Task talk(string text, string output_file = @"outputs\tara.wav")
	{
		Stopwatch sw = new();
		sw.Start();
		System.Timers.Timer timer = new(100);
		if (streaming)
		{
			Task _t = Task.Run(async () =>
			{
				timer.Elapsed += TimerCallback;
				Console.WriteLine($"[ INFO ] Waiting for {SPEECH_START_DELAY}ms");
				await Task.Delay(SPEECH_START_DELAY); // let speech_gen() generate tokens and fill the buffer
				timer.Start();
				player.Play();
			});
		}
		Console.WriteLine("\n[ EVENT ] calling server()");
		await decode_server_reply_and_fill_audio(await send_to_server(text));
		timer.Stop();
		sw.Stop();
		if (!streaming)
		{
			FileStream wav_file = File.OpenWrite(output_file);
			memory_stream.WriteTo(wav_file);
			wav_file.Close();
			long generation_time = sw.ElapsedMilliseconds;
			double audio_duration = new MediaFoundationReader(output_file).TotalTime.TotalMilliseconds;
			Console.WriteLine($"[ EVENT ] Finished, generation_time: {generation_time} ms, rtf: {audio_duration / generation_time}");
		}
	}
}

public partial class Program
{
	public static void Main()
	{
		string server_url = "http://localhost:5000";
		Client client = new(server_url);
		Listener listener = new();
		listener.PROMPT_READY += async (string text) =>
		{
			listener.state = listener_state.SPEAKING;
			Console.Write($"\n[ EVENT ] prompt_ready(): {text}");
			await client.talk(text);
			listener.state = listener_state.LISTENING_SILENCE;
		};
		Console.ReadLine();
	}
}
