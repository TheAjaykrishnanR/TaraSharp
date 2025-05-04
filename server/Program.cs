using LLama;
using LLama.Common;
using LLama.Sampling;
using Microsoft.AspNetCore.Builder;

public class Tara
{
	// llamasharp
	StatelessExecutor executor;

	public Tara()
	{
		string modelPath = @"E:\ai\orpheus-tts\orpheus_gguf\orpheus-3b-0.1-ft-q4_k_m.gguf";
		ModelParams modelParams = new(modelPath)
		{
			ContextSize = 1200,
			GpuLayerCount = 30,
			Threads = 10,
			BatchSize = 1024,
			FlashAttention = true
		};
		Console.WriteLine("[ INFO ] Loading Oepheus-TTS-1b...");
		executor = new(LLamaWeights.LoadFromFile(modelParams), modelParams);
	}

	static string makeOrpheusPrompt(string text)
	{
		string voice = "tara";
		string start = "<|audio|>";
		string end = "<|eot_id|>";
		string content = $"{voice}: {text}";

		return $"{start}{content}{end}";
	}

	InferenceParams inferPrams = new()
	{
		MaxTokens = -1,
		SamplingPipeline = new DefaultSamplingPipeline()
		{
			RepeatPenalty = 1.1f,
			Temperature = 0.6f,
			TopP = 0.95f,
		},
	};

	public IAsyncEnumerable<string> do_inference(string text)
	{
		string prompt = makeOrpheusPrompt(text);
		return executor.InferAsync(prompt, inferenceParams: inferPrams);
	}
}

public partial class Program
{
	public static async Task Main()
	{
		var app = WebApplication.Create();
		//Tara tara = new();
		app.MapPost("/", async (HttpContext context) =>
		{
			IFormCollection formCollection = await context.Request.ReadFormAsync();
			string text_to_speak = formCollection["text"];
			Console.WriteLine($"recieved POST: {text_to_speak}");
			StreamWriter sw = new(context.Response.Body);
			/*
			await foreach (string text_token in tara.do_inference(text_to_speak))
			{
				await sw.WriteAsync(text_token);
				Console.WriteLine(text_token);
			}*/
			foreach (string text_token in File.ReadAllLines("lined_tokens.txt"))
			{
				await sw.WriteAsync(text_token);
				await sw.FlushAsync();
				Console.WriteLine(text_token);
				//await Task.Delay(500);
			}
		});
		app.Run();
	}
}
