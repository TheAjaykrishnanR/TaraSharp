using LLama;
using LLama.Native;
using LLama.Common;
using LLama.Sampling;
using Microsoft.AspNetCore.Builder;

public class Tara
{
	// llamasharp
	StatelessExecutor executor;

	public Tara()
	{
		string modelPath = @"models/orpheus-3b-0.1-ft-q4_k_m.gguf";
		ModelParams modelParams = new(modelPath)
		{
			ContextSize = 8600,
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
		NativeLibraryConfig.All
			.WithCuda()
			.SkipCheck(true)
			.WithAutoFallback(false)
			.WithLogCallback((level, message) => Console.Write($"{level}: {message}"));
		NativeApi.llama_empty_call();

		var app = WebApplication.Create();
		Tara tara = new();
		app.MapPost("/", async (HttpContext context) =>
		{
			IFormCollection formCollection = await context.Request.ReadFormAsync();
			string text_to_speak = formCollection["text"];
			Console.WriteLine($"recieved POST: {text_to_speak}");
			StreamWriter sw = new(context.Response.Body);
			await foreach (string text_token in tara.do_inference(text_to_speak))
			{
				await sw.WriteAsync(text_token);
				await sw.FlushAsync();
			}
		});
		app.Run();
	}
}
