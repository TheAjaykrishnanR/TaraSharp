using NAudio.Wave;
using Vosk;
using Newtonsoft.Json;

public class Listener
{
	WaveFormat wave_format;
	BufferedWaveProvider capture_buffer;
	WaveInEvent ears = new();
	Model vosk_model = new(@"vosk\vosk-model-small-en-us-0.15");
	VoskRecognizer rec;

	// a continuous string of words cpatured in one go without interruptions
	string last_sentence = "";

	listener_state state;
	long last_spoken;

	// PROMPT_READY event
	public delegate void prompt_ready(string prompt);
	public event prompt_ready PROMPT_READY = delegate { };

	public Listener(int sample_rate = 16000)
	{
		wave_format = new(sample_rate, 1);
		capture_buffer = new(wave_format);
		capture_buffer.DiscardOnBufferOverflow = true;
		ears.WaveFormat = wave_format;
		ears.DataAvailable += on_data_available_callback;
		rec = new(vosk_model, sample_rate);
		rec.SetWords(true);
		rec.SetMaxAlternatives(0);
		ears.StartRecording();
	}

	int counter = 0;
	string live_detection;
	void on_data_available_callback(object? sender, WaveInEventArgs e)
	{
		byte[] recorded_buffer = e.Buffer;
		int recorded_byte_length = e.BytesRecorded;
		capture_buffer.AddSamples(recorded_buffer.Take(recorded_byte_length).ToArray(), 0, recorded_byte_length);
		if (rec.AcceptWaveform(recorded_buffer, recorded_byte_length))
		{
			string full_detection = JsonConvert.DeserializeObject<dynamic>(rec.Result())["text"].Value;
			//Console.WriteLine($"full_extracted: {full_detection}");
		}
		else
		{
			live_detection = JsonConvert.DeserializeObject<dynamic>(rec.PartialResult())["partial"].Value;
			if (live_detection == "")
			{
				state = listener_state.LISTENING_SILENCE;
				if (last_sentence.Length > 0 && DateTimeOffset.Now.ToUnixTimeMilliseconds() - last_spoken > 500)
				{
					state = listener_state.THINKING;
					Console.WriteLine($"Sending querry: {last_sentence}");
					PROMPT_READY(last_sentence);
					last_sentence = "";
				}
			}
			else
			{
				state = listener_state.LISTENING_WORDS;
				last_sentence = live_detection;
				last_spoken = DateTimeOffset.Now.ToUnixTimeMilliseconds();
			}
		}
		Console.WriteLine($"{counter} => STATE: {state}, LAST_SENTENCE: {last_sentence}, LIVE: {live_detection}");
		counter++;
	}

	~Listener()
	{
		ears.StopRecording();
	}
}

// it is rather dogmatic and simplistic to assume that speaking supersedes thinking and that these two processes happen sequentially
// rather than in an interweaved fashion. But it is also reasonable to assume that the human brain doesnt multitask either and what
// happens when someone is speaking is some fashion of rapid switching between bursts of thinking and speaking
public enum listener_state
{
	LISTENING_WORDS,
	LISTENING_SILENCE,
	THINKING,
	// this has to be set from the outside 
	SPEAKING,
}
