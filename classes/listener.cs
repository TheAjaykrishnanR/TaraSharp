using NAudio.Wave;
using Vosk;

public class Listener
{
	WaveFormat wave_format;
	BufferedWaveProvider capture_buffer;
	WaveInEvent ears = new();
	Model vosk_model = new(@"vosk\vosk-model-small-en-us-0.15");
	VoskRecognizer rec;

	public Listener(int sample_rate = 24000)
	{
		wave_format = new(sample_rate, 1);
		capture_buffer = new(wave_format);
		capture_buffer.DiscardOnBufferOverflow = true;
		ears.WaveFormat = wave_format;
		ears.DataAvailable += on_data_available_callback;
		rec = new(vosk_model, 24000);
		rec.SetWords(true);
		rec.SetMaxAlternatives(0);
		ears.StartRecording();
	}

	void on_data_available_callback(object? sender, WaveInEventArgs e)
	{
		byte[] recorded_buffer = e.Buffer;
		int recorded_byte_length = e.BytesRecorded;
		capture_buffer.AddSamples(recorded_buffer.Take(recorded_byte_length).ToArray(), 0, recorded_byte_length);
		if (rec.AcceptWaveform(recorded_buffer, recorded_byte_length))
		{
			Console.WriteLine(rec.Result());
		}
		else
		{
			Console.WriteLine(rec.PartialResult());
		}
	}
}
