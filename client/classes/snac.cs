using TorchSharp;
using static TorchSharp.torch;
using TorchSharp.Modules;
using System.Text.Json;
using TorchSharp.PyBridge;

class configJSON
{
	public int sampling_rate { get; set; }
	public int encoder_dim { get; set; }
	public int[] encoder_rates { get; set; }
	public int latent_dim { get; set; }
	public int decoder_dim { get; set; }
	public int[] decoder_rates { get; set; }
	public int? attn_window_size { get; set; }
	public int codebook_size { get; set; }
	public int codebook_dim { get; set; }
	public int[] vq_strides { get; set; }
	public bool noise { get; set; }
	public bool depthwise { get; set; }
}

public class Snac : nn.Module
{
	int sampling_rate;
	int encoder_dim;
	int[] encoder_rates;
	int latent_dim;
	int decoder_dim;
	int[] decoder_rates;
	int? attn_window_size;
	int codebook_size;
	int codebook_dim;
	int[] vq_strides;
	bool noise;
	bool depthwise;

	int hop_length;
	int n_codebooks;

	Encoder encoder;
	Decoder decoder;
	public ResidualVectorQuantize quantizer;

	public Snac(
		int sampling_rate = 44100,
		int encoder_dim = 64,
		int[] encoder_rates = null,
		int? latent_dim = null,
		int decoder_dim = 1536,
		int[] decoder_rates = null,
		int? attn_window_size = 32,
		int codebook_size = 4096,
		int codebook_dim = 8,
		int[] vq_strides = null,
		bool noise = true,
		bool depthwise = true
	) : base("Snac")
	{
		this.sampling_rate = sampling_rate;
		this.encoder_dim = encoder_dim;
		this.encoder_rates = encoder_rates ?? [3, 3, 7, 7];
		this.latent_dim = latent_dim ?? (int)(encoder_dim * (Math.Pow(2, encoder_rates.Length)));
		this.decoder_dim = decoder_dim;
		this.decoder_rates = decoder_rates ?? [7, 7, 3, 3];
		this.attn_window_size = attn_window_size;
		this.codebook_size = codebook_size;
		this.codebook_dim = codebook_dim;
		this.vq_strides = vq_strides ?? [8, 4, 2, 1];
		this.noise = noise;
		this.depthwise = depthwise;

		hop_length = 1;
		foreach (int i in this.encoder_rates)
		{
			hop_length *= i;
		}
		n_codebooks = this.vq_strides.Length;

		encoder = new(this.encoder_dim, this.encoder_rates, depthwise: this.depthwise, attn_window_size: this.attn_window_size);
		quantizer = new(input_dim: this.latent_dim, codebook_size: this.codebook_size, codebook_dim: this.codebook_dim, vq_strides: this.vq_strides);
		decoder = new(this.latent_dim, this.decoder_dim, this.decoder_rates, this.noise, depthwise: this.depthwise, attn_window_size: this.attn_window_size);

		RegisterComponents();
	}

	int _lcm(int a, int b)
	{
		int _a = a;
		int _b = b;
		while (a != b)
		{
			if (a > b) { a -= b; }
			else { b -= a; }
		}
		return (_a * _b) / a;
	}

	public Tensor preprocess(Tensor audio_data)
	{
		long length = audio_data.shape[^1];
		int __lcm = _lcm(vq_strides[0], attn_window_size ?? 1);
		int pad_to = hop_length * __lcm;
		int right_pad = Convert.ToInt32(Math.Ceiling((double)length / pad_to) * pad_to - length);
		audio_data = nn.functional.pad(audio_data, (0, right_pad));
		return audio_data;
	}

	public (Tensor, List<Tensor>) forward(Tensor audio_data)
	{
		long length = audio_data.shape[^1];
		audio_data = preprocess(audio_data);
		Tensor z = encoder.forward(audio_data);
		(Tensor z_q, List<Tensor> codes) = quantizer.forward(z);
		Tensor audio_hat = decoder.forward(z_q);
		return (audio_hat.narrow(dim: -1, start: 0, length: length), codes);
	}

	public List<Tensor> encode(Tensor audio_data)
	{
		audio_data = preprocess(audio_data);
		Tensor z = encoder.forward(audio_data);
		(_, List<Tensor> codes) = quantizer.forward(z);
		return codes;
	}

	public Tensor decode(List<Tensor> codes)
	{
		Tensor z_q = quantizer.from_codes(codes);
		//Console.WriteLine(z_q.print());
		Tensor audio_hat = decoder.forward(z_q); // leaking 
		return audio_hat;

	}

	public static Snac from_config(string config_path)
	{
		string text = File.ReadAllText(config_path);
		configJSON config = JsonSerializer.Deserialize<configJSON>(text);
		return new(
			sampling_rate: config.sampling_rate,
			encoder_dim: config.encoder_dim,
			encoder_rates: config.encoder_rates,
			decoder_dim: config.decoder_dim,
			decoder_rates: config.decoder_rates,
			attn_window_size: config.attn_window_size,
			codebook_size: config.codebook_size,
			codebook_dim: config.codebook_dim,
			vq_strides: config.vq_strides,
			noise: config.noise,
			depthwise: config.depthwise
		);
	}

	public static Snac from_pretrained(string config_path, string model_path)
	{
		Snac model = from_config(config_path);
		model.load_py(model_path);
		model.eval();
		return model;
	}
}

// <summary>
// Encoder & Decoder
// </summary>

class Snake1d : nn.Module<Tensor, Tensor>
{
	Parameter alpha;

	public Snake1d(int channels) : base("Snake1d")
	{
		alpha = new(ones(new long[] { 1, channels, 1 }));

		RegisterComponents();
	}

	Tensor snake(Tensor x, Parameter alpha)
	{
		long[] shape = x.shape;
		x = x.reshape(shape[0], shape[1], -1);
		x += (alpha + 1e-9).reciprocal() * sin(alpha * x).pow(2);
		x = x.reshape(shape);
		return x;
	}

	public override Tensor forward(Tensor x)
	{
		return snake(x, alpha);
	}
}

class ResidualUnit : nn.Module<Tensor, Tensor>
{
	List<nn.Module<Tensor, Tensor>> layers = new();
	Sequential block;
	public ResidualUnit(int dim = 16, int dilation = 1, int kernel = 7, int groups = 1) : base("ResidualUnit")
	{
		int pad = ((kernel - 1) * dilation) / 2;
		layers.Add(new Snake1d(dim));
		layers.Add(nn.Conv1d(dim, dim, kernel_size: kernel, dilation: dilation, padding: pad, groups: groups));
		layers.Add(new Snake1d(dim));
		layers.Add(nn.Conv1d(dim, dim, kernel_size: 1));

		block = nn.Sequential(layers);

		RegisterComponents();
	}

	public override Tensor forward(Tensor x)
	{
		Tensor y = block.forward(x);
		long pad = (x.shape.Last() - y.shape.Last()) / 2;
		if (pad > 0)
		{
			x = x.slice(-1, start: pad, finish: -pad, step: 1);
		}
		return x + y;
	}
}

class EncoderBlock : nn.Module<Tensor, Tensor>
{
	List<nn.Module<Tensor, Tensor>> layers = new();
	Sequential block;
	public EncoderBlock(int outputDim = 16, int? inputDim = null, int stride = 1, int groups = 1) : base("EncoderBlock")
	{
		inputDim = inputDim == null ? outputDim / 2 : inputDim;
		layers.Add(new ResidualUnit(inputDim.Value, dilation: 1, groups: 1));
		layers.Add(new ResidualUnit(inputDim.Value, dilation: 3, groups: 1));
		layers.Add(new ResidualUnit(inputDim.Value, dilation: 6, groups: 1));
		layers.Add(new Snake1d(inputDim.Value));
		layers.Add(nn.Conv1d(
			inputDim.Value,
			outputDim,
			kernel_size: 2 * stride,
			stride: stride,
			padding: (long)Math.Ceiling((double)stride / 2)
		));
		block = nn.Sequential(layers);

		RegisterComponents();
	}

	public override Tensor forward(Tensor x)
	{
		return block.forward(x);
	}
}
class Encoder : nn.Module
{
	List<nn.Module<Tensor, Tensor>> layers = new();
	Sequential block;

	public Encoder(
		int d_model = 64,
		int[]? strides = null,
		bool depthwise = false,
		int? attn_window_size = 32
	) : base("Encoder")
	{
		strides = strides ?? [3, 3, 7, 7];
		layers.Add(nn.Conv1d(1, d_model, kernel_size: 7, padding: 3));
		int groups;
		foreach (int stride in strides)
		{
			d_model *= 2;
			groups = depthwise ? d_model / 2 : 1;
			layers.Add(new EncoderBlock(outputDim: d_model, stride: stride, groups: groups));

		}
		if (attn_window_size != null)
		{
			layers.Add(new LocalMHA(dim: d_model, window_size: attn_window_size));
		}
		groups = depthwise ? d_model : 1;
		layers.Add(nn.Conv1d(d_model, d_model, kernel_size: 7, padding: 3, groups: groups));
		block = nn.Sequential(layers);

		RegisterComponents();

	}

	public Tensor forward(Tensor x)
	{
		return block.forward(x);
	}
}

class NoiseBlock : nn.Module<Tensor, Tensor>
{
	Conv1d linear;
	public NoiseBlock(int dim) : base("NoiseBlock")
	{
		linear = nn.Conv1d(dim, dim, kernel_size: 1, bias: false);

		RegisterComponents();
	}

	public override Tensor forward(Tensor x)
	{
		(long B, long C, long T) = (x.shape[0], x.shape[1], x.shape[2]);
		Tensor noise = randn([B, 1, T], device: x.device, dtype: x.dtype);
		Tensor h = linear.forward(x);
		Tensor n = noise * h;
		x += n;
		return x;
	}
}

class DecoderBlock : nn.Module<Tensor, Tensor>
{
	List<nn.Module<Tensor, Tensor>> layers = new();
	Sequential block;

	public DecoderBlock(
		int input_dim = 16,
		int output_dim = 8,
		int stride = 1,
		bool noise = false,
		int groups = 1
	) : base("DecoderBlock")
	{
		layers.Add(new Snake1d(input_dim));
		layers.Add(nn.ConvTranspose1d(
			input_dim,
			output_dim,
			kernel_size: 2 * stride,
			stride: stride,
			padding: (long)Math.Ceiling((double)stride / 2),
			output_padding: stride % 2
		));
		if (noise)
		{
			layers.Add(new NoiseBlock(output_dim));
		}
		layers.Add(new ResidualUnit(output_dim, dilation: 1, groups: groups));
		layers.Add(new ResidualUnit(output_dim, dilation: 3, groups: groups));
		layers.Add(new ResidualUnit(output_dim, dilation: 9, groups: groups));
		block = nn.Sequential(layers);

		RegisterComponents();
	}

	public override Tensor forward(Tensor x)
	{
		return block.forward(x);
	}
}

class Decoder : nn.Module
{

	List<nn.Module<Tensor, Tensor>> layers = new();
	Sequential model;

	public Decoder(
		int input_channels,
		int channels,
		int[] rates,
		bool noise = false,
		bool depthwise = false,
		int? attn_window_size = 32,
		int d_out = 1
	) : base("Decoder")
	{
		if (depthwise)
		{
			layers.Add(nn.Conv1d(input_channels, input_channels, kernel_size: 7, padding: 3, groups: input_channels));
			layers.Add(nn.Conv1d(input_channels, channels, kernel_size: 1));
		}
		else
		{
			layers.Add(nn.Conv1d(input_channels, channels, kernel_size: 7, padding: 3));
		}
		if (attn_window_size != null)
		{
			layers.Add(new LocalMHA(dim: channels, window_size: attn_window_size));
		}

		int output_dim = 0;
		foreach (var (i, stride) in rates.Index())
		{
			int input_dim = (int)(channels / Math.Pow(2, i));
			output_dim = (int)(channels / Math.Pow(2, i + 1));
			int groups = depthwise ? output_dim : 1;
			layers.Add(new DecoderBlock(input_dim, output_dim, stride, noise, groups: groups));
		}

		layers.Add(new Snake1d(output_dim));
		layers.Add(nn.Conv1d(output_dim, d_out, kernel_size: 7, padding: 3));
		layers.Add(nn.Tanh());

		model = nn.Sequential(layers);

		RegisterComponents();
	}

	public Tensor forward(Tensor x)
	{
		return model.forward(x);
	}
}

//<summary>
//Attention
//</summary>

class SinusoidalEmbeddings : nn.Module
{
	Tensor inv_freq;
	Tensor scale;
	bool use_xpos;
	int? scale_base;

	public SinusoidalEmbeddings(int dim, int? scale_base = null, bool use_xpos = false) : base("SinusoidalEmbeddings")
	{
		inv_freq = 1.0 / (pow(10000, (arange(0, dim, 2).@float() / dim)));
		this.register_buffer("inv_freq", inv_freq);
		this.use_xpos = use_xpos;
		this.scale_base = scale_base;
		scale = (arange(0, dim, 2) + 0.4 * dim) / (1.4 * dim);
		this.register_buffer("scale", scale, persistent: false);

		RegisterComponents();
	}

	public (Tensor, Tensor) forward(Tensor x)
	{
		(long seq_len, Device device) = (x.shape[^2], x.device);
		Tensor t = arange(seq_len, device: device).type_as(inv_freq);
		Tensor freqs = t.view(-1, 1) * inv_freq.view(1, -1);
		freqs = cat([freqs, freqs], dim: -1);
		if (!use_xpos)
		{
			return (freqs, ones(1, device: device));
		}
		Tensor power = (t - floor(seq_len / 2)) / (scale_base);
		scale = pow(scale, power.view(-1, 1));
		scale = cat([scale, scale], dim: -1);
		return (freqs, scale);
	}
}

class LocalMHA : nn.Module<Tensor, Tensor>
{
	nn.Module norm;
	int heads;
	int? window_size;
	Linear to_qkv;
	SinusoidalEmbeddings? rel_pos;
	Linear to_out;

	public LocalMHA(int dim = 1024, int? window_size = 32, int dim_head = 64, bool use_rotary_pos_emb = true) : base("LocalMHA")
	{

		this.norm = nn.LayerNorm(dim);
		this.heads = dim / dim_head;
		this.window_size = window_size;
		this.to_qkv = nn.Linear(dim, dim * 3, hasBias: false);
		if (use_rotary_pos_emb)
		{
			rel_pos = new(dim_head, scale_base: (int)Math.Floor((double)window_size / 2));
		}
		else
		{
			rel_pos = null;
		}
		this.to_out = nn.Linear(dim, dim, hasBias: false);

		RegisterComponents();
	}

	Tensor rotate_half(Tensor x)
	{
		int r = 2;
		long[] new_shape = x.shape.Take(x.shape.Length - 1).Concat([r, x.shape.Last() / 2]).ToArray();
		x = x.view(new_shape);
		(Tensor x1, Tensor x2) = (
			x.unbind(dimension: -2)[0],
			x.unbind(dimension: -2)[1]
		);

		return cat([-x2, x1], dim: -1);
	}

	(Tensor, Tensor) apply_rotary_pos_emb(Tensor q, Tensor k, Tensor freqs, Tensor scale)
	{

		long q_len = q.shape[^2];
		long dim = freqs.shape.Length - 2;
		long size = freqs.shape[dim];
		long start = size - q_len;
		Tensor q_freqs = freqs.narrow(dim, start, q_len);
		Tensor inv_scale = pow(scale, -1);
		if (scale.ndim == 2)
		{
			scale = scale.narrow(0, scale.size(0) - q_len, q_len);
		}
		q = (q * q_freqs.cos() * scale) + (rotate_half(q) * q_freqs.sin() * scale);
		k = (k * freqs.cos() * inv_scale) + (rotate_half(k) * freqs.sin() * inv_scale);
		return (q, k);
	}

	public override Tensor forward(Tensor x)
	{
		(long B, long C, long T) = (x.shape[0], x.shape[1], x.shape[2]);
		Tensor residual = x;
		x = norm(x.transpose(1, 2));
		long windows = (long)(T / window_size);
		(Tensor q, Tensor k, Tensor v) = (
			to_qkv.forward(x).chunk(3, dim: -1)[0],
			to_qkv.forward(x).chunk(3, dim: -1)[1],
			to_qkv.forward(x).chunk(3, dim: -1)[2]
		);

		q = q.view(B, windows, (long)window_size, heads, -1).permute(0, 3, 1, 2, 4);
		k = k.view(B, windows, (long)window_size, heads, -1).permute(0, 3, 1, 2, 4);
		v = v.view(B, windows, (long)window_size, heads, -1).permute(0, 3, 1, 2, 4);

		if (rel_pos != null)
		{
			(Tensor pos_emb, Tensor scale) = rel_pos.forward(k);
			(q, k) = apply_rotary_pos_emb(q, k, pos_emb, scale);
		}
		Tensor _out = nn.functional.scaled_dot_product_attention(q, k, v);
		(long b, long h, long w, long n, long d) = (
			_out.shape[0],
			_out.shape[1],
			_out.shape[2],
			_out.shape[3],
			_out.shape[4]
		);
		long[] new_shape = { b, w * n, h * d };
		_out = _out.view(new_shape);
		_out = to_out.forward(_out);
		return _out.transpose(1, 2) + residual;
	}
}

// <summary>
// vq
// </summary>

public class VectorQuantize : nn.Module<Tensor, (Tensor, Tensor)>
{

	int codebook_size;
	int codebook_dim;
	public int stride;
	Conv1d in_proj;
	public Conv1d out_proj;
	public Embedding codebook;

	public VectorQuantize(
		int input_dim,
		int codebook_size,
		int codebook_dim,
		int stride = 1
	) : base("VectorQuantize")
	{
		this.codebook_size = codebook_size;
		this.codebook_dim = codebook_dim;
		this.stride = stride;
		this.in_proj = nn.Conv1d(input_dim, codebook_dim, kernel_size: 1);
		this.out_proj = nn.Conv1d(codebook_dim, input_dim, kernel_size: 1);
		this.codebook = nn.Embedding(codebook_size, codebook_dim);
		//Console.WriteLine($"codebook.weight: {this.codebook.weight[0].print()}");

		RegisterComponents();
	}

	public Tensor embed_code(Tensor embed_id)
	{
		return codebook.forward(embed_id);
	}

	public Tensor decode_code(Tensor embed_id)
	{
		Tensor embed = embed_code(embed_id).transpose(1, 2);
		return embed;
	}

	(Tensor, Tensor) decode_latents(Tensor latents)
	{
		(long b, long d, long t) = (
			latents.shape[0],
			latents.shape[1],
			latents.shape[2]
		);
		long[] new_shape = { b * t, d };
		Tensor encodings = latents.view(new_shape);
		Tensor codebook = this.codebook.weight;
		encodings = nn.functional.normalize(encodings);
		codebook = nn.functional.normalize(codebook);

		Tensor dist = encodings.pow(2).sum(1, keepdim: true) - 2 * matmul(encodings, codebook.t()) + codebook.pow(2).sum(1, keepdim: true).t();

		Tensor max_indexes = (-dist).max(1).indexes;
		Tensor indices = max_indexes.view([b, max_indexes.shape[0] / b]);
		Tensor z_q = decode_code(indices);
		return (z_q, indices);
	}

	public override (Tensor, Tensor) forward(Tensor z)
	{
		if (stride > 1)
		{
			z = nn.functional.avg_pool1d(z, stride, stride);
		}
		Tensor z_e = in_proj.forward(z);
		(Tensor z_q, Tensor indices) = decode_latents(z_e);
		z_q = z_e + (z_q - z_e).detach();
		z_q = out_proj.forward(z_q);
		if (stride > 1)
		{
			z_q = z_q.repeat_interleave(stride, dim: -1);
		}
		return (z_q, indices);
	}
}

public class ResidualVectorQuantize : nn.Module
{

	int input_dim;
	int codebook_size;
	int codebook_dim;
	int[] vq_strides;
	int n_codebooks;

	public ModuleList<VectorQuantize> quantizers = new();

	public ResidualVectorQuantize(
		int input_dim = 512,
		int codebook_size = 1024,
		int codebook_dim = 8,
		int[]? vq_strides = null
	) : base("ResidualVectorQuantize")
	{
		this.input_dim = input_dim;
		this.codebook_size = codebook_size;
		this.codebook_dim = codebook_dim;
		this.vq_strides = vq_strides ?? [1, 1, 1, 1];
		this.n_codebooks = this.vq_strides.Length;

		foreach (int stride in this.vq_strides)
		{
			quantizers.Add(new VectorQuantize(input_dim, codebook_size, codebook_dim, stride));
		}

		RegisterComponents();
	}

	public (Tensor, List<Tensor>) forward(Tensor z)
	{
		Tensor z_q = 0;
		Tensor residual = z;
		List<Tensor> codes = new();

		foreach (var (i, quantizer) in quantizers.Index())
		{
			(Tensor z_q_i, Tensor indices_i) = quantizer.forward(residual);
			z_q += z_q_i;
			residual -= z_q_i;
			codes.Add(residual);
		}
		return (z_q, codes);
	}

	// <VERIFIED>
	// Identical to the PyTorch version
	// z_q: [1x768x16]
	// </VERIFIED>
	public Tensor from_codes(List<Tensor> codes)
	{
		Tensor z_q = 0.0;

		for (int i = 0; i < n_codebooks; i++)
		{
			Tensor z_p_i = quantizers[i].decode_code(codes[i]);
			//Console.WriteLine($"{z_p_i}: {z_p_i.print()}");
			Tensor z_q_i = quantizers[i].out_proj.forward(z_p_i);
			z_q_i = z_q_i.repeat_interleave(quantizers[i].stride, dim: -1);
			z_q += z_q_i;
		}
		//Console.WriteLine($"{z_q}: {z_q.print()}");
		return z_q;
	}
}
