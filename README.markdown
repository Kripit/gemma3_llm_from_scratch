# Yo, Check Out My Gemma3 Project

## What’s This All About?

So, I built this thing called **Gemma3**, a small language model that’s basically a transformer on a budget. It’s got all the cool stuff—attention mechanisms, embeddings, and all that jazz—and it can generate text from prompts like “Once upon a time…” without totally choking. I coded it up in a Jupyter notebook (`gemma3_from_scratch.ipynb`) using PyTorch, and it’s got everything from model setup to training to spitting out stories. Here’s the lowdown on what I did and how you can mess with it.

## The Model Vibe

Gemma3 is a transformer, but not one of those massive beasts that need a supercomputer. It’s got:
- **Embeddings**: Turns words into vectors (size 640 for the 270M version).
- **Transformer Blocks**: 18 layers of goodness, with multi-head attention (4 heads) that switches between global attention (sees everything) and sliding-window attention (looks at 512 tokens at a time).
- **RoPE**: Fancy positional encodings so the model knows what order words are in. I set it up with different bases for local (10,000) and global (1,000,000) attention.
- **Output Layer**: Maps the final output to a vocab of 50,257 tokens for picking the next word.
- **Config**: I used `GEMMA3_CONFIG_270M` with a context length of 32,768 tokens and `bfloat16` for faster math on GPUs.

It’s lightweight but still packs a punch. I alternated attention types to keep it efficient without losing too much smarts. Wanna know more about transformers? What part of this setup sounds coolest to you?

## Getting It Running

Here’s how to fire this up:
1. **Grab the Dependencies**: You’ll need Python 3.6+ and some libraries. Run this:
   ```bash
   pip install torch matplotlib tqdm
   ```
   If you’re on Colab with a GPU, you’re golden—no extra setup for CUDA.

2. **Data Prep**: I used some dataset files (like `train-00000-of-00004-2d5a1467fff108.parquet`) in a `data/` folder. You’ll need something similar or tweak the `get_batch` function to load your own data. What kind of data are you working with?

3. **Get the Notebook**: Snag `gemma3_from_scratch.ipynb` from the repo or upload it to Colab.

4. **Google Drive (Optional)**: If you’re saving checkpoints on Colab, mount your Drive:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```

## Training It Up

Training this thing was a journey. Here’s how I set it up:
- **Optimizer**: AdamW with a learning rate of 1e-4, some fancy betas (0.9, 0.95), and weight decay (0.1) to keep things chill.
- **Scheduler**: Linear warmup for 1,000 steps, then cosine decay to 5e-4 over 150,000 iterations.
- **Batch Size**: 32, with 128-token blocks to catch longer patterns.
- **Gradient Accumulation**: 32 steps to smooth out the training without blowing up my GPU.
- **Mixed Precision**: Used `bfloat16` or `float16` with AMP to make it run faster.
- **Eval**: Checked training and validation loss every 500 steps and saved the best model as `best_model_params.pt`.

The training loop plots loss curves with Matplotlib so you can see how it’s doing. It’s a grind—150,000 iterations—but it gets there. Got a GPU? You’ll want one, ‘cause CPU training is sloooow. What hardware you running this on?

## Generating Text

Once it’s trained, you can make it write stuff. The `generate` method takes a prompt, runs it through the model, and spits out up to 200 new tokens. I played with temperature and top-k sampling to keep it creative but not too wild. Here’s an example:

```python
sentence = "Once upon a time there was a pumpkin."
context = torch.tensor(enc.encode_ordinary(sentence)).unsqueeze(dim=0)
y = model.generate(context, max_new_tokens=200)
print(enc.decode(y.squeeze().tolist()))
```

I tried prompts like “A little girl went to the woods” and got some decent stories. What kind of prompts would you throw at this thing?

## What You’ll See

The notebook has a plot of training vs. validation loss, so you can see how the model learns. It’s not perfect, but it gets better with more training and a good dataset. Speaking of, you’ll need to sort out the `get_batch` function for your data. What’s your dataset like? Need help with that part?

## Pro Tips

- **Tokenizer**: I used an `enc` object (think `tiktoken`). Swap it out if you’ve got another tokenizer.
- **Hardware**: A T4 GPU on Colab works fine. CPU’s doable but painful.
- **Tweaks**: Mess with `GEMMA3_CONFIG_270M` to change the model size or attention setup. What changes would you wanna try?
- **Checkpoints**: Saves to `best_model_params.pt`, so you don’t lose your progress.

## License

MIT License, so feel free to fork it and play around. Check the LICENSE file for the boring stuff.