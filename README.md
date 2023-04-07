# alpaca-lorainf
Inference for alpaca-lora in `fp16` or `int8`.

# Install

Requirements can be found in `requirements.txt` file.

You also need [Huggingface-converted](https://huggingface.co/docs/transformers/main/model_doc/llama#:~:text=the%20research%20community.-,Tips,-%3A) LLaMA weights and Alpaca-LoRA weights for your system.
There is a custom finetune script in this repo for a non-quantified training.

`MODEL_LLAMA_BASEPATH` and `MODEL_ALPACA_LORA_BASEPATH` environmet variables should be set.

# Usage

```
$ python alpaca.py -h
usage: Alpaca-LoRA [-h] [--device {single,auto}] [--model {7B,13B,30B,65B}]
                   [--max-size MAX_SIZE] [--8bit] [--compile] [--input]

Question-answer system based on LLaMa model

options:
  -h, --help            show this help message and exit
  --device {single,auto}
                        device to map the model (default: auto)
  --model {7B,13B,30B,65B}
                        size of the model (default: 13B)
  --max-size MAX_SIZE   max generation size (default: 512)
  --8bit                use int8 quantification (default: False)
  --compile             compile model after load (default: False)
  --input               use additional context as input (default: False)
```

## Home use

RTX4090 can be used with `13B` model in `8bit` mode. Please use `--device single` on Windows machines.
Windows also doesn't support `torch.compile`, so the basic usage will be like:
```
$ python alpaca.py --8bit --device single
[...]
Print (exit) as a question to quit.
Question:
Continue the song: Old McDonald had
Answer:
a farm, E-I-E-I-O.

*****

Question:
What is an adenoassociated virus?
Answer:
Adenoassociated virus (AAV) is a small, non-enveloped, single-stranded DNA virus
belonging to the Parvoviridae family. It is a naturally occurring virus found in 
humans and other mammals. AAV is used in gene therapy to deliver therapeutic 
genes into cells.

*****

Question:
Что ты знаешь о компании BIOCAD?
Answer:
I know that BIOCAD is a Russian biotechnology company that specializes in the 
development, manufacturing, and commercialization of innovative diagnostic and 
therapeutic products. The company was founded in 2001 and is headquartered in 
St. Petersburg, Russia.

*****

Question:
What is pembrolizumab?
Answer:
Pembrolizumab is a monoclonal antibody that is used to treat certain types of 
cancer. It works by targeting a protein called PD-1, which is found on the 
surface of certain immune cells. By blocking PD-1, pembrolizumab helps the 
immune system recognize and attack cancer cells.
```
