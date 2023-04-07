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
usage: Alpaca-LoRA [-h] [--model {7B,13B,30B,65B}] [--template TEMPLATE]
                   [--max-size MAX_SIZE] [--8bit] [--no-lora] [--input]
                   [--prompt PROMPT] [--temperature TEMPERATURE]
                   [--top_p TOP_P] [--top_k TOP_K] [--num_beans NUM_BEANS]

Question-answer system based on LLaMa model

options:
  -h, --help            show this help message and exit
  --model {7B,13B,30B,65B}
                        size of the model (default: 13B)
  --template TEMPLATE   use specific template from template/ dir (default:
                        disabled)
  --max-size MAX_SIZE   max generation size (default: 1024)
  --8bit                use int8 quantification (default: false)
  --no-lora             do not load LoRA weights and use plain LLaMA (default:
                        false)
  --input               use additional context as input (default: false)
  --prompt PROMPT       run single shot on a specific request (default:
                        disables)
  --temperature TEMPERATURE
                        generation temperature (default: 1)
  --top_p TOP_P         generation top_p (default: 0.75)
  --top_k TOP_K         generation top_k (default: 40)
  --num_beans NUM_BEANS
                        generation number of beams (default: 4)
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

*****

Question:
Write a Python code that defines a list of tuples with name, birth year and death year of each Russian Emperor from Peter the Great to Nicholas II. Put the content of the list into a csv file with a correct header line.
Answer:
import csv

russian_emperors = [('Peter the Great', 1682, 1725), ('Catherine the Great', 1729, 1796), ('Paul I', 1754, 1801), ('Alexander I', 1777, 1825), ('Nicholas I', 1796, 1855), ('Alexander II', 1818, 1881), ('Alexander III', 1845, 1894), ('Nicholas II', 1868, 1918)]

with open('russian_emperors.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerow(['Name', 'Birth Year', 'Death Year'])
    for emperor in russian_emperors:
        writer.writerow([emperor[0], emperor[1], emperor[2]])
```
