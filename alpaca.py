#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os
import os.path as osp
import sys
import torch
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer
from peft import PeftModel

from utils.prompter import Prompter
from utils.io_utils import wait_and_answer, labeled_input

try:
    BASE_MODEL = osp.join(os.getenv("MODEL_LLAMA_BASEPATH"), "hf")
    LORA_MODEL = osp.join(os.getenv("MODEL_ALPACA_LORA_BASEPATH"), "")
except TypeError:
    print("Model environment variables (MODEL_LLAMA_BASEPATH and MODEL_ALPACA_LORA_BASEPATH) are not set")
    sys.exit(1)

if torch.cuda.is_available():
    DEVICE = 'cuda'
else:
    DEVICE = 'cpu'

try:
    if torch.backends.mps.is_available():
        DEVICE = 'mps'
except AttributeError:
    pass


class AlpacaLora(object):
    """Question-answer generation model"""
    def __init__(self, config):
        self.tokenizer = None
        self.model = None
        self.prompter = Prompter(config.template)
        self.generation_config = GenerationConfig(temperature=config.temperature,
                                                 top_p=config.top_p,
                                                 top_k=config.top_k,
                                                 num_beams=config.num_beams)
        self.max_new_tokens = config.max_size
        self._load(config)

    def _load(self, config):
        self.tokenizer = LlamaTokenizer.from_pretrained(osp.join(BASE_MODEL, config.model_size))
        self.model = LlamaForCausalLM.from_pretrained(osp.join(BASE_MODEL, config.model_size),
                                                      load_in_8bit=config.load_in_8bit,
                                                      torch_dtype=torch.float16,
                                                      device_map='auto')
        if config.use_lora:
            device_map = 'auto'
            if sys.platform == "win32":
                device_map = {'':0}
            self.model = PeftModel.from_pretrained(self.model,
                                                   osp.join(LORA_MODEL, config.model_size),
                                                   torch_dtype=torch.float16,
                                                   device_map=device_map)
        if not config.load_in_8bit:
            self.model.half()
        self.model.eval()
        if torch.__version__ >= "2" and sys.platform != "win32":
            self.model = torch.compile(self.model)

    def _tokenize(self, request, context):
        prompt = self.prompter.generate_prompt(request, context)
        inputs = self.tokenizer(prompt, return_tensors="pt")
        return inputs['input_ids'].to(DEVICE)

    def generate(self, request, context=None):
        """Generate answer on specified question"""
        input_ids = self._tokenize(request, context)

        with torch.no_grad():
            generation_output = self.model.generate(
                input_ids=input_ids,
                generation_config=self.generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=self.max_new_tokens,
            )
        s = generation_output.sequences[0]
        output = self.tokenizer.decode(s)
        return self.prompter.get_response(output)


def one_shot(alpaca, prompt):
    """Generates single output for specified prompt"""
    wait_and_answer(lambda: alpaca.generate(prompt))


def repl(alpaca, use_input=False):
    """Use alpaca as a repl"""
    request = None
    context = None
    while True:
        request = labeled_input('Request:')
        if len(request) == 0:
            return
        context = labeled_input('Context:') if use_input else None
        wait_and_answer(lambda: alpaca.generate(request, context))


def main():
    """Main function"""
    parser = argparse.ArgumentParser(prog='Alpaca-LoRA',
                                     description='Question-answer system based on LLaMa model')
    parser.add_argument('--model', dest='model_size',
                        choices=['7B', '13B', '30B', '65B'],
                        default='13B', help='size of the model (default: 13B)')
    parser.add_argument('--template', dest='template', default=None,
                        help='use specific template from template/ dir (default: disabled)')
    parser.add_argument('--max-size', dest='max_size',
                        default=512, type=int, help='max generation size (default: 1024)')
    parser.add_argument('--8bit', action='store_true', dest='load_in_8bit',
                        help='use int8 quantification (default: false)')
    parser.add_argument('--no-lora', action='store_false', dest='use_lora',
                        help='do not load LoRA weights and use plain LLaMA (default: false)')
    parser.add_argument('--input', action='store_true', dest='use_input',
                        help='use additional context as input (default: false)')
    parser.add_argument('--prompt', type=str, dest='prompt', default=None,
                        help='run single shot on a specific request (default: disables)')
    parser.add_argument('--temperature', type=float, dest='temperature', default=1,
                        help='generation temperature (default: 1)')
    parser.add_argument('--top_p', type=float, dest='top_p', default=0.75,
                        help='generation top_p (default: 0.75)')
    parser.add_argument('--top_k', type=int, dest='top_k', default=40,
                        help='generation top_k (default: 40)')
    parser.add_argument('--num_beams', type=int, dest='num_beams', default=4,
                        help='generation number of beams (default: 4)')
    args = parser.parse_args()

    alpaca = AlpacaLora(args)

    os.system('color')
    if args.prompt:
        one_shot(alpaca, args.prompt)
    else:
        repl(alpaca, args.use_input)


if __name__ == '__main__':
    main()
