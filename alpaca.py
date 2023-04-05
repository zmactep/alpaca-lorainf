#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import argparse
import os
import sys
import os.path as osp
from typing import Union
import torch
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer
from peft import PeftModel

try:
    BASE_MODEL = osp.join(os.getenv("MODEL_LLAMA_BASEPATH"), "hf")
    LORA_MODEL = osp.join(os.getenv("MODEL_ALPACA_LORA_BASEPATH"), "")
except TypeError:
    print("Model environment variables are not set")
    sys.exit(1)

class Prompter(object):
    __slots__ = ("template", "_verbose")

    def __init__(self, template_name: str = "", verbose: bool = False):
        self._verbose = verbose
        if not template_name:
            # Enforce the default here, so the constructor can be called with '' and will not break.
            template_name = "alpaca"
        file_name = osp.join("templates", f"{template_name}.json")
        if not osp.exists(file_name):
            raise ValueError(f"Can't read {file_name}")
        with open(file_name) as fp:
            self.template = json.load(fp)
        if self._verbose:
            print(
                f"Using prompt template {template_name}: {self.template['description']}"
            )

    def generate_prompt(
        self,
        instruction: str,
        input: Union[None, str] = None,
        label: Union[None, str] = None,
    ) -> str:
        # returns the full prompt from instruction and optional input
        # if a label (=response, =output) is provided, it's also appended.
        if input:
            res = self.template["prompt_input"].format(
                instruction=instruction, input=input
            )
        else:
            res = self.template["prompt_no_input"].format(
                instruction=instruction
            )
        if label:
            res = f"{res}{label}"
        if self._verbose:
            print(res)
        return res

    def get_response(self, output: str) -> str:
        return output.split(self.template["response_split"])[1].strip()


def load_model(model_size, load_in_8bit=True, compile_model=False):
    tokenizer = LlamaTokenizer.from_pretrained(osp.join(BASE_MODEL, model_size))

    model = LlamaForCausalLM.from_pretrained(osp.join(BASE_MODEL, model_size),
                                             load_in_8bit=load_in_8bit,
                                             torch_dtype=torch.float16,
                                             device_map='auto')
    model = PeftModel.from_pretrained(model,
                                      osp.join(LORA_MODEL, model_size),
                                      torch_dtype=torch.float16,
                                      device_map={'':0})
    if not load_in_8bit:
        model.half()

    model.eval()
    if compile_model:
        model = torch.compile(model)

    return tokenizer, model


def get_input():
    line = ""
    user_input = ""
    try:
        while True:
            line = input()
            if len(user_input) > 0:
                line = "\n" + line
            user_input += line
    except EOFError:
        pass
    return user_input


def main_cycle(tokenizer, model, use_input=False, template=None, max_new_tokens=1024):
    prompter = Prompter(template)
    generation_config = GenerationConfig(temperature=1, top_p=0.75, top_k=40, num_beams=4)
    request = None
    rinput = None

    print("Print (exit) as a question to quit.")
    while True:
        print('Question:')
        request = get_input()
        if request == '(exit)':
            break
        if use_input:
            print('Input:')
            rinput = get_input()

        prompt = prompter.generate_prompt(request, rinput)
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs['input_ids'].to('cuda')

        with torch.no_grad():
            generation_output = model.generate(
                input_ids=input_ids,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=max_new_tokens,
            )

        s = generation_output.sequences[0]
        output = tokenizer.decode(s)
        print("Answer:")
        print(prompter.get_response(output))
        print("\n*****\n\n")


def main():
    parser = argparse.ArgumentParser(
                        prog='Alpaca-LoRA',
                        description='Question-answer system based on LLaMa model')
    parser.add_argument('--model', dest='model',
                        choices=['7B', '13B', '30B', '65B'],
                        default='13B', help='size of the model (default: 13B)')
    parser.add_argument('--max-size', dest='max_size',
                        default=512, type=int, help='max generation size (default: 512)')
    parser.add_argument('--8bit', action='store_true', dest='load_in_8bit',
                        help='use int8 quantification (default: False)')
    parser.add_argument('--compile', action='store_true', dest='compile_model',
                        help='compile model after load (default: False)')
    parser.add_argument('--input', action='store_true', dest='use_input',
                        help='use additional context as input (default: False)')
    args = parser.parse_args()

    tokenizer, model = load_model(args.model, compile_model=args.compile_model, load_in_8bit=args.load_in_8bit)
    main_cycle(tokenizer, model, max_new_tokens=args.max_size, use_input=args.use_input)


if __name__ == '__main__':
    main()
