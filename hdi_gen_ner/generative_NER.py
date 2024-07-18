"""
Use this script to run the generation pipeline.

The script generate the text using defined models and write the results in the
"/results/{model}" folder in the form "generation_{prompt_type}_shots={bool}.json"
"""
import sys
from tqdm import tqdm
import json
from copy import deepcopy
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import os
os.environ["TRANSFORMERS_VERBOSITY"] = "error"


def process_example(pipe, tokenizer, example, system, iteration, shots=None):
    prompt_message = example[1]
    gold = example[2]
    prompt_template = format_prompt(tokenizer, system, prompt_message, shots)
    result = pipe(prompt_template, do_sample=False,
                  pad_token_id=pipe.tokenizer.eos_token_id)[0]['generated_text']
    return (iteration, prompt_message, (result, gold))


def format_prompt(tokenizer, system, prompt_message, shots=None):
    shots_list = deepcopy(shots)
    if shots_list:
        first_shot = shots_list.pop(0)
        combined_prompt = f'{system}\n{first_shot["query"]}'
        messages = [{"role": "user", "content": combined_prompt},
                    {"role": "assistant", "content": f"{first_shot['answer']}"}]
        for shot in shots_list:
            messages.append({"role": "user", "content": f"{shot['query']}"})
            messages.append(
                {"role": "assistant", "content": f"{shot['answer']}"})
        messages.append({"role": "user", "content": f"{prompt_message}"})
    else:
        messages = [{"role": "user", "content": f'{system}{prompt_message}'}]
    tokenized_message = tokenizer.apply_chat_template(messages,
                                                      tokenize=True,
                                                      add_generation_prompt=True,
                                                      return_tensors="pt")
    return tokenizer.decode(tokenized_message[0])


def process_dataset(model, tokenizer, dataset, system, shots=None):
    pipe = pipeline(
        "text-generation",
        return_full_text=False,
        max_new_tokens=512,
        model=model,
        tokenizer=tokenizer
    )
    results = [process_example(pipe,
                               tokenizer,
                               example,
                               system=system,
                               iteration=i,
                               shots=shots)
               for i, example in enumerate(tqdm(dataset))]
    return results


def run_inference(model_name_or_path, raw_prompt, dataset, shots):
    generative_model = AutoModelForCausalLM.from_pretrained(model_name_or_path,
                                                            device_map="cuda:0",
                                                            trust_remote_code=True,
                                                            revision="main")
    generative_tokenizer = AutoTokenizer.from_pretrained(model_name_or_path,
                                                         use_fast=True)
    formatted_prompt = format_prompt(generative_tokenizer,
                                     raw_prompt,
                                     'TEXT HERE',
                                     shots)
    results = process_dataset(generative_model,
                              generative_tokenizer,
                              dataset,
                              raw_prompt,
                              shots=shots)
    return results, formatted_prompt


if __name__ == '__main__':
    model_choice_map = {'mistral': "TheBloke/Mistral-7B-Instruct-v0.2-GPTQ",
                        'phi': "kaitchup/Phi-3-mini-4k-instruct-gptq-4bit"}
    model_selection = sys.argv[1]
    selected_model = model_choice_map[model_selection]
    prompt_selection = sys.argv[2]
    uses_few_shots = sys.argv[3].lower() == 'true'
    with open(f'prompts/system_prompts/system_{prompt_selection}.txt', 'r') as fp:
        prompt = fp.read()
    few_shots = None
    if uses_few_shots:
        with open(f'prompts/few_shots/{prompt_selection}_shots.json', 'r') as fp:
            few_shots = json.load(fp)['shots']
    with open('data/hdi_extractive_entity_recognition.json', 'r') as fp:
        data = json.load(fp)['dataset']
    with open('data/used_subset_idx.txt', 'r') as fp:
        split_idx = [int(idx) for idx in fp.readlines()]
    selected_data = [sentence for sentence in data if sentence[0] in split_idx]
    results, formatted_prompt = run_inference(selected_model,
                                              prompt,
                                              selected_data,
                                              few_shots)
    shots_mapping = {'true': 'few_shots', 'false': '0_shots'}
    shots = str(uses_few_shots).lower()
    with open(f'results/{model_selection}/{shots_mapping[shots]}/generation_{prompt_selection}_shots={shots}.json', 'w') as fp:
        json.dump({"results": results, "prompt": formatted_prompt}, fp)
