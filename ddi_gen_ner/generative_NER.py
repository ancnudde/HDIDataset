import sys
from tqdm import tqdm
import json
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


def process_example(pipe, tokenizer, example, system, iteration, shots=None):
    prompt_message = example[1]
    gold = example[2]
    prompt_template = format_prompt(tokenizer, system, prompt_message, shots)
    result = pipe(prompt_template, do_sample=False,
                  pad_token_id=pipe.tokenizer.eos_token_id)[0]['generated_text']
    return (iteration, prompt_message, (result, gold))


def format_prompt(tokenizer, system, prompt_message, shots=None):
    if shots:
        first_shot = shots.pop(0)
        combined_prompt = f'{system}\n{first_shot["query"]}'
        messages = [{"role": "user", "content": combined_prompt},
                    {"role": "assistant", "content": f"{first_shot['answer']}"}]
        for shot in shots:
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


if __name__ == '__main__':
    model_choice_map = {'mistral': "TheBloke/Mistral-7B-Instruct-v0.2-GPTQ",
                        'phi': "kaitchup/Phi-3-mini-4k-instruct-gptq-4bit"}
    model_selection = sys.argv[1]
    selected_model = model_choice_map[model_selection]
    prompt_selection = sys.argv[2]
    few_shots = sys.argv[3].lower() == 'true'
    with open(f'prompts/{prompt_selection}_prompt.txt', 'r') as fp:
        prompt = fp.read()
    shots = None
    if few_shots:
        with open('prompts/few_shots.json', 'r') as fp:
            shots = json.load(fp)['shots']
    with open('data/ddi_ner.json', 'r') as fp:
        data = json.load(fp)['dataset']
    model_name_or_path = selected_model
    generative_model = AutoModelForCausalLM.from_pretrained(model_name_or_path,
                                                            device_map="cuda:0",
                                                            trust_remote_code=True,
                                                            revision="main")
    generative_tokenizer = AutoTokenizer.from_pretrained(model_name_or_path,
                                                         use_fast=True)
    formatted_prompt = format_prompt(generative_tokenizer, prompt, '', shots)
    results = process_dataset(generative_model,
                              generative_tokenizer,
                              data,
                              prompt,
                              shots)
    with open(f'results/{model_selection}/generation_{prompt_selection}_shots={few_shots}.json', 'w') as fp:
        json.dump({"results": results, "prompt": formatted_prompt}, fp)
    del generative_tokenizer
    del generative_model
