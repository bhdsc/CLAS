
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, MllamaForConditionalGeneration, AutoProcessor 
from transformers import BitsAndBytesConfig, Gemma3ForCausalLM
# from janus.models import MultiModalityCausalLM, VLChatProcessor
from utils import harmful_dataset
from neural_controllers import NeuralController
from utils import LLMType
from collections import namedtuple 
import json
import requests
from PIL import Image
import io
from tqdm import tqdm 
import pickle
import gc

# from janus.utils.io import load_pil_images
from generation_utils import extract_image
import utils

SEED = 0
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
np.random.seed(SEED)


LLM = namedtuple('LLM', ['language_model', 'tokenizer', 'processor', 'name', 'model_type'])


def select_llm(model_type, MODEL_VERSION='3.1', MODEL_SIZE='8B'):

    if model_type=='llama':

        if MODEL_VERSION == '3.1' and MODEL_SIZE == '8B':
            model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
        elif MODEL_VERSION == '3.1' and MODEL_SIZE == '70B':
            model_id = "unsloth/Meta-Llama-3.1-70B-Instruct-bnb-4bit"
        elif MODEL_VERSION == '3.3' and MODEL_SIZE == '70B':
            model_id = "unsloth/Llama-3.3-70B-Instruct-bnb-4bit"

        language_model = AutoModelForCausalLM.from_pretrained(
            model_id, device_map="cuda",
            # torch_dtype=torch.bfloat16,
        )

        use_fast_tokenizer = "LlamaForCausalLM" not in language_model.config.architectures
        tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=use_fast_tokenizer, padding_side="left", legacy=False)
        tokenizer.pad_token_id = 0 
        # model_name='llama_3_8b_it'
        if MODEL_VERSION == '3.1' and MODEL_SIZE == '8B':
            model_name='llama_3_8b_it_eng_only'
        elif MODEL_VERSION == '3.1' and MODEL_SIZE == '70B':
            model_name = "llama_3.1_70b_it_eng_only"
        elif MODEL_VERSION == '3.3' and MODEL_SIZE == '70B':
            model_name = "llama_3.3_70b_it_eng_only"

        processor = None
        llm_type = LLMType.TEXT
        # print(tokenizer.chat_template)

    elif model_type=='gemma':
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        if MODEL_VERSION == '3.1' and MODEL_SIZE == '1B':
            model_id = "google/gemma-3-1b-it"

        tokenizer = AutoTokenizer.from_pretrained(model_id)

        language_model = Gemma3ForCausalLM.from_pretrained(
            model_id, quantization_config=quantization_config
        ).eval()

        if MODEL_VERSION == '3.1' and MODEL_SIZE == '1B':
            model_name='gemma_3_1b_it_eng_only'
        processor = None
        llm_type = LLMType.GEMMA_TEXT



    llm = LLM(language_model, tokenizer, processor, model_name, llm_type)
    # print(llm.language_model)
    return llm



def generate(concept, llm, prompt, image=None, coefs=[0.4], control_method='rfm', max_tokens=100, gen_orig=True):
    controller = NeuralController(
        llm,
        llm.tokenizer,
        rfm_iters=8,
        control_method=control_method,
        n_components=1
    )

    controller.load(concept=concept, model_name=llm.name, path='directions/')

    # No steering 
    if gen_orig:
        original_output = controller.generate(prompt, image=image, max_new_tokens=max_tokens, do_sample=False)#, temperature=0)
        print(original_output)

    outputs = []

    target_keys = set(range(-1, -80, -1)) 

    print(controller.hidden_layers & target_keys)

    for coef in coefs:
        print(f"Coeff: {coef} ==========================================================")
        steered_output = controller.generate(prompt,
                                            image=image, 
                                            layers_to_control=controller.hidden_layers & target_keys,
                                            control_coef=coef,
                                            max_new_tokens=max_tokens,
                                            do_sample=False)
        outputs.append((coef, steered_output))
        print(steered_output)
    return outputs


def read_file(fname, lower=True):

    concepts = []
    with open(fname, encoding="utf-8") as f: 
        for line in f:
            if lower:
                concepts.append(line.strip().lower())
            else:
                concepts.append(line.strip())
    concepts = sorted(list(set(concepts)))
    return concepts


def main():

    torch.backends.cudnn.benchmark = True        
    torch.backends.cuda.matmul.allow_tf32 = True        

    fnames = ['data/fears/fears.txt',
              'data/personalities/personalities.txt',
              'data/moods/moods.txt',
              'data/places/places.txt',
              'data/personas/personas.txt',]
    lowers = [True, True, True, False, False]
    dataset_labels = ['fears', 'personalities', 'moods', 'places', 'personas']

    # COEFS = [.35, .36, .37, .38, .39]
    # COEFS = [.3, .35, .4, .45, .5]#, 
    # COEFS = [.5, .51, .52, .53, .54, .55]
    # COEFS = [.55, .56, .57, .58, .59, .60]
    # COEFS = [.45, .46, .47, .48, .49, .50]
    # COEFS = [.3, .35, .4, .45, .5, .55]
    # COEFS = [.1, .15, .2, .25, .3]
    COEFS = [.4, .41, .42, .43, .44, .45]  # FOR LLAMA 3.1, 3.3 70b

    # COEFS = [.45, .475, .5, .525, .55]  # 
    # COEFS = [.45, .5, .55, .6, .65, .7]
    # COEFS = [.75, .76, .77, .78, .79, .80] # 
    # COEFS = [.55, .6, .65, .7, .75, .8] # FOR LLAMA 3.1 8b

    # COEFS = [.3, .35, .4, .425, .45, .475, .5]#, .55, .6]
    # COEFS = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    METHOD = 'rfm'

    model_type = 'llama'
    MODEL_VERSION = '3.3'
    MODEL_SIZE = '70B'
    llm = select_llm(model_type, MODEL_VERSION=MODEL_VERSION, MODEL_SIZE=MODEL_SIZE)

    VERSIONS = [1, 2, 3, 4, 5]

    for VERSION in VERSIONS:        
        if VERSION >= 2: 
            VERSION_LABEL = f'_v{VERSION}'
        else:
            VERSION_LABEL = f''
        for f_idx, fname in enumerate(fnames):
            # if f_idx != 4:
            #     continue
            if f_idx != 0:
                continue
            # if f_idx == 0 or f_idx == 3:
            #     continue
            concepts = read_file(fname, lower=lowers[f_idx])
            dataset_label = dataset_labels[f_idx]
            all_outputs = []

            for concept_idx, concept in enumerate(tqdm(concepts)):
                if concept != 'dogs':
                    continue
                # if concept != 'bad breath':
                #     continue
                print(f"=============================================CONCEPT={concept}=============================================")

                if dataset_label == 'fears':
                    if VERSION == 1:
                        prompt = "What is the scariest thing in the world? Answer in 20 words or less."
                    elif VERSION == 2:
                        prompt = "Is there anything you secretly hope never happens to you? Answer in 20 words or less."
                    elif VERSION == 3:
                        prompt = "What would a movie villain based on your worst fear be like? Answer in 20 words or less."
                    elif VERSION == 4:
                        prompt = "If a fear had to visit you in your sleep, what form would it take? Answer in 20 words or less."                
                    elif VERSION == 5:
                        prompt = "What would a fear-themed amusement park ride, made just for you, be like? Answer in 20 words or less."
                elif dataset_label == 'personalities':
                    if VERSION == 1:
                        prompt = "What is your favorite subject? Answer in 20 words or less." 
                    elif VERSION == 2:
                        prompt = "What's the kind of problem that makes you forget to eat or sleep? Answer in 20 words or less."
                    elif VERSION == 3: 
                        prompt = "If you had unlimited resources, what would you work on? Answer in 20 words or less."
                    elif VERSION == 4: 
                        prompt = "Describe your work in a way a 5-year-old can understand. Answer in 20 words or less."
                    elif VERSION == 5: 
                        prompt = "What's a question that got you hooked? Answer in 20 words or less."
                elif dataset_label == 'personas':
                    if VERSION == 1:
                        prompt = "Tell me about your work.  Answer in 20 words or less."
                    elif VERSION == 2:
                        prompt = "What's something you've always been drawn to? Answer in 20 words or less."
                    elif VERSION == 3: 
                        prompt = "If someone gave you the keys to everything — money, power, influence — what's the first thing you'd do? Answer in 20 words or less."
                    elif VERSION == 4: 
                        prompt = "Describe your work in a way a 5-year-old can understand. Answer in 20 words or less."
                    elif VERSION == 5: 
                        prompt = "If you could teach everyone one thing, what would it be? Answer in 20 words or less."
                elif dataset_label == 'moods':
                    if VERSION == 1:
                        prompt = "How are you feeling today? Answer in 20 words or less." 
                    elif VERSION == 2: 
                        prompt = "What's one sentence that would sum up your current vibe? Answer in 20 words or less."
                    elif VERSION == 3: 
                        prompt = "Tell me about your day. Answer in 20 words or less."
                    elif VERSION == 4: 
                        prompt = "What would a song that describes your current mood be called? Answer in 20 words or less."
                    elif VERSION == 5: 
                        prompt = "Where would you go if you could go anywhere right now? Answer in 20 words or less."
                elif dataset_label == 'places':
                    if VERSION == 1:
                        prompt = "What is your favorite place? Answer in 20 words or less." 
                    elif VERSION == 2:
                        prompt = "If you could instantly master any skill, what would it be? Answer in 20 words or less."
                    elif VERSION == 3:
                        prompt = "If you could build a shrine to one place, what would you put inside it? Answer in 20 words or less."
                    elif VERSION == 4:
                        prompt = "If someone filmed a movie about your life, where would the opening scene be set? Answer in 20 words or less."
                    elif VERSION == 5:
                        prompt = "Where were you the last time you felt completely at peace? Answer in 20 words or less."
                outputs = generate(concept, llm, prompt, image=None, coefs=COEFS, 
                                control_method=METHOD, max_tokens=50, gen_orig=False)
                all_outputs.append((concept, outputs))
                torch.cuda.empty_cache()
                # break
            return 
            with open(f"cached_outputs/{METHOD}_{dataset_label}_steered_500_concepts_{model_type}_{MODEL_VERSION}_{MODEL_SIZE}_english_only{VERSION_LABEL}.pkl", "wb") as file:
                pickle.dump(all_outputs, file)
            gc.collect()

    # return 



if __name__ == "__main__":
    main()