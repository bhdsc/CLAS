import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, MllamaForConditionalGeneration, AutoProcessor
from transformers import Gemma3ForCausalLM, BitsAndBytesConfig
from neural_controllers import NeuralController
from utils import LLMType
from collections import namedtuple 
from tqdm import tqdm 
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

        # print(tokenizer.chat_template)

    llm = LLM(language_model, tokenizer, processor, model_name, llm_type)
    # print(llm.language_model)
    return llm


def compute_save_directions(llm, dataset, concept, control_method='rfm'):
    concept_types = [concept]
    for concept_type in concept_types:
        controller = NeuralController(
            llm,
            llm.tokenizer,
            rfm_iters=8,
            control_method=control_method,
            n_components=1,
        )
        controller.compute_directions(dataset[concept_type]['train']['inputs'], dataset[concept_type]['train']['labels'])
        controller.save(concept=f'{concept_type}', model_name=llm.name, path='directions/')        



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


    model_type = 'llama'
    MODEL_VERSION='3.1'
    MODEL_SIZE='70B'
    llm = select_llm(model_type, MODEL_VERSION=MODEL_VERSION, MODEL_SIZE=MODEL_SIZE)

    METHOD = 'rfm'
    orig_lang = 'python'
    other_lang = 'c++'
    dataset = utils.programming_language_dataset(llm, orig_lang, other_lang)
    compute_save_directions(llm, dataset, other_lang, control_method=METHOD)



if __name__ == "__main__":
    main()