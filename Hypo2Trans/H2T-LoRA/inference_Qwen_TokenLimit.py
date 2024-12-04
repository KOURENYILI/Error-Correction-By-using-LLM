# import os
# import sys
# import editdistance
# import torch
# import transformers
# from peft import PeftModel
# from transformers import GenerationConfig, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
# import re
# from utils.callbacks import Iteratorize, Stream
# from utils.prompter import Prompter
# from argparse import ArgumentParser
# import json
# import logging

# if torch.cuda.is_available():
#     device = "cuda"
# else:
#     device = "cpu"

# try:
#     if torch.backends.mps.is_available():
#         device = "mps"
# except:  # noqa: E722
#     pass

# load_8bit = True

# def calculate_wer(pre, ref):
#     wer_score = editdistance.eval(pre, ref) / len(ref)
#     return wer_score

# if __name__ == '__main__':
#     parser = ArgumentParser()
#     parser.add_argument("--ckpt_path", type=str, required=True)
#     parser.add_argument("--test_data_path", type=str, required=True, help='<your_test_data>.json')
#     args = parser.parse_args()

#     prompter = Prompter("H2T-LoRA")
    
#     print("Loading model and tokenizer...")
    
#     # Simplified quantization config
#     bnb_config = BitsAndBytesConfig(
#         load_in_8bit=True
#     )
    
#     # Load Qwen tokenizer and model
#     tokenizer = AutoTokenizer.from_pretrained(
#         'Qwen/Qwen2.5-7B-Instruct',
#         trust_remote_code=True
#     )
    
#     model = AutoModelForCausalLM.from_pretrained(
#         'Qwen/Qwen2.5-7B-Instruct',
#         quantization_config=bnb_config,
#         device_map="auto",
#         trust_remote_code=True
#     )
    
#     # Load the fine-tuned LoRA weights
#     model = PeftModel.from_pretrained(
#         model,
#         args.ckpt_path,
#         device_map={'': 0},
#     )

#     if not load_8bit:
#         model.half()
#     model.eval()

#     if torch.__version__ >= "2" and sys.platform != "win32":
#         model = torch.compile(model)

#     def qwen_infer(input1, input2=None):
#         if input2 is not None:
#             # Calculate min and max tokens from hypotheses
#             hypotheses = input2.split('.')
#             hypotheses = [h.strip() for h in hypotheses if h.strip()]
            
#             # Get token lengths for each hypothesis
#             hypothesis_lengths = [len(tokenizer.encode(h)) for h in hypotheses]
#             min_tokens = min(hypothesis_lengths)
#             max_tokens = max(hypothesis_lengths)
            
#             prompt = prompter.generate_prompt(input=input1, input2=input2)
#         else:
#             prompt = prompter.generate_prompt(input=input1)
            
#         inputs = tokenizer(prompt, return_tensors="pt")
#         input_ids = inputs["input_ids"].to(device)
        
#         generation_config = GenerationConfig(
#             do_sample=True,
#             temperature=0.1,
#             top_p=0.75,
#             top_k=40,
#             num_beams=4,
#             max_new_tokens=max_tokens if input2 is not None else 256,
#             min_new_tokens=min_tokens if input2 is not None else 10,
#             pad_token_id=tokenizer.pad_token_id,
#             eos_token_id=tokenizer.eos_token_id
#         )
        
#         with torch.no_grad():
#             generation_output = model.generate(
#                 input_ids=input_ids,
#                 generation_config=generation_config,
#                 return_dict_in_generate=True,
#                 output_scores=True
#             )

#         s = generation_output.sequences[0]
#         output = tokenizer.decode(s, skip_special_tokens=True)
#         return prompter.get_response(output)

#     print(f"Loading test data from {args.test_data_path}")
#     # Load and process test data
#     with open(args.test_data_path, 'r') as f:
#         test_data = json.load(f)

#     print(f"Processing {len(test_data)} test samples...")
    
#     before = 0
#     after = 0
#     ignore = 0

#     for i in range(len(test_data)):
#         print(f"Processing sample {i+1}/{len(test_data)}", end='\r')
        
#         best_hypo = test_data[i]['input1']
#         input2 = test_data[i]['input2']
#         ground_truth = test_data[i]['output']
        
#         prediction = qwen_infer(input1=best_hypo, input2=input2)

#         # Clean up the text
#         prediction = re.sub('</s>', '', prediction).lower()
#         prediction = re.sub(r'[^\w\s]|[\d]', '', prediction)
#         best_hypo = re.sub(r'[^\w\s]|[\d]', '', best_hypo)
#         ground_truth = re.sub(r'[^\w\s]|[\d]', '', ground_truth)
#         prediction = re.sub(r'\n+.+', '', prediction)

#         try:
#             wer_prediction = calculate_wer(prediction.split(), ground_truth.split())
#             wer_best_hypo = calculate_wer(best_hypo.split(), ground_truth.split())
#         except Exception as e:
#             print(f"\nError processing sample {i+1}: {str(e)}")
#             ignore += 1
#             continue

#         before = before + wer_best_hypo
#         after = after + wer_prediction
        
#         if wer_best_hypo != wer_prediction:
#             print(f"\nSample {i+1}:")
#             print('before:::', best_hypo)
#             print('after :::', prediction)
#             print('answer:::', ground_truth)
#             print('before score', wer_best_hypo)
#             print('after score', wer_prediction)
#             print('-' * 50)

#     print('\nFinal Results:')
#     print('Average WER before:', before / (len(test_data) - ignore))
#     print('Average WER after:', after / (len(test_data) - ignore))
#     print('Number of samples ignored:', ignore)

import os
import sys
import editdistance
import torch
import transformers
from peft import PeftModel
from transformers import GenerationConfig, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import re
from utils.callbacks import Iteratorize, Stream
from utils.prompter import Prompter
from argparse import ArgumentParser
import json
import logging
from jiwer import wer as jiwer_wer

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

try:
    if torch.backends.mps.is_available():
        device = "mps"
except:  # noqa: E722
    pass

load_8bit = True

def calculate_wer(pre, ref):
    wer_score = editdistance.eval(pre, ref) / len(ref)
    return wer_score

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--test_data_path", type=str, required=True, help='<your_test_data>.json')
    parser.add_argument("--output_dir", type=str, default="outputs", help='Directory to store outputs')
    parser.add_argument("--test_size", type=int, default=None, help='Number of samples to test. If not provided, uses all data')
    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    prompter = Prompter("H2T-LoRA")
    
    print("Loading model and tokenizer...")
    
    # Simplified quantization config
    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True
    )
    
    # Load Qwen tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(
        'Qwen/Qwen2.5-7B-Instruct',
        trust_remote_code=True
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        'Qwen/Qwen2.5-7B-Instruct',
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    
    # Load the fine-tuned LoRA weights
    model = PeftModel.from_pretrained(
        model,
        args.ckpt_path,
        device_map={'': 0},
    )

    if not load_8bit:
        model.half()
    model.eval()

    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    def qwen_infer(input1, input2=None):
        if input2 is not None:
            # Calculate min and max tokens from hypotheses
            hypotheses = input2.split('.')
            hypotheses = [h.strip() for h in hypotheses if h.strip()]
            
            # Get token lengths for each hypothesis
            hypothesis_lengths = [len(tokenizer.encode(h)) for h in hypotheses]
            min_tokens = min(hypothesis_lengths)
            max_tokens = max(hypothesis_lengths)
            
            prompt = prompter.generate_prompt(input=input1, input2=input2)
        else:
            prompt = prompter.generate_prompt(input=input1)
            
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)
        
        generation_config = GenerationConfig(
            do_sample=True,
            temperature=0.1,
            top_p=0.75,
            top_k=40,
            num_beams=4,
            max_new_tokens=max_tokens if input2 is not None else 256,
            min_new_tokens=min_tokens if input2 is not None else 10,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
        
        with torch.no_grad():
            generation_output = model.generate(
                input_ids=input_ids,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True
            )

        s = generation_output.sequences[0]
        output = tokenizer.decode(s, skip_special_tokens=True)
        return prompter.get_response(output)

    print(f"Loading test data from {args.test_data_path}")
    # Load and process test data
    with open(args.test_data_path, 'r') as f:
        test_data = json.load(f)

    # Limit test size if specified
    if args.test_size is not None:
        test_size = min(args.test_size, len(test_data))
        test_data = test_data[:test_size]
        print(f"Using {test_size} samples for testing")
    else:
        print(f"Using all {len(test_data)} samples for testing")
    
    before = 0
    after = 0
    ignore = 0
    
    # Lists to store all predictions and ground truths for jiwer
    all_predictions = []
    all_ground_truths = []
    all_best_hypos = []
    
    # List to store detailed outputs
    outputs = []

    for i in range(len(test_data)):
        print(f"Processing sample {i+1}/{len(test_data)}", end='\r')
        
        best_hypo = test_data[i]['input1']
        input2 = test_data[i]['input2']
        ground_truth = test_data[i]['output']
        
        prediction = qwen_infer(input1=best_hypo, input2=input2)

        # Clean up the text
        prediction = re.sub('</s>', '', prediction).lower()
        prediction = re.sub(r'[^\w\s]|[\d]', '', prediction)
        best_hypo = re.sub(r'[^\w\s]|[\d]', '', best_hypo)
        ground_truth = re.sub(r'[^\w\s]|[\d]', '', ground_truth)
        prediction = re.sub(r'\n+.+', '', prediction)

        try:
            wer_prediction = calculate_wer(prediction.split(), ground_truth.split())
            wer_best_hypo = calculate_wer(best_hypo.split(), ground_truth.split())
            
            # Add to lists for jiwer calculation
            all_predictions.append(prediction)
            all_ground_truths.append(ground_truth)
            all_best_hypos.append(best_hypo)
            
        except Exception as e:
            print(f"\nError processing sample {i+1}: {str(e)}")
            ignore += 1
            continue

        before = before + wer_best_hypo
        after = after + wer_prediction
        
        # Store output details
        output_detail = {
            "sample_id": i + 1,
            "before": best_hypo,
            "after": prediction,
            "ground_truth": ground_truth,
            "before_score": wer_best_hypo,
            "after_score": wer_prediction
        }
        outputs.append(output_detail)
        
        if wer_best_hypo != wer_prediction:
            print(f"\nSample {i+1}:")
            print('before:::', best_hypo)
            print('after :::', prediction)
            print('answer:::', ground_truth)
            print('before score', wer_best_hypo)
            print('after score', wer_prediction)
            print('-' * 50)

    # Calculate average WER
    avg_wer_before = before / (len(test_data) - ignore)
    avg_wer_after = after / (len(test_data) - ignore)
    
    # Calculate jiwer WER
    jiwer_wer_before = jiwer_wer(all_ground_truths, all_best_hypos)
    jiwer_wer_after = jiwer_wer(all_ground_truths, all_predictions)

    # Save detailed outputs
    output_file = os.path.join(args.output_dir, "detailed_outputs.json")
    with open(output_file, 'w') as f:
        json.dump(outputs, f, indent=2)

    # Save WER results
    wer_results = {
        "test_size": len(test_data),
        "average_wer": {
            "before": avg_wer_before,
            "after": avg_wer_after
        },
        "jiwer_wer": {
            "before": jiwer_wer_before,
            "after": jiwer_wer_after
        },
        "samples_ignored": ignore
    }
    
    wer_file = os.path.join(args.output_dir, "wer_results.json")
    with open(wer_file, 'w') as f:
        json.dump(wer_results, f, indent=2)

    print('\nFinal Results:')
    print('Test size:', len(test_data))
    print('Average WER before:', avg_wer_before)
    print('Average WER after:', avg_wer_after)
    print('JiWER WER before:', jiwer_wer_before)
    print('JiWER WER after:', jiwer_wer_after)
    print('Number of samples ignored:', ignore)
    print(f'\nResults saved to {args.output_dir}')