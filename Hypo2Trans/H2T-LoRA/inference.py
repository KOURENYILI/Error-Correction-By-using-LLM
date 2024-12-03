import os
import sys
import editdistance
import torch
import transformers
from peft import PeftModel
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer
import re
from utils.callbacks import Iteratorize, Stream
from utils.prompter import Prompter
from argparse import ArgumentParser
import json
import logging
from vllm import LLM, SamplingParams
import csv
import gc

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

# if __name__ == '__main__':
#     parser = ArgumentParser()
#     parser.add_argument("--lora_model_path", type=str, required=True, help='/root/Hypo2Trans/merge_lora')
#     parser.add_argument("--test_data_path", type=str, required=True, help='data/test_wsj.json')
#     args = parser.parse_args()

#     prompter = Prompter("alpaca")
#     # tokenizer = LlamaTokenizer.from_pretrained(lora_model_path)
#     tokenizer = LlamaTokenizer.from_pretrained(args.lora_model_path)

#     # model = LlamaForCausalLM.from_pretrained(
#     #     '/root/Hypo2Trans/merge_lora',
#     #     load_in_8bit=load_8bit,
#     #     torch_dtype=torch.float16,
#     #     device_map='auto',
#     # )
#     # model = PeftModel.from_pretrained(
#     #     model,
#     #     # args.ckpt_path,
#     #     torch_dtype=torch.float16,
#     #     device_map={'': 0},
#     # )

#     # model.config.pad_token_id = 0
#     # model.config.bos_token_id = 1
#     # model.config.eos_token_id = 2
#     # if not load_8bit:
#     #     model.half()
#     # model.eval()

#     # if torch.__version__ >= "2" and sys.platform != "win32":
#     #     model = torch.compile(model)
        
#     # Initialize vLLM
#     # model = lora_model_path
#     model = args.lora_model_path
#     llm = LLM(model=model,tokenizer=model, device=device)
#     sampling_params = SamplingParams(
#         temperature=0.1,
#         top_p=0.75,
#         top_k=40,
#         # num_beams=4,
#         # max_new_tokens=256,
#     )

#     def alpaca_infer(input1, input2=None):
#         if input2 is not None:
#             prompt,_ = prompter.generate_prompt(input=input1, input2=input2)
#         else:
#             prompt,_ = prompter.generate_prompt(input=input1)
#         # print(prompt)
#         # inputs = tokenizer(prompt, return_tensors="pt")
#         # input_ids = inputs["input_ids"].to(device)
        
#         # Use vLLM for inference
#         with torch.no_grad():
#             generation_output = llm.generate(
#                 prompt,
#                 sampling_params
#             )
#         for output in generation_output:
#             output1 = output.outputs[0].text
#         # s = generation_output.sequences[0]
#         # output = tokenizer.decode(s)
#         # print("output",output)
#         # print(fefefe)
#         # return prompter.get_response(output)
#         return output1



#     with open(args.test_data_path, 'r') as f:
#         test_data = json.load(f)

#     with open("wer_results.txt", "w") as result_file:
#         result_file.write("Before_WER\tAfter_WER\tBest_Hypothesis\tPrediction\tGround_Truth\n")  # Header


#     before = 0
#     after = 0

#     ignore = 0

#     for i in range(len(test_data)):
#         best_hypo = test_data[i]['input1']
#         input2 = test_data[i]['input2']
#         ground_truth = test_data[i]['output']
#         prediction = alpaca_infer(input1=best_hypo, input2=input2)
#         prediction = re.sub('</s>', '', prediction).lower()
#         prediction = re.sub(r'\.\s.*', '', prediction)  # MODIFY
#         prediction = re.sub(r'[^\w\s]|[\d]', '', prediction)
#         best_hypo = re.sub(r'[^\w\s]|[\d]', '', best_hypo)
#         ground_truth = re.sub(r'[^\w\s]|[\d]', '', ground_truth)
#         prediction = re.sub(r'\n+.+', '', prediction)
#         try:
#             wer_prediction = calculate_wer(prediction.split(), ground_truth.split())
#             wer_best_hypo = calculate_wer(best_hypo.split(), ground_truth.split())
#         except Exception:
#             ignore += 1
#             continue
        

#         before = before + wer_best_hypo
#         after = after + wer_prediction
#         if wer_best_hypo != wer_prediction:
#             print('before:::', best_hypo)
#             print('after :::', prediction)
#             print('answer:::', ground_truth)
#             print('before score', wer_best_hypo)
#             print('after score', wer_prediction)

#             result_file.write(f"{wer_best_hypo}\t{wer_prediction}\t{best_hypo}\t{prediction}\t{ground_truth}\n")



#     print('before', before / (len(test_data) - ignore))
#     print('after', after / (len(test_data) - ignore))

#     result_file.write(f"\nAverage Before WER: {before / (len(test_data) - ignore)}\n")
#     result_file.write(f"Average After WER: {after / (len(test_data) - ignore)}\n")


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--lora_model_path", type=str, required=True, help='/root/Hypo2Trans/merge_lora')
    parser.add_argument("--test_data_path", type=str, required=True, help='data/test_wsj.json')
    args = parser.parse_args()

    prompter = Prompter("alpaca")
    # tokenizer = LlamaTokenizer.from_pretrained(lora_model_path)
    tokenizer = LlamaTokenizer.from_pretrained(args.lora_model_path)

    # The original code used LlamaForCausalLM and PeftModel to load and perform inference with the base model and LoRA model.
    # The modified code removes the loading logic for LlamaForCausalLM and PeftModel, replacing it with vLLM for inference.
    
    # model = LlamaForCausalLM.from_pretrained(
    #     '/root/Hypo2Trans/merge_lora',
    #     load_in_8bit=load_8bit,
    #     torch_dtype=torch.float16,
    #     device_map='auto',
    # )
    # model = PeftModel.from_pretrained(
    #     model,
    #     # args.ckpt_path,
    #     torch_dtype=torch.float16,
    #     device_map={'': 0},
    # )

    # model.config.pad_token_id = 0
    # model.config.bos_token_id = 1
    # model.config.eos_token_id = 2
    # if not load_8bit:
    #     model.half()
    # model.eval()

    # if torch.__version__ >= "2" and sys.platform != "win32":
    #     model = torch.compile(model)
        
    # Initialize vLLM
    # model = lora_model_path
    model = args.lora_model_path
    llm = LLM(model=model,tokenizer=model, device=device, max_model_len=2048)
    
    # vLLM uses SamplingParams to configure sampling parameters. GenerationConfig has been removed
    # and replaced with SamplingParams from vLLM for setting sampling parameters during inference.
    sampling_params = SamplingParams(
        temperature=0.1,
        top_p=0.75,
        top_k=40,
        max_tokens=512,
        # num_beams=4,
        # max_new_tokens=256,
        stop_token_ids=[32002],
    )

    def alpaca_infer(input1, input2=None):
        if input2 is not None:
            prompt,_ = prompter.generate_prompt(input=input1, input2=input2)
            # print(f"prompt::: {prompt}\n##############")
            token_len=len(tokenizer.encode(prompt))
            # print(f"token length::{token_len}")
        else:
            prompt,_ = prompter.generate_prompt(input=input1)
        # print(prompt)
        # inputs = tokenizer(prompt, return_tensors="pt")
        # input_ids = inputs["input_ids"].to(device)
        
        # Use vLLM for inference
        with torch.no_grad():
            generation_output = llm.generate(
                prompt,
                sampling_params
            )
        for output in generation_output:
            output1 = output.outputs[0].text
        # s = generation_output.sequences[0]
        # output = tokenizer.decode(s)
        # print("output",output)
        # print(fefefe)
        # return prompter.get_response(output)

        # This is to adapt to vLLM's inference mechanism. vLLM does not require encoding the input as before, 
        # but instead generates the output directly from the prompt. This improves the simplicity and performance of the code.
        return output1


    with open(args.test_data_path, 'r') as f:
        test_data = json.load(f)

    with open("only_lora_results.csv", "w", newline="") as csvfile:
    # with open("only_evo_results.csv", "w", newline="") as csvfile:
    # with open("evo_lora_results.csv", "w", newline="") as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["Best_Hypothesis", "Prediction", "Ground_Truth", "Before_WER", "After_WER"])

        before = 0
        after = 0
    
        ignore = 0
    
        def extract_responses(txt):
    
          pattern = r"(?:### Response:|Response:|### True Transcription:)\s*(.*?)(?=(\n###|\Z))"
          matches = re.findall(pattern, txt, re.DOTALL)
          if matches:
              res = matches[0][0].strip().lower()
              res = re.sub(r'[^\w\s]','',res)
              return res
                
          return ""
            
        for i in range(len(test_data)):
            best_hypo = test_data[i]['input1']
            input2 = test_data[i]['input2']
            ground_truth = test_data[i]['output']
            prediction = alpaca_infer(input1=best_hypo, input2=input2)
            # print(f"res ::: {prediction}")
            prediction = extract_responses(prediction)
            gc.collect()
            torch.cuda.empty_cache()
            # print(f"res_ext ::: {prediction}")
            #prediction = re.sub('</s>', '', prediction).lower()
    
            # This regular expression matches the content after a period and removes it.
            # The purpose is to delete the part of the prediction result that comes after the period.
            #prediction = re.sub(r'\.\s.*', '', prediction)  # MODIFY
            #prediction = re.sub(r'[^\w\s]|[\d]', '', prediction)
            best_hypo = re.sub(r'[^\w\s]|[\d]', '', best_hypo)
            ground_truth = re.sub(r'[^\w\s]|[\d]', '', ground_truth)
            #prediction = re.sub(r'\n+.+', '', prediction)
            # try:
            #     wer_prediction = calculate_wer(prediction.split(), ground_truth.split())
            #     wer_best_hypo = calculate_wer(best_hypo.split(), ground_truth.split())
            # except Exception:
            #     ignore += 1
            #     continue
    
            try:
                wer_prediction = calculate_wer(prediction.split(), ground_truth.split())
                wer_best_hypo = calculate_wer(best_hypo.split(), ground_truth.split())
            except Exception as e:
                print(f"Exception encountered: {e}")
                ignore += 1
                continue
    
    
            before = before + wer_best_hypo
            after = after + wer_prediction
    
            csv_writer.writerow([best_hypo, prediction, ground_truth, wer_best_hypo, wer_prediction])
    
            if wer_best_hypo != wer_prediction:
                print('before:::', best_hypo)
                print('after :::', prediction)
                print('answer:::', ground_truth)
                print('before score', wer_best_hypo)
                print('after score', wer_prediction)


        print('before', before / (len(test_data) - ignore))
        print('after', after / (len(test_data) - ignore))
    
        csv_writer.writerow([])
        csv_writer.writerow(["Average Before WER", before / (len(test_data) - ignore)])
        csv_writer.writerow(["Average After WER", after / (len(test_data) - ignore)])





