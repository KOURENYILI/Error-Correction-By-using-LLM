import os
import sys
import editdistance
import torch
import transformers
from transformers import GenerationConfig, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import re
import json
from utils.prompter import Prompter
from argparse import ArgumentParser
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

def load_filter_keywords(file_path):
    """
    从 JSON 文件加载过滤关键词
    """
    with open(file_path, "r") as f:
        filter_keywords = json.load(f)
    explanation_starts = []
    for category, keywords in filter_keywords.items():
        explanation_starts.extend(keywords)
    return explanation_starts

def filter_explanation_response(prediction, input_text, explanation_starts):
    """
    根据过滤关键词过滤预测结果
    """
    prediction_clean = prediction.strip().lower()
    prediction_no_punctuation = re.sub(r'[^a-z]', '', prediction_clean)

    for pattern in explanation_starts:
        if prediction_no_punctuation.startswith(pattern):
            return input_text
    if prediction_clean.startswith("quote") and prediction_clean.endswith("unquote"):
        return prediction.strip()
    if len(prediction.split()) > len(input_text.split()) * 1.5:
        return input_text
    return prediction.strip()

def postprocess_prediction(prediction):
    quote_match = re.match(r'quote (.+?) unquote', prediction, re.IGNORECASE)
    if quote_match:
        return prediction.strip()

    numeric_mapping = {
        "three point two five": "three and a quarter",
        "fifteen point five": "fifteen and a half",
        "point five": "and a half",
        "point two five":"and a quarter",
        "1.5": "one and a half",
        "3.25": "three and a quarter",
        "2.5": "two and a half"
    }
    for numeric, word in numeric_mapping.items():
        prediction = re.sub(rf'\b{re.escape(numeric)}\b', word, prediction)
    spelling_corrections = {
        "recieve": "receive",
        "adress": "address",
        "teh": "the"
    }
    for wrong, correct in spelling_corrections.items():
        prediction = re.sub(rf'\b{re.escape(wrong)}\b', correct, prediction)
    return prediction

def remove_head_tail_redundancy(prediction):
    high_frequency_words = {"the", "a", "an", "of", "and", "to", "in", "on", "at", "for", "with", "by"}
    words = prediction.split()
    if len(words) > 1 and words[0] == words[-1] and words[0] not in high_frequency_words:
        words = words[:-1]
    return " ".join(words)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--test_data_path", type=str, required=True, help='<your_test_data>.json')
    parser.add_argument("--output_dir", type=str, default="outputs", help='Directory to store outputs')
    parser.add_argument("--max_samples", type=int, default=None, help='Number of samples to test. If not provided, uses all data')
    parser.add_argument("--output_file", type=str, default="results.txt", help='Name of the output file')
    args = parser.parse_args()

    # 加载过滤关键词
    filter_keywords_file = "filter_keywords.json"
    explanation_starts = load_filter_keywords(filter_keywords_file)

    os.makedirs(args.output_dir, exist_ok=True)
    output_txt = os.path.join(args.output_dir, args.output_file)

    prompter = Prompter("H2T-LoRA")
    
    print("Loading model and tokenizer...")
    
    bnb_config = BitsAndBytesConfig(load_in_8bit=True)
    
    tokenizer = AutoTokenizer.from_pretrained(
        'TinyLlama/TinyLlama-1.1B-Chat-v1.0',
        trust_remote_code=True
    )
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        'TinyLlama/TinyLlama-1.1B-Chat-v1.0',
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )

    if not load_8bit:
        model.half()
    model.eval()

    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    def qwen_infer(input1, input2=None):
        if input2 is not None:
            hypotheses = input2.split('.')
            hypotheses = [h.strip() for h in hypotheses if h.strip()]
            hypothesis_lengths = [len(tokenizer.encode(h)) for h in hypotheses]
            min_tokens = min(hypothesis_lengths)
            max_tokens = max(hypothesis_lengths)
            
            prompt = prompter.generate_prompt(input=input1, input2=input2)
        else:
            prompt = prompter.generate_prompt(input=input1)
            
        inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)
        
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
                attention_mask=attention_mask,
                generation_config=generation_config
            )

        output = tokenizer.decode(generation_output[0], skip_special_tokens=True)
        return prompter.get_response(output)

    print(f"Loading test data from {args.test_data_path}")
    with open(args.test_data_path, 'r') as f:
        test_data = json.load(f)

    if args.max_samples is not None:
        test_size = min(args.max_samples, len(test_data))
        test_data = test_data[:test_size]
        print(f"Using {test_size} samples for testing")
    else:
        print(f"Using all {len(test_data)} samples for testing")
    
    before, after, ignore = 0, 0, 0
    all_predictions, all_ground_truths, all_best_hypos = [], [], []

    with open(output_txt, 'w') as f:
        for i, data in enumerate(test_data):
            print(f"Processing sample {i+1}/{len(test_data)}")
            best_hypo = data['input1']
            input2 = data['input2']
            ground_truth = data['output']
            try:
                raw_prediction = qwen_infer(input1=best_hypo, input2=input2)
                filtered_prediction = filter_explanation_response(raw_prediction, best_hypo, explanation_starts)
                postprocessed_prediction = postprocess_prediction(filtered_prediction)
                final_prediction = remove_head_tail_redundancy(postprocessed_prediction)

                prediction = re.sub('</s>', '', final_prediction).lower()
                prediction = re.sub(r'[^\w\s]|[\d]', '', prediction)
                best_hypo = re.sub(r'[^\w\s]|[\d]', '', best_hypo)
                ground_truth = re.sub(r'[^\w\s]|[\d]', '', ground_truth)

                wer_prediction = calculate_wer(prediction.split(), ground_truth.split())
                wer_best_hypo = calculate_wer(best_hypo.split(), ground_truth.split())
                
                all_predictions.append(prediction)
                all_ground_truths.append(ground_truth)
                all_best_hypos.append(best_hypo)

                before += wer_best_hypo
                after += wer_prediction

                # 打印到终端
                print(f"Best hypothesis (before): {best_hypo}")
                print(f"Prediction (after): {prediction}")
                print(f"Ground truth: {ground_truth}")
                print(f"Before WER score: {wer_best_hypo:.4f}")
                print(f"After WER score: {wer_prediction:.4f}")
                print("-" * 50)

                # 保存到文件
                f.write(f"before::: {best_hypo}\n")
                f.write(f"after ::: {prediction}\n")
                f.write(f"answer::: {ground_truth}\n")
                f.write(f"before score: {wer_best_hypo:.4f}\n")
                f.write(f"after score: {wer_prediction:.4f}\n\n")

            except Exception as e:
                print(f"\nError processing sample {i+1}: {e}")
                ignore += 1
                continue

        avg_wer_before = before / (len(test_data) - ignore)
        avg_wer_after = after / (len(test_data) - ignore)
        jiwer_wer_before = jiwer_wer(all_ground_truths, all_best_hypos)
        jiwer_wer_after = jiwer_wer(all_ground_truths, all_predictions)

        # 打印总体结果
        print(f"\nOverall before WER: {avg_wer_before:.4f}")
        print(f"Overall after WER: {avg_wer_after:.4f}")
        print(f"JiWER before: {jiwer_wer_before:.4f}")
        print(f"JiWER after: {jiwer_wer_after:.4f}")

        # 保存总体结果到文件
        f.write(f"before overall WER: {avg_wer_before:.4f}\n")
        f.write(f"after overall WER: {avg_wer_after:.4f}\n")
        f.write(f"JiWER before: {jiwer_wer_before:.4f}\n")
        f.write(f"JiWER after: {jiwer_wer_after:.4f}\n")
    print(f"\nResults saved to {output_txt}")
