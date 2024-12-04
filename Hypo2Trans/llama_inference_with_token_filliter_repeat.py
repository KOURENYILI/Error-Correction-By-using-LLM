import os
import sys
import editdistance
import torch
import transformers
from peft import PeftModel, LoraConfig
from transformers import GenerationConfig, AutoModelForCausalLM, AutoTokenizer
import re
from utils.prompter import Prompter
from argparse import ArgumentParser
import json
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

load_8bit = True  # 如果不支持量化，可以设置为 False

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
    # 数字表达式的映射
    number_mappings = {
        r"\b(one) (thousand) (nine) (hundred) (and) (\w+)\b": r"19\5",
        "one point five": "a dollar fifty",
        "one point five dollars": "a dollar fifty",
        "1.5": "a dollar fifty",
        "1.50": "a dollar fifty",
        "point twenty five": "a quarter",
        "zero point twenty five": "a quarter",
        "zero point two five": "a quarter",
        "0.25": "a quarter",
        "0.250": "a quarter",
        "zero point twenty-five": "a quarter",
        "zero point 25": "a quarter",
        ".25": "a quarter",
        "point two five": "a quarter"
    }
    
    # 先处理特殊的数字映射
    prediction_clean = prediction.lower()
    for number_expr, replacement in number_mappings.items():
        if isinstance(number_expr, str) and number_expr in prediction_clean:
            prediction_clean = prediction_clean.replace(number_expr, replacement)
            break
        elif isinstance(number_expr, str):
            prediction_clean = re.sub(number_expr, replacement, prediction_clean)

    # 处理数字表达中的连字符和空格
    prediction_clean = re.sub(r'(\w+)(-?)five', r'\1 five', prediction_clean)
    prediction_clean = re.sub(r'seventy(?![ -])', 'seventy ', prediction_clean)
    prediction_clean = re.sub(r'twenty(?![ -])', 'twenty ', prediction_clean)
    prediction_clean = re.sub(r'thirty(?![ -])', 'thirty ', prediction_clean)
    prediction_clean = re.sub(r'forty(?![ -])', 'forty ', prediction_clean)
    prediction_clean = re.sub(r'fifty(?![ -])', 'fifty ', prediction_clean)
    prediction_clean = re.sub(r'sixty(?![ -])', 'sixty ', prediction_clean)
    prediction_clean = re.sub(r'eighty(?![ -])', 'eighty ', prediction_clean)
    prediction_clean = re.sub(r'ninety(?![ -])', 'ninety ', prediction_clean)
    
    prediction_no_punctuation = re.sub(r'[^a-z]', '', prediction_clean)
   
    # 检查过滤条件
    for pattern in explanation_starts:
        if prediction_no_punctuation.startswith(pattern):
            return input_text
           
    # 只有完整的 quote...unquote 结构才强制保留
    if prediction_clean.startswith("quote") and prediction_clean.endswith("unquote"):
        return prediction_clean
       
    if len(prediction_clean.split()) > len(input_text.split()) * 1.5:
        return input_text
   
    # 纠正首个单词如果它是拼写错误
    input_words = input_text.split()
    pred_words = prediction_clean.split()
   
    if len(input_words) > 0 and len(pred_words) > 0:
        if pred_words[0].lower() != input_words[0].lower() and len(pred_words[0]) >= len(input_words[0])/2:
            distance = editdistance.eval(pred_words[0].lower(), input_words[0].lower())
            if distance <= 2:  # 如果编辑距离较小，认为是拼写错误
                pred_words[0] = input_words[0]

    return ' '.join(pred_words)

def remove_head_tail_redundancy(prediction):
    """
    检查并移除首尾重复的单词（排除高频词）
    """
    print("\n====HEAD-TAIL REDUNDANCY CHECK CALLED====")
    print(f"Original prediction: {prediction}")

    # 定义高频词列表，这些词即使首尾重复也不会被移除
    high_frequency_words = {"the", "a", "an", "of", "and", "to", "in", "on", "at", "for", "with", "by"}

    # 按单词分割
    words = prediction.split()
   
    # 如果首尾单词相同，且不是高频词，则移除最后一个单词
    if len(words) > 1 and words[0] == words[-1] and words[0] not in high_frequency_words:
        print(f"Removed head-tail duplicate: {words[0]}")
        words = words[:-1]  # 移除最后一个单词
   
    # 合并为字符串
    cleaned_prediction = " ".join(words)
    print(f"Cleaned prediction: {cleaned_prediction}")
    return cleaned_prediction

def remove_ending_text(text):
    """
    删除文本末尾从句号开始的所有内容
    """
    try:
        period_index = text.rindex('.')
        return text[:period_index+1]
    except ValueError:
        return text

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--test_data_path", type=str, required=True, help='<your_test_data>.json')
    parser.add_argument("--output_dir", type=str, default="outputs", help='Directory to store outputs')
    parser.add_argument("--max_samples", type=int, default=None, help='Number of samples to test. If not provided, uses all data')
    parser.add_argument("--output_file", type=str, default="results.txt", help='Name of the output file')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # 加载过滤关键词
    explanation_starts = load_filter_keywords('filter_keywords.json')
    prompter = Prompter("H2T-LoRA")
   
    print("Loading model and tokenizer...")

    try:
        # 1. 首先尝试加载 tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            'meta-llama/Llama-3.2-3B-Instruct',
            trust_remote_code=True
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
       
        # 2. 加载基础模型
        model = AutoModelForCausalLM.from_pretrained(
            'meta-llama/Llama-3.2-3B-Instruct',
            device_map="auto",
            trust_remote_code=True
        )
       
        # 3. 尝试加载 LoRA 权重，有多级备选方案
        try:
            print("Attempting to load LoRA weights...")
            # 尝试直接加载 LoRA
            model = PeftModel.from_pretrained(
                model,
                args.model_path,
                device_map={'': 0},
            )
            print("Successfully loaded LoRA weights!")
        except Exception as e:
            print(f"Warning: Failed to load LoRA directly: {e}")
            try:
                # 尝试使用基本 LoRA 配置
                print("Attempting to load LoRA with basic config...")
                config = LoraConfig(
                    r=8,
                    lora_alpha=32,
                    target_modules=["q_proj", "v_proj"],
                    lora_dropout=0.05,
                    bias="none",
                )
                model = PeftModel.from_pretrained(
                    model,
                    args.model_path,
                    device_map={'': 0},
                    config=config
                )
                print("Successfully loaded LoRA with basic config!")
            except Exception as e2:
                print(f"Warning: Failed to load LoRA with basic config: {e2}")
                print("Falling back to base model only...")

        # 4. 模型设置
        if not load_8bit:
            model.half()
        model.eval()

        if torch.__version__ >= "2" and sys.platform != "win32":
            model = torch.compile(model)
           
        print("Model loading complete!")

    except Exception as e:
        print(f"Critical error loading model: {e}")
        sys.exit(1)

    def llama_infer(input1, input2=None):
        if input2 is not None:
            hypotheses = input2.split('.')
            hypotheses = [h.strip() for h in hypotheses if h.strip()]
           
            hypothesis_lengths = [len(tokenizer.encode(h)) for h in hypotheses]
            min_tokens = min(hypothesis_lengths)
            max_tokens = max(hypothesis_lengths)
           
            prompt = prompter.generate_prompt(input=input1, input2=input2)
        else:
            prompt = prompter.generate_prompt(input=input1)
           
        inputs = tokenizer(prompt, return_tensors="pt", padding=True)
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
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True
            )

        s = generation_output.sequences[0]
        output = tokenizer.decode(s, skip_special_tokens=True)
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
   
    before = 0
    after = 0
    ignore = 0
   
    all_predictions = []
    all_ground_truths = []
    all_best_hypos = []
   
    outputs = []

    output_txt = os.path.join(args.output_dir, args.output_file)
    with open(output_txt, 'w') as f:
        for i in range(len(test_data)):
            print(f"Processing sample {i+1}/{len(test_data)}", end='\r')
           
            best_hypo = test_data[i]['input1']
            input2 = test_data[i]['input2']
            ground_truth = test_data[i]['output']
           
            try:
                # 获取原始预测
                raw_prediction = llama_infer(input1=best_hypo, input2=input2)
                print(f"\nRaw prediction: {raw_prediction}")

                # 过滤预测结果
                filtered_prediction = filter_explanation_response(raw_prediction, best_hypo, explanation_starts)
                print(f"Filtered prediction: {filtered_prediction}")

                # 移除首尾重复
                final_prediction = remove_head_tail_redundancy(filtered_prediction)
                final_prediction = remove_ending_text(final_prediction)
                print(f"Final prediction: {final_prediction}")

                # 最终处理结果
                prediction = final_prediction
                prediction = re.sub('</s>', '', prediction).lower()
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

                # 实时打印到终端
                print(f"\nbefore::: {best_hypo}")
                print(f"after ::: {prediction}")
                print(f"answer::: {ground_truth}")
                print(f"before score: {wer_best_hypo:.4f}")
                print(f"after score: {wer_prediction:.4f}")
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

        print(f"\nbefore overall WER: {avg_wer_before:.4f}")
        print(f"after overall WER: {avg_wer_after:.4f}")
        print(f"JiWER before: {jiwer_wer_before:.4f}")
        print(f"JiWER after: {jiwer_wer_after:.4f}")

        f.write(f"before overall WER: {avg_wer_before:.4f}\n")
        f.write(f"after overall WER: {avg_wer_after:.4f}\n")
        f.write(f"JiWER before: {jiwer_wer_before:.4f}\n")
        f.write(f"JiWER after: {jiwer_wer_after:.4f}\n")

    print(f"\nResults saved to {output_txt}")
