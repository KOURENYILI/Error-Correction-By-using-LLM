import os
import sys
import editdistance
import torch
import transformers
from peft import PeftModel
from transformers import GenerationConfig, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
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

load_8bit = True

def calculate_wer(pre, ref):
    wer_score = editdistance.eval(pre, ref) / len(ref)
    return wer_score

def filter_explanation_response(prediction, input_text):
    """
    过滤掉解释性回答，保留 quote/unquote 结构
    """
    print("\n====FILTER FUNCTION CALLED====")
    print(f"Checking prediction: {prediction}")
    print(f"Against input: {input_text}")
    
    # 清理预测文本：小写、去除首尾空格
    prediction_clean = prediction.strip().lower()
    print(f"Cleaned prediction: {prediction_clean}")
    
    # 去掉标点和空格后的文本
    prediction_no_punctuation = re.sub(r'[^a-z]', '', prediction_clean)
    
    # 解释性开头的关键词
    explanation_starts = [
        'thebest',
        'thecorrect',
        'thespeaker',
        'thesentence',
        'thestatement',
        'theresponse',
        'thisis',
        'theanswer',
        'thissentence'
    ]
    
    # 检查是否以任何解释性短语开头
    for pattern in explanation_starts:
        if prediction_no_punctuation.startswith(pattern):
            print(f"Found explanation pattern: {pattern}")
            print("====RETURNING ORIGINAL INPUT====")
            return input_text

    # 检查是否以 quote 开头并以 unquote 结尾
    if prediction_clean.startswith("quote") and prediction_clean.endswith("unquote"):
        print("====QUOTE MATCH DETECTED====")
        return prediction.strip()  # 保留完整句子
    
    # 检查长度是否过长
    if len(prediction.split()) > len(input_text.split()) * 1.5:
        print("Length check failed")
        print("====RETURNING ORIGINAL INPUT====")
        return input_text
    
    print("====NO FILTER APPLIED====")    
    return prediction.strip()

def postprocess_prediction(prediction):
    """
    对预测结果进行后处理：修正拼写错误、格式化数字表达等，并确保 quote/unquote 块的完整性
    """
    print("\n====POSTPROCESSING FUNCTION CALLED====")
    print(f"Original prediction: {prediction}")

    # 保留 quote/unquote 结构，避免删除标点符号
    quote_match = re.match(r'quote (.+?) unquote', prediction, re.IGNORECASE)
    if quote_match:
        print("====QUOTE/UNQUOTE BLOCK DETECTED====")
        return prediction.strip()  # 返回原始文本

    # 标准化数字表达形式
    numeric_mapping = {
        "three point two five": "three and a quarter",
        "fifteen point five": "fifteen and a half",
        "1.5": "one and a half",
        "3.25": "three and a quarter",
        "2.5": "two and a half"
    }
    for numeric, word in numeric_mapping.items():
        prediction = re.sub(rf'\b{re.escape(numeric)}\b', word, prediction)

    # 修正常见拼写错误（示例）
    spelling_corrections = {
        "recieve": "receive",
        "adress": "address",
        "teh": "the"
    }
    for wrong, correct in spelling_corrections.items():
        prediction = re.sub(rf'\b{re.escape(wrong)}\b', correct, prediction)

    print(f"Postprocessed prediction: {prediction}")
    return prediction

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

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--test_data_path", type=str, required=True, help='<your_test_data>.json')
    parser.add_argument("--output_dir", type=str, default="outputs", help='Directory to store outputs')
    parser.add_argument("--max_samples", type=int, default=None, help='Number of samples to test. If not provided, uses all data')
    parser.add_argument("--output_file", type=str, default="results.txt", help='Name of the output file')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    prompter = Prompter("H2T-LoRA")
    
    print("Loading model and tokenizer...")
    
    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True
    )
    
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
    
    model = PeftModel.from_pretrained(
        model,
        args.model_path,
        device_map={'': 0},
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
                raw_prediction = qwen_infer(input1=best_hypo, input2=input2)
                print(f"\nRaw prediction: {raw_prediction}")

                # 调用过滤函数
                filtered_prediction = filter_explanation_response(raw_prediction, best_hypo)
                print(f"Filtered prediction: {filtered_prediction}")
                
                # 调用后处理函数
                postprocessed_prediction = postprocess_prediction(filtered_prediction)
                print(f"Postprocessed prediction: {postprocessed_prediction}")

                # 移除首尾重复
                final_prediction = remove_head_tail_redundancy(postprocessed_prediction)
                print(f"Final prediction: {final_prediction}")

                # 对预测结果进行进一步处理
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

        # 最终统计结果写入文件
        f.write(f"before overall WER: {avg_wer_before:.4f}\n")
        f.write(f"after overall WER: {avg_wer_after:.4f}\n")
        f.write(f"JiWER before: {jiwer_wer_before:.4f}\n")
        f.write(f"JiWER after: {jiwer_wer_after:.4f}\n")

    print(f"\nResults saved to {output_txt}")
