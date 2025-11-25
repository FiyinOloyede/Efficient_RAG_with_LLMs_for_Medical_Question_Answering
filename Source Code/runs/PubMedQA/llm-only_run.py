import warnings, time, re
import pandas as pd
from sklearn.metrics import f1_score, classification_report
from data_preprocessing import load_and_process_pubmedqa
from models import login_hf, load_model_and_tokenizer

warnings.filterwarnings('ignore')

# Login
login_hf("hf_token_here")  # Only if gated 

# Load any of the model in 8-bit ("meta-llama/Meta-Llama-3-8B-Instruct", "deepseek-ai/deepseek-llm-7b-chat",  "mistralai/Mistral-7B-Instruct-v0.3")
model_pipeline = load_model_and_tokenizer("meta-llama/Meta-Llama-3-8B-Instruct", "8bit")

# Load data
df = load_and_process_pubmedqa()

# Build prompts
def make_prompt(row):
    return (
        "You are a helpful biomedical assistant. Answer the question strictly with a single word: 'yes', 'no', or 'maybe'.\n"

        f"Question: {row['question']}\n"
        "Answer:"
    )

df["prompt"] = df.apply(make_prompt, axis=1)


# Benchmark standalone LLM 
results = []
total = len(df) # number of samples to evaluate (1000)

for i, (prompt, expected) in enumerate(zip(df["prompt"], df["final_decision"])):
    if i >= total:
        break

    start = time.time()
    prefill_words = len(prompt.split())
    output = model_pipeline(
        prompt, max_new_tokens=5, do_sample=False, temperature=0.0
    )[0]["generated_text"]
    latency = time.time() - start

    # Extract prediction after 'Answer'
    after_answer = output.lower().split("answer")[-1]

    # Try to directly extract yes/no/maybe with regex
    found = re.search(r"\b(yes|no|maybe)\b", after_answer) 
    predicted = found.group(1) if found else "?"

    # Look at the last line if extraction failed
    if predicted == "?":
        last_line = output.strip().splitlines()[-1].lower()
        if "yes" in last_line: predicted = "yes"
        elif "no" in last_line: predicted = "no"
        elif "maybe" in last_line: predicted = "maybe"

    results.append({
        "expected": expected,
        "predicted": predicted,
        "is_correct": predicted == expected,
        "latency": latency,
        "prefill_words": prefill_words
    })

    # Print first few and any failed cases for quick inspection
    if i < 5 or predicted == "?":
        print(f"\nPrompt:\n{prompt}\nOutput:\n{output}")
        print(f"Expected: {expected}, Predicted: {predicted}")

# Evaluation
df_results = pd.DataFrame(results) # Compile results dataframe
# Compute metrics
accuracy = df_results["is_correct"].mean()
avg_latency = df_results["latency"].mean()
avg_prefill_words = df_results["prefill_words"].mean()

# Compute F1 scores
macro_f1 = f1_score(df_results["expected"], df_results["predicted"], average="macro", zero_division=0)
micro_f1 = f1_score(df_results["expected"], df_results["predicted"], average="micro", zero_division=0)
weighted_f1 = f1_score(df_results["expected"], df_results["predicted"], average="weighted", zero_division=0)

# Print summary metrics
print(f"\nMCQ Benchmark completed on {total} samples")
print(f"Accuracy: {accuracy:.2%}")
print(f"Avg latency: {avg_latency:.2f}s")
print(f"Avg prefill words: {avg_prefill_words:.0f}")
print(f"Macro F1: {macro_f1:.4f}, Micro F1: {micro_f1:.4f}, Weighted F1: {weighted_f1:.4f}")
print("\nClassification report:\n", classification_report(df_results["expected"], df_results["predicted"], zero_division=0))

# Save results to CSV file for further analysis
df_results.to_csv("llm-only_llama3_results.csv", index=False)