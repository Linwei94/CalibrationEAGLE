# Environment Setup

```bash
git clone https://github.com/Linwei94/CalibrationEAGLE.git
cd CalibrationEAGLE
uv sync
```
# Test CalibrationEagle
## Download Hugging Face weights (two minimal test cases)
```bash
hf download meta-llama/Llama-3.1-8B-Instruct
hf download yuhuili/EAGLE3-LLaMA3.1-Instruct-8B
```

# Evaluation
```bash
python -m eagle.evaluation.gen_ea_answer_llama3chat --ea-model-path yuhuili/EAGLE3-LLaMA3.1-Instruct-8B --base-model-path meta-llama/Llama-3.1-8B-Instruct --use_eagle3
```