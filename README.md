 NaijaCXR-VLM: Domain-Adapted Vision-Language Model for Nigerian Chest X-Ray Report Generation

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)](https://pytorch.org)
[![HuggingFace](https://img.shields.io/badge/🤗-Transformers-yellow)](https://huggingface.co)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

## Overview

**NaijaCXR-VLM** is a fine-tuned Vision-Language Model (VLM) for automated radiology report
generation from Nigerian chest X-rays (NaijaCXR dataset). Built on
[Google MedGemma-4b-it](https://huggingface.co/google/medgemma-4b-it) and fine-tuned using
Parameter-Efficient Fine-Tuning (PEFT/LoRA), this model bridges the gap between globally
pre-trained medical AI and the specific clinical context of Nigerian radiological practice.

This work is part of a PhD research project investigating domain adaptation between the
widely-used MIMIC-CXR dataset (US-based) and the NaijaCXR dataset (Nigeria-based), with
the goal of developing clinically reliable AI-assisted radiology tools for low-to-middle
income country (LMIC) healthcare settings.

---

## Key Results

| Metric | Value |
|--------|-------|
| Normal/Abnormal Agreement | **91.5%** (54/59 valid predictions) |
| Abnormal Cases Correctly Flagged | **22/25** (88%) |
| Unique Predictions / 60 test samples | **15** (vs. 3 in domain-adapted version) |
| Prediction Normal Rate | 58.3% (matches ground truth exactly) |
| Best Eval Loss (epoch 3) | **1.400** |


## Model Architecture

MedGemma-4b-it (base)
├── SigLIP Vision Encoder (native 896×896 → 256 tokens, frozen)
├── Multi-Modal Projector (fully fine-tuned)
└── Gemma-3 4B Language Model
└── LoRA Adapters (r=16, α=32)
└── Targets: q_proj, k_proj, v_proj, o_proj,
gate_proj, up_proj, down_pro
### Training Configuration

| Parameter | Value |
|-----------|-------|
| Base model | google/medgemma-4b-it |
| Quantization | 4-bit NF4 (QLoRA) |
| LoRA rank | 16 |
| LoRA alpha | 32 |
| LoRA dropout | 0.05 |
| Learning rate | 1.5e-4 |
| LR scheduler | Cosine |
| Warmup steps | 50 |
| Epochs | 3 (early stopping at best eval loss) |
| Batch size | 2 (effective 16 with grad accumulation) |
| Label smoothing | 0.05 |
| Optimizer | paged_adamw_8bit |
| Trainable params | 35,738,752 / 2,525,961,712 (1.41%) |

---

## Dataset

### NaijaCXR
- **Source**: Rasheed Shekoni Federal University Teaching Hospital (RSFUTH), Nigeria
- **Size**: 1,400 unique chest X-ray / radiology report pairs
- **Pathologies**: Normal, Cardiomegaly, Hypertensive Heart Disease,
  Congestive Cardiac Failure, Peripartum Cardiac Failure,
  Pulmonary Tuberculosis, Pneumonia, Incipient Cardiac Failure
- **Format**: PA chest X-rays with structured FINDINGS + IMPRESSION reports
  written in Nigerian clinical radiology style

### MIMIC-CXR (auxiliary)
- **Source**: Beth Israel Deaconess Medical Center, USA (PhysioNet)
- **Subset used**: 1,006 examples (balanced sample)
- **Purpose**: Supplementary training signal; provides report structure diversity

### Data Balancing Strategy
- NaijaCXR downsampled to match MIMIC count (1,006 each)
- Normal/abnormal class balance preserved within NaijaCXR (50/50 split)
- Combined training set: 2,012 examples (90/10 train/eval split)

---

## Installation

```bash
git clone https://github.com/El-amin/NaijaCXR-VLM.git
cd NaijaCXR-VLM
pip install -r requirements.txt
```

### Requirements
torch>=2.0.0
transformers>=4.40.0
peft>=0.10.0
bitsandbytes>=0.43.0
accelerate>=0.27.0
datasets>=2.18.0
Pillow>=10.0.0
pandas>=2.0.0
scikit-learn>=1.3.0
evaluate>=0.4.0
rouge_score>=0.1.2
bert_score>=0.3.13
opencv-python>=4.8.0
matplotlib>=3.7.0
tqdm>=4.65.0


---

## Usage

### Report Generation

```python
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig
from peft import PeftModel
from PIL import Image
import torch

# Load model
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

base_model = AutoModelForImageTextToText.from_pretrained(
    "google/medgemma-4b-it",
    quantization_config=bnb_config,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

model = PeftModel.from_pretrained(base_model, "path/to/saved/adapter")
processor = AutoProcessor.from_pretrained("google/medgemma-4b-it")
model.eval()

# Generate report
def generate_report(image_path, model, processor, max_new_tokens=300,
                     prompt="Describe the chest X-ray findings and impression."):
    image = Image.open(image_path).convert("RGB")
    messages = [{"role": "user", "content": [
        {"type": "image"},
        {"type": "text", "text": prompt}
    ]}]
    chat_prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=chat_prompt, images=image, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.2,
        )

    generated = output_ids[0][inputs["input_ids"].shape[1]:]
    return processor.decode(generated, skip_special_tokens=True)

# Example
report = generate_report("chest_xray.jpg", model, processor)
print(report)
```

### Expected Output Format

FINDINGS:
There is cardiomegaly with multichamber configuration. There is engorgement
of the central pulmonary vasculature with upper lobe blood diversion. There
is haziness of the lower lung zones bilaterally. There is blunting of the
costophrenic sulci. The bony thorax and overlying soft tissues are within
normal limits.

IMPRESSION:
Congestive Cardiac Failure


---

## Repository Structure

NaijaCXR-VLM/
│
├── notebooks/
│ ├── NaijaCXR_VLM_Baseline_NoDA.ipynb # Main training notebook (no domain adaptation)
│ ├── NaijaCXR_VLM_MedGemma_4b_FIXED.ipynb # DA pipeline notebook (for reference/comparison)
│ └── NaijaCXR_Domain_Adaptation.ipynb # Adversarial SigLIP adaptation (standalone)
│
├── results/
│ ├── NaijaCXR-VLM_baseline_preds.csv # Baseline model predictions (60 test samples)
│ ├── NaijaCXR-VLM_preds.csv # DA pipeline predictions (for comparison)
│ └── training_logs.json # Training loss curves
│
├── figures/
│ ├── tsne_domain_adaptation.png # t-SNE before/after domain adaptation
│ ├── gradcam_results.png # Grad-CAM visualizations
│ ├── occlusion_attribution.png # Occlusion-based causal attribution
│ └── attention_rollout.png # Attention rollout visualization
│
├── requirements.txt
├── LICENSE
└── README.md


---


## Limitations

1. **Dataset scale**: 140 unique NaijaCXR examples is a small fine-tuning set. Performance
   on rare pathologies (e.g., PTB, pneumonia) is limited by the number of training examples
   for those classes.

2. **Upsampling**: NaijaCXR examples are repeated ~7x per epoch to match MIMIC count.
   While the label-alignment fix prevents hard collapse, soft over-representation of
   majority-class phrasing remains a risk across longer training runs.

3. **Spatial grounding**: Occlusion testing showed that the model\'s diagnostic confidence
   is not strongly spatially concentrated over the clinically relevant anatomical regions,
   suggesting the model relies partly on global image statistics rather than fine-grained
   localized features.

4. **Domain adaptation**: Adversarial SigLIP domain adaptation (DANN-style) was
   investigated but found to degrade performance relative to the native-config baseline.
   The 896×896/256-token native MedGemma configuration is recommended over any forced
   384×384/729-token surgery for fine-tuning on this dataset.


---

## Citation

If you use this work, please cite:

```bibtex
@misc{naijacxr_vlm_2025,
  title     = {NaijaCXR-VLM: Fine-Tuning MedGemma for Nigerian Chest X-Ray
               Report Generation},
  author    = {Aminu Musa},
  year      = {2025},
  note      = {African University of Science and Technology, Abuja},
  url       = {https://github.com/El-amin/NaijaCXR-VLM}
}
```

---

## Acknowledgements

- [Google DeepMind](https://deepmind.google/) for MedGemma-4b-it
- [PhysioNet / MIMIC-CXR](https://physionet.org/content/mimic-cxr/) for the auxiliary training data
- [Rasheed Shekoni Federal University Teaching Hospital](https://rsfuth.edu.ng/),
  Dutse, Nigeria for the NaijaCXR dataset
- HuggingFace for the `transformers`, `peft`, and `datasets` libraries

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

> Note on data: The NaijaCXR dataset is noot publicly released in this repository
> due to patient privacy and institutional data governance requirements. Access requests
> for research purposes can be directed to the data custodians at RSFUTH.
'''

