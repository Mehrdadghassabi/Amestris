<div align="center">
  <img src="https://github.com/user-attachments/assets/083c889b-433d-4564-9710-841336308f7f"/>
</div>
<p align="center">
📃 <a href="https://arxiv.org/abs/2604.25702" target="_blank">Paper</a> ｜🤗 <a href="https://huggingface.co/gaokerena/amestris_1b" target="_blank">huggingface repository</a>
</p>


## 📒 Table of Contents
- [📒 Table of Contents](#-table-of-contents)
- [📍 Overview](#-overview)
- [🏃 Training process](#-training-process)
- [📊 Results](#-Results)
- [⛔️ License](#-license)
- [🤝 Collaborators](#-collaborators)

---

## 📍 Overview
Amestris is a specialized translation model built on Gemma3-1B, fine-tuned using a hybrid pipeline of Backtranslation and Direct Preference Optimization (DPO).
while standard translation models often struggle with synthetic data noise, our framework utilizes backtranslation to generate diverse candidate pairs and DPO to align the model with preferred, human-like linguistic outputs. The result is a lightweight, highly efficient translation engine that punches well above its weight class.

## 🏃 Training process
We propose a DPO-based framework to improve the translation ability of a baseline language model, where high-quality preference pairs are constructed through expert translation, back-translation, and quality filtering, then used to fine-tune the student model with LoRA. Starting from a source-language monolingual corpus, expert translations are generated and then back-translated by the student model; low-quality back-translations are identified using BLEU and COMET-based filtering, with only sufficiently poor samples retained to form preference triplets (x,yw,yl). These curated preferences are then used in DPO to increase the likelihood of expert-level translations over flawed ones, without requiring a separate reward model. The process is applied iteratively, refreshing the dataset after each update to progressively reduce translationese artifacts and improve fluency, adequacy, and semantic consistency in the model’s outputs.

<img width="517" height="341" alt="fig1" src="https://github.com/user-attachments/assets/12e8ffd3-ba06-4af4-b695-731e12dd467a" />

## 📊 Results
We evaluated our DPO-based framework on the English-to-German translation task using the WMT14 dataset and the compact Gemma3-1B model as the baseline, producing the trained student model Amestris-1B.
notice that in all of the mentioned metrics higer value means better performance except for TER which is reverse.

|                       | gemma3-1b (baseline) | amestris-1b |
|-----------------------|--------------------|---------------------------|
| **BLEU**  | **0.1572**           | 15.00                  | 
| **COMET22**      | 0.7698          | **0.7810**        | 
| **COMET_KIWI22**      | 0.7031              | **0.7476**     | 
| **METEOR**     | 0.3861          | **0.3969**        | 
| **TER**  | 0.7765          | **0.7621**        | 
| **chrF++**      | 0.4193          | **0.4382**          | 


## ⛔️ License
[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) (non-commercial use only)

## 🤝 Collaborators
1. Mehrdad Ghassabi
2. Sepehr Rajabi
3. Dr. Hamid Reza Baradaran Kashani
4. Sadra Hakim
5. Mahshid Keivandarian
6. Amirhossein Jahani Bahnamiri
