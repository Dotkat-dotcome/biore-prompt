# biore-prompt
prompt-based learning experiments on Chemprot dataset.


## Requirements
```bash
git clone ...
cd biore-prompt
```

```bash
conda env create -f env.yml
```

## Experiments


```bash
mkdir -p $OUTPUT_DIR $OUTPUT_DIR/train $OUTPUT_DIR/val $OUTPUT_DIR/test

```

```bash
cd code_script
```

```bash
python3 run_prompt.py \
--data_dir ../datasets/chemprot \
--output_dir $OUTPUT_DIR \
--model_type roberta \
--model_name_or_path roberta-base \
--per_gpu_train_batch_size 4 \
--gradient_accumulation_steps 1 \
--max_seq_length 512 \
--warmup_steps 500 \
--learning_rate 3e-5 \
--learning_rate_for_new_token 1e-5 \
--num_train_epochs 5 \
--weight_decay 1e-2 \
--adam_epsilon 1e-6 \
--seed 42 \
--temps ./temp/combined.txt

```

The temps can be
- `./temp/freq.txt`, `./temp/freq-spec.txt`, `./temp/sim.txt`, `./temp/combined.txt`, `./temp/random.txt`



## Note
Unlike previous works evalute across 5 relations, excluding the class: no relation. We keep the training and evaluation on all classes.

## Citation
```
@InProceedings{yeh-lavergne-zweigenbaum:2022:LREC,
  author    = {Yeh, Hui-Syuan  and  Lavergne, Thomas  and  Zweigenbaum, Pierre},
  title     = {Decorate the Examples: A Simple Method of Prompt Design for Biomedical Relation Extraction},
  booktitle      = {Proceedings of the Language Resources and Evaluation Conference},
  month          = {June},
  year           = {2022},
  address        = {Marseille, France},
  publisher      = {European Language Resources Association},
  pages     = {3780--3787},
  abstract  = {Relation extraction is a core problem for natural language processing in the biomedical domain. Recent research on relation extraction showed that prompt-based learning improves the performance on both fine-tuning on full training set and few-shot training. However, less effort has been made on domain-specific tasks where good prompt design can be even harder. In this paper, we investigate prompting for biomedical relation extraction, with experiments on the ChemProt dataset. We present a simple yet effective method to systematically generate comprehensive prompts that reformulate the relation extraction task as a cloze-test task under a simple prompt formulation. In particular, we experiment with different ranking scores for prompt selection. With BioMed-RoBERTa-base, our results show that prompting-based fine-tuning obtains gains by 14.21 F1 over its regular fine-tuning baseline, and 1.14 F1 over SciFive-Large, the current state-of-the-art on ChemProt. Besides, we find prompt-based learning requires fewer training examples to make reasonable predictions. The results demonstrate the potential of our methods in such a domain-specific relation extraction task.},
  url       = {https://aclanthology.org/2022.lrec-1.403}
}

```