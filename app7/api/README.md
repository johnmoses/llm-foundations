# API

Running fine-tuning

```bash
python app.py --mode basic_ft --dataset ../data/basic_finetune.csv
python app.py --mode peft_ft --dataset ../data/peft_finetune.csv
python app.py --mode rlhf_train --dataset ../data/rlhf_finetune.csv
python app.py --mode ../data/rlhf_preference.csv
```
