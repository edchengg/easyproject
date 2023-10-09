# NER

We use the code base and script adapted from MasakhaNER:
[Script](https://github.com/masakhane-io/masakhane-ner/blob/main/MasakhaNER2.0/scripts/mdeberta.sh)


## NER & EasyProject data
Google drive: [link](https://drive.google.com/drive/folders/1zJpS_VqNM21SfsW7m4hM7qvbsm3Os7rq?usp=drive_link)
```
NER (for evaluation): data_{masakahner,wikiann}
EasyProject (for training): output_nllb_3Bft_{wikiann,conll}
```

## EasyProject post-processing script
We use the following script to perform post-processing for translation data. This step assign labels to entities inside the brackets (e.g., [ ]). The post processed data are stored in output_nllb_3Bft_{wikiann,conll}. The original data are stored in {conll,wikiann}_nllb_3B_ft.pkl files in the google drive.

Wikiann:
```
python decode_marker_wikiann.py
```

Masakhaner:
```
python decode_marker_conll.py
```

## NER training
We use the following script with slurm to run experiments - please adjust accordingly.

Wikiann:
```
bash xlmr_en_marker_transfer_3bft.sh
```

Masakhaner:
```
bash mdeberta_en_marker_transfer_3bft.sh
```

## NER results
I put evaluation results in the Google drive above.
```
save_ckpts_{wikiann,conll}
```