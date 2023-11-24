# EasyProject

[Frustratingly Easy Label Projection for Cross-lingual Transfer](https://arxiv.org/abs/2211.15613) (Findings of ACL2023)

![EasyProject GIF](https://raw.githubusercontent.com/edchengg/easyproject/main/asset/easyproject.gif)

[Gradio Demo](https://ychennlp-easyproject.hf.space/)

## Checkpoints
Update (May 30, 2023): Update checkpoints due to an issue in [Huggingface NLLB tokenization](https://github.com/huggingface/transformers/pull/22313).
- [NLLB-200-1.3B](https://huggingface.co/ychenNLP/nllb-200-distilled-1.3B-easyproject)
- [NLLB-200-3.3B](https://huggingface.co/ychenNLP/nllb-200-3.3b-easyproject)

## Data
- [Translated Data](https://drive.google.com/drive/folders/15LTRv2TMbrI67slWLWyClhVMaYOFD78b?usp=share_link)
- [MT Training Data](https://drive.google.com/drive/folders/15LTRv2TMbrI67slWLWyClhVMaYOFD78b?usp=share_link)

## Code

We use the code base and script adapted from MasakhaNER:
[Script](https://github.com/masakhane-io/masakhane-ner/blob/main/MasakhaNER2.0/scripts/mdeberta.sh)


### NER & EasyProject data
Google drive: [link](https://drive.google.com/drive/folders/1zJpS_VqNM21SfsW7m4hM7qvbsm3Os7rq?usp=drive_link)
```
NER (for evaluation): data_{masakahner,wikiann}
EasyProject (for training): output_nllb_3Bft_{wikiann,conll}
```

### EasyProject post-processing script
We use the following script to perform post-processing for translation data. This step assign labels to entities inside the brackets (e.g., [ ]). The post processed data are stored in output_nllb_3Bft_{wikiann,conll}. The original data are stored in {conll,wikiann}_nllb_3B_ft.pkl files in the google drive.

Wikiann:
```
python decode_marker_wikiann.py
```

Masakhaner:
```
python decode_marker_conll.py
```

### NER training
We use the following script with slurm to run experiments - please adjust accordingly.

Wikiann:
```
bash xlmr_en_marker_transfer_3bft.sh
```

Masakhaner:
```
bash mdeberta_en_marker_transfer_3bft.sh
```

## Citation
Please cite if you use the above resources for your research
```
@inproceedings{chen2023easyproject,
  title={Frustratingly Easy Label Projection for Cross-lingual Transfer},
  author={Chen, Yang and Jiang, Chao and Ritter, Alan and Xu, Wei},
  booktitle={Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Findings)},
  year={2023}
}
```

## Funding Acknowledgment
This material is based in part on research sponsored by IARPA via the BETTER program (2019-19051600004).
