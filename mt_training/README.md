# MT Training

- Please use the updated transformers library as it fixed an issue with NLLB-200 tokenization
- The training script is used to fine-tuned NLLB translation model to learn adding markers during translation
- The training data is automatically constructed to put markers around entities on parallel sentences
- Please note EasyProject was only fine-tuned on parallel data from English to {Germany, Spanish, Dutch, Chinese, Arabic}
