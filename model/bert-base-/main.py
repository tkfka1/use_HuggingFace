from transformers import AutoModelForMaskedLM, AutoTokenizer, pipeline

MODEL_NAME = 'bert-base-multilingual-cased'
model = AutoModelForMaskedLM.from_pretrained(MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

text = "이순신은 [MASK] 중기의 무신이다."

tokenizer.tokenize(text)


kor_mask_fill = pipeline(task='fill-mask', model=model, tokenizer=tokenizer)
