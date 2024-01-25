from transformers import pipeline

# 텍스트 생성 파이프라인을 초기화합니다. 여기서는 GPT-2 모델을 사용합니다.
generator = pipeline('text-generation', model='gpt2')

# 생성할 텍스트의 시작 부분을 제공합니다.
prompt = "오늘 날씨는 "

# 모델에 프롬프트를 제공하여 텍스트를 생성합니다.
generated_texts = generator(prompt, max_length=50, num_return_sequences=1)

# 생성된 텍스트를 출력합니다.
for generated_text in generated_texts:
    print(generated_text['generated_text'])
