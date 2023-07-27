# 사양
- Windows 11 Pro Ram 32GB
- NVIDIA Geforce RTX 3060 12GB Driver 531.79
- CUDA 11.8.89
- cudnn 8.9.2
- Python 3.11.4
- PyTorch 2.0.1+cu118

# PyTorch 설치
1. [여기](https://pytorch.kr/get-started/locally/)에서 PyTorch 버전 호환 및 CUDA 버전 확인 
2. [여기](https://developer.nvidia.com/rdp/cudnn-archive)에서 CUDA Tookit 설치
3. 호환되는 버전의 cudnn 압축파일 다운로드
4. 압축 해제 후 C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\{버전} 폴더 안으로 **bin, include, lib** 폴더 이동
5. 1에서 생성된 pip 명령어 실행
```
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

# PyTorch Quick Start
- [PyTorchTutorial/torch.ipynb](PyTorchTutorial/torch.ipynb)

# BERT
- [참고블로그](https://heekangpark.github.io/nlp/huggingface-bert#kramdown_%EA%B5%AC%ED%98%84)

## Transformer Model
: 문장 속 단어와 같은 순차 데이터 내의 관계를 추적해 맥락과 의미를 학습하는 신경망으로 데이터를 처리하는 대형 CODEC 블록에 해당

## Bert VS GPT
1. Bert: Bidirectional Encoder Representations from Transformers
2. GPT: Generative Pre-trained Transformers
3. NLP: Natural Language Processing, 자연어 처리

|       | Bert                                    | GPT                            |
|-------|-----------------------------------------|--------------------------------|
| 구조    | Transformer 기반 양방향 인코더                  | Transformer 기반 단방향 인코더         |
| 학습 방식 | MLM(Masked Language Modeling)           | CLM(Causal Language Modeling)  |
| 문맥 이해 | 앞뒤 문맥 모두 고려                             | 이전 문맥만 고려                      |
| 사용 사례 | 개체 인식, 개체 관계 추출, 감성 분석 등 다양한 NLP 작업에 유용 | 텍스트 생성, 요약, 기계 번역 등의 생성 작업에 유용 |

## Hugging Face
- [Hugging Face, bert-base-uncased](https://huggingface.co/bert-base-uncased)

### pip install
- pytorch
- transformers
- ipywidgets(Jupyter Notebook, Lab)

### Jupyter Notebook
- [Bert/hf_bert_base_uncased.ipynb](BERT/hf_bert_base_uncased.ipynb)