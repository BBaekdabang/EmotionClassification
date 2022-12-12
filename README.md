# Emotion Recognition in Conversation
# 월간 데이콘 발화자의 감정인식 AI 경진대회
<img width="1000" img height="200" alt="Dacon" src="https://user-images.githubusercontent.com/113493692/206966548-bb71d381-a828-4a6f-8806-d63be4e37419.png">

#### 데이터셋 다운로드 : [월간 데이콘 발화자의 감정인식 AI 경진대회](https://dacon.io/competitions/official/236027/data#)

대회 참여 기간 : 2022.11.21. ~ 2022.12.12.

### 최종 순위 : Public 3등

<img width="1000" alt="image" src="https://user-images.githubusercontent.com/113493692/206967116-f16f8827-ad37-4a32-ad29-24cdbee13b7d.png">


---

# Code

<table>
    <thead>
        <tr>
            <th>목록</th>
            <th>파일명</th>
            <th>설명</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td rowspan=1>SAM Optimizer</td>
            <td>
                <a href="https://github.com/HappyBusDay/Korean_ABSA/blob/main/code/Data_Augmentation.ipynb">SAM_Optimizer.ipynb</a>
            </td>
            <td> SAM Optimizer </td>
        </tr>
            <td>Training</td>
            <td>
                <a href="https://github.com/HappyBusDay/Korean_ABSA/blob/main/code/train.ipynb">Train.ipynb</a>     
            <td> Model과 Data를 불러와 학습시키는 Code </td>
        </tr>
        <tr>
            <td>Inference</td>
            <td>
                <a href="https://github.com/HappyBusDay/Korean_ABSA/blob/main/code/test.ipynb">Inference.ipynb</a>     
            <td> 학습 시킨 Model을 불러와서 Test Data의 label을 Inference하는 Code </td>
        </tr>        
        <tr>
            <td rowspan=2>Model Ensemble</td>       
            <td>
                <a href="https://github.com/HappyBusDay/Korean_ABSA/blob/main/code/Manual_Ensemble.ipynb">HardVoting.ipynb</a>
            <td> Hard Voting</td>
        </tr>
        <tr>
            <td>
                <a href="https://github.com/HappyBusDay/Korean_ABSA/blob/main/code/Auto_Ensemble.ipynb">SoftVoting.ipynb</a>
            <td>Soft Voting</td>
        </tr>
        
   </tbody>
</table>

---

# 가. 개발 환경

     Google Colab Pro
     
     huggingface-hub==0.10.1
     datasets==2.6.1
     tokenizers==0.13.2
     torch==1.12.1+cu113
     torchvision==0.13.1+cu113
     transformers==4.24.0
     tqdm==4.64.1
     scikit-learn==1.0.2
     sentencepiece==0.1.97


---

# 나. 데이터 예시 
 출처 : Dacon, 월간 데이콘 발화자의 감정인식 AI 경진대회 

    {"ID": "TRAIN-0002", "Utterance": "That I did. That I did.", "Speaker": "Chandler", "Dialogue_ID": "0", "Target": "neutral"}
    {"ID": "TRAIN-0007", "Utterance": "But there’ll be perhaps 30 people under you so you can dump a certain amount on them.", "Speaker": "The Interviewer", "Dialogue_ID": "0", "Target": "neutral"}
    {"ID": "TRAIN-0017", "Utterance": "No, I-I-I-I don't, I actually don't know", "Speaker": "Rachel", "Dialogue_ID": "1", "Target": "fear"}

---


# 라. 주요 소스 코드

- ## Model Load From Hugging Face
   
   
    <table>
    <thead>
        <tr>
            <th>Model</th>
            <th>Pre-Trained Dataset</th>
            <th>링크(HuggingFace)</td>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td> DistilBERT</td>
            <td>                 
                <a href="https://huggingface.co/datasets/viewer/?dataset=emotion">Twitter-Sentiment-Analysis</a></td>
            <td>
                <a href="https://huggingface.co/bhadresh-savani/distilbert-base-uncased-emotion">bhadresh-savani/distilbert-base-uncased-emotion</a>
        </tr>
        <tr>
            <td> BERT</td>            
            <td>                 
                <a href="https://huggingface.co/datasets/viewer/?dataset=emotion">Twitter-Sentiment-Analysis</a></td>
            <td>
                <a href="https://huggingface.co/bhadresh-savani/bert-base-uncased-emotion">bhadresh-savani/bert-base-uncased-emotion</a>
        </tr>
        <tr>
            <td> EmoBERTa-base</td>
            <td>                
                <a href="https://github.com/tae898/multimodal-datasets/tree/a36101638a8121b422ce4a2a17746b25f23335b8">multimodal-datasets</a></td>
            <td>
                <a href="https://huggingface.co/tae898/emoberta-base">tae898/emoberta-base</a>
        </tr>
        <tr>
            <td> EmoBERTa-large</td>
            <td>                
                <a href="https://github.com/tae898/multimodal-datasets/tree/a36101638a8121b422ce4a2a17746b25f23335b8">multimodal-datasets</a></td>
            <td>
                <a href="https://huggingface.co/tae898/emoberta-large">tae898/emoberta-large</a>
        </tr>
    </tbody>
    </table>
      
    
   ```c
    # HuggingFace에서 불러오기
    from transformers import AutoTokenizer, AutoModel
    base_model = "HuggingFace주소"

    Model = AutoModel.from_pretrained(base_model)
    tokenizer = AutoTokenizer.from_pretrained(base_model)
   ```


- ## Data Load: jsonlload
    데이터가 line별로 저장된 json 파일( jsonl )이기 때문에 데이터 로드를 할 때 해당 코드로 구현함

    ```c
    import json
    import pandas as pd
    def jsonlload(fname, encoding="utf-8"):
        json_list = []
        with open(fname, encoding=encoding) as f:
            for line in f.readlines():
                json_list.append(json.loads(line))
        return json_list
    df = pd.DataFrame(jsonlload('/content/sample.jsonl'))
    ```


- ## Inference: predict_from_korean_form
   predict_from_korean_form 6가지의 방법 중 일부
   
   > [HappyBusDay/Korean_ABSA/code/test.ipynb 참조](https://github.com/HappyBusDay/Korean_ABSA/blob/main/code/test.ipynb)

    - #### 방법 1: Force ( Force evaluation of a Argument )
    
         빈칸 " [ ] " 에 대해서 가장 높은 확률의 카테고리를 강제로 뽑아내는 방법 

         [Force에 대한 설명](https://rdrr.io/r/base/force.html)

         ```c

        def predict_from_korean_form_kelec_forcing(tokenizer_kelec, ce_model, pc_model, data):

            ...

            자세한 코드는 code/test.ipynb 참조

            return data
         ```

 
    - #### 방법 2: DeBERTa(RoBERTa)와 ELECTRA 
        
         모델 별 tokenizer를 이용한 inference 진행하는 방법

         ```c

        def predict_from_korean_form_deberta(tokenizer_deberta, tokenizer_kelec, ce_model, pc_model, data):

            ...

           자세한 코드는 code/test.ipynb 참조

            return data
         ```

     
    - #### 방법 3: Threshold
     
         확률 기반으로 annotation을 확실한 것만 가져오는 방법
         
         확실한 것만 잡고 확률값이 낮은 것은 그냥 " [ ] "으로 결과값 도출 

         ```c

        def predict_from_korean_form_kelec_threshold(tokenizer_kelec, ce_model, pc_model, data):

            ...

           자세한 코드는 code/test.ipynb 참조

            return data
         ```



- ## Pipeline 및 Ensemble

   > [HappyBusDay/Korean_ABSA/code/test.ipynb 참조](https://github.com/HappyBusDay/Korean_ABSA/blob/main/code/test.ipynb)
   
    - #### Pipeline: 여러 모델을 불러 결과값 도출
        
        해당 코드는 **12종류[category{6종류} + polarity{6종류}]의 모델**을 불러옴

        " [ ] " 을 최소화 하기 위해 DeBERTa와 ELECTRA 등 여러 모델의 Weight파일을 불러 진행

        ```c
        def Win():

            print("Deberta!!")

            tokenizer_kelec = AutoTokenizer.from_pretrained(base_model_elec)
            tokenizer_deberta = AutoTokenizer.from_pretrained(base_model_deberta)
            tokenizer_roberta = AutoTokenizer.from_pretrained(base_model_roberta)

            num_added_toks_kelec = tokenizer_kelec.add_special_tokens(special_tokens_dict)
            num_added_toks_deberta = tokenizer_deberta.add_special_tokens(special_tokens_dict)
            num_added_toks_roberta = tokenizer_roberta.add_special_tokens(special_tokens_dict)

            ...    

            자세한 코드는 code/test.ipynb 참조

            return pd.DataFrame(jsonlload('/content/drive/MyDrive/Inference_samples.jsonl'))
        ```
    
    
    - #### Ensemble: 위의 Inference의 결과로 만들어진 jsonl파일을 불러와 Hard Voting을 진행
        > [Ensemble.ipynb 참조](https://github.com/HappyBusDay/Korean_ABSA/blob/main/code/Ensemble.ipynb)

        > [Auto_Ensemble.ipynb 참조](https://github.com/HappyBusDay/Korean_ABSA/blob/main/code/Auto_Ensemble.ipynb)

       
        <img width="450" alt="KakaoTalk_20221113_222631386" src="https://user-images.githubusercontent.com/73925429/201582648-93ae75da-affe-4198-83a5-fb5280c54bdd.png">

        ( Hard Voting )

     


---

# 마. Reference

[1] [EDA: Easy Data Augmentation](https://arxiv.org/pdf/1901.11196.pdf): Wei, Jason, and Kai Zou. "Eda: Easy data augmentation techniques for boosting performance on text classification tasks." arXiv preprint arXiv:1901.11196 (2019).

[2] [Back-Trainslation](https://proceedings.neurips.cc/paper/2020/file/44feb0096faa8326192570788b38c1d1-Paper.pdf): Xie, Qizhe, et al. "Unsupervised data augmentation for consistency training." Advances in Neural Information Processing Systems 33 (2020): 6256-6268.

[3] [ELECTRA](https://arxiv.org/pdf/2003.10555.pdf): Clark, Kevin, et al. "Electra: Pre-training text encoders as discriminators rather than generators." arXiv preprint arXiv:2003.10555 (2020).

[4] [RoBERTa](https://arxiv.org/pdf/1907.11692.pdf): Liu, Yinhan, et al. "Roberta: A robustly optimized bert pretraining approach." arXiv preprint arXiv:1907.11692 (2019).

[5] [DeBERTa](https://arxiv.org/pdf/2006.03654.pdf): He, Pengcheng, et al. "Deberta: Decoding-enhanced bert with disentangled attention." arXiv preprint arXiv:2006.03654 (2020).

[6] [teddysum/korean_ABSA_baseline](https://github.com/teddysum/korean_ABSA_baseline): GitHub

[7] [catSirup/KorEDA](https://github.com/catSirup/KorEDA): GitHub

---

# 바. Members
Yongjae Kim | dydwo322@naver.com<br>
Hyein Oh | gpdls741@naver.com<br>
Seungyong Guk | kuksy77@naver.com<br>
Jaehyeog Lee | tysl4545@naver.com<br>
Hyoje Jung | flash1253@naver.com<br>
Hyojin Kang | khj94111@gmail.com
