# 캡스톤프로젝트 - PA_SASRec
도서 추천 시스템

## Description
사용자의 구매 정보와 보조 정보를 활용한 SASRec 성능 향상 프로젝트  

## Environment
CPU : Intel(R) Core(TM) i7-10750H CPU  
GPU : NVIDIA GeForce GTX 1660 Ti  
RAM : 16GB

python version : 3.9.7  
torch version : 1.10.0+cu113  

SASRec Reference : https://github.com/pmixer/SASRec.pytorch

## Model
<img src="https://github.com/et007693/PA_SASRec/blob/main/img/model.png?raw=true" width="600" height="600"></img>

  Embedding Layer  
  모든 유저 구매기록 10개로 slicing, 구매기록 3개 이하인 유저 기록 삭제  
  
  Parallel multi-head attention  
  이질적인 보조 정보를 독립적으로 활용하기 위해 보조정보수와 동일하게 parallel attention block 구성  
  각 attention block에서 나온 결과를 point-wise sum

## 실행 코드
``` python
main.py --dataset=book_transactions --train_dir=default --maxlen=10 --dropout_rate=0.2 --device=cuda
```
