# MLOps-Example
[Blogging](https://byeongjo-kim.tistory.com/7)

## System Design
![png](https://raw.githubusercontent.com/byeongjokim/MLOps-Example/main/png/system_design.png)

## CI/CD
![png](https://raw.githubusercontent.com/byeongjokim/MLOps-Example/main/png/cicd0.png)

![png](https://raw.githubusercontent.com/byeongjokim/MLOps-Example/main/png/cicd1.png)

### CI
- [yaml file](https://raw.githubusercontent.com/byeongjokim/MLOps-Example/main/.github/workflows/ci.yml)
```yaml
name: CI

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  build:
    runs-on: ubuntu-latest
    steps:          
      - name: Check out
        uses: actions/checkout@v2
      
      - name: Docker Login
        uses: docker/login-action@v1.8.0
        with:
          username: ${{ secrets.REGISTRY_USERNAME }}
          password: ${{ secrets.REGISTRY_PASSWORD }}

      - name: Build Images
        run: |
          docker build kubeflow_pipeline/0_data -t byeongjokim/mnist-pre-data
          ..(생략)..
          docker push byeongjokim/mnist-deploy
      
      - name: Slack Notification
        if: always()
        uses: rtCamp/action-slack-notify@v2
        env:
          SLACK_ICON_EMOJI: ':bell:'
          SLACK_CHANNEL: mnist-project
          SLACK_MESSAGE: 'Build/Push Images :building_construction: - ${{job.status}}'
          SLACK_USERNAME: Github
          SLACK_WEBHOOK: ${{ secrets.SLACK_WEBHOOK_URL }}
```

### CD
- [yaml file](https://raw.githubusercontent.com/byeongjokim/MLOps-Example/main/.github/workflows/cd.yml)
```yaml
name: CD

on:
  workflow_run:
    workflows: ["ci"]
    branches: [main]
    types: 
      - completed

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:          
      - name: Check out
        uses: actions/checkout@v2
          
      - uses: actions/setup-python@v2
        with:
          python-version: '3.6.12'
          architecture: x64
      
      - uses: BSFishy/pip-action@v1
        with:
          packages: |
            kfp==1.3.0
      - name: run pipeline to kubeflow
        run: python kubeflow_pipeline/pipeline.py

      - name: Slack Notification
        if: always()
        uses: rtCamp/action-slack-notify@v2
        env:
          SLACK_ICON_EMOJI: ':bell:'
          SLACK_CHANNEL: mnist-project
          SLACK_MESSAGE: 'Upload & Run pipeline :rocket: - ${{job.status}}'
          SLACK_USERNAME: Github
          SLACK_WEBHOOK: ${{ secrets.SLACK_WEBHOOK_URL }}
```

## CT
### alert using slack when new data is coming
![png](https://raw.githubusercontent.com/byeongjokim/MLOps-Example/main/png/ct1.png)
- using Slack Trigger (slack_sdk)
- private repository in Github..
- when to train?
    - every Tuesday
    - new data is coming

## kubeflow pipelines
- 0_data
    - **data 수집**
    - **전처리 후 npy 저장(train/test/validation)**
    - npy_interval 사용하여 데이터 나누어 저장
    - embedding 학습에 사용하는 데이터와 faiss 학습 시 사용되는 데이터 구분
- 1_validate_data
    - **전처리된 npy 검증**
    - shape, type 등 전처리 결과 확인
- 2_train_model
    - **embedding 모델 학습**
    - ArcFace 사용
    - multip gpu 사용
    - 학습 완료된 모델 torch.jit.script 저장
- 3_embedding
    - faiss 사용될 데이터 **embedding 전처리 후 npy 저장**
    - torch.jit.script 저장된 모델 load 해서 사용
- 4_train_faiss
    - embedding npy로 **faiss 학습**
    - faiss index 저장
- 5_analysis_model
    - **전체 모델 성능 평가**
    - mlpipeline-metrics, mlpipeline-ui-metadata로 시각화
    - class 별 accuracy 측정 (confusion matrix)
    - dsl.Condition 사용하여 배포할지 결정
- 6_deploy
    - 모델 버전 관리
    - config 관리
    - service, deployment 배포
    - torchserve 사용

### Serving Model using TorchServe
![png](https://raw.githubusercontent.com/byeongjokim/MLOps-Example/main/png/serving.png)

### uploaded pipelines
![png](https://raw.githubusercontent.com/byeongjokim/MLOps-Example/main/png/pipelines0.png)

### confusion matrix(mlpipeline-ui-metadata)
![png](https://raw.githubusercontent.com/byeongjokim/MLOps-Example/main/png/pipelines1.png)

### alrert results using slack
![png](https://raw.githubusercontent.com/byeongjokim/MLOps-Example/main/png/pipelines2.png)