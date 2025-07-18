# Mask 출력 형식 기능

## 개요

X-AnyLabeling에 새로운 "mask" 출력 형식이 추가되었습니다. 이 기능을 통해 사용자는 포인트 프롬프트를 사용하여 segmentation 모델이 생성한 마스크를 직접 편집할 수 있습니다.

## 주요 기능

### 1. Mask 출력 형식 지원
- **polygon**, **rectangle**, **rotation** 외에 **mask** 옵션 추가
- Segmentation 모델들이 마스크를 직접 출력할 수 있음
- 마스크는 binary 형태로 저장되고 시각화됨

### 2. Point Prompt → Mask 생성
- 사용자가 포인트를 찍으면 segmentation 모델이 자동으로 마스크 생성
- SAM, SAM2, GroundingSAM 등 다양한 모델 지원
- 실시간 마스크 생성 및 시각화

### 3. 브러시 편집 기능
- 생성된 마스크를 브러시로 직접 편집 가능
- 마우스 클릭으로 픽셀 추가/제거
- Ctrl + 클릭으로 지우개 모드

### 4. 마스크 저장 및 로드
- 마스크는 base64 인코딩으로 JSON 파일에 저장
- 마스크 형태, 데이터 타입 정보도 함께 저장
- 파일 로드 시 마스크 자동 복원

## 사용법

### 1. Mask 모드 선택
1. Auto Labeling 패널에서 모델 선택
2. Output 드롭다운에서 "mask" 선택
3. 포인트 추가 버튼으로 프롬프트 포인트 설정

### 2. 마스크 생성
1. 이미지에서 원하는 객체에 포인트 클릭
2. "Run" 버튼 클릭 또는 단축키 사용
3. 모델이 자동으로 마스크 생성

### 3. 마스크 편집
1. 편집 모드에서 마스크 선택
2. 마우스 클릭으로 브러시 편집
3. Ctrl + 클릭으로 지우개 모드

### 4. 마스크 저장
- 자동 저장 또는 Ctrl+S로 수동 저장
- 마스크는 JSON 파일에 base64로 인코딩되어 저장

## 지원 모델

다음 모델들이 mask 출력 형식을 지원합니다:

- **Segment Anything (SAM)**
- **Segment Anything 2 (SAM2)**
- **GroundingSAM**
- **GroundingSAM2**
- **SAM-HQ**
- **SAM-Med2D**
- **EdgeSAM**
- **EfficientViT-SAM**

## 기술적 세부사항

### 마스크 저장 형식
```json
{
  "label": "object",
  "shape_type": "mask",
  "mask": "base64_encoded_mask_data",
  "mask_shape": [height, width],
  "mask_dtype": "uint8"
}
```

### 브러시 편집 알고리즘
- 원형 브러시 사용
- 실시간 픽셀 수정
- numpy 배열 기반 효율적 처리

### 시각화
- 선택된 마스크: 흰색 반투명 오버레이
- 일반 마스크: 녹색 반투명 오버레이
- 실시간 업데이트

## 단축키

- **M**: 마스크 생성 모드
- **Ctrl + 클릭**: 마스크 지우개 모드
- **일반 클릭**: 마스크 브러시 모드

## 제한사항

1. 마스크는 현재 binary 형태만 지원
2. 브러시 크기는 고정 (10픽셀)
3. 복잡한 마스크의 경우 편집 성능이 저하될 수 있음

## 향후 개선 계획

1. **다중 클래스 마스크** 지원
2. **가변 브러시 크기** 설정
3. **RLE 압축** 지원으로 파일 크기 최적화
4. **마스크 스무딩** 기능
5. **마스크 병합/분할** 도구

## 문제 해결

### 마스크가 보이지 않는 경우
1. 마스크가 선택되었는지 확인
2. 캔버스 줌 레벨 확인
3. 마스크 데이터가 올바르게 로드되었는지 확인

### 브러시 편집이 작동하지 않는 경우
1. 편집 모드인지 확인
2. 마스크가 선택되었는지 확인
3. 마우스 위치가 이미지 범위 내인지 확인

### 저장 오류
1. 마스크 데이터 크기 확인
2. 디스크 공간 확인
3. 파일 권한 확인 