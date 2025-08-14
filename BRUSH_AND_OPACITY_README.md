# 🎨 Brush Mode & Opacity Control Features

X-AnyLabeling에 새로 구현된 고급 어노테이션 기능들입니다.

## 📋 목차

- [브러시 모드 (Brush Mode)](#-브러시-모드-brush-mode)
- [투명도 조절 시스템 (Opacity Control System)](#-투명도-조절-시스템-opacity-control-system)
- [성능 최적화](#-성능-최적화)
- [사용법 가이드](#-사용법-가이드)
- [기술적 구현 세부사항](#-기술적-구현-세부사항)

---

## 🖌️ 브러시 모드 (Brush Mode)

### 개요
기존의 폴리곤, 사각형, 회전 도형을 실시간으로 마스크로 변환하여 픽셀 단위의 정밀한 편집이 가능한 브러시 도구입니다.

### 주요 기능

#### 1. **스마트 도형 변환**
- `polygon`, `rectangle`, `rotation` 타입의 도형을 자동으로 마스크로 변환
- 원본 도형 타입 정보 보존 (`_original_shape_type` 속성)
- 브러시 모드 해제 시 원래 도형 타입으로 자동 복원

#### 2. **실시간 마스크 편집**
- 브러시 스트로크로 즉시 픽셀 추가/제거
- 실시간 시각적 피드백
- 마스크가 완전히 지워지면 자동 도형 삭제

#### 3. **듀얼 모드 지원**
- **그리기 모드**: 마스크에 픽셀 추가
- **지우개 모드**: 마스크에서 픽셀 제거 (Ctrl 키 또는 UI 버튼)

#### 4. **적응형 브러시 크기**
- 이미지 크기와 줌 레벨에 따른 자동 브러시 크기 조정
- 화면 좌표에서 일관된 브러시 크기 유지

### 키보드 단축키

| 단축키 | 기능 |
|--------|------|
| `M` | 브러시 모드 토글 (켜기/끄기) |
| `Ctrl + 브러시` | 지우개 모드 |
| `마우스 휠` | 브러시 크기 조정 (브러시 모드 활성화 시) |

---

## 🔍 투명도 조절 시스템 (Opacity Control System)

### 개요
마스크 오버레이와 채우기 색상에 대한 개별 투명도 제어를 제공하는 시스템입니다.

### 주요 기능

#### 1. **듀얼 투명도 제어**
- **마스크 투명도**: 브러시로 수정된 도형의 투명도
- **채우기 투명도**: 일반 도형의 채우기 색상 투명도

#### 2. **실시간 미리보기**
- 투명도 변경 시 즉시 시각적 피드백
- 0-255 범위의 세밀한 투명도 제어

#### 3. **배치 작업 지원**
- 선택된 도형들에 대한 일괄 투명도 설정
- 전체 도형에 대한 일괄 투명도 설정

### API 사용법

#### 개별 도형 투명도 설정
```python
# 마스크 투명도 설정 (0-255)
canvas.set_shape_opacity(shape, 128, is_mask=True)

# 채우기 투명도 설정
canvas.set_shape_opacity(shape, 200, is_mask=False)
```

#### 선택된 도형들 투명도 설정
```python
# 선택된 모든 도형의 마스크 투명도 설정
canvas.set_selected_shapes_opacity(150, is_mask=True)

# 선택된 모든 도형의 채우기 투명도 설정
canvas.set_selected_shapes_opacity(180, is_mask=False)
```

#### 전체 도형 투명도 설정
```python
# 모든 도형의 마스크 투명도 설정
canvas.set_all_shapes_opacity(120, is_mask=True)

# 모든 도형의 채우기 투명도 설정
canvas.set_all_shapes_opacity(160, is_mask=False)
```

---

## 🚀 성능 최적화

### 1. **지연된 렌더링**
- 브러시 업데이트를 8ms 지연으로 배치 처리
- 약 120fps의 부드러운 편집 경험

### 2. **배치 처리**
- 연속된 브러시 스트로크를 한 번에 처리
- 렌더링 호출 최소화

### 3. **캐시 관리**
- 마스크 QImage 캐시 자동 무효화
- 메모리 사용량 최적화

### 4. **스마트 브러시 크기 조정**
```python
def calculate_optimal_brush_size(self, image_width, image_height):
    """이미지 크기와 디스플레이 스케일에 따른 최적 브러시 크기 계산"""
    # 기본 디스플레이 대각선 기준으로 스케일링
    # 줌 레벨을 고려한 이미지 좌표 브러시 크기 반환
```

---

## 📖 사용법 가이드

### 브러시 모드 시작하기

1. **모드 전환**
   - `M` 키를 눌러 브러시 모드 활성화
   - 또는 UI에서 브러시 모드 버튼 클릭

2. **도형 선택**
   - 편집하고 싶은 도형을 클릭하여 선택
   - 자동으로 마스크로 변환됨

3. **브러시 편집**
   - 마우스 드래그로 브러시 스트로크
   - `Shift` 키를 누른 상태로 지우개 모드

4. **모드 종료**
   - `M` 키를 다시 눌러 브러시 모드 비활성화
   - 자동으로 원래 도형 타입으로 복원

### 투명도 조절하기

1. **개별 도형 투명도**
   - 도형 선택 후 투명도 슬라이더 조정
   - 마스크/채우기 투명도 개별 설정

2. **일괄 투명도 설정**
   - 여러 도형 선택 후 투명도 조정
   - 전체 도형에 대한 투명도 설정

---

## 🔧 기술적 구현 세부사항

### 핵심 클래스 및 메서드

#### Canvas 클래스 확장
```python
class Canvas(QtWidgets.QWidget):
    # 새로운 인스턴스 변수들
    is_brush_mode = False          # 브러시 모드 상태
    brush_radius = 10              # 브러시 반지름
    eraser_mode = False            # 지우개 모드 상태
    _brush_target_shape = None     # 브러시 대상 도형
    brush_modified = False         # 브러시 수정 여부
```

#### 주요 메서드들

##### `set_brush_mode(enabled: bool, radius: int = 10)`
- 브러시 모드 활성화/비활성화
- 도형을 마스크로 변환 및 복원
- 선택 상태 동기화

##### `edit_mask_with_brush(shape, pos, radius, add)`
- 브러시로 마스크 직접 수정
- 성능 최적화된 마스크 조작
- 자동 도형 삭제 처리

##### `set_eraser_mode(enabled: bool)`
- 지우개 모드 토글
- Ctrl 키와 연동

### 마스크 변환 로직

#### 도형 → 마스크 변환
```python
if shape.shape_type != "mask":
    # 원본 타입 보존
    shape._original_shape_type = shape.shape_type
    
    # 폴리곤을 마스크로 변환
    h, w = self.pixmap.height(), self.pixmap.width()
    shape.mask = polygon_to_mask(
        [(int(p.x()), int(p.y())) for p in shape.points],
        (h, w)
    )
    shape.shape_type = "mask"
    shape.points = []
```

#### 마스크 → 도형 복원
```python
if shape.shape_type == "mask":
    # 마스크를 폴리곤으로 변환
    points = mask_to_polygon(shape.mask, simplify=False)
    if points and len(points) >= 3:
        # 원본 타입으로 복원
        original_type = shape._original_shape_type
        # 색상, 투명도 정보 복사
        # 방향 정보 복사 (rotation 타입의 경우)
```

### 성능 최적화 기법

#### 1. **지연된 업데이트**
```python
self._brush_update_timer = QtCore.QTimer()
self._brush_update_timer.setSingleShot(True)
self._brush_update_timer.timeout.connect(self._delayed_brush_update)
self._brush_update_timer.start(8)  # 8ms 지연
```

#### 2. **배치 스트로크 처리**
```python
def _batch_process_brush_stroke(self, stroke_points):
    """여러 브러시 스트로크를 한 번에 처리"""
    for pos, add in stroke_points:
        shape.mask = apply_brush_to_mask(
            shape.mask, pos.x(), pos.y(), 
            radius=radius, add=add
        )
    # 캐시 무효화는 한 번만
    self._invalidate_cache(shape)
```

#### 3. **스마트 보간**
```python
# 거리가 멀 때만 보간 처리
if dist > image_brush_radius * 0.5:
    step_size = max(2, image_brush_radius // 3)
    steps = max(1, int(dist // step_size))
    for i in range(steps + 1):
        t = i / steps
        x = prev.x() * (1 - t) + curr.x() * t
        y = prev.y() * (1 - t) + curr.y() * t
        self.edit_mask_with_brush(shape, QPointF(x, y), radius, add)
```

---

## 📝 변경사항 요약

### 추가된 기능
- ✅ 브러시 모드로 도형 편집
- ✅ 실시간 마스크 변환 및 편집
- ✅ 듀얼 모드 (그리기/지우개)
- ✅ 적응형 브러시 크기
- ✅ 마스크/채우기 투명도 개별 제어
- ✅ 배치 투명도 설정
- ✅ 성능 최적화된 렌더링

### 수정된 파일
- `anylabeling/views/labeling/widgets/canvas.py` - 메인 Canvas 클래스

### 새로운 시그널
- `brush_mode_changed` - 브러시 모드 변경 알림
- `eraser_mode_changed` - 지우개 모드 변경 알림

### 키보드 단축키
- `M` - 브러시 모드 토글
- `Ctrl + 브러시` - 지우개 모드

---

## 🎯 사용 시나리오

### 1. **세그멘테이션 작업**
- 폴리곤으로 대략적인 영역 지정
- 브러시 모드로 세밀한 경계 조정
- 투명도 조절로 다른 레이어와 비교

### 2. **마스크 정제**
- 자동 생성된 마스크의 경계 다듬기
- 브러시로 세부 영역 추가/제거
- 투명도로 원본 이미지와 비교

### 3. **복합 어노테이션**
- 여러 도형의 투명도 조절
- 레이어 간 가시성 최적화
- 브러시로 도형 간 경계 조정

---

## 🔮 향후 개선 계획

### 단기 계획
- [ ] 브러시 크기 미리보기 개선
- [ ] 브러시 히스토리/실행 취소 기능
- [ ] 다양한 브러시 모양 지원

### 장기 계획
- [ ] AI 기반 브러시 자동화
- [ ] 브러시 프리셋 시스템
- [ ] 3D 투명도 제어

---

## 📞 지원 및 문의

이 기능과 관련된 질문이나 개선 제안이 있으시면:
- GitHub Issues에 등록
- 또는 개발팀에 직접 문의

---

*이 문서는 X-AnyLabeling v3.1.1+ 버전의 새로운 기능들을 설명합니다.* 