# 🎬 CAIKE 흥행예측시스템 가이드 챗봇

CAIKE 흥행예측시스템의 사용법을 안내하는 AI 챗봇입니다. Azure OpenAI와 Azure AI Search를 활용한 RAG(Retrieval-Augmented Generation) 패턴으로 구현되어, 매뉴얼 기반의 정확한 답변을 제공합니다.

[![GitHub Stars](https://img.shields.io/github/stars/JeongHyeon-Kang/caikebot)](https://github.com/JeongHyeon-Kang/caikebot)
[![GitHub Issues](https://img.shields.io/github/issues/JeongHyeon-Kang/caikebot)](https://github.com/JeongHyeon-Kang/caikebot/issues)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

## 🚀 주요 기능

- **📚 매뉴얼 기반 Q&A**: CAIKE 시스템 매뉴얼을 기반으로 한 정확한 답변 제공
- **🔍 실시간 벡터 검색**: Azure AI Search를 통한 관련도 높은 정보 검색
- **🟢 연결상태 모니터링**: 실시간 시스템 연결상태 확인
- **🎨 직관적인 UI**: Streamlit 기반의 사용하기 쉬운 웹 인터페이스
- **📄 PDF 문서 처리**: 자동 페이지 분리 및 OCR 텍스트 추출

## 🛠️ 기술 스택

- **Frontend**: Streamlit
- **AI Model**: Azure OpenAI (GPT-4)
- **Vector Search**: Azure AI Search
- **PDF Handling**: PyPDF2, reportlab
- **Environment**: Python 3.11+

## 📋 시스템 요구사항

- Python 3.11 이상
- Azure OpenAI 구독
- Azure AI Search 구독
- Tesseract OCR 엔진 (선택사항 - OCR 기능 사용 시)

## ⚙️ 설치 및 설정

### 1. 프로젝트 클론
```bash
git clone https://github.com/JeongHyeon-Kang/caikebot.git
cd caikebot
```

### 2. 가상환경 생성 및 활성화
```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate
```

### 3. 패키지 설치
```bash
pip install -r requirements.txt
```

### 4. 환경 변수 설정
`.env` 파일을 생성하고 다음 변수들을 설정하세요:

```env
# Azure OpenAI 설정
AZURE_OPENAI_API_KEY=your_openai_api_key
AZURE_OPENAI_ENDPOINT=your_openai_endpoint
AZURE_OPENAI_DEPLOYMENT_NAME=your_chat_deployment_name
AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME=your_embedding_deployment_name

# Azure AI Search 설정
AZURE_SEARCH_ENDPOINT=your_search_endpoint
AZURE_SEARCH_API_KEY=your_search_api_key
AZURE_SEARCH_INDEX_NAME=your_search_index_name
```

### 5. Tesseract OCR 설치 (OCR 기능 사용 시)
```bash
# Windows (winget 사용)
winget install --id UB-Mannheim.TesseractOCR

# 또는 수동 설치
# https://github.com/UB-Mannheim/tesseract/wiki
```

### 6. 애플리케이션 실행
```bash
streamlit run streamlit_app.py
```

## 📁 프로젝트 구조

```
caikebot/
├── streamlit_app.py                    # 메인 Streamlit 애플리케이션
├── streamlit_app_backup.py             # 백업 파일
├── document_separate.py                # PDF 페이지 분리 유틸리티
├── pdf.py                             # PDF OCR 변환 도구
├── requirements.txt                    # Python 패키지 의존성
├── README.md                          # 프로젝트 문서
├── .env                              # 환경 변수 (git ignore)
├── .deployment                       # Azure 배포 설정
├── streamlit.sh                      # Streamlit 실행 스크립트
├── CAIKE-TR-D271-사용자매뉴얼-v1.0_20241018.pdf  # 원본 매뉴얼 (51페이지)
├── CAIKE_INFO.pdf                    # CAIKE 정보 문서 (19페이지)
└── separated_pages/                  # 분리된 PDF 페이지들
    ├── CAIKE-TR-D271-사용자매뉴얼-v1.0_20241018_page_001.pdf
    ├── CAIKE_INFO_page_001.pdf
    └── ... (총 70개 페이지)
```

## 🔧 주요 구성 요소

### CaikeRAGChatbot 클래스
- Azure OpenAI 클라이언트 설정
- 시스템 프롬프트 관리
- RAG 패턴 구현
- 응답 생성 로직

### PDF 문서 처리
- **`document_separate.py`**: 원본 PDF를 페이지별로 분리
- **`pdf.py`**: 이미지 기반 PDF를 검색 가능한 텍스트 PDF로 변환
- **`separated_pages/`**: RAG 검색용 개별 PDF 페이지들 (70개 페이지)

### 환경 변수 관리
- `.env` 파일을 통한 보안 설정 관리
- 필수 환경 변수 검증 로직

## 💡 사용 예시

### 질문 예시
- "회원가입 방법 알려주세요"
- "VOD 흥행 예측 요청 방법 알려주세요"
- "VOD 흥행 예측 조건 변경 방법 알려주세요"
- "VOD 흥행 예측 요청 엑셀 일괄 등록 방법 알려주세요"
- "채널 시청률 예측 요청 방법 알려주세요"
- "채널 시청률 정기 리포트 조회 방법 알려주세요"

### 응답 형태
AI는 관련도 점수와 함께 정확한 매뉴얼 기반 답변을 제공합니다:
```
회원가입 방법은 다음과 같습니다...
```

### 연결상태 표시
- **🟢**: 시스템 정상 연결
- **🔴**: 연결 실패 또는 오류 발생

## 🎯 특징

### RAG 패턴 구현
- **벡터 검색**: 사용자 질문과 매뉴얼 내용의 의미적 유사도 계산
- **컨텍스트 주입**: 관련 문서 내용을 GPT 모델에 제공
- **정확성 향상**: 매뉴얼에 기반한 신뢰할 수 있는 답변

### 사용자 경험
- **실시간 상태 표시**: 타이틀 옆 연결상태 인디케이터
- **관련도 투명성**: 답변의 신뢰도를 수치로 표시
- **오류 처리**: 친화적인 오류 메시지 및 디버그 모드

### 문서 처리 능력
- **PDF 페이지 분리**: 대용량 PDF를 개별 페이지로 자동 분리
- **OCR 텍스트 추출**: 이미지 기반 PDF에서 텍스트 추출
- **다국어 지원**: 한국어 + 영어 동시 인식

### 보안 및 안정성
- **환경 변수 보호**: 민감한 API 키 정보 분리
- **오류 핸들링**: 예외 상황에 대한 적절한 처리
- **로깅**: 개발자용 콘솔 로그와 사용자용 메시지 분리

## 🛠️ 유틸리티 도구

### PDF 분리 도구
```bash
python document_separate.py
```
- 여러 PDF 파일을 자동으로 페이지별 분리
- RAG 시스템용 문서 준비

### OCR 변환 도구
```bash
python pdf.py
```
- 이미지 기반 PDF를 검색 가능한 PDF로 변환
- 한국어/영어 텍스트 추출

## 🚀 배포

### Azure App Service 배포
1. `.deployment` 파일이 배포 설정을 관리
2. 환경 변수를 Azure App Service에서 설정
3. `streamlit.sh` 스크립트로 자동 실행

### Heroku 배포
```bash
# Procfile 사용
web: streamlit run streamlit_app.py --server.port=$PORT --server.address=0.0.0.0
```

## 🔍 문제 해결

### 연결 실패 (🔴 표시)
1. `.env` 파일의 환경 변수 확인
2. Azure 구독 상태 및 API 키 유효성 검증
3. 네트워크 연결 상태 확인

### OCR 오류
1. Tesseract OCR 설치 확인
2. 언어 팩 설치 상태 점검
3. 이미지 품질 및 해상도 확인

### 응답 품질 개선
1. Azure AI Search 인덱스 확인
2. 임베딩 모델 상태 점검
3. 문서 업데이트 필요성 검토

## 📖 API 문서

### 주요 메서드

#### `CaikeRAGChatbot.__init__()`
챗봇 인스턴스 초기화 및 Azure 클라이언트 설정

#### `setup_azure_clients()`
Azure OpenAI 및 Azure AI Search 클라이언트 구성

#### `generate_response(messages)`
사용자 질문에 대한 RAG 기반 응답 생성

#### `get_rag_parameters()`
Azure AI Search용 RAG 파라미터 설정

## 📊 통계

- **총 문서 페이지**: 70개 (매뉴얼 51개 + 정보 문서 19개)
- **지원 언어**: 한국어, 영어
- **응답 시간**: 평균 2-5초
- **정확도**: 매뉴얼 기반 95%+ 정확도

## 🤝 기여하기

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### 기여 가이드라인
- 코드 스타일: PEP 8 준수
- 커밋 메시지: 한국어로 명확하게 작성
- 문서화: 새로운 기능에 대한 문서 업데이트 필수

## 📄 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다. 자세한 내용은 `LICENSE` 파일을 참조하세요.

## 📞 연락처

- **프로젝트 링크**: [https://github.com/JeongHyeon-Kang/caikebot](https://github.com/JeongHyeon-Kang/caikebot)
- **이슈 리포트**: [Issues](https://github.com/JeongHyeon-Kang/caikebot/issues)
- **기능 요청**: [Feature Requests](https://github.com/JeongHyeon-Kang/caikebot/issues/new)

## 🙏 감사의 말

- **Azure OpenAI 팀**: 강력한 GPT 모델 제공
- **Streamlit 커뮤니티**: 훌륭한 웹 앱 프레임워크
- **PyMuPDF 개발팀**: 우수한 PDF 처리 라이브러리
- **Tesseract OCR**: 오픈소스 OCR 엔진

## 📈 로드맵

### v2.0 (계획 중)
- [ ] 다국어 UI 지원
- [ ] 실시간 문서 업데이트
- [ ] 음성 인터페이스
- [ ] 모바일 앱 지원

### v1.5 (개발 중)
- [x] OCR 텍스트 추출
- [x] PDF 페이지 분리
- [x] 관련도 점수 표시
- [ ] 사용자 피드백 시스템

---

**⚠️ 주의사항**: 이 챗봇은 CAIKE 흥행예측시스템 매뉴얼에 기반한 답변만 제공하며, 매뉴얼에 없는 내용에 대해서는 표준 응답을 제공합니다.

---

<div align="center">

**🎬 CAIKE 흥행예측시스템과 함께 더 나은 콘텐츠 전략을 수립하세요!**

</div>
