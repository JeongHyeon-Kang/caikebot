import os
import streamlit as st
from dotenv import load_dotenv
from openai import AzureOpenAI

# 환경 변수 로드
load_dotenv()

class CaikeRAGChatbot:
    def __init__(self):
        """CAIKE 흥행예측시스템 RAG 챗봇 초기화"""
        self.setup_azure_clients()
        self.setup_system_prompt()
        
    def setup_azure_clients(self):
        """Azure OpenAI 클라이언트 설정"""
        try:
            self.openai_api_key = os.getenv("AZURE_OPENAI_API_KEY")
            self.openai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
            self.chat_deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
            self.embedding_deployment_name = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME")
            self.search_endpoint = os.getenv("AZURE_SEARCH_ENDPOINT")
            self.search_api_key = os.getenv("AZURE_SEARCH_API_KEY")
            self.search_index_name = os.getenv("AZURE_SEARCH_INDEX_NAME")
            
            # 필수 환경 변수 확인
            if not all([
                self.openai_api_key, self.openai_endpoint, self.chat_deployment_name,
                self.embedding_deployment_name, self.search_endpoint, 
                self.search_api_key, self.search_index_name
            ]):
                raise ValueError("필수 환경 변수가 설정되지 않았습니다.")
            
            # Azure OpenAI 클라이언트 초기화
            self.chat_client = AzureOpenAI(
                api_version="2024-12-01-preview",
                azure_endpoint=self.openai_endpoint,
                api_key=self.openai_api_key
            )
            
        except Exception as e:
            error_msg = f"❌ Azure OpenAI 클라이언트 설정 중 오류: {str(e)}"
            # 로그만 출력하고 Streamlit 상태는 나중에 표시
            print(error_msg)
            raise e
    
    def setup_system_prompt(self):
        """시스템 프롬프트 설정"""
        self.system_prompt = """
        당신은 CAIKE 흥행예측시스템의 전문 가이드 챗봇입니다. 
        사용자의 질문에 대해 제공된 매뉴얼 정보를 바탕으로 정확하고 도움이 되는 답변을 제공해주세요.
        
        CAIKE 흥행예측시스템은 영화 흥행 예측을 위한 AI 시스템으로, 다음과 같은 기능들을 제공합니다:
        - 영화 데이터 분석 및 처리
        - AI 기반 흥행 예측 모델링
        - 시각화 및 리포트 생성
        - 사용자 맞춤형 대시보드
        - 예측 결과 해석 및 인사이트 제공
        
        답변 시 다음 규칙을 따라주세요:
        1. 제공된 컨텍스트 정보를 우선적으로 활용하세요
        2. 단계별로 명확하게 설명해주세요
        3. 구체적인 예시나 화면 설명이 있다면 포함해주세요
        4. 친근하고 전문적인 톤을 유지해주세요
        5. 한국어로 답변해주세요
        """
    
    def get_rag_parameters(self):
        """RAG 패턴을 위한 파라미터 설정"""
        return {
            "data_sources": [
                {
                    "type": "azure_search",
                    "parameters": {
                        "endpoint": self.search_endpoint,
                        "index_name": self.search_index_name,
                        "authentication": {
                            "type": "api_key",
                            "key": self.search_api_key,
                        },
                        "query_type": "vector",
                        "embedding_dependency": {
                            "type": "deployment_name",
                            "deployment_name": self.embedding_deployment_name,
                        },
                    }
                }
            ],
        }
    
    def generate_response(self, messages):
        """사용자 질문에 대한 응답 생성"""
        try:
            # RAG 파라미터 설정
            rag_params = self.get_rag_parameters()
            
            # Azure OpenAI API 호출
            response = self.chat_client.chat.completions.create(
                model=self.chat_deployment_name,
                messages=messages,
                extra_body=rag_params,
                temperature=0.7,
                max_tokens=1000
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            return f"죄송합니다. 응답 생성 중 오류가 발생했습니다: {str(e)}"


def main():
    """메인 Streamlit 애플리케이션"""
    # 페이지 설정
    st.set_page_config(
        page_title="CAIKE 흥행예측시스템 가이드 챗봇",
        page_icon="🎬",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # 제목 및 설명
    st.title("🎬 CAIKE 흥행예측시스템 가이드 챗봇")
    st.markdown("**CAIKE 흥행예측시스템 사용법에 대해 궁금한 것이 있으시면 언제든 물어보세요!**")
    
    # 사이드바 설정
    with st.sidebar:
        st.header("⚙️ 시스템 정보")
        
        # 환경 변수 확인
        with st.expander("🔧 환경 설정 확인"):
            env_vars = [
                "AZURE_OPENAI_API_KEY",
                "AZURE_OPENAI_ENDPOINT", 
                "AZURE_OPENAI_DEPLOYMENT_NAME",
                "AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME",
                "AZURE_SEARCH_ENDPOINT",
                "AZURE_SEARCH_API_KEY",
                "AZURE_SEARCH_INDEX_NAME"
            ]
            
            env_status = {}
            for var in env_vars:
                value = os.getenv(var)
                env_status[var] = bool(value)
                
            # 요약 상태 표시
            total_vars = len(env_vars)
            configured_vars = sum(env_status.values())
            
            if configured_vars == total_vars:
                st.success(f"✅ 모든 환경 변수 설정 완료 ({configured_vars}/{total_vars})")
            else:
                st.warning(f"⚠️ 환경 변수 설정 필요 ({configured_vars}/{total_vars})")
            
            # 상세 정보 (선택적 표시)
            if st.checkbox("상세 정보 보기", key="env_details"):
                for var, is_set in env_status.items():
                    if is_set:
                        st.success(f"✅ {var}")
                    else:
                        st.error(f"❌ {var}")
        
        st.markdown("---")
        
        # 시스템 상태 확인
        if st.button("🔄 연결 상태 확인", key="check_connection"):
            if "chatbot" in st.session_state and st.session_state.chatbot is not None:
                try:
                    # 간단한 테스트 호출
                    test_message = [{"role": "user", "content": "안녕하세요"}]
                    test_response = st.session_state.chatbot.generate_response(test_message)
                    if "오류" not in test_response:
                        st.success("✅ 시스템 연결 정상")
                    else:
                        st.warning("⚠️ 시스템 응답에 문제가 있을 수 있습니다")
                except Exception as e:
                    st.error(f"❌ 연결 실패: {str(e)}")
            else:
                st.error("❌ 챗봇이 초기화되지 않았습니다")
        
        st.markdown("---")
        st.markdown("### 💡 사용 팁")
        st.markdown("""
        - "시스템 설치 방법을 알려주세요"
        - "데이터 업로드는 어떻게 하나요?"
        - "예측 모델 학습 과정을 설명해주세요"
        - "결과 해석 방법을 알고 싶어요"
        - "오류가 발생했는데 어떻게 해결하나요?"
        """)
        
        st.markdown("---")
        st.markdown("### 📋 시스템 구성")
        st.markdown("""
        **Azure OpenAI**: GPT 모델 및 임베딩  
        **Azure AI Search**: 벡터 검색 (인덱스: rag-1751860390373)  
        **RAG 패턴**: 매뉴얼 기반 정확한 답변 제공
        """)
    
    # 세션 상태 초기화
    if "messages" not in st.session_state:
        # 임시 챗봇 인스턴스로 시스템 프롬프트만 가져오기
        try:
            temp_chatbot = CaikeRAGChatbot()
            system_prompt = temp_chatbot.system_prompt
        except Exception:
            system_prompt = "CAIKE 흥행예측시스템 가이드 챗봇입니다."
        
        st.session_state.messages = [
            {
                "role": "system", 
                "content": system_prompt
            },
            {
                "role": "assistant", 
                "content": "안녕하세요! 🎬 CAIKE 흥행예측시스템 가이드 챗봇입니다.\n\n시스템 사용법, 기능 설명, 문제 해결 등 무엇이든 궁금한 것을 물어보세요. 매뉴얼을 기반으로 정확한 답변을 드리겠습니다! 😊"
            }
        ]
    
    if "chatbot" not in st.session_state:
        try:
            st.session_state.chatbot = CaikeRAGChatbot()
            st.success("✅ 챗봇이 성공적으로 초기화되었습니다.")
        except Exception as e:
            st.error(f"챗봇 초기화 중 오류가 발생했습니다: {str(e)}")
            st.info("환경 변수 설정을 확인해주세요.")
            st.stop()
    
    # 채팅 기록 표시 (시스템 메시지 제외)
    for message in st.session_state.messages:
        if message["role"] != "system":
            with st.chat_message(message["role"]):
                st.write(message["content"])
    
    # 사용자 입력 처리
    if prompt := st.chat_input("CAIKE 시스템에 대해 궁금한 것을 물어보세요..."):
        # 사용자 메시지 추가
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.write(prompt)
        
        # 챗봇 응답 생성
        with st.chat_message("assistant"):
            with st.spinner("🤔 매뉴얼을 검색하고 답변을 준비하는 중..."):
                try:
                    # 챗봇이 초기화되었는지 확인
                    if "chatbot" not in st.session_state or st.session_state.chatbot is None:
                        error_message = "챗봇이 초기화되지 않았습니다. 페이지를 새로고침해주세요."
                        st.error(error_message)
                        st.session_state.messages.append({"role": "assistant", "content": error_message})
                        st.stop()
                    
                    # 현재 대화 컨텍스트 준비 (최근 10개 메시지만 사용)
                    recent_messages = st.session_state.messages[-10:]
                    
                    # 응답 생성
                    response = st.session_state.chatbot.generate_response(recent_messages)
                    
                    # 응답 표시
                    st.write(response)
                    
                    # 응답을 세션에 추가
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    
                except Exception as e:
                    error_message = f"죄송합니다. 응답 생성 중 오류가 발생했습니다: {str(e)}"
                    st.error(error_message)
                    
                    # 오류 상세 정보 (디버그 모드에서만 표시)
                    if st.sidebar.checkbox("디버그 모드", key="debug_mode"):
                        with st.expander("🔍 오류 상세 정보"):
                            import traceback
                            st.code(traceback.format_exc())
                    
                    st.session_state.messages.append({"role": "assistant", "content": error_message})
    
    # 하단 정보
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666;'>
        💡 이 챗봇은 Azure OpenAI와 Azure AI Search를 활용한 RAG(Retrieval-Augmented Generation) 패턴을 사용합니다.<br>
        정확한 답변을 위해 CAIKE 시스템 매뉴얼을 실시간으로 검색하여 응답합니다.
        </div>
        """, 
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
