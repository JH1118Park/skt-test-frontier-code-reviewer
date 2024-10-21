import streamlit as st
from openai import OpenAI

# Streamlit UI 구성
def main():
    st.title('Langchain과 GPT-4o-mini를 활용한 텍스트 요약 앱')
    
    # OpenAI API 키 입력란 추가
    openai_api_key = st.sidebar.text_input('OpenAI API Key', type='password')
    
    if not openai_api_key:
        st.warning('API Key를 입력해주세요.')
        return
    
    # OpenAI API 키 설정
    #openai.api_key = openai_api_key

    client = OpenAI(api_key=openai_api_key)  # OpenAI 라이브러리 임포트

    # 사용자 입력 텍스트
    input_text = st.text_area("요약하고 싶은 텍스트를 입력하세요.")

    if st.button('요약하기') and input_text:
        # 텍스트 요약 실행
        with st.spinner('요약 중입니다...'):
            try:
                response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "user", "content": f"다음 텍스트를 요약해주세요:\n{input_text}"}
                    ],
                    max_tokens=150,
                    temperature=0.7
                )
                summary = response.choices[0].message.content.strip()
                st.success('요약이 완료되었습니다!')
                st.text_area('요약 결과', value=summary, height=200)
            except Exception as e:
                st.error(f'요약 중 오류가 발생했습니다: {str(e)}')

if __name__ == "__main__":
    main()
