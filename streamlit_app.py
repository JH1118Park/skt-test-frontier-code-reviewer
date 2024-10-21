import streamlit as st
import pandas as pd # pandas는 데이터 조작과 분석을 위한 라이브러리입니다
# CSV 파일에서 데이터를 로드합니다.
df = pd.read_csv('movies_2024.csv')
# 언어별 영화 수를 계산합니다
language_data = df['original_language'].value_counts()
# Streamlit 앱 제목
st.title('Movie Language Distribution')
# 데이터 출력
language_data
# 막대 차트 생성 및 표시
st.bar_chart(language_data)