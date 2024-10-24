import json
import streamlit as st
from openai import OpenAI
import sys
import subprocess
import os
import requests

# UTF-8 인코딩 설정
sys.stdout.reconfigure(encoding='utf-8')

# 화면 레이아웃 설정
st.set_page_config(layout="wide")

def get_gitlab_branches(repo_url):
    result = subprocess.run(
        ['git', 'ls-remote', '--heads', repo_url],
        capture_output=True,
        text=True
    )
    if result.returncode == 0:
        lines = result.stdout.strip().split('\n')
        branches = [line.split('/')[-1] for line in lines]
        return branches
    else:
        print("Error:", result.stderr)
        return []

# 특정 경로의 마지막 커밋의 diff 가져오기
def get_last_commit_diff(gitlab_url, gitlab_token, project_path, branch_selection, commit_diff_count):
    # GitLab 인스턴스 URL 및 개인 액세스 토큰 설정
    # GITLAB_URL = "https://gitlab.tde.sktelecom.com"
    # PRIVATE_TOKEN = "tde2-N35zS8ZvYio_KD3shyDa"  # 개인 액세스 토큰으로 교체하세요
    # PROJECT_PATH = "CRD/sdk/engine/nrtc_media_engine"  # 프로젝트 경로

    # GitLab 프로젝트 ID 가져오기
    response = requests.get(
        f"{gitlab_url}/api/v4/projects",
        headers={"PRIVATE-TOKEN": gitlab_token},
        params={"search": project_path.split('/')[-1]}
    )

    project = next((p for p in response.json() if p['path_with_namespace'] == project_path), None)
    if not project:
        raise ValueError("프로젝트를 찾을 수 없습니다.")

    project_id = project['id']

    # Diff 정보 가져오기
    response = requests.get(
        f"{gitlab_url}/api/v4/projects/{project_id}/repository/compare",
        headers={"PRIVATE-TOKEN": gitlab_token},
        params={"from": f"HEAD~{commit_diff_count}", "to": "HEAD"}
    )

    if response.status_code == 200:
        diff_data = response.json().get("diffs", [])
        # for diff in diff_data:
        #     st.warning(f"파일: {diff['new_path']}")
        #     st.warning(diff['diff'])
    else:
        st.warning(f"Diff를 가져오는 데 실패했습니다. 상태 코드: {response.status_code}")
        
    return diff_data
    # try:
    #     # 작업 디렉토리를 지정한 경로로 변경
    #     os.chdir(repo_path)

    #     # 브랜치로 체크아웃
    #     subprocess.run(["git", "checkout", branch_name], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    #     # 브랜치로 Pull
    #     subprocess.run(["git", "pull"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    #     # git diff 명령 실행 (HEAD~1과 HEAD 사이의 diff)
    #     result = subprocess.run(
    #         ["git", "diff", f"{branch_name}~{commit_diff_count}", branch_name],
    #         stdout=subprocess.PIPE,
    #         stderr=subprocess.DEVNULL,  # 에러 출력 suppressed
    #         text=True,
    #         check=True
    #     )
    #     return result.stdout if result.stdout else None
    # except subprocess.CalledProcessError as e:
    #     st.warning(f"An error occurred: {e.stderr}")
    #     return None
    # except FileNotFoundError:
    #     st.warning(f"The path {repo_path} does not exist.")
    #     return None

# 파일별 diff를 나누는 함수
def split_diff_by_file(diff):
    files_diff = diff.split("diff --git")
    return ["diff --git" + file_diff for file_diff in files_diff if file_diff.strip()]

# 파일별 diff를 요약하는 함수
def summarize_diff_by_file(diff, client):
    # files_diff = split_diff_by_file(diff)
    summarized_diffs = []

    for file_diff in diff:
        #st.warning(file_diff)
        try:
            response = client.chat.completions.create(
                model="gpt-4o",  # GPT-4 모델 사용
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert code reviewer. "
                                   "Tone - Professional and informative, with a focus on clarity and precision."
                                   "Understanding User Intent - Determine core objectives by analyzing keywords and context."
                                   "Understanding User Intent - Focus on extracting relevant information from user queries."
                                   "Understanding User Intent - Adapt responses to the specific needs or tasks indicated by the user."
                                   "Maintaining Contextual Awareness - Track the ongoing conversation history for multi-turn interactions."
                                   "Maintaining Contextual Awareness - Build on previous responses to ensure continuity and relevance."
                                   "Maintaining Contextual Awareness - Reference earlier queries or information where appropriate to maintain consistency."
                                   "Providing Concise and Informative Responses"
                                   "Providing Concise and Informative Responses - Prioritize essential information and avoid unnecessary details."
                                   "Providing Concise and Informative Responses - Ensure clarity by simplifying explanations and avoiding complex jargon unless requested."
                                   "Providing Concise and Informative Responses - Offer additional insights or elaboration if the user requests further information."
                                   "Language Adaptation and Tone Matching - Respond in the same language or tone used by the user to match their preferences."
                                   "Language Adaptation and Tone Matching -Ensure language consistency throughout the conversation for a personalized interaction."
                                   "Language Adaptation and Tone Matching -Adapt technical explanations based on the user’s level of familiarity and comfort."
                                   "Encouraging Engagement and Follow-up Interaction - Suggest additional topics or questions when there are waiting periods (e.g., during processing or analysis)."
                                   "Encouraging Engagement and Follow-up Interaction - Provide prompts to encourage further exploration of related subjects."
                                   "Encouraging Engagement and Follow-up Interaction - Offer to assist with other aspects of the user's project or queries."
                                   "Ensuring User-Centricity and Clarity - Structure responses to be clear and understandable, regardless of the user’s technical background."
                                   "Ensuring User-Centricity and Clarity - Use analogies or simplified explanations to make complex concepts more accessible."
                                   "Ensuring User-Centricity and Clarity -  Tailor explanations based on the user’s knowledge level or previous information they’ve shared."
                                   "Adhering to Scope and Boundaries - Remain focused on relevant tasks, avoiding discussions beyond core functionalities."
                                   "Adhering to Scope and Boundaries - Limit explanations about internal workings or APIs unless needed for clarity."
                                   "Adhering to Scope and Boundaries - Redirect users to their objectives if discussions veer off-topic."
                                   "Emphasizing Security and Privacy - Avoid requesting personal or sensitive information from users."
                                   "Emphasizing Security and Privacy - Provide reminders about privacy considerations if the situation calls for it."
                                   "Emphasizing Security and Privacy - Reassure users that data is handled securely and responses are designed to respect their privacy."
                                   "Handling Errors and Unclear Queries - Request clarifications if a query is vague or ambiguous."
                                   "Handling Errors and Unclear Queries - Suggest possible interpretations or ways the user could refine their query."
                                   "Handling Errors and Unclear Queries - Offer polite explanations when a task is beyond capabilities, suggesting alternatives."
                                   "Offering Programmatic and Task-Based Responses - Provide code snippets, analysis results, or data visualizations directly in response to user tasks."
                                   "Offering Programmatic and Task-Based Responses - Focus on minimizing outputs (e.g., suppressing verbose logs) when providing code."
                                   "Offering Programmatic and Task-Based Responses - Follow specific instructions when performing data-related tasks, ensuring clarity and relevance."
                                   "Supporting Educational and Developmental Interactions - Provide explanations suited to the user’s age and skill level (e.g., middle school English)."
                                   "Supporting Educational and Developmental Interactions - Adapt examples and coding lessons to match the user’s development stage and understanding level."
                                   "Supporting Educational and Developmental Interactions - Offer encouragement and support to build the user’s confidence and engagement."
                    },
                    {
                        "role": "user",
                        "content": (
                            f"다음 git diff를 검토하고 요약해 주세요:\n\n{file_diff}\n\n"
                            "Ask the user to provide the C++ code snippet they want analyzed or optimized."
                            "Inquire about the specific areas they want help with, such as performance optimization, bug fixing, or general code review."
                            "Offer a summary of the code's functionality before diving into specific recommendations."
                            "Provide detailed suggestions, corrections, or optimizations based on the code review."
                            "Encourage the user to ask follow-up questions or request further explanations if needed."
                            "Ensure all recommendations adhere to the latest C++ standards. - Provide clear, actionable advice that the user can implement immediately."
                            "Maintain a balance between technical depth and clarity, ensuring that all users can follow along regardless of their expertise level."
                            "Querying About Technical Analysis - Asking for specific data analysis tasks (e.g., Inter-Arrival Time in pcap files)."
                            "Querying About Technical Analysis - Requesting code snippets or Python functions for specific programming needs."
                            "Querying About Technical Analysis - Inquiring about methods to automate processes or improve efficiency (e.g., Git command outputs)."
                            "Maintaining Multi-Turn Conversations - Following up on previous discussions with relevant queries (e.g., 'How long will it take?')."
                            "Maintaining Multi-Turn Conversations - Providing new information that builds on past exchanges, requiring the system to maintain context."
                            "Maintaining Multi-Turn Conversations - Asking for updates or further elaboration based on prior instructions or responses."
                            "Requesting Personalized Assistance - Preferring concise and task-specific solutions, particularly for coding-related queries."
                            "Requesting Personalized Assistance - Requesting code output that aligns with the user’s current setup (e.g., using a Mac)."
                            "Requesting Personalized Assistance - Asking for analysis tailored to their project goals, such as variable integration or packet analysis."
                            "Language and Educational Needs - Asking for explanations or instructions in Korean to match their language preference."
                            "Language and Educational Needs - Requesting information suited to their learning level, such as middle school English."
                            "Language and Educational Needs - Seeking to learn new programming or technical skills in a structured, age-appropriate manner."
                            "Engaging in Data-Related Queries - Requesting methods for visualizing data, especially with Python (e.g., visualizing packet data)."
                            "Engaging in Data-Related Queries - Seeking guidance on data integration techniques, focusing on combining variables based on criteria."
                            "Engaging in Data-Related Queries - Inquiring about efficient coding practices for data processing or analysis tasks."
                            "Exploring Git and Version Control - Asking about Git functionalities and seeking methods to manage repositories efficiently."
                            "Exploring Git and Version Control - Requesting help with suppressing output or configuring Git commands programmatically."
                            "Exploring Git and Version Control - Inquiring about log analysis or specific Git commands for code management."
                            "Ensuring Privacy and Security in Queries - Avoiding the inclusion of sensitive information when seeking assistance."
                            "Ensuring Privacy and Security in Queries - Asking for methods to analyze or manage data securely and efficiently."
                            "Ensuring Privacy and Security in Queries - Seeking assurance on how data is handled or processed when interacting with the system."
                            "Testing and Experimenting with Code - Providing specific coding scenarios for testing outputs or behavior (e.g., Python coding tasks)."
                            "Testing and Experimenting with Code - Requesting optimizations or changes to existing code based on their project requirements."
                            "Testing and Experimenting with Code - Engaging in coding practices that align with their technical and educational growth."
                            "Requesting Continuous Feedback and Support - Asking for feedback on code snippets or technical explanations to gauge understanding."
                            "Requesting Continuous Feedback and Support - Seeking ongoing assistance in learning programming concepts or using tools effectively."
                            "Requesting Continuous Feedback and Support - Inquiring about additional resources or suggestions to further their technical skills."
                            "변경 사항을 요약하고, 코드의 문법상 오류가 없는지 꼼꼼하게 확인해줘. "
                            "그리고 조금이라도 오류가 있을만한 코드는 수정해서 추천해줘. "
                            "그리고 코드의 기능적 영향과 잠재적인 이슈에 대해 한국어로 자세히 설명해줘. "
                            "코드 리뷰를 충실히 수행하는 것처럼 변경 사항의 의도와 그에 따른 장단점을 상세하게 분석해줘."
                        )
                    }
                ],
                max_tokens=1000
            )
            summary = response.choices[0].message.content.strip()


            col1, col2 = st.columns(2)

            if file_diff:
                # diff 내용만 가져오기
                diff_content = file_diff['diff']

                # 함수 호출하여 diff 내용 출력
                diffs = print_diff(diff_content)
                with col1:
                    st.text_area('**수정 파일**',value=diffs, height=500)
            else:
                st.warning("Failed to get the diff.")

            if summary:
                with col2:
                    st.text_area('**요약 결과**', value=summary, height=500)
            else:
                st.warning("Failed to summarize the diff.")

            summarized_diffs.append(f"******************************************************************************************************************************************************\n")
            summarized_diffs.append(f"Diff for file:\n{diffs}\n")
            summarized_diffs.append(f"------------------------------------------------------------------------------------------------------------------------------------------------------\n")
            summarized_diffs.append(f"Summary for file:\n{summary}\n")

            sendto_webhook(diffs, summary)

        except Exception as e:
            st.warning(f"An error occurred during summarization: {e}")
            summarized_diffs.append("Error during summarization.")

    return "\n".join(summarized_diffs)

# diff 내용을 보기 좋게 출력하는 함수
def print_diff(diff):
    summarized_diffs = []
    lines = diff.split("\\n")
    for line in lines:
        summarized_diffs.append(line)
    return "\n".join(summarized_diffs)

def sendto_webhook(diff, summary):
    # 파일에서 데이터를 읽기
    # file_path = '/Users/1111792/Documents/log_mo.txt'  # 출력 파일 경로
    # with open(file_path, 'r') as file:
    # output_data = file.read()

    # Make Webhook URL
    webhook_url = "https://hook.us2.make.com/t7urq2yybrhc57fcd6ccbwzbbxujr3d7"

    # 전송할 데이터 (JSON 형태로 설정)
    payload = {
        "diff": diff,
        "summary": summary
    }

    # 요청 헤더 설정
    headers = {
        "Content-Type": "application/json"
    }

    # POST 요청 전송
    response = requests.post(webhook_url, json=payload, headers=headers)

    # 응답 확인
    if response.status_code == 200:
        st.warning("데이터가 성공적으로 전송되었습니다.")
    else:
        st.warning(f"오류 발생: {response.status_code}")
        st.warning(response.text)

# Streamlit UI 구성
def main():
    st.image('https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRlRT8bWQBnlAliiZRWLJ2MmW_3p8oB5bn2dQ&s', width=100)
    st.markdown(
    """
    <h1 style="text-align: left; color: black; font-family: 'Arial Black', sans-serif;">
        에이닷 통화 소스코드 Commit Review 앱
    </h1>
    """,
    unsafe_allow_html=True
    )

    gitlab_url = "https://gitlab.tde.sktelecom.com"
    
    # OpenAI API 키 입력란 추가
    openai_api_key = st.sidebar.text_input('OpenAI API Key', type='password')

    # Gitlab 키 입력란 추가
    gitlab_token = st.sidebar.text_input('Gitlab Token', type='password')

    # 저장소 선택 추가
    project_path = st.sidebar.selectbox(
        '저장소를 선택하세요',
        ('CRD/sdk/engine/nrtc_media_engine', 'CRD/sdk/android/nrtc_sdk_android', 'CRD/app/nrtc_sample_android', 'CRD/app/nrtc_sample_ios', 'CRD/sdk/engine/nrtc_pjsip', 'CRD/sdk/ios/nrtc_sdk_ios')
    )

    # 브랜치 선택 추가
    repo_url = f"{gitlab_url}/{project_path}.git"
    branches = get_gitlab_branches(repo_url)
   
    default_branches = ['develop', 'master', 'release']
    all_branches = default_branches + branches
    branch_selection = st.sidebar.selectbox(
        '브랜치를 선택하세요',
        (all_branches)
    )

    # Commit Count(from HEAD to HEAD~n)
    commit_diff_count = st.sidebar.selectbox(
        'Commit Count (from HEAD to HEAD~n)',
        ('1', '2', '3', '4', '5', '6', '7', '8', '9', '10')
    )

    if st.sidebar.button('SLACK 전송'):
        st.sidebar.write('SLACK 전송 버튼 클릭됨')
    if st.sidebar.button('TELEGRAM 전송'):
        st.sidebar.write('TELEGRAM 전송 버튼 클릭됨')
    
    if not openai_api_key:
        st.warning('API Key를 입력해주세요.')
        return
    
    if not gitlab_token:
        st.warning('Gitlab Token을 입력해주세요.')
        return

    client = OpenAI(api_key=openai_api_key)  # OpenAI 라이브러리 임포트

    #repo_path = "/Users/1111792/Documents/Source/torri/temp/" + repo_selection  # 여기에 로컬 저장소 경로를 입력하세요
    diff = get_last_commit_diff(gitlab_url, gitlab_token, project_path, branch_selection, commit_diff_count)
    if diff:
        summary = summarize_diff_by_file(diff, client)
        # sendto_webhook(summary)
        # if summary:
        #     st.text_area('요약 결과', value=summary, height=5000)
        # else:
        #     st.warning("Failed to summarize the diff.")
    else:
        st.warning("No diff to display.")

if __name__ == "__main__":
    main()