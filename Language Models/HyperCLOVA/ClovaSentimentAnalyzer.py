import sys
import requests
import json

class ClovaSentimentAnalyzer:
    def __init__(self, 
                 content,
                 client_id = "--hide--",
                 client_secret = "--hide--",
                 url = "https://naveropenapi.apigw.ntruss.com/sentiment-analysis/v1/analyze"):
        self.content = content
        self.url = url
        self.headers = {"X-NCP-APIGW-API-KEY-ID": client_id,
                        "X-NCP-APIGW-API-KEY": client_secret,
                        "Content-Type": "application/json"}
        self.data = {"content": content}

    def send_request(self):
        API_data = json.dumps(self.data, indent=4, sort_keys=True) # API 전송을 위한 json 형식으로 전환
        response = requests.post(self.url, data=API_data, headers=self.headers) # API 요청 전송
        rescode = response.status_code

        return rescode, response
    
    def get_result(self):
        rescode, response = self.send_request()

        # 에러 확인 및 결과를 json 형식으로 바꾸기
        if(rescode == 200):
            json_response = response.json()
        else:
            print("Error : " + response.text)

        return json_response
    
    def show_result(self):
        json_response = self.get_result()

        result = json_response['document']['sentiment'] # 감정 분석 결과
        result_probs = json_response['document']['confidence'][result] # 감정 분석 결과 확률
        
        # 형식대로 출력
        print(f'✅분석결과 {round(result_probs, 2)}% 확률로 {result}한 문장입니다. \n🔽아래는 각 문장을 세부적으로 분석한 결과입니다.\n')
        
        # 각 문장마다 분석의 핵심이 된 단어 추출
        num_sentences = len(json_response['sentences'])

        for i in range(num_sentences):
            sentence = json_response['sentences'][i]
            
            # 각 문장 감정 분석 결과
            result = sentence['sentiment']
            result_probs = sentence['confidence'][result]

            # 핵심 부분 시작 인덱스와 길이
            idx = sentence['highlights'][0]['offset']
            length = sentence['highlights'][0]['length']

            # 핵심 표현 추출
            main_sentence = sentence['content'][idx:idx+length]

            print(f'''{i+1}번째 문장은 {round(result_probs*100, 2)}% 확률로 {result}한 문장입니다. \n▶ "{main_sentence}"라는 부분 때문입니다.\n''')


if __name__ == '__main__':
    ### 쿠팡 리뷰 중 1점짜리 리뷰
    content = '''어지간해서 리뷰는 귀찮아서 안쓰는데 종이빨대 처음 체감해보고 식겁함. 
    맛은 문제 없는데 조이빨대 음료에 꽂아놓고 1분 이상 오래 둔것도 아닌데 볓번 빨자마자 종이빨대가 녹아서 입에 흐물흐물한게 묻어나옴. 
    그리고 플라스틱 빨대로 비교했을 때도 다 마시고난 다음에 찝찝한 느낌이 굉장히 강했음. 
    환경생각하자는거 다 알겠는데 만들어도 제대로 만들던가 그냥 빨대 없이 마시게 만들던가. 
    환경생각하자면서 소비자는 그냥 개엿으로 아는듯'''

    analyzer = ClovaSentimentAnalyzer(content)
    analyzer.show_result()