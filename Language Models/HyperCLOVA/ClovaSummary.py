import requests
import json

class ClovaSummary:
    def __init__(self, 
                 title, # 제목
                 content, # 요약하고자 하는 문서
                 client_id = '--hide--', 
                 client_secret = '--hide--', 
                 url= "https://naveropenapi.apigw.ntruss.com/text-summary/v1/summarize", # HTTP 요청 주소
                 language = "ko", # 문서의 언어 (ko, ja)
                 model = "news", # 요약에 사용될 모델 (general : 일반 문서 요약, news : 뉴스 요약)
                 tone = "2", # 요약문의 tone을 조절 (0, 1, 2, 3)
                 summaryCount = "3"  # 요약문의 sentence 개수 (default 3)
                 ):
        self.headers = {"X-NCP-APIGW-API-KEY-ID" : client_id, 
                        "X-NCP-APIGW-API-KEY": client_secret, 
                        "Content-Type": "application/json"} # 요청 헤더
        self.url = url 
        self.title = title 
        self.content = content

        # 문자열로 입력 받아야 하는 파라미터를 문자열로 변환
        self.language = str(language)
        self.model = str(model)
        self.tone = str(tone)
        self.summaryCount = str(summaryCount)

        # For Debugging
        print(f"Initialized ClovaSummary with language={self.language}, model={self.model}, tone={self.tone}, summaryCount={self.summaryCount}")

    def send_request(self):
        data = {"document" : {"title": self.title, "content" : self.content}, 
                "option" : {"language": self.language, "model": self.model, "tone": self.tone, "summaryCount" : self.summaryCount}} # 요청 바디 결합
        API_data = json.dumps(data, indent=4, sort_keys=True) # API 전송을 위한 json 형식으로 전환
        response = requests.post(self.url, data=API_data, headers=self.headers)
        rescode = response.status_code # API 요청 전송

        return rescode, response

    def get_summary(self):
        rescode, response = self.send_request()
        if rescode == 200:
            json_response = response.json() # json 형식으로 변환
            if 'summary' in json_response:
                print(json_response['summary'])
            else:
                print("Error: 'summary' key not found in the response")
        else:
            print("Error: " + response.text) # rescode가 400 혹은 500인 경우, 에러가 존재하는 것임


if __name__ == '__main__':
    title= "'하루 2000억' 판 커지는 간편송금 시장"
    content = "간편송금 이용금액이 하루 평균 2000억원을 넘어섰다. 한국은행이 17일 발표한 '2019년 상반기중 전자지급서비스 이용 현황'에 따르면 올해 상반기 간편송금서비스 이용금액(일평균)은 지난해 하반기 대비 60.7% 증가한 2005억원으로 집계됐다. 같은 기간 이용건수(일평균)는 34.8% 늘어난 218만건이었다. 간편 송금 시장에는 선불전자지급서비스를 제공하는 전자금융업자와 금융기관 등이 참여하고 있다. 이용금액은 전자금융업자가 하루평균 1879억원, 금융기관이 126억원이었다. 한은은 카카오페이, 토스 등 간편송금 서비스를 제공하는 업체 간 경쟁이 심화되면서 이용규모가 크게 확대됐다고 분석했다. 국회 정무위원회 소속 바른미래당 유의동 의원에 따르면 카카오페이, 토스 등 선불전자지급서비스 제공업체는 지난해 마케팅 비용으로 1000억원 이상을 지출했다. 마케팅 비용 지출규모는 카카오페이가 491억원, 비바리퍼블리카(토스)가 134억원 등 순으로 많았다."
    language = "ko"
    model = "news"
    tone = "2"
    summaryCount = "3"

    model = ClovaSummary(title, content)
    model.get_summary()