"""
执行
CUDA_VISIBLE_DEVICES=0 lmdeploy serve api_server /share/new_models/Shanghai_AI_Laboratory/internlm2-chat-1_8b --server-port 23333 --api-keys internlm2
后
streamlit run chat_ui.py

"""



from openai import OpenAI

client = OpenAI(
    api_key = "internlm2",
    base_url = "http://0.0.0.0:23333/v1"
)
while True:    
    question = input("请输入>>>>\n")
    if question == "q":
        break
    response = client.chat.completions.create(
        model=client.models.list().data[0].id,
        messages=[
            {"role": "system", "content": question}
        ]
    )
    
    print(response.choices[0].message.content,'\n')