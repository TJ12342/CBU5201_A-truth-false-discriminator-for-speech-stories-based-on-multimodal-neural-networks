import os
import speech_recognition as sr
import pandas as pd

# 创建一个 Recognizer 实例
r = sr.Recognizer()

# 指定 PocketSphinx 的模型路径
model_directory = r"D:\books\machine_learning\project\pocketsphinx-data\cmusphinx-zh-cn-5.2"
acoustic_model_path = os.path.join(model_directory, "zh_cn.cd_cont_5000")
language_model_path = os.path.join(model_directory, "zh_cn.lm.bin")
dictionary_path = os.path.join(model_directory, "zh_cn.dic")

df=pd.read_csv(r'D:\books\machine_learning\project\data\CBU0521DD_stories_attributes.csv')


audio_directory=r'D:\books\machine_learning\project\data\CBU0521DD_stories'
output_directory=audio_directory

for i in range(1, 101):
    # 生成文件名
    file_number = str(i).zfill(5)
    file_name = file_number + ".wav"
    audio_file_path = os.path.join(audio_directory, f"{file_number}.wav")
    output_file_path = os.path.join(output_directory, f"{file_number}.txt")

    # 检查音频文件是否存在
    if not os.path.isfile(audio_file_path):
        print(f"音频文件 {audio_file_path} 不存在")
        continue

    # 读取音频文件
    with sr.AudioFile(audio_file_path) as source:
        print(f"读取音频文件 {file_number}.wav...")
        audio_data = r.record(source)  # 读取整个音频文件
        print("识别中...")

    # 使用 PocketSphinx 进行离线识别
    try:
        if (df.loc[df['filename']==file_name]['Language']=='English').bool():
            text = r.recognize_sphinx(audio_data, language='en-US')
        else:
            text = r.recognize_sphinx(audio_data, language=(acoustic_model_path, language_model_path, dictionary_path))
        print("识别结果: {}".format(text))
        with open(output_file_path, "w", encoding="utf-8") as file:
            file.write(text)
        print(f"已保存识别结果到 {file_number}.txt")
    except sr.UnknownValueError:
        print("PocketSphinx 无法识别语音")
    except sr.RequestError as e:
        print("请求 PocketSphinx 时发生错误: {}".format(e))

