import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
import parselmouth
import librosa
from transformers import BertModel, BertTokenizer
import os
import pandas as pd


def extract_f0(audio_file):
    sound = parselmouth.Sound(audio_file)
    pitch = sound.to_pitch()
    f0_values = pitch.selected_array['frequency']
    f0_values = f0_values[f0_values != 0]  # 去除无声部分
    return np.mean(f0_values)

def extract_formants(audio_file):
    sound = parselmouth.Sound(audio_file)
    formant = sound.to_formant_burg()
    formant_values = []
    for i in range(1, 6):
        values = [formant.get_value_at_time(i, t) for t in formant.ts()]
        values = [v for v in values if not np.isnan(v)]  # 过滤掉 NaN 值
        if values:
            formant_values.append(np.mean(values))
        else:
            formant_values.append(0)  # 如果没有有效值，用 0 替换
    return formant_values

def extract_intensity(audio_file):
    sound = parselmouth.Sound(audio_file)
    intensity = sound.to_intensity()
    return np.mean(intensity.values)

def extract_mfcc(audio_file, n_mfcc=13):
    y, sr = librosa.load(audio_file, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return mfcc.mean(axis=1)

def extract_audio_features(audio_file):
    mfcc = extract_mfcc(audio_file)
    f0 = extract_f0(audio_file)
    formant = extract_formants(audio_file)
    intensity = extract_intensity(audio_file)
    
    return {
        'mfcc': mfcc,
        'f0': f0,
        'formant': formant,
        'intensity': intensity
    }




# 使用预训练的 BERT 模型提取文本特征
class TextFeatureExtractor:
    def __init__(self):
        # 指定本地模型目录
        model_dir = r'D:\books\machine_learning\project\bert-base-uncased'
        
        # 从本地加载模型和分词器
        self.tokenizer = BertTokenizer.from_pretrained(model_dir)
        self.model = BertModel.from_pretrained(model_dir)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)  # 将模型移动到 GPU
        self.model.eval()  # 设置为评估模式，不训练
        print(self.device)

    def extract_text_features(self, text):
        inputs = self.tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
        inputs = {key: value.to(self.device) for key, value in inputs.items()}  # 将输入移动到 GPU
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()  # 将结果移动回 CPU

text_extractor = TextFeatureExtractor()

texts = ['hello', 'world']

text_features = np.array([text_extractor.extract_text_features(t) for t in texts])

print(text_features)


data_file=r'D:\books\machine_learning\project\data\CBU0521DD_stories'

df=pd.read_csv(r'D:\books\machine_learning\project\data\CBU0521DD_stories_attributes.csv')


audio_latent = [[],[],[],[]]  # 语音文件列表
texts_latent = []  # 对应的文本列表
labels = []  # 对应的标签，0表示假，1表示真

for i in range(1, 101):
    # 生成文件名
    file_number = str(i).zfill(5)
    file_name = file_number + ".wav"
    audio_file_path = os.path.join(data_file, f"{file_number}.wav")
    text_file_path = os.path.join(data_file, f"{file_number}.txt")

    audio_latents=extract_audio_features(audio_file_path)
    audio_latent[0].append(audio_latents['mfcc'])
    audio_latent[1].append(audio_latents['f0'])
    audio_latent[2].append(audio_latents['formant'])
    audio_latent[3].append(audio_latents['intensity'])

    if i==1:
        print('size of mfcc:',len(audio_latents['mfcc']))
        print(audio_latents['mfcc'])
        print('size of f0:',1)
        print(audio_latents['f0'])
        print('size of formant:',len(audio_latents['formant']))
        print(audio_latents['formant'])
        print('size of intensity:',1)
        print(audio_latents['intensity'])


    texts_latent.append(text_extractor.extract_text_features(text_file_path))

    if (df.loc[df['filename']==file_name]['Language']=='English').bool():
        labels.append(0)
    else:
        labels.append(1)


    

print(audio_latent)


#concat
audio_latent2=[]

for i in range(0,100):
    audio_latent2.append(np.concatenate((audio_latent[0][i],audio_latent[1][i],audio_latent[2][i],audio_latent[3][i])))


# normalize
audio_latent2 = (audio_latent2 - np.mean(audio_latent2, axis=0)) / np.std(audio_latent2, axis=0)
texts_latent = (texts_latent - np.mean(texts_latent, axis=0)) / np.std(texts_latent, axis=0)


audio_features=torch.tensor(audio_latent2, dtype=torch.float32)
text_features=torch.tensor(texts_latent, dtype=torch.float32)
labels=torch.tensor(labels, dtype=torch.long)



# 定义特征映射的 MLP 模型
class FeatureMapper(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(FeatureMapper, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )
    
    def forward(self, x):
        return self.fc(x)

# 定义二分类的 MLP 模型
class Classifier(nn.Module):
    def __init__(self, input_dim):
        super(Classifier, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )
    
    def forward(self, x):
        return self.fc(x)

# 初始化模型
audio_mapper = FeatureMapper(audio_features.shape[1], 64)
text_mapper = FeatureMapper(text_features.shape[1], 64)
classifier = Classifier(128)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(list(audio_mapper.parameters()) + list(text_mapper.parameters()) + list(classifier.parameters()), lr=0.001)

# 创建数据集和数据加载器
dataset = TensorDataset(audio_features, text_features, labels)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

# 训练模型
for epoch in range(20):
    audio_mapper.train()
    text_mapper.train()
    classifier.train()
    
    for audio, text, label in train_loader:
        optimizer.zero_grad()
        
        audio_mapped = audio_mapper(audio)
        text_mapped = text_mapper(text)
        combined_features = torch.cat((audio_mapped, text_mapped), dim=1)
        
        output = classifier(combined_features)
        loss = criterion(output, label)
        
        loss.backward()
        optimizer.step()
    
    print(f'Epoch {epoch+1}, Training Loss: {loss.item()}')

# 测试模型
audio_mapper.eval()
text_mapper.eval()
classifier.eval()

correct = 0
total = 0

with torch.no_grad():
    for audio, text, label in test_loader:
        audio_mapped = audio_mapper(audio)
        text_mapped = text_mapper(text)
        combined_features = torch.cat((audio_mapped, text_mapped), dim=1)
        
        output = classifier(combined_features)
        _, predicted = torch.max(output.data, 1)
        
        total += label.size(0)
        correct += (predicted == label).sum().item()

accuracy = correct / total
print(f'Accuracy: {accuracy}')


