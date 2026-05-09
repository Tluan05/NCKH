import gradio as gr
import torch
import torchaudio
import numpy as np
from torchvision import models
import torch.nn as nn
import sounddevice as sd

# --- SETUP MODEL ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASSES = np.load('classes.npy', allow_pickle=True)
REPEL_DB = {'Achetadomesticus': 22000, 'Tettigoniacantans': 19000}

model = models.resnet18()
model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
model.fc = nn.Linear(model.fc.in_features, 66)
model.load_state_dict(torch.load('insect_detector.pth', map_location=DEVICE))
model.to(DEVICE)
model.eval()

mel_transform = torchaudio.transforms.MelSpectrogram(sample_rate=22050, n_mels=128).to(DEVICE)
db_transform = torchaudio.transforms.AmplitudeToDB().to(DEVICE)

# --- HÀM TẠO ÂM THANH ĐUỔI ---
def generate_repel_tone(frequency, duration=2.0):
    fs = 44100
    # Dùng tần số 1000Hz nếu là siêu âm để người dùng nghe được khi test
    test_freq = 1000 if frequency > 18000 else frequency
    
    t = np.linspace(0, duration, int(fs * duration), False)
    tone = np.sin(test_freq * t * 2 * np.pi)
    # Trả về dưới dạng (sample_rate, numpy_array) cho Gradio
    return fs, tone

def predict_and_repel(audio):
    if audio is None: return "Chưa có âm thanh", "0%", "N/A", None
    
    sr, data = audio
    waveform = torch.from_numpy(data).float()
    
    # Đưa giá trị về khoảng [-1.0, 1.0] (Gradio mặc định trả về Int16)
    if waveform.abs().max() > 1.0:
        waveform = waveform / 32768.0
        
    # Đưa về mono nếu là stereo
    if waveform.ndim > 1: waveform = waveform.mean(dim=1)
    if waveform.ndim == 1: waveform = waveform.unsqueeze(0)
    
    # Chuẩn hóa âm lượng (Normalize) để AI nghe rõ hơn
    if waveform.abs().max() > 0:
        waveform = waveform / waveform.abs().max()
    
    if sr != 22050:
        resampler = torchaudio.transforms.Resample(sr, 22050).to(DEVICE)
        waveform = resampler(waveform.to(DEVICE))
    
    # CHUẨN HÓA ĐỘ DÀI VỀ ĐÚNG 5 GIÂY (110250 samples)
    target_samples = 22050 * 5
    if waveform.shape[1] > target_samples:
        waveform = waveform[:, :target_samples] # Cắt nếu dài hơn
    elif waveform.shape[1] < target_samples:
        padding = target_samples - waveform.shape[1]
        waveform = torch.nn.functional.pad(waveform, (0, padding)) # Bù im lặng nếu ngắn hơn

    with torch.no_grad():
        spec = mel_transform(waveform.to(DEVICE))
        spec_db = db_transform(spec)
        outputs = model(spec_db.unsqueeze(0))
        probs = torch.nn.functional.softmax(outputs, dim=1)
        conf, predicted = torch.max(probs, 1)
        
        species = CLASSES[predicted.item()]
        confidence = conf.item()
        hz = REPEL_DB.get(species, 20000)
        
        # TẠO ÂM THANH ĐUỔI ĐỂ PHÁT TRÊN WEB
        repel_audio = generate_repel_tone(hz)
        
        # Phát trực tiếp trên máy tính luôn (nếu muốn)
        sd.play(repel_audio[1], repel_audio[0])
        
    return species, f"{confidence*100:.2f}%", f"{hz} Hz", repel_audio

# --- GIAO DIỆN ---
with gr.Blocks(title="Insect Repellent AI") as demo:
    gr.Markdown("# 🦟 Hệ thống AI Nhận diện và Đuổi Côn trùng")
    gr.Markdown("Thu âm tiếng côn trùng -> AI nhận diện -> Tạo sóng âm phản hồi.")
    
    with gr.Row():
        audio_input = gr.Audio(sources=["microphone", "upload"], label="1. Thu âm tiếng côn trùng")
        with gr.Column():
            out_species = gr.Textbox(label="Kết quả nhận diện")
            out_conf = gr.Textbox(label="Độ tin cậy")
            out_hz = gr.Textbox(label="Tần số đuổi (Hz)")
            out_audio = gr.Audio(label="2. Âm thanh đuổi mô phỏng (Bấm Play để nghe)", interactive=False)
            
    btn = gr.Button("PHÂN TÍCH VÀ PHÁT SÓNG ĐUỔI", variant="primary")
    btn.click(predict_and_repel, inputs=audio_input, outputs=[out_species, out_conf, out_hz, out_audio])

if __name__ == "__main__":
    demo.launch()
