# ==========================================
# 1. KURULUMLAR VE SİSTEM HAZIRLIĞI
# ==========================================
import os, torch, gc, base64, time
from transformers import AutoTokenizer, AutoModelForCausalLM
# from TTS.api import TTS # Moved after installation
from IPython.display import Audio, display, Javascript
from google.colab import output

os.environ["COQUI_TOS_AGREED"] = "1"

def clear_mem():
    gc.collect()
    torch.cuda.empty_cache()

print("Sistem hazırlanıyor, lütfen bekleyin...")
!apt-get update -qq && apt-get install -y -qq ffmpeg espeak-ng libsndfile1 > /dev/null
!pip install -q coqui-tts transformers accelerate bitsandbytes git+https://github.com/openai/whisper.git

from TTS.api import TTS # Moved after installation
import whisper # Moved after installation

# ==========================================
# 2. MODELLERİN YÜKLENMESİ
# ==========================================
device = "cuda" if torch.cuda.is_available() else "cpu"
clear_mem()

print("Modeller yükleniyor (Gemma-2-2B & XTTS v2)...")
model_id = "unsloth/gemma-2-2b-it-bnb-4bit"
tokenizer_llm = AutoTokenizer.from_pretrained(model_id)
llm_model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", torch_dtype=torch.bfloat16)
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
stt_model = whisper.load_model("base", device="cpu")

print("\n🚀 BANKA ASİSTANI ÇEVRİMİÇİ!")

# ==========================================
# 3. SES VE SOHBET FONKSİYONLARI
# ==========================================

def record_audio_js():
    js = Javascript("""
    async function record() {
      const stream = await navigator.mediaDevices.getUserMedia({audio: true});
      const recorder = new MediaRecorder(stream);
      let chunks = [];
      const btn = document.createElement('button');
      btn.textContent = '🎤 KONUŞMAK İÇİN BASILI TUTUN VEYA TIKLAYIN';
      btn.style = 'padding:25px; background:#0056b3; color:white; border-radius:15px; border:none; cursor:pointer; font-size:20px; width:100%; box-shadow: 0 4px 15px rgba(0,0,0,0.3);';
      document.body.appendChild(btn);

      return new Promise(resolve => {
        btn.onclick = () => {
          if(recorder.state === 'inactive'){
            recorder.start();
            btn.textContent = '🛑 DİNLİYORUM... DURDURMAK İÇİN TEKRAR BASIN';
            btn.style.background = '#d32f2f';
          } else {
            recorder.stop();
            btn.textContent = '⌛ SESİNİZ İŞLENİYOR...';
            btn.disabled = true;
          }
        };
        recorder.ondataavailable = e => chunks.push(e.data);
        recorder.onstop = async () => {
          const blob = new Blob(chunks);
          const reader = new FileReader();
          reader.readAsDataURL(blob);
          reader.onloadend = () => {
            resolve(reader.result);
            btn.remove();
          };
        };
      });
    }
    """)
    display(js)
    data = output.eval_js('record()')
    return base64.b64decode(data.split(',')[1])

chat_history = []

def banka_asistani_akisi(referans_ses):
    global chat_history
    print("\n--- SOHBET BAŞLADI ---")

    while True:
        try:
            # 1. DİNLEME
            raw_audio = record_audio_js()
            with open('in.wav', 'wb') as f: f.write(raw_audio)

            result = stt_model.transcribe('in.wav', language="tr")
            user_text = result["text"].strip()

            if not user_text:
                continue

            print(f"\n[Müşteri]: {user_text}")

            # 2. ÖZEL DURUM KONTROLLERİ VE SENARYO AKIŞI
            user_lower = user_text.lower()
            response_text = ""
            vakti_geldi_mi_ayriligin = False

            # Senaryo Adım 1: İlk Karşılama ve Kimlik Bilgisi İsteme
            if len(chat_history) == 0:
                response_text = "Merhaba, hoş geldiniz! Ben bankanın müşteri hizmetleri temsilcisiyim. Size nasıl yardımcı olabilirim? Öncelikle, adınızı ve müşteri numaranızı veya T.C. kimlik numaranızı, telefon numaranızı alabilir miyim? Böylece sizinle daha hızlı ve etkili bir şekilde iletişim kurabilirim."

            # Senaryo Adım 2: Vedalaşma
            elif any(x in user_lower for x in ["teşekkür ederim", "teşekkürler", "yeterli", "bu kadar", "hoşçakal", "iyi günler"]):
                response_text = "Size yardımcı olabildiysem ne mutlu bana! Başka bir konuda yardımcı olmamı isterseniz lütfen bana bildirin. Bizleri aradığınız için teşekkür ederiz efendim. Görüşmek dileğiyle, hoşça kalın, iyi günler dilerim!"
                vakti_geldi_mi_ayriligin = True

            # Senaryo Adım 3: Genel Bilgi ve Detaylı Faiz/Kampanya Sorguları (LLM)
            else:
                # LLM'e tüm banka bilgilerini sistem komutu olarak veriyoruz
                system_prompt = (
                    "Sen resmi ve nazik bir banka müşteri hizmetleri temsilcisisin. Müşteriye 'Efendim' diye hitap et. "
                    "BANKA BİLGİLERİ VE KAMPANYALAR:\n"
                    "- Hizmetler: Hesap açma, kredi kartı başvurusu, kredi talepleri, yatırım danışmanlığı, sigorta (Ev, araç, sağlık, seyahat).\n"
                    "- Kredi Kartı Detayları: Vergi ödemesi, kırtasiye ve sağlık harcamalarına 3 ay vade farksız taksit; Elektronik ve beyaz eşyaya 6 ay %3.4 faiz oranı.\n"
                    "- Finansman Kredi Faizleri: 12 ay %4.5, 24 ay %3.4, 36-48 ay %2.6, 60 ay %1.9 faiz uygulanır.\n"
                    "KURALLAR:\n"
                    "1. Müşteri bir hizmeti sorduğunda genel bilgi ver.\n"
                    "2. Eğer müşteri 'detay' isterse veya 'faizler ne kadar' gibi spesifik sorarsa parantez içindeki yukarıdaki rakamları sırala.\n"
                    "3. Müşteri sorunluysa 'Az önceki müşterimizde de benzer bir sorun vardı ama hemen çözdük, merak etmeyin' de."
                )

                prompt = f"<bos><start_of_turn>system\n{system_prompt}<end_of_turn>\n"

                for turn in chat_history[-4:]:
                    prompt += f"<start_of_turn>{turn['role']}\n{turn['content']}<end_of_turn>\n"

                prompt += f"<start_of_turn>user\n{user_text}<end_of_turn>\n<start_of_turn>model\n"

                inputs = tokenizer_llm(prompt, return_tensors="pt").to(device)
                with torch.no_grad():
                    outputs = llm_model.generate(**inputs, max_new_tokens=200, do_sample=True, temperature=0.6)

                response_text = tokenizer_llm.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)

            print(f"[Asistan]: {response_text}")

            # 4. SESLENDİRME (XTTS v2)
            tts.tts_to_file(
                text=response_text,
                speaker_wav=referans_ses,
                language="tr",
                file_path="out.wav"
            )

            display(Audio("out.wav", autoplay=True))

            # Hafızayı güncelle
            chat_history.append({"role": "user", "content": user_text})
            chat_history.append({"role": "model", "content": response_text})

            if vakti_geldi_mi_ayriligin:
                time.sleep(6) # Sesin bitmesini bekle
                print("\n--- SOHBET SONLANDIRILDI ---")
                break

            time.sleep(1)
            clear_mem()

        except Exception as e:
            print(f"Hata: {e}")
            break

# ==========================================
# 4. ÇALIŞTIRMA
# ==========================================
ref = "benim_sesim.wav"

if os.path.exists(ref):
    banka_asistani_akisi(ref)
else:
    print(f"HATA: '{ref}' bulunamadı. Lütfen sol panelden sesini yükle.")
