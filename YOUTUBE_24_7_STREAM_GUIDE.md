# 📺 Raspberry Pi ile YouTube 7/24 Canlı Yayın Kurulum Rehberi

Bu rehber, Raspberry Pi üzerinde yerel olarak çalışan MixtureOfExpertCrypto siberpunk HUD arayüzünü ekran kartı veya monitör gerektirmeden (Headless) yakalayıp, ses efektleriyle birlikte **YouTube üzerinde 7/24 canlı yayın** olarak yayınlamak için gerekli adımları içermektedir.

---

## 🏗️ Genel Mimari (Nasıl Çalışır?)

Raspberry Pi OS Lite (Ekransız sürüm) üzerinde fiziksel bir monitör çıkışı olmadığı için ekranı doğrudan yakalayamayız. Bunun yerine şu yapıyı kullanırız:
1. **Xvfb (X Virtual Framebuffer):** Bellek üzerinde sanal bir ekran (örneğin 1280x720 piksel çözünürlüğünde) oluşturur.
2. **Chromium (Headless/Kiosk):** Bu sanal ekran içerisinde tam ekran modunda `http://localhost:8888` adresini açar.
3. **FFmpeg:** Sanal ekran belleğini (`x11grab` kullanarak) ve sistem seslerini yakalar. Raspberry Pi'nin donanımsal H.264 işlemcisini (`h264_v4l2m2m`) kullanarak görüntüyü kodlar ve YouTube RTMP sunucularına canlı yayın olarak aktarır.

---

## 1️⃣ Gerekli Sistem Paketlerinin Kurulması

Raspberry Pi terminaline SSH ile bağlanın ve gerekli tüm video/ses sanallaştırma araçlarını yükleyin:

```bash
sudo apt update
sudo apt install -y xvfb chromium-browser ffmpeg alsa-utils pulseaudio
```

*Not: Ses efektlerinin (synth tınıları ve bildirim sesleri) yayına aktarılabilmesi için sanal bir ses aygıtı (PulseAudio loopback) kullanılacaktır.*

---

## 2️⃣ Sanal Ses Kartı Yapılandırması (Ses Efektleri İçin)

Canlı yayına tarayıcıdan çıkan seslerin de aktarılması için sanal bir ses loopback modülü yüklenmelidir:

```bash
# PulseAudio'yu başlatın ve loopback modülünü yükleyin
pulseaudio --start
pactl load-module module-null-sink sink_name=virtual-cable sink_properties=device.description="Virtual_Cable"
pactl load-module module-loopback latency_msec=1
```

---

## 3️⃣ Canlı Yayın Başlatma Scripti (`start_stream.sh`)

Tüm bileşenleri sırasıyla başlatan, çökmeleri önleyen ve yayını otomatize eden bir kabuk scripti oluşturalım.

1. Proje dizininde scripti oluşturun:
   ```bash
   nano ~/MixtureOfExpertCrypto/start_stream.sh
   ```
2. İçeriğini aşağıdaki gibi düzenleyin (YouTube Yayın Anahtarınızı eklemeyi unutmayın):

```bash
#!/bin/bash

# Yapılandırma Ayarları
YOUTUBE_URL="rtmp://a.rtmp.youtube.com/live2"
YOUTUBE_KEY="BURAYA_YOUTUBE_YAYIN_ANAHTARINIZI_YAZIN" # YouTube Studio'dan alınır
RESOLUTION="1280x720"
DISPLAY_NUM=":99"

echo "🚀 Canlı yayın hazırlıkları başlatılıyor..."

# 1. Eski sanal ekranları temizle
pkill -f Xvfb
pkill -f chromium-browser
pkill -f ffmpeg

sleep 2

# 2. Xvfb ile Sanal Ekran Oluştur (720p, 24-bit renk derinliği)
echo "🖥️ Sanal ekran oluşturuluyor (${RESOLUTION})..."
Xvfb $DISPLAY_NUM -screen 0 ${RESOLUTION}x24 -ac +extension GLX +render -noreset &
export DISPLAY=$DISPLAY_NUM
sleep 3

# 3. Chromium'u sanal ekranda Kiosk modunda başlat
echo "🌐 Tarayıcı HUD arayüzü başlatılıyor..."
chromium-browser \
    --display=$DISPLAY_NUM \
    --kiosk \
    --no-first-run \
    --no-sandbox \
    --disable-infobars \
    --disable-session-crashed-bubble \
    --window-size=1280,720 \
    --window-position=0,0 \
    --autoplay-policy=no-user-gesture-required \
    http://localhost:8888 &
    
sleep 5

# 4. FFmpeg ile Yayını YouTube'a Aktar (Donanım İvmeli H.264)
echo "📹 Canlı yayın akışı YouTube'a yönlendiriliyor..."
ffmpeg -f x11grab -video_size $RESOLUTION -framerate 30 -i $DISPLAY_NUM.0 \
    -f pulse -i virtual-cable.monitor \
    -c:v h264_v4l2m2m -b:v 2500k -pix_fmt yuv420p -g 60 \
    -c:a aac -b:a 128k -ar 44100 \
    -f flv "$YOUTUBE_URL/$YOUTUBE_KEY"
```

3. Script dosyasını kaydedip çıkın (`Ctrl+X`, sonra `Y` ve `Enter`).
4. Çalıştırma izni verin:
   ```bash
   chmod +x ~/MixtureOfExpertCrypto/start_stream.sh
   ```

---

## 4️⃣ 7/24 Kesintisiz Çalışma İçin Systemd Servisi Kurulumu

Yayın scriptinin arka planda kesintisiz çalışması, Pi kapansa dahi otomatik başlaması ve FFmpeg çöktüğünde kendi kendine ayağa kalkması için bir sistem servisi kuralım.

1. Servis dosyasını oluşturun:
   ```bash
   sudo nano /etc/systemd/system/moe-stream.service
   ```
2. İçeriğe şunları ekleyin:

```ini
[Unit]
Description=MixtureOfExpert HUD YouTube 24/7 Stream Service
After=network.target moe-web.service

[Service]
Type=simple
User=pi
WorkingDirectory=/home/pi/MixtureOfExpertCrypto
ExecStart=/bin/bash /home/pi/MixtureOfExpertCrypto/start_stream.sh
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
```

3. Servisi kaydedin, sistem servis listesini güncelleyin ve aktifleştirin:
   ```bash
   sudo systemctl daemon-reload
   sudo systemctl enable moe-stream
   sudo systemctl start moe-stream
   ```

*Yayının durumunu ve FFmpeg çıktılarını anlık izlemek için:*
```bash
sudo journalctl -u moe-stream -f
```

---

## 5️⃣ Yayının Kesilmemesi İçin Raspberry Pi Optimizasyonları

7/24 canlı yayınlar yüksek işlemci gücü gerektirir. Pi'nin aşırı ısınmasını ve yayında donmaları önlemek için şu kurallara uymanız önem arz eder:

1. **Çözünürlük ve Kare Hızı Sınırı:**
   Yayını **1080p 60fps** yerine **720p 30fps** olarak kilitleyin. Yukarıdaki script bu şekilde optimize edilmiştir. 720p kalitesi, finansal grafikler ve siberpunk HUD akışı için fazlasıyla yeterli netlik sunacaktır.
2. **Aktif Soğutma Fanı Kullanın:**
   Raspberry Pi 4 veya 5 üzerinde sürekli FFmpeg kodlaması (encoding) çalışacağı için işlemci sıcaklığı hızla yükselecektir. Sıcaklık 80°C'yi aşarsa Pi kendini yavaşlatır (thermal throttling) ve yayında kasmalar başlar. Kaliteli bir fanlı kasa (örneğin Armor Case veya Ice Tower) kullanılması zorunludur.
3. **Kablolu Bağlantı:**
   Yayın kararlılığı için Pi'yi internete WiFi yerine doğrudan Ethernet kablosu ile bağlayın.
4. **YouTube Yayın Ayarları (Stream Settings):**
   YouTube Studio paneline gidin, yayın ayarlarına tıklayın ve gecikmeyi en aza indirmek için **"Low Latency" (Düşük Gecikme)** seçeneğini işaretleyin. Ayrıca yayın bağlantısı koptuğunda izleyicilerin yayının kapandığını görmemesi için **"Enable Auto-start"** seçeneğini açık tutun.

---

## 6️⃣ Alternatif Masaüstü Yöntemi: GUI + OBS Studio

Eğer Raspberry Pi'nizde **Masaüstü Sürümü** (GUI) yüklüyse ve monitöre bağlı olarak kullanıyorsanız:

1. Terminalden OBS kurun:
   ```bash
   sudo apt install -y obs-studio
   ```
2. OBS programını açın ve yeni bir **"Browser Source" (Tarayıcı Kaynağı)** ekleyin.
3. URL kısmına `http://localhost:8888` girin, genişliği `1280` ve yüksekliği `720` olarak ayarlayın.
4. Çıkış (Output) ayarlarından Video Bitrate değerini `2500 Kbps` yapın. Encoder olarak işlemci yükünü azaltmak için donanımsal **`Hardware (V4L2 Video Encoder)`** seçeneğini seçin.
5. YouTube Yayın Anahtarınızı ayarlar kısmına yapıştırarak yayını başlatın.
