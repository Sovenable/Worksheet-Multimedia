# Tugas Implementasi: Real-time Remote Photoplethysmography (rPPG)
# Link Repository : https://github.com/Sovenable/Worksheet-Multimedia.git
# Referensi Teknis Pengerjaan: https://chatgpt.com/share/69317daf-bc58-800a-bfc3-4571839d3fb4
## Deskripsi Tugas
Tugas ini bertujuan memberikan pengalaman praktis bagi mahasiswa dalam mengimplementasikan teknologi Remote Photoplethysmography (rPPG) yang dibahas di kelas. Mahasiswa diminta membangun sebuah sistem perangkat lunak yang mampu mendeteksi detak jantung seseorang secara real-time menggunakan kamera (webcam) tanpa kontak fisik. Sistem ini mereplikasi pipeline rPPG dasar — dari deteksi wajah sampai estimasi BPM — dan menambahkan visualisasi realtime agar hasil dapat langsung diamati.

## Langkah Langkah perngerjaan
### 1. Deteksi Wajah
Pertama sistem mendeteksi dan melacak wajah menggunakan MediaPipe Face Mesh sehingga posisi ROI dapat ditentukan otomatis tiap frame. Deteksi ini memastikan ROI selalu mengikuti gerakan kepala ringan dan mengurangi kebutuhan crop manual. Dengan tracking otomatis, ekstraksi sinyal bisa berlangsung terus-menerus saat pengguna bergerak sedikit.
### 2. Ekstraksi Sinyal

Dari ROI yang terdeteksi, sistem melakukan spatial averaging pada kanal warna hijau untuk mendapatkan sinyal mentah rPPG. Selain green-channel, ada juga mekanisme POS (plane orthogonal to skin) sebagai fallback untuk meningkatkan ketahanan terhadap perubahan pencahayaan atau gerakan ringan. Data warna rata-rata disimpan dalam buffer sliding-window untuk diproses lebih lanjut.

### 3. Pemrosesan Sinyal
Sinyal mentah didetrend untuk menghilangkan drift, lalu diterapkan bandpass filter pada rentang fisiologis (0.67–4.0 Hz) untuk menahan noise di luar rentang denyut jantung. Setelah itu sinyal diberi windowing  dan dipersiapkan untuk transformasi frekuensi. Tahap ini penting untuk meningkatkan rasio sinyal terhadap noise sebelum estimasi frekuensi dominan.

### 4. Konversi Frekuensi → BPM
Sistem melakukan FFT pada sinyal yang telah diproses untuk menemukan frekuensi dominan di rentang bandpass. Frekuensi puncak tersebut dikonversi ke Beats Per Minute (BPM) dengan cara BPM = peak_freq (Hz) × 60. Untuk stabilitas, hasil BPM disaring (smoothing + hold) sehingga angka di layar tidak kedip atau berubah drastis bila ada gangguan singkat.

## Perbedaan dengan di demonstrasi dosen di kelas
### 1. Pemrosesan Real-time
Pada demonstrasi di kelas, dosen memproses video yang sudah direkam sebelumnya. Setiap frame dalam video diproses secara berurutan dari awal hingga akhir, sehingga hasil BPM baru muncul setelah seluruh video selesai dianalisis. Pendekatan ini cocok untuk penelitian offline, tetapi tidak memberikan interaksi langsung kepada pengguna. Pada implementasi yang saya buat, sistem bekerja secara real-time menggunakan webcam. Setiap frame yang masuk langsung diproses saat itu juga tanpa menunggu frame lain, sehingga BPM dapat muncul secara langsung di layar. Dengan cara ini, pengguna bisa langsung melihat perubahan BPM secara instan, bisa memperbaiki posisi wajah kalau ROI tidak tepat.
### 2. Pemilihan ROI Secara Otomatis pada Dahi
Saat demo di kelas, dosen memilih ROI secara manual dengan melakukan cropping pada bagian wajah tertentu di video. Hal ini membuat proses bergantung pada input manual dan tidak bisa beradaptasi secara otomatis jika posisi wajah berubah. Pada implementasi saya, ROI dipilih secara otomatis menggunakan MediaPipe Face Mesh dengan fokus pada area dahi (forehead). Bagian dahi dipilih karena merupakan area kulit yang cukup stabil, mudah dideteksi, dan relatif minim gangguan seperti rambut dan gerakan mulut. Dengan ROI otomatis, pengguna tidak perlu mengatur area wajah secara manual — sistem akan mengikuti posisi wajah selama kamera masih dapat mendeteksinya.
### 3. BPM Ditampilkan Langsung pada Video Overlay
Dalam demo dosen, BPM hanya dicetak di terminal sebagai teks biasa. Pada implementasi saya, BPM tidak hanya ditampilkan di terminal, tetapi juga muncul langsung sebagai overlay text di bagian bawah layar kamera. Ini membuat pengguna dapat melihat hasil estimasi tanpa perlu melihat terminal. Selain BPM, saya juga menampilkan nilai rata-rata kanal hijau (Green channel) untuk memberi gambaran apakah sinyal wajah sedang kuat atau tidak. ROI dahi juga ditandai dengan kotak hijau di video, sehingga area yang diekstraksi terlihat jelas. Tampilan ini membuat sistem jauh lebih informatif dan user-friendly.
### 4. Visualisasi Spektrum Frekuensi Secara Real-time
Pada demo di kelas, grafik atau plot hanya ditampilkan secara statis setelah seluruh proses selesai. Dalam implementasi saya, spektrum frekuensi hasil FFT ditampilkan secara real-time pada panel di bawah video. Spektrum ini menunjukkan puncak frekuensi yang sedang dominan pada sinyal rPPG, sehingga pengguna dapat melihat secara visual bagaimana sinyal berubah dari waktu ke waktu. Plot ini diperbarui setiap kali estimasi BPM dilakukan, sehingga pengguna bisa langsung melihat apakah sinyal sedang stabil atau tidak. Visualisasi semacam ini sangat membantu untuk debugging dan memberi pemahaman lebih jelas tentang bagaimana sinyal rPPG diproses.