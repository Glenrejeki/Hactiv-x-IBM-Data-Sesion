# Hactiv\_x\_IBM\_Data\_Sesion — Dokumentasi Notebook Google Colab

**Deskripsi singkat**
Notebook ini berisi contoh eksperimen dasar di Google Colab: instalasi paket, pengambilan token dari Colab, pemanggilan API model (Replicate & Hugging Face), contoh penggunaan `transformers` (Flan-T5), serta contoh loop dan pembacaan CSV. README ini menjelaskan setiap bagian, cara setup, masalah umum, dan saran perbaikan.

📂 **Link ke Notebook Colab:** [Buka di Google Colab](https://colab.research.google.com/drive/1BbV_ZsSlhclE9wFd2-oun67hEpjx7iGa?usp=sharing)

---

## Daftar isi

1. Tujuan
2. Prasyarat
3. Instalasi & dependensi
4. Mengatur token / environment variable
5. Penjelasan isi sel (cell-by-cell)
6. Masalah umum & cara perbaikan
7. Tips menjalankan di Colab (GPU vs CPU)
8. File data
9. Contoh `requirements.txt`
10. Keamanan & best practice
11. Lisensi

---

## 1. Tujuan

Notebook ini bertujuan sebagai *starter* untuk:

* Praktik instalasi paket ML/LLM
* Memanggil model via Hugging Face Inference API
* Mencoba pipeline `transformers` (text2text-generation)
* Melatih kebiasaan membaca CSV dan looping di Python

Catatan: notebook ini tampak dibuat untuk percobaan — ada beberapa baris yang berulang/typo yang perlu dibenahi agar lebih stabil.

## 2. Prasyarat

* Akun GitHub (jika ingin menyimpan notebook)
* Akun Hugging Face (untuk `hf_token`) dan token akses
* (Opsional) Akun Replicate (untuk `REPLICATE_API_TOKEN`) bila ingin pakai API Replicate
* Google Colab (direkomendasikan karena mudah) atau environment Python lokal

## 3. Instalasi & dependensi

Gabungkan dan jalankan satu blok instalasi supaya runtime bersih:

```bash
# update pip dulu (opsional)
pip install -U pip

# paket utama yang dipakai di notebook
pip install replicate langchain langchain-community huggingface_hub==0.23.0 transformers pandas
```

> Catatan: di notebook asli ada beberapa perintah `!pip install` yang duplikat. Cukup jalankan satu perintah ringkas di atas. Versi `huggingface_hub==0.23.0` dicantumkan sesuai contoh; kalau terjadi konflik, hapus versi spesifik atau sesuaikan dengan versi `transformers` yang Anda gunakan.

## 4. Mengatur token / environment variable

Notebook memakai dua token (contoh):

* `REPLICATE_API_TOKEN` — token untuk Replicate (jika dipakai)
* `HF_TOKEN` — token untuk Hugging Face

**Contoh cara set di Colab (aman):**

```py
# cara interaktif: jangan commit token ke repo!
from getpass import getpass
import os

os.environ["REPLICATE_API_TOKEN"] = getpass("Enter your Replicate token: ")
os.environ["HF_TOKEN"] = getpass("Enter your Hugging Face token: ")
```

Notebook asli menggunakan `google.colab.userdata` (jika tersedia) seperti:

```py
from google.colab import userdata
api_token = userdata.get("api_token")
os.environ["REPLICATE_API_TOKEN"] = api_token

hf_token = userdata.get("hf_token")
```

> **PENTING**: ada typo di notebook asli — `os.environ["REPLIACATE_API_TOKEN"]` (salah tulis). Gunakan `REPLICATE_API_TOKEN`.

## 5. Penjelasan isi sel (cell-by-cell)

Berikut ringkasan tiap bagian kode di notebook yang Anda kirim:

1. **Variabel sederhana & print**

   ```py
   nama = 'Glen'
   umur = 20
   print('Halo nama saya adalah', nama, ' dan umur saya ', umur)
   ```

   * Contoh sederhana untuk cek environment Python.

2. **Instalasi paket (beberapa kali diulang)**

   * `!pip install replicate`, `!pip install langchain_community`, dll.
   * Saran: gabungkan jadi satu blok instalasi seperti di bagian 3.

3. **Ambil token dari Colab dan set environment**

   ```py
   from google.colab import userdata
   import os
   api_token = userdata.get("api_token")
   os.environ["REPLICATE_API_TOKEN"] = api_token  # perbaiki typo jika perlu
   ```

   * Pastikan `userdata.get(...)` tidak mengembalikan `None`.

4. **Hugging Face InferenceClient**

   ```py
   from huggingface_hub import InferenceClient
   hf_token = userdata.get("hf_token")
   client = InferenceClient(model="mistralai/Mistral-7B-Instruct-v0.2", token=hf_token)
   response = client.chat_completion(messages=[{"role":"user","content":"Dimana Batam?"}], max_tokens=128)
   print(response.choices[0].message["content"])
   ```

   * Pastikan token valid dan model tersedia untuk akun Anda (beberapa model butuh akses khusus).

5. **Contoh loop & prompt untuk klasifikasi**

   * Menampilkan elemen list, membuat prompt sederhana untuk klasifikasi sentiment.
   * Contoh prompt belum terhubung ke API — hanya mencetak prompt.

6. **Load CSV dengan pandas**

   ```py
   df = pd.read_csv('personality_datasert.csv')
   df.head()
   ```

   * Pastikan file `personality_datasert.csv` diupload ke Colab atau path benar.

7. **Transformers pipeline (google/flan-t5-base)**

   ```py
   from transformers import pipeline
   generator = pipeline("text2text-generation", model="google/flan-t5-base")
   resp = generator("Apa itu introvert?")
   print(resp[0]["generated_text"])
   ```

   * Versi berikutnya memakai parameter `device=-1` (CPU). Di Colab kalau ada GPU, set `device=0`.
   * Contoh lanjutan menggunakan `num_beams`, `do_sample`, `top_p` untuk mengontrol keluaran.

## 6. Masalah umum & cara perbaikan

* **Tidak ada token / `None`**: `userdata.get(...)` mengembalikan `None` → masukkan token lewat `getpass` atau `os.environ` manual.
* **Typo environment variable**: `REPLIACATE_API_TOKEN` salah ketik. Gunakan `REPLICATE_API_TOKEN`.
* **ModuleNotFoundError**: setelah `pip install`, restart runtime (Runtime → Restart runtime) sebelum `import` agar modul yang baru terinstal terdeteksi.
* **Model tidak dapat diakses**: model di InferenceClient mungkin memerlukan akses spesial atau subscription — periksa akses model di Hugging Face.
* **Out of memory** saat memakai `transformers` di GPU: gunakan model yang lebih kecil atau jalankan di CPU.

## 7. Tips menjalankan di Colab (GPU vs CPU)

* Untuk mempercepat `transformers`, aktifkan Runtime → Change runtime type → Hardware accelerator → GPU.
* Di `pipeline`, gunakan `device=0` untuk GPU. Contoh:

```py
from transformers import pipeline
generator = pipeline(
    "text2text-generation",
    model="google/flan-t5-base",
    device=0  # pakai GPU (0)
)
```

* Jika pakai CPU, set `device=-1`.

## 8. File data

* `personality_datasert.csv` harus diupload ke Colab (gunakan menu Files → Upload) atau simpan di Google Drive dan mount drive.
* Pastikan kolom dan encoding CSV sesuai (biasanya UTF-8).

## 9. Contoh `requirements.txt`

```
replicate
langchain
langchain-community
huggingface_hub==0.23.0
transformers
pandas
```

## 10. Keamanan & best practice

* **Jangan** commit token ke GitHub. Selalu gunakan secret manager (Colab `userdata`, GitHub Secrets, atau `getpass`).
* Jika share notebook publik, hapus baris yang berisi token plain text.
* Pin versi paket saat reproducibility penting.

## 11. Lisensi

Bebas untuk dipakai untuk belajar. Jika ingin dipublikasikan, tambahkan lisensi (mis. MIT) di repo.

---

## Bantuan selanjutnya

Kalau mau, aku bisa:

* Buatkan versi README bahasa Inggris.
* Buat file `requirements.txt` atau `runtime.txt` untuk Colab.
* Benahi notebook: perbaiki typo, gabungkan pip install, dan tambahkan cell interaktif untuk memasukkan token.

Kalau mau, beri tahu mana yang mau kamu perbaiki dulu — aku langsung editkan.
