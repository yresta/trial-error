import streamlit as st
from telethon.sync import TelegramClient
from telethon.tl.functions.messages import GetHistoryRequest
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import pandas as pd
import asyncio
import nest_asyncio
import re
from collections import Counter

# Import fungsi dari file lavidaloca.py
from heredacode import (
    integrate_clustering_with_keywords,
    clean_text_for_clustering,
    is_unimportant_sentence,
    find_optimal_clusters,
    load_spelling_corrections,
    get_sentence_model,
    build_spelling_pattern
)

# Patch asyncio agar bisa berjalan di Streamlit
nest_asyncio.apply()

# --- KONFIGURASI ---
api_id = 21469101
api_hash = '3088140cd7e561ecdadcfbd9871cf3f0'
session_name = 'session_utama'
wib = ZoneInfo("Asia/Jakarta")

# --- Load model ---
sentence_model = get_sentence_model()

# --- Load spelling corrections ---
spelling = load_spelling_corrections('kata_baku.csv')

# --- Topik dan keyword ---
topik_keywords = {
    # Topik dengan logika "DAN" (semua kata harus ada)
    "Status Bast": [
        ["bast"],
        ["stuck", "bast"]
    ],
    "Verifikasi Toko": [
        ["verifikasi", "toko"],
        ["verivikasi", "toko"],
        ["cek", "id", "toko"],
        ["nib"]
    ],
    "Verifikasi Pembayaran": [
        ["verifikasi", "pembayaran"],
        ["verifikasi", "pesanan"],
        ["verivikasi", "pembayaran"],
        ["minta", "verifikasi"],
        ["konfirmasi"],
        ["notif", "error"],
        ["verifikasi"],
        ["verivikasi"]
    ],
    "Penerusan Dana": [
        ["penerusan", "dana"],
        ["dana", "diteruskan"],
        ["uang", "diteruskan"],
        ["penerusan"],
        ["diteruskan"],
        ["meneruskan"],
        ["dana", "teruskan"],
        ["uang", "teruskan"],
        ["penyaluran"],
        ["di teruskan"],
        ["salur"]
    ],
    "Dana Belum Masuk": [
        ["dana", "belum", "masuk"],
        ["uang", "belum", "masuk"],
        ["dana", "masuk", "belum"],
        ["uang", "masuk", "belum"],
        ["dana", "tidak", "masuk"],
        ["uang", "tidak", "masuk"],
        ["dana", "gagal", "masuk"],
        ["uang", "gagal", "masuk"],
        ["belum", "masuk", "rekening"],
        ["belum", "transfer", "masuk"],
        ["belum", "masuk"]

    ],
    "Jadwal Cair Dana": [
        ["bos", "cair"],
        ["bop", "cair"],
        ["jadwal", "cair"],
        ["kapan", "cair"],
        ["gelombang", "2"],
        ["tahap", "2"],
        ["pencairan"]
    ],
    "Kendala Akses" : [
        ["kendala", "akses"],
        ["gagal", "akses"],
        ["tidak", "bisa", "akses"],
        ["tidak", "bisa", "login"],
        ["tidak", "bisa", "masuk"],
        ["gagal", "login"],
        ["gagal", "masuk"],
        ["gagal", "akses"],
        ["reset", "akun"],
        ["reset", "password"],
        ["ganti", "password"],
        ["ganti", "akun"],
        ["ganti", "email"],
        ["ganti", "nomor"],
        ["ganti", "no hp"],
        ["ganti", "no telepon"],
        ["ganti", "telepon"],
        ["eror", "akses"],
        ["eror", "login"],
        ["eror"],
        ["web", "dibuka"],
        ["gk", "bisa", "masuk"],
        ["belum", "lancar"],
        ["bisa", "diakses"],
        ["gangguan"],
        ["gangguannya"],
        ["belum", "normal", "webnya"],
        ["trobel"],
        ["trobelnya"],
        ["ga", "bisa", "akses"],
        ["ga", "bisa", "log", "in"],
        ["ga", "bisa", "masuk"],
        ["ga", "bisa", "web"],
        ["g", "masuk2"],
        ["gk", "bisa2"],
        ["web", "troubel"],
        ["jaringan"],
        ["belum", "bisa", "masuk", "situs"],
        ["belum", "normal", "web"],
        ["vpn"],
        ["gabisa", "login"],
        ["gabisa", "akses"],
        ["g", "bisa", "akses"],
        ["g", "bisa", "login"],
        ["tidak", "bisa", "di", "buka"],
        ["bermasalah", "login"],
        ["login", "trouble"],
        ["maintenance"],
        ["di block"],
        ["normal"],
        ["error"],
        ["trouble"],
        ["maintainance"]
    ],
    "Kendala Autentikasi": [
        ["kendala", "autentikasi"],
        ["gagal", "autentikasi"],
        ["tidak", "bisa", "autentikasi"],
        ["gagal", "otentikasi"],
        ["tidak", "bisa", "otentikasi"],
        ["authenticator", "reset"], 
        ["autentikasi"],
        ["autentifikasi"],
        ["otentikasi"],
        ["otp", "gagal"],
        ["otp", "tidak", "bisa"],
        ["otp", "tidak", "muncul"],
        ["otp", "tidak", "tampil"],
        ["otp", "tidak", "ada"],
        ["reset", "barcode"],
        ["authenticator"],
        ["aktivasi"]
    ],
    "Kendala Upload": [
        ["kendala", "upload"],
        ["gagal", "upload"],
        ["tidak", "bisa", "upload"],
        ["gagal", "unggah"],
        ["tidak", "bisa", "unggah"],
        ["produk", "tidak", "muncul"],
        ["produk", "tidak", "tampil"],
        ["produk", "tidak", "ada"],
        ["produk", "massal"],
        ["produk", "masal"],
        ["template", "upload"],
        ["template", "unggah"],
        ["unggah", "produk"],
        ["menambahkan"],
        ["menambah", "produk"],
        ["tambah", "produk"],
        ["tambah", "barang"],
        ["unggah", "foto"],
        ["unggah", "gambar"],
        ["unggah", "foto", "produk"],
        ["unggah", "gambar", "produk"],
        ["upload", "produk"],
        ["upload", "barang"]
    ],
    "Kendala Pengiriman": [
        ["tidak", "bisa", "pengiriman"],
        ["barang", "rusak"],
        ["barang", "hilang"],
        ["status", "pengiriman"]
    ],
    "Tanda Tangan Elektronik (TTE)": [
        ["tanda", "tangan", "elektronik"],
        ["ttd", "elektronik"],
        ["tte"],
        ["ttd"],
        ["tt elektronik"],
        ["e", "sign"],
        ["elektronik", "dokumen"]
    ],
    "Ubah Data Toko": [
        ["ubah", "data", "toko"],
        ["edit", "data", "toko"],
        ["ubah", "nama", "toko"],
        ["edit", "nama", "toko"],
        ["ubah", "rekening"],
        ["edit", "rekening"],
        ["ubah", "status", "toko"],
        ["edit", "status", "toko"],
        ["ubah", "status", "umkm"],
        ["edit", "status", "umkm"],
        ["ubah", "status", "pkp"],
        ["ganti"]
    ],
    "Seputar Akun Pengguna": [
        ["ganti", "email"],
        ["ubah", "email"],
        ["ganti", "nama", "akun"],
        ["ubah", "nama", "akun"],
        ["ganti", "akun"],
        ["ubah", "akun"],
        ["gagal", "ganti", "akun"],
        ["gagal", "ubah", "akun"]
    ],
    "Pengajuan Modal": [
        ["pengajuan", "modal"],
        ["ajukan", "modal"],
        ["modal", "kerja"],
        ["dana", "talangan"],
        ["dibatalkan", "pengajuan"],
        ["tidak", "bisa", "ajukan"],
        ["modal", "talangan"],
        ["modal", "kerja"],
        ["dana", "talangan"],
        ["dana", "kerja"],
        ["modal", "bantuan"],
        ["modal", "usaha"],
        ["modal", "bantuan", "usaha"]
    ],
    "Pajak": [
        ["pajak", "ppn"],
        ["pajak", "invoice"],
        ["pajak", "npwp"],
        ["pajak", "penghasilan"],
        ["e-billing"],
        ["dipotong", "pajak"],
        ["pajak", "keluaran"],
        ["potongan", "pajak"],
        ["coretax"],
        ["pajak"],
        ["ppn"],
        ["npwp"],
        ["e-faktur"],
        ["efaktur"],
        ["e-billing"],
        ["dpp"],
        ["pph"]
    ],
    "Etika Penggunaan": [
        ["bendahara", "dapat", "untung"],
        ["bendahara", "dagang"],
        ["bendahara", "etik"],
        ["distributor", "dilarang"],
        ["etik", "distributor"],
        ["etik", "larangan"],
        ["etik", "juknis"],
        ["larangan"]
    ],
    "Waktu Proses": [
        ["kapan"],
        ["estimasi"],
        ["waktu", "proses"],
        ["waktu", "penyelesaian"],
        ["waktu", "selesai"],
    ],
    "Pembatalan Pesanan": [
        ["batalkan", "pesanan"],
        ["pembatalan", "pesanan"],
        ["batalkan", "order"],
        ["pembatalan", "order"],
        ["batalin", "pesanan"],
        ["batalin", "order"],
        ["cancel"]
    ],
    
    # Topik dengan logika "ATAU" (salah satu kata cukup)
    "Pembayaran Dana": ["transfer", "dana masuk", "pengembalian", "bayar", "pembayaran", "dana", "dibayar", "notif pembayaran", "transaksi", "expired"],
    "Pengiriman Barang": ["pengiriman", "barang rusak", "kapan dikirim", "status pengiriman", "diproses"],
    "Penggunaan Siplah": ["pakai siplah", "siplah", "laporan siplah", "pembelanjaan", "tanggal pembelanjaan", "ubah tanggal", "dokumen", "bisa langsung dipakai", "terhubung arkas"],
    "Kurir Pengiriman": ["ubah kurir", "ubah jasa kirim", "jasa pengiriman", "jasa kurir"],
    "Status": ["cek"],
    "Bantuan Umum": ["ijin tanya", "minta tolong", "tidak bisa", "cara", "masalah", "mau tanya", "input", "pkp", "pesanan gantung", "di luar dari arkas", "di bayar dari"],
    "lainnya": []
}

# --- TAMPILAN STREAMLIT ---
st.set_page_config(page_title="Scraper & Analisis Telegram", layout="wide")
st.title("💥 Analisis Topik Pertanyaan Grup Telegram")

# Input grup Telegram dan tanggal
group = st.text_input("Masukkan username atau ID grup Telegram:", "@contohgroup")
today = datetime.now(wib).date()
week_ago = today - timedelta(days=7)
col1, col2 = st.columns(2)
with col1:
    start_date_scrape = st.date_input("Scrape dari tanggal", week_ago)
with col2:
    end_date_scrape = st.date_input("Scrape sampai tanggal", today)

# --- Fungsi pendukung ---
def is_question_like(text: str) -> bool:
    """Deteksi apakah teks valid pertanyaan dengan filter ketat."""
    if pd.isna(text) or not isinstance(text, str):
        return False
    
    text_lower = text.strip().lower()
    if not text_lower:
        return False

    # --- 1. Filter noise awal (hanya sapaan, tanda, atau <3 kata tanpa konteks) ---
    non_question_patterns = [
        r'^(min|admin|pak|bu|kk|kak|om|bro|sis)[\s\?]*$',  # cuma panggilan
        r'^[\?\.]+$',                                     # cuma tanda baca
        r'^(iya|ya+|oke+|ok|sip|noted+|oh+|lah+|loh+)$',   # respon singkat
        r'^(anggota baru|selamat berlibur|wah keren|mantap+|mantul+).*$',  # basa basi
    ]
    for pat in non_question_patterns:
        if re.match(pat, text_lower):
            return False

    # --- 2. Cek tanda tanya (tetapi wajib ada >=3 kata bermakna) ---
    words = text_lower.split()
    if "?" in text_lower and len(words) >= 3:
        return True

    # --- 3. Kata tanya baku ---
    question_words = ["apa","apakah","siapa","kapan","mengapa","kenapa","bagaimana",
                      "gimana","dimana","berapa","kok","kenapakah","bagaimanakah"]
    if any(q in words for q in question_words):
        # wajib ada kata konteks (biar ga lolos 'dari mana kk?' yg no konteks)
        context_keywords = ["dana","uang","pembayaran","verifikasi","akun","login","akses","upload",
                            "barang","produk","toko","pengiriman","rekening","modal","npwp","pajak"]
        if any(kw in text_lower for kw in context_keywords):
            return True
        else:
            return False

    # --- 4. Frasa pemicu pertanyaan ---
    question_phrases = [
        "ada yang tahu", "mau tanya", "izin bertanya", "boleh tanya",
        "butuh bantuan", "ada solusi", "minta saran", "rekomendasi",
        "sudah diproses belum", "kok belum", "kapan cair", "gimana prosesnya",
        "cek status", "caranya gimana", "kenapa gagal"
    ]
    if any(phrase in text_lower for phrase in question_phrases):
        return True

    return False

async def scrape_messages(group, start_dt, end_dt):
    all_messages = []
    sender_cache = {}
    progress_bar = st.progress(0, text="Menghubungkan ke Telegram...")
    try:
        async with TelegramClient(session_name, api_id, api_hash) as client:
            entity = await client.get_entity(group)
            offset_id = 0
            limit = 100
            while True:
                history = await client(GetHistoryRequest(
                    peer=entity, limit=limit, offset_id=offset_id, offset_date=None,
                    max_id=0, min_id=0, add_offset=0, hash=0
                ))
                if not history.messages: break
                messages = history.messages
                msg_date_wib_oldest = messages[-1].date.astimezone(wib)
                if msg_date_wib_oldest < start_dt:
                    messages = [msg for msg in messages if msg.date.astimezone(wib) >= start_dt]
                for msg in messages:
                    if not msg.message or not msg.date or not msg.sender_id: continue
                    msg_date_wib = msg.date.astimezone(wib)
                    if start_dt <= msg_date_wib <= end_dt:
                        sender_id = msg.sender_id
                        sender_name = sender_cache.get(sender_id)
                        if sender_name is None:
                            try:
                                sender = await client.get_entity(sender_id)
                                sender_name = f"{sender.first_name or ''} {sender.last_name or ''}".strip()
                                if not sender_name:
                                    sender_name = sender.username or f"User ID: {sender_id}"
                                sender_cache[sender_id] = sender_name
                            except Exception:
                                sender_name = f"User ID: {sender_id}"
                                sender_cache[sender_id] = sender_name
                        all_messages.append({
                            'id': msg.id, 'sender_id': sender_id, 'sender_name': sender_name,
                            'text': msg.message, 'date': msg_date_wib.strftime("%Y-%m-%d %H:%M:%S")
                        })
                if not messages or msg_date_wib_oldest < start_dt:
                    break
                offset_id = messages[-1].id
                progress_bar.progress(min(0.9, 0.1 + len(all_messages)/2000), text=f"Mengambil pesan... Total: {len(all_messages)}")
    except Exception as e:
        st.error(f"Terjadi kesalahan saat scraping: {e}")
        return None
    progress_bar.progress(1.0, text="Selesai mengambil pesan!")
    return pd.DataFrame(all_messages)

def analyze_all_topics(df_questions):
    if df_questions.empty:
        st.warning("Tidak ada data pertanyaan yang bisa dianalisis.")
        return
    num_messages = len(df_questions)
    if num_messages <= 100:
        num_auto_clusters = find_optimal_clusters(df_questions['processed_text'].tolist(), 5, 8)
    elif num_messages <= 300:
        num_auto_clusters = find_optimal_clusters(df_questions['processed_text'].tolist(), 10, 15)
    elif num_messages <= 1000:
        num_auto_clusters = find_optimal_clusters(df_questions['processed_text'].tolist(), 15, 25)
    else:
        num_auto_clusters = find_optimal_clusters(df_questions['processed_text'].tolist(), 25, 40)

    df_for_clustering = df_questions.copy()
    df_for_clustering['text'] = df_for_clustering['processed_text']
    df_questions_with_topics = integrate_clustering_with_keywords(
        df_for_clustering, 
        topik_keywords, 
        spelling_corrections=spelling,
        num_auto_clusters=num_auto_clusters
    )
    topik_counter = Counter(df_questions_with_topics["final_topic"])
    st.subheader("Ringkasan Topik Teratas")
    summary_data = [{"Topik": topik, "Jumlah Pertanyaan": count} for topik, count in topik_counter.most_common()]
    st.dataframe(pd.DataFrame(summary_data), use_container_width=True)

    st.subheader("Detail Pertanyaan per Topik")
    for topik, count in topik_counter.most_common():
        with st.expander(f"Topik: {topik} ({count} pertanyaan)"):
            questions_for_topic = df_questions_with_topics[df_questions_with_topics["final_topic"] == topik]["text"].tolist()
            for q in questions_for_topic:
                st.markdown(f"- {q.strip()}")

# --- Tombol eksekusi ---
if st.button("🚀 Mulai Proses dan Analisis"):
    if not group or group == "@contohgroup":
        st.warning("⚠ Mohon isi nama grup Telegram yang valid terlebih dahulu.")
        st.stop()
    start_dt = datetime.combine(start_date_scrape, datetime.min.time()).replace(tzinfo=wib)
    end_dt = datetime.combine(end_date_scrape, datetime.max.time()).replace(tzinfo=wib)
    df_all = asyncio.run(scrape_messages(group, start_dt, end_dt))

    if df_all is not None and not df_all.empty:
        st.success(f"✅ Berhasil mengambil {len(df_all)} pesan mentah.")
        df_all = df_all.sort_values('date').reset_index(drop=True)
        df_all['text'] = df_all['text'].str.lower()
        df_all['text'] = df_all['text'].str.replace(r'http\S+|www\.\S+', '', regex=True)
        df_all = df_all[df_all['text'].str.strip() != '']
        df_all.drop_duplicates(subset=['sender_id', 'text', 'date'], keep='first', inplace=True)
        df_all = df_all[~df_all['sender_name'].isin(['CS TokoLadang', 'Eko | TokLa', 'Vava'])]

        df_all['is_question'] = df_all['text'].apply(is_question_like)
        df_questions = df_all[df_all['is_question']].copy()

        # --- Load spelling corrections ---
        spelling = load_spelling_corrections("kata_baku.csv")   # dict: {"gpp": "tidak apa apa", ...}
        apply_spelling = build_spelling_pattern(spelling)       # function regex replacer

        # --- Preprocessing dengan spelling ---
        df_questions['processed_text'] = df_questions['text'].apply(lambda x: clean_text_for_clustering(x, apply_spelling))
        df_questions = df_questions[~df_questions['processed_text'].apply(is_unimportant_sentence)]

        tab1, tab2 = st.tabs(["❓ Daftar Pertanyaan", "📊 Analisis Topik"])
        with tab1:
            st.subheader(f"❓ Ditemukan {len(df_questions)} Pesan Pertanyaan")
            if not df_questions.empty:
                st.dataframe(df_questions[['date', 'sender_name', 'text']], use_container_width=True)
            else:
                st.info("Tidak ada pesan yang terdeteksi sebagai pertanyaan pada periode ini.")
        with tab2:
            analyze_all_topics(df_questions)
        st.success("Analisis Selesai!")