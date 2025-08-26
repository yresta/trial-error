import streamlit as st

# === Setup umum & imports ===
from telethon import TelegramClient, errors
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import pandas as pd
import asyncio
import nest_asyncio
import re
from collections import Counter, defaultdict
import numpy as np
import logging

# NLP & clustering
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from keybert import KeyBERT

# Semantic models
from sentence_transformers import SentenceTransformer

# Optional libs
try:
    import hdbscan
    HDBSCAN_AVAILABLE = True
except Exception:
    HDBSCAN_AVAILABLE = False

try:
    import umap
    UMAP_AVAILABLE = True
except Exception:
    UMAP_AVAILABLE = False

# NLTK data
nltk_packages = ['stopwords', 'punkt']
for pkg in nltk_packages:
    try:
        nltk.data.find(f'corpora/{pkg}')
    except Exception:
        nltk.download(pkg)

nest_asyncio.apply()
logging.getLogger("telethon").setLevel(logging.ERROR)

st.set_page_config(page_title="Scraper & Analisis Telegram", layout="wide")
st.title("🏦 Analisis Topik Pertanyaan Grup Telegram")

# === Secrets / Credentials ===
api_id = st.secrets["TELEGRAM_API_ID"]
api_hash = st.secrets["TELEGRAM_API_HASH"]
session_name = "session_utama"
wib = ZoneInfo("Asia/Jakarta")

# === Konstanta & heuristik ===
CENTROID_SIMILARITY_THRESHOLD = 0.8
MIN_EXAMPLES_FOR_CENTROID = 2
MIN_CLUSTER_SHOW = 2
MERGE_CLUSTER_SIM_THRESHOLD = 0.9

# Stopwords Indonesia + tambahan
stop_words_id = set()
try:
    stop_words_id = set(stopwords.words('indonesian'))
except Exception:
    stop_words_id = set(['yang','dan','di','ke','dari','ini','itu','ada','untuk','dengan','sudah','belum','bisa','tidak'])

tambahan_stop_words = [
    'yg', 'ga', 'gak', 'gk', 'ya', 'dong', 'sih', 'aja', 'kak', 'min', 'gan', 'om', 'pak', 'bu',
    'mohon', 'bantu', 'tolong', 'bantuan', 'solusi', 'tanya', 'nanya', 'bertanya', 'mas',
    'gimana', 'bagaimana', 'kenapa', 'mengapa', 'apa', 'apakah', 'kapan', 'siapa', 'kah',
    'nya', 'nih', 'tuh', 'deh', 'kok', 'kek', 'admin', 'customer', 'service', 'cs', 'halo',
    'terima', 'kasih', 'assalamualaikum', 'waalaikumsalam', 'saya', 'aku', 'anda', 'kami', 'kita',
    'ada', 'ini', 'itu', 'di', 'ke', 'dari', 'dan', 'atau', 'tapi', 'untuk', 'dengan', 'sudah', 'belum',
    'po', 'kode', 'nomor', 'nama', 'toko', 'masuk', 'jam', 'website', 'admin', 'min', 'bapak', 'ibu',
    'pak', 'bu', 'kak', 'halo', 'selamat', 'malam', 'pagi', 'siang', 'sore', 'langsung', 'kenapa',
    'apa', 'siapa', 'berapa', 'dimana', 'bagaimana', 'tu', 'kalo', 'knpa', 'izin', 'promo',
    'produk', 'sekolah', 'input', 'pesanan', 'gagal', 'error', 'trouble', 'maintenance', 'rekening',
    'akun', 'email', 'no', 'hp', 'telepon', 'reset', 'ganti', 'ubah', 'edit', 'status', 'umkm', 'pkp',
    'modal', 'kerja', 'usaha', 'bantuan', 'penyaluran', 'transfer', 'bayar', 'pembayaran', 'dana',
    'uang', 'notif', 'expired', 'pengiriman', 'barang', 'rusak', 'hilang', 'diproses', 'laporan',
    'tanggal', 'dokumen', 'terhubung', 'arkas', 'jasa', 'kurir', 'cek', 'ijin', 'masalah', 'ka','min min'
]
stop_words_id.update(tambahan_stop_words)
stemmer = PorterStemmer()

custom_stopwords = {
    "ready", "izin", "siang", "silakan", "tolong", "kak", "min", 
    "mohon", "terima", "kasih", "minta", "halo", "ya", "oke"
}

# === Aturan Topik Keyword (rule-based) ===
topik_keywords = {
    # Topik logika "DAN" 
    "status_bast": [
        ["bast"],
        ["stuck", "bast"]
    ],
    "verifikasi_toko": [
        ["verifikasi", "toko"],
        ["verivikasi", "toko"],
        ["cek", "id", "toko"]
    ],
    "verifikasi_pembayaran": [
        ["verifikasi", "pembayaran"],
        ["verifikasi", "pesanan"],
        ["verivikasi", "pembayaran"],
        ["minta", "verifikasi"],
        ["konfirmasi"],
        ["notif", "error"],
        ["verifikasi"],
        ["verivikasi"]
    ],
    "penerusan_dana": [
        ["penerusan", "dana"],
        ["dana", "diteruskan"],
        ["uang", "diteruskan"],
        ["penerusan"],
        ["diteruskan"],
        ["meneruskan"],
        ["dana", "teruskan"],
        ["uang", "teruskan"],
        ["penyaluran"],
        ["di teruskan"]
    ],
    "dana_belum_masuk": [
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
    "jadwal_cair_dana": [
        ["bos", "cair"],
        ["bop", "cair"],
        ["jadwal", "cair"],
        ["kapan", "cair"],
        ["gelombang", "2"],
        ["tahap", "2"],
        ["pencairan"]
    ],
    "modal_talangan": [
        ["modal", "talangan"],
        ["modal", "kerja"],
        ["dana", "talangan"],
        ["dana", "kerja"],
        ["modal", "bantuan"],
        ["modal", "usaha"],
        ["modal", "bantuan", "usaha"]
    ],
    "kendala_akses" : [
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
        ["error"],
        ["kapan", "normal"],
        ["trouble"],
        ["ganguan"],
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
        ["maintenance"]
    ],
    "kendala_autentikasi": [
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
        ["google", "authenticator"],
        ["gogle", "authenticator"],
        ["aktivasi", "2 langkah"]
    ],
    "kendala_upload": [
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
        ["unggah", "gambar", "produk"]
    ],
    "kendala_pengiriman": [
        ["tidak", "bisa", "pengiriman"],
        ["barang", "rusak"],
        ["barang", "hilang"],
        ["status", "pengiriman"]
    ],
    "tanda_tangan_elektronik": [
        ["tanda", "tangan", "elektronik"],
        ["ttd", "elektronik"],
        ["tte"],
        ["ttd"],
        ["tt elektronik"],
        ["e", "sign"],
        ["elektronik", "dokumen"]
    ],
    "ubah_data_toko": [
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
        ["ubah", "status", "pkp"]
    ],
    "akun_pengguna": [
        ["ganti", "email"],
        ["ubah", "email"],
        ["ganti", "nama", "akun"],
        ["ubah", "nama", "akun"],
        ["ganti", "akun"],
        ["ubah", "akun"],
        ["gagal", "ganti", "akun"],
        ["gagal", "ubah", "akun"]
    ],
    "pengajuan_modal": [
        ["pengajuan", "modal"],
        ["ajukan", "modal"],
        ["modal", "kerja"],
        ["dana", "talangan"],
        ["dibatalkan", "pengajuan"],
        ["tidak", "bisa", "ajukan"]
    ],
    "pajak": [
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
        ["e-billing"]
    ],
    "etika_penggunaan": [
        ["bendahara", "dapat", "untung"],
        ["bendahara", "dagang"],
        ["bendahara", "etik"],
        ["distributor", "dilarang"],
        ["etik", "distributor"],
        ["etik", "larangan"],
        ["etik", "juknis"],
        ["larangan"]
    ],
    "waktu_proses": [
        ["kapan"],
        ["estimasi"],
        ["waktu", "proses"],
        ["waktu", "penyelesaian"],
        ["waktu", "selesai"],
    ],
    
    # Topik logika "ATAU" 
    "pembayaran_dana": ["transfer", "dana masuk", "pengembalian", "bayar", "pembayaran", "dana", "dibayar", "notif pembayaran", "transaksi", "expired"],
    "pengiriman_barang": ["pengiriman", "barang rusak", "kapan dikirim", "status pengiriman", "diproses"],
    "penggunaan_siplah": ["pakai siplah", "siplah", "laporan siplah", "pembelanjaan", "tanggal pembelanjaan", "ubah tanggal", "dokumen", "bisa langsung dipakai", "terhubung arkas"],
    "kurir_pengiriman": ["ubah kurir", "ubah jasa kirim", "jasa pengiriman", "jasa kurir"],
    "status": ["cek"],
    "bantuan_umum": ["ijin tanya", "minta tolong", "tidak bisa", "cara", "masalah", "mau tanya", "input", "pkp", "pesanan gantung", "di luar dari arkas", "di bayar dari"],
    "lainnya": []
}

# === Preprocessing ===

def clean_text_for_clustering(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'http\S+|www\.\S+', '', text)    # URL
    text = re.sub(r'@\w+', '', text)                # mentions
    text = re.sub(r'#\w+', '', text)                # hashtags
    text = re.sub(r'\d+', '', text)                 # numbers
    text = re.sub(r'[^a-z\s]', '', text)            # non-letters
    text = re.sub(r"(.)\1{2,}", r"\1", text)        # repeated chars
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in stop_words_id and len(t) > 2]
    tokens = [stemmer.stem(t) for t in tokens]
    cleaned = ' '.join(tokens)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned

# Load dictionary kata baku (opsional)
try:
    dictionary_df = pd.read_csv("kata_baku.csv")
    spelling_correction = dict(zip(dictionary_df['tidak_baku'], dictionary_df['kata_baku']))
except Exception as e:
    st.warning(f"Gagal memuat kata_baku.csv: {e}")
    spelling_correction = {}

def correct_spelling(text, corrections):
    if not isinstance(text, str):
        return text
    words = text.split()
    corrected_words = [corrections.get(word, word) for word in words]
    return ' '.join(corrected_words)

def is_unimportant_sentence(text: str) -> bool:
    if not isinstance(text, str):
        return True
    txt = text.strip().lower()
    unimportant_phrases = [
        "siap", "noted", "oke", "ok", "baik", "sip", "thanks", "makasih", "terima kasih",
        "info apa", "info ni", "info nya", "trus ini", "terus ini", "ini saja", "ini aja",
        "ini min", "ini ya", "ini??", "ini?", "ini.", "ini", "sudah", "udah", "iya", "ya", "oh", "ohh"
    ]
    kata_tanya = ['apa','bagaimana','kenapa','siapa','kapan','dimana','mengapa','gimana','kok']
    if len(txt.split()) <= 2 and not any(q in txt for q in kata_tanya):
        if any(phrase == txt or phrase in txt for phrase in unimportant_phrases):
            return True
    return False

def is_question_like(text: str) -> bool:
    if pd.isna(text) or not isinstance(text, str):
        return False
    txt = text.strip().lower()
    if not txt:
        return False
    if '?' in txt:
        return True
    if len(txt.split()) < 3:
        if any(k in txt for k in ['apa','apakah','kapan','siapa','dimana','kenapa','bagaimana']):
            return True
        return False
    question_phrases = [
        # ==== 1. Permintaan Informasi Umum ====
        "ada yang tahu", "ada yg tau", "ada yg tahu", "ada yang tau ga", "ada yang tau gak",
        "ada yg punya info", "ada yg punya kabar", "ada kabar ga", "ada berita", "ada yg denger",
        "ada yg liat", "ada yg nemu", "ada yg ngalamin", "ada yang pernah", "yg udah tau",
        "udah ada yang tau", "ada info dong", "ada info gak", "info dong", "info donk",
        "kasih info dong", "kasih tau dong", "denger2 katanya", "bener gak sih",
        "tau ga", "tau gak", "kalian ada info?", "siapa yang tau?", "dengar kabar",
        "kabar terbaru apa", "yang tau share dong", "bisa kasih info?", "ada update?",

        # ==== 2. Tanya Langsung / Izin Bertanya ====
        "mau tanya", "pengen tanya", "pingin tanya", "ingin bertanya", "izin bertanya",
        "izin nanya", "boleh tanya", "boleh nanya", "numpang tanya", "tanya dong",
        "tanya donk", "nanya dong", "nanya ya", "aku mau nanya", "saya mau tanya",
        "penasaran nih", "penasaran banget", "penasaran donk", "mau nanya nih",
        "mau nanya ya", "btw mau tanya", "eh mau tanya", "boleh tau nggak",
        "pingin nanya", "penasaran aja", "bisa tanya gak", "lagi cari info nih",

        # ==== 3. Permintaan Bantuan / Solusi ====
        "minta tolong", "tolong dong", "tolongin dong", "tolong bantu", "bisa bantu",
        "butuh bantuan", "mohon bantuan", "mohon bantuannya", "minta bantuannya",
        "bisa tolong", "perlu bantuan nih", "ada solusi ga", "ada solusi gak",
        "apa solusinya", "gimana solusinya", "solusinya gimana", "ada yang bisa bantu",
        "ada yg bisa bantuin", "bisa bantuin gak", "butuh pertolongan", "bantu dong",
        "help dong", "help me", "minta tolong ya", "bantuin ya", "ada yang bisa nolong",

        # ==== 4. Permintaan Saran / Pendapat ====
        "ada saran", "minta sarannya", "butuh saran", "rekomendasi dong", "rekomendasi donk",
        "minta rekomendasi", "saran dong", "saran donk", "menurut kalian", "menurut agan",
        "gimana menurut kalian", "bagusnya gimana", "lebih baik yang mana", "kalian pilih yang mana",
        "kira-kira lebih bagus mana", "lebih enak mana", "mending yg mana", "menurutmu gimana",
        "kira2 pilih yg mana", "enaknya pilih yg mana", "bantu saran dong", "bantu milih dong",

        # ==== 5. Konfirmasi / Cek Status ====
        "sudah diproses belum", "udah masuk belum", "udah diapprove belum", "kok belum masuk",
        "belum cair ya", "pencairannya kapan", "kapan cair", "gimana prosesnya", "statusnya gimana",
        "sudah dicek belum", "cek status dong", "minta dicek", "mohon dicek", "sampai kapan ya",
        "bener ga", "ini valid gak", "ini udah benar?", "masih pending ya", "belum juga nih",
        "harus nunggu berapa lama", "status pending kah", "udah diproses kah", "masih dalam proses?",
        "sudah disetujui belum", "udah dikirim belum", "cek dulu dong", "konfirmasi dong",

        # ==== 6. Tanya Cara / Langkah ====
        "cara pakainya gimana", "cara pakenya gimana", "cara daftar gimana", "cara aksesnya gimana",
        "gimana caranya", "caranya gimana", "apa langkahnya", "apa tahapannya",
        "gimana stepnya", "step by step dong", "bisa kasih tutorial?", "tutorial dong",
        "cara install gimana", "cara setup gimana", "gimana setupnya", "konfigurasinya gimana",
        "gimana mulai", "cara mulainya gimana", "cara ngisi gimana", "cara input gimana",
        "login gimana", "cara reset gimana", "cara klaim gimana",

        # ==== 7. Kata Tanya Baku ====
        "apa", "apakah", "siapa", "kapan", "mengapa", "kenapa", "kenapa ya", "bagaimana",
        "gimana", "gimana ya", "gimana sih", "di mana", "dimana", "di mana ya", "berapa",
        "knp ya", "knp sih", "knp bisa", "apa ya", "yang mana ya", "kenapa begitu",
        "mengapakah", "kok bisa", "apa itu", "kenapa tidak",

        # ==== 8. Gaya Chat / Singkatan Umum ====
        "gmn ya", "gmn caranya", "gmn dong", "gmna sih", "gmna ini", "blh mnt",
        "mnt bantu", "mnt saran", "mnt info", "cek donk", "ini knp ya", "ini bgmn ya",
        "ini harus gimana", "ga ngerti", "bngung nih", "bingung banget", "bingung gw",
        "bisa dijelasin", "minta penjelasan", "bingung jelasin dong",

        # ==== 9. Seputar Pembayaran / Transaksi ====
        "va belum aktif ya", "va nya apa", "va nya belum keluar", "kode pembayaran mana",
        "kenapa pending", "kenapa gagal", "tf nya masuk belum", "rekeningnya mana",
        "sudah bayar belum", "bayar kemana", "no rek nya mana", "status tf nya apa",
        "konfirmasi pembayaran gimana", "bayar pakai apa", "pembayaran berhasil ga", "verifikasi donk",
        "rek belum masuk", "sudah transfer", "sudah tf", "uangnya belum masuk", "status transfer",
        "no pembayaran mana", "kode bayar belum muncul", "tf udah masuk?", "rek sudah benar belum"
    ]
    if any(phrase in txt for phrase in question_phrases):
        return True
    first_word = txt.split()[0]
    if first_word in ['apa','apakah','siapa','kapan','mengapa','kenapa','bagaimana','dimana','berapa','gimana']:
        return True
    return False

# === Load Models ===

@st.cache_resource(show_spinner=False)
def load_sentence_model(model_name: str = 'paraphrase-multilingual-mpnet-base-v2'):
    return SentenceTransformer(model_name)

# === Scraping ===

async def scrape_messages_iter(group, start_dt, end_dt, limit_per_call=1000, max_messages=None):
    all_messages = []
    sender_cache = {}
    progress_text = st.empty()
    progress = st.progress(0.0)
    try:
        async with TelegramClient(session_name, api_id, api_hash) as client:
            entity = await client.get_entity(group)
            fetched = 0
            async for msg in client.iter_messages(entity, limit=max_messages):
                if not getattr(msg, 'message', None) or not getattr(msg, 'date', None):
                    continue
                msg_date_wib = msg.date.astimezone(wib)
                if msg_date_wib < start_dt:
                    break
                if msg_date_wib > end_dt:
                    continue
                sender_id = getattr(msg, 'sender_id', None)
                sender_name = sender_cache.get(sender_id)
                if sender_name is None:
                    try:
                        sender = await client.get_entity(sender_id)
                        first_name = getattr(sender, 'first_name', '') or ''
                        last_name = getattr(sender, 'last_name', '') or ''
                        sender_name = f"{first_name} {last_name}".strip()
                        if not sender_name:
                            sender_name = getattr(sender, 'username', f"User ID: {sender_id}")
                    except Exception:
                        sender_name = f"User ID: {sender_id}"
                    sender_cache[sender_id] = sender_name
                all_messages.append({
                    'id': getattr(msg, 'id', None),
                    'sender_id': sender_id,
                    'sender_name': sender_name,
                    'text': getattr(msg, 'message', ''),
                    'date': msg_date_wib.strftime("%Y-%m-%d %H:%M:%S"),
                    'date_dt': msg_date_wib
                })
                fetched += 1
                if fetched % 50 == 0:
                    progress.progress(min(0.95, fetched / (max_messages or 2000)))
                    progress_text.text(f"Memproses pesan...")
            progress.progress(1.0)
    except errors.RPCError as e:
        st.error(f"Error Telethon RPC: {e}")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Terjadi kesalahan saat scraping: {e}")
        return pd.DataFrame()
    if not all_messages:
        return pd.DataFrame()
    df = pd.DataFrame(all_messages)
    return df

# === Keyword Extraction & Embedding ===

def extract_top_keywords_from_texts(texts, top_n=2, max_features=1000):
    if not texts:
        return []
    try:
        vectorizer = TfidfVectorizer(
            max_features=max_features, 
            stop_words=list(stop_words_id), 
            ngram_range=(1,2)
            )
        X = vectorizer.fit_transform(texts)
        scores = np.asarray(X.sum(axis=0)).ravel()
        features = vectorizer.get_feature_names_out()
        top_idx = np.argsort(scores)[::-1][:top_n]
        return [features[i] for i in top_idx]
    except Exception:
        return []

def embed_texts(texts, sentence_model):
    if not texts:
        return np.zeros((0, sentence_model.get_sentence_embedding_dimension()))
    preproc = [clean_text_for_clustering(t) for t in texts]
    emb = sentence_model.encode(preproc, show_progress_bar=False)
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    emb = emb / norms
    return emb

def assign_by_topic_centroid(unclassified_texts, unclassified_indices, topik_per_pertanyaan, sentence_model,
                             sim_threshold=CENTROID_SIMILARITY_THRESHOLD, min_examples=MIN_EXAMPLES_FOR_CENTROID):
    labeled_by_topic = defaultdict(list)
    for item in topik_per_pertanyaan:
        if item['topik'] != 'lainnya':
            labeled_by_topic[item['topik']].append(item['pertanyaan'])
    topics_list, centroids = [], []
    for topic, texts in labeled_by_topic.items():
        if len(texts) >= min_examples:
            emb = embed_texts(texts, sentence_model)
            if emb.shape[0] > 0:
                centroids.append(emb.mean(axis=0))
                topics_list.append(topic)
    if not centroids:
        return {}, list(range(len(unclassified_texts)))
    centroid_matrix = np.vstack(centroids)
    u_emb = embed_texts(unclassified_texts, sentence_model)
    if u_emb.shape[0] == 0:
        return {}, list(range(len(unclassified_texts)))
    sims = np.dot(u_emb, centroid_matrix.T)
    assigned, remaining_local_idxs = {}, []
    for i in range(len(unclassified_texts)):
        best_j = int(np.argmax(sims[i]))
        best_sim = float(sims[i, best_j])
        if best_sim >= sim_threshold:
            assigned[i] = {'topic': topics_list[best_j], 'score': best_sim}
        else:
            remaining_local_idxs.append(i)
    return assigned, remaining_local_idxs

# === Clustering Helpers ===

def _auto_umap_params(n_texts: int):
    if n_texts <= 50:
        return dict(n_neighbors=10, n_components=5, metric='cosine', random_state=42)
    if n_texts <= 200:
        return dict(n_neighbors=12, n_components=5, metric='cosine', random_state=42)
    return dict(n_neighbors=min(15, max(8, n_texts//20)), n_components=5, metric='cosine', random_state=42)

def _auto_hdbscan_params(n_texts: int):
    mcs = max(3, n_texts // 40)  
    return dict(
        min_cluster_size=mcs,
        min_samples=max(2, mcs // 2),   
        metric="euclidean",
        cluster_selection_method="eom"
    )

def _merge_similar_clusters(embeddings: np.ndarray, labels: np.ndarray, sim_threshold: float = MERGE_CLUSTER_SIM_THRESHOLD) -> np.ndarray:
    uniq = sorted(set(labels) - {-1})
    if len(uniq) <= 1:
        return labels
    centroids = {lab: embeddings[labels == lab].mean(axis=0) for lab in uniq}
    merged_to = {}
    for i, l1 in enumerate(uniq):
        for l2 in uniq[i+1:]:
            sim = float(cosine_similarity([centroids[l1]], [centroids[l2]])[0][0])
            if sim >= sim_threshold:
                rep = min(l1, l2)
                other = max(l1, l2)
                merged_to[other] = rep
    def find_rep(x):
        while x in merged_to:
            x = merged_to[x]
        return x
    new_labels = labels.copy()
    for idx, lab in enumerate(labels):
        if lab == -1:
            continue
        new_labels[idx] = find_rep(lab)
    return new_labels

def _contiguous_labels(labels: np.ndarray) -> np.ndarray:
    mapping, next_id = {}, 0
    new = labels.copy()
    for i, lab in enumerate(labels):
        if lab == -1:
            new[i] = -1
            continue
        if lab not in mapping:
            mapping[lab] = next_id
            next_id += 1
        new[i] = mapping[lab]
    return new

# === Main Clustering Function ===

kw_model = KeyBERT()

def semantic_clustering_auto(texts, sentence_model):
    if not texts:
        return None, {}, {}, None, None

    # Stopwords gabungan
    from nltk.corpus import stopwords
    import nltk
    try:
        stopwords_id = set(stopwords.words('indonesian'))
    except LookupError:
        nltk.download('stopwords')
        stopwords_id = set(stopwords.words('indonesian'))
    stopwords_en = set(stopwords.words('english'))
    stopwords_custom = {'yg','ya','gak','nggak','banget','nih','sih','dong','aja',
                        'kayak','dulu','udah','lagi','bisa','akan','sama','ke','di','itu','ini'}
    all_stopwords = list(stopwords_id | stopwords_en | stopwords_custom)

    # Embedding
    embeddings = sentence_model.encode(texts, show_progress_bar=False, normalize_embeddings=True)

    # Dimensionality reduction (opsional dengan UMAP)
    if len(texts) >= 5:
        reducer = umap.UMAP(n_neighbors=15, n_components=5, metric="cosine", random_state=42)
        X_umap = reducer.fit_transform(embeddings)
        X_for_cluster = X_umap
    else:
        X_umap = None
        X_for_cluster = embeddings

    # === Clustering utama: HDBSCAN dengan cosine ===
    if len(texts) >= 5:
        clusterer = hdbscan.HDBSCAN(min_cluster_size=5,
                                    metric='cosine',
                                    cluster_selection_method='eom')
        labels = clusterer.fit_predict(embeddings)
    else:
        labels = np.zeros(len(texts), dtype=int)

    # Fallback kalau semua -1
    if all(lbl == -1 for lbl in labels):
        max_k = min(10, max(2, len(texts)//2))
        best_k, best_score, best_labels = 2, -1, None
        for k in range(2, max_k+1):
            km = KMeans(n_clusters=k, n_init=10, random_state=42)
            lbl = km.fit_predict(X_for_cluster)
            try:
                score = silhouette_score(X_for_cluster, lbl)
            except Exception:
                score = -1
            if score > best_score:
                best_k, best_score, best_labels = k, score, lbl
        labels = best_labels if best_labels is not None else np.zeros(len(texts), dtype=int)

    # === Merge cluster mirip (berdasarkan cosine centroid) ===
    cluster_ids = [c for c in set(labels) if c != -1]
    centroids = {}
    for c in cluster_ids:
        idxs = [i for i, l in enumerate(labels) if l == c]
        centroids[c] = np.mean(embeddings[idxs], axis=0)

    merged_map = {}
    used = set()
    for i, c1 in enumerate(cluster_ids):
        if c1 in used:
            continue
        merged_map[c1] = c1
        for c2 in cluster_ids[i+1:]:
            if c2 in used:
                continue
            sim = cosine_similarity([centroids[c1]], [centroids[c2]])[0,0]
            if sim > 0.8:  # threshold merge
                merged_map[c2] = c1
                used.add(c2)

    labels = np.array([merged_map.get(lbl, lbl) for lbl in labels])

    # === Keyword & contoh tiap cluster ===
    cluster_keywords, cluster_example = {}, {}
    cluster_texts_map = defaultdict(list)

    for idx, lbl in enumerate(labels):
        if lbl == -1:
            continue
        cluster_texts_map[lbl].append(texts[idx])

    for cluster_id, cluster_texts in cluster_texts_map.items():
        if len(cluster_texts) == 1:
            # pakai KeyBERT langsung
            kws = kw_model.extract_keywords(cluster_texts[0], keyphrase_ngram_range=(1,3), stop_words=all_stopwords, top_n=3)
            cluster_keywords[cluster_id] = [w for w,_ in kws]
            cluster_example[cluster_id] = cluster_texts[0]
        else:
            # TF-IDF untuk konsistensi
            vectorizer = TfidfVectorizer(max_features=2000,
                                         stop_words=all_stopwords,
                                         ngram_range=(1,3),
                                         min_df=1)
            tfidf_matrix = vectorizer.fit_transform(cluster_texts)
            feature_names = vectorizer.get_feature_names_out()
            mean_tfidf = np.asarray(tfidf_matrix.mean(axis=0)).flatten()
            sorted_idx = np.argsort(mean_tfidf)[::-1]
            top_keywords = [feature_names[i] for i in sorted_idx[:5]]
            if not top_keywords:  # fallback KeyBERT
                kws = kw_model.extract_keywords(" ".join(cluster_texts), keyphrase_ngram_range=(1,3),
                                                stop_words=all_stopwords, top_n=3)
                top_keywords = [w for w,_ in kws]
            cluster_keywords[cluster_id] = top_keywords

            # contoh representatif
            idxs = [i for i,l in enumerate(labels) if l == cluster_id]
            cluster_embeddings = embeddings[idxs]
            centroid = np.mean(cluster_embeddings, axis=0)
            sims = np.dot(cluster_embeddings, centroid) / (np.linalg.norm(cluster_embeddings, axis=1) * np.linalg.norm(centroid) + 1e-9)
            best_local = int(np.argmax(sims))
            cluster_example[cluster_id] = texts[idxs[best_local]]

    # === Hitung silhouette (opsional) ===
    silhouette = None
    try:
        valid_labels = [l for l in labels if l != -1]
        if len(set(valid_labels)) > 1:
            silhouette = silhouette_score(X_for_cluster, labels)
    except Exception:
        pass

    return labels, cluster_keywords, cluster_example, silhouette, X_umap

# === Topic Assignment & Keyword Extraction ===

def split_stuck_words(word: str) -> str:
    patterns = [
        (r'(dibatalkan)(retur)', r'\1 \2'),
        (r'(retur)(dibatalkan)', r'\1 \2'),
        (r'(bank)(terpotong)', r'\1 \2'),
        (r'(terpotong)(bank)', r'\1 \2'),
        (r'(verifikasi)(pembayaran)', r'\1 \2'),
        (r'(pembayaran)(dana)', r'\1 \2')
    ]
    for pat, repl in patterns:
        word = re.sub(pat, repl, word)
    return word

def get_main_keyword(keywords, stop_words_id, top_n=2):
    if not keywords:
        return None

    generic_words = {
            "baru", "topik", "ready", "izin", "silakan", "hai", "halo", 
            "tolong", "bantu", "tanya", "nanya", "kak", "min", "gan", "om",
            "pak", "bu", "cs", "tokla", "customer", "service", "baru","topik",
            "up","ngalami","proses","info","solusi","kendala","admin","toko",
            "tokla","pakai","bantu","mohon","permohonan","data", "ya","loh",
            "kok","tahi","anjing","brow","wkwk","haha","heh","hmm","mantap",
            "sip","udah","dong","nih"
            }

    tokens = []
    for kw in keywords:
        if not kw:
            continue
        kw_clean = kw.strip().lower()
        kw_clean = split_stuck_words(kw_clean)

        for tok in kw_clean.split():
            if (tok not in stop_words_id and 
                tok not in generic_words and 
                len(tok) > 2):
                tokens.append(tok)

    tokens = list(dict.fromkeys(tokens))
    if not tokens:
        return keywords[0].lower()
    tokens = tokens[:top_n]
    return "_".join(tokens)

def clean_main_keyword(keyword: str) -> str:
    if not keyword:
        return ""
    keyword = keyword.lower().strip()
    keyword = re.sub(r'@\w+', '', keyword)
    keyword = re.sub(r'\b(cs[_-]*\w+)\b', '', keyword)
    keyword = re.sub(r'[^a-z]', ' ', keyword)
    tokens = keyword.split()
    tokens = list(dict.fromkeys(tokens))
    if not tokens:
        return ""
    return "_".join(tokens)

def generate_topic_name(cluster_id, cluster_keywords, cluster_example, stop_words_id):
    main_keyword = get_main_keyword(cluster_keywords.get(cluster_id, []), stop_words_id, top_n=2)
    main_keyword = clean_main_keyword(main_keyword)

    if not main_keyword or len(main_keyword) < 3:
        example_text = cluster_example.get(cluster_id, "")
        if example_text:
            kws = kw_model.extract_keywords(example_text, keyphrase_ngram_range=(1,3),
                                            stop_words=stop_words_id, top_n=1)
            if kws:
                main_keyword = kws[0][0].replace(" ", "_")

    if not main_keyword:
        example_text = cluster_example.get(cluster_id, "")
        if len(example_text.split()) > 5:
            main_keyword = "_".join(example_text.split()[:5])
        else:
            main_keyword = example_text.replace(" ", "_")

    return f"(new) {main_keyword}"

# === Full Analysis Pipeline ===

def analyze_all_topics(df_questions: pd.DataFrame, sentence_model):
    if df_questions is None or df_questions.empty:
        st.warning("Tidak ada data pertanyaan yang bisa dianalisis.")
        return

    topik_counter = Counter()
    topik_per_pertanyaan = []
    unclassified_texts, unclassified_indices = [], []

    for idx, row in df_questions.iterrows():
        text = row.get('text', '')
        if pd.isna(text) or not isinstance(text, str):
            continue
        text_lc = text.lower()
        found_topik = []
        for topik, patterns in topik_keywords.items():
            if isinstance(patterns, list) and patterns and isinstance(patterns[0], list):
                if any(all(p in text_lc for p in group) for group in patterns):
                    found_topik.append(topik)
            else:
                if isinstance(patterns, list) and any(p in text_lc for p in patterns):
                    found_topik.append(topik)
        if not found_topik:
            unclassified_texts.append(text)
            unclassified_indices.append(idx)
            selected_topik = 'lainnya'
        else:
            spesifik_topik = [t for t in found_topik if t != 'bantuan_umum']
            selected_topik = spesifik_topik[0] if spesifik_topik else found_topik[0]
        topik_counter[selected_topik] += 1
        topik_per_pertanyaan.append({'topik': selected_topik, 'pertanyaan': text, 'index': idx})

    # Centroid assignment
    assigned_map = {}
    remaining_local_idxs = list(range(len(unclassified_texts)))
    if unclassified_texts:
        try:
            assigned_map, remaining_local_idxs = assign_by_topic_centroid(
                unclassified_texts, unclassified_indices, topik_per_pertanyaan, sentence_model,
                sim_threshold=CENTROID_SIMILARITY_THRESHOLD, min_examples=MIN_EXAMPLES_FOR_CENTROID)
        except Exception:
            assigned_map, remaining_local_idxs = {}, list(range(len(unclassified_texts)))

    if assigned_map:
        for local_i, info in assigned_map.items():
            orig_idx = unclassified_indices[local_i]
            topic = info['topic']
            if topik_counter.get('lainnya', 0) > 0:
                topik_counter['lainnya'] -= 1
            topik_counter[topic] += 1
            for item in topik_per_pertanyaan:
                if item['index'] == orig_idx:
                    item['topik'] = topic
                    break

    # Clustering semantik otomatis
    remaining_texts = [unclassified_texts[i] for i in remaining_local_idxs]
    remaining_indices = [unclassified_indices[i] for i in remaining_local_idxs]

    new_topics_found = {}
    if len(remaining_texts) >= MIN_CLUSTER_SHOW:
        st.subheader("🔍 Mendeteksi Topik Baru")
        with st.spinner("Menghitung embedding & clustering..."):
            labels, cluster_keywords, cluster_example, silhouette, X_umap = semantic_clustering_auto(
                remaining_texts, sentence_model
            )
        if labels is not None and len(labels) == len(remaining_texts):
            cluster_counts = Counter(labels)
            for cluster_id, count in cluster_counts.items():
                if cluster_id == -1:
                    continue

                # === pakai generate_topic_name ===
                new_topic_name = generate_topic_name(
                    cluster_id,
                    cluster_keywords,
                    cluster_example,
                    stop_words_id
                )

                new_topics_found[cluster_id] = {
                    'name': new_topic_name,
                    'keywords': cluster_keywords.get(cluster_id, []),
                    'count': count,
                    'texts': [],
                    'example': cluster_example.get(cluster_id, "")
                }

            for local_i, label in enumerate(labels):
                if label in new_topics_found:
                    orig_idx = remaining_indices[local_i]
                    orig_text = remaining_texts[local_i]
                    new_topics_found[label]['texts'].append(orig_text)
                    topik_counter[new_topics_found[label]['name']] += 1
                    for item in topik_per_pertanyaan:
                        if item['index'] == orig_idx and item['topik'] == 'lainnya':
                            item['topik'] = new_topics_found[label]['name']
                    if topik_counter.get('lainnya', 0) > 0:
                        topik_counter['lainnya'] -= 1
                        
            if new_topics_found:
                st.success(f"✅ Ditemukan {len(new_topics_found)} potensi topik baru!")
    else:
        st.info(f"Jumlah teks yang tidak terklasifikasi ({len(remaining_texts)}) kurang untuk clustering.")

    # Update ulang counter biar sinkron
    topik_counter = Counter([item['topik'] for item in topik_per_pertanyaan])

    # Ringkasan akhir
    st.subheader("📊 Ringkasan Topik Teratas")
    if not topik_counter:
        st.write("Tidak ada topik yang teridentifikasi.")
        return

    summary_data = pd.DataFrame([{"Topik": t, "Jumlah Pertanyaan": c} for t, c in topik_counter.most_common()])
    st.dataframe(summary_data, use_container_width=True)

    st.subheader("📝 Detail Pertanyaan per Topik")
    mapping = defaultdict(list)
    for item in topik_per_pertanyaan:
        mapping[item['topik']].append(item['pertanyaan'])
    for topik, count in topik_counter.most_common():
        with st.expander(f"Topik: {topik} ({count} pertanyaan)"):
            for q in mapping.get(topik, [])[:200]:
                st.markdown(f"- *{q.strip()}*")

# === Streamlit UI ===

group = st.text_input("Masukkan username atau ID grup Telegram:", "@contohgroup")
today = datetime.now(wib).date()
week_ago = today - timedelta(days=7)

col1, col2 = st.columns(2)
with col1:
    start_date_scrape = st.date_input("Scrape dari tanggal", week_ago, format="YYYY-MM-DD")
with col2:
    end_date_scrape = st.date_input("Scrape sampai tanggal", today, format="YYYY-MM-DD")

run_button = st.button("🚀 Mulai Proses dan Analisis", type="primary")

if run_button:
    model_name = 'paraphrase-multilingual-mpnet-base-v2'
    with st.spinner(f"Memuat model sentence-transformers: {model_name} ..."):
        sentence_model = load_sentence_model(model_name)
        if not group or group.strip() == "" or group.strip() == "@contohgroup":
            st.warning("⚠️ Mohon isi nama grup Telegram yang valid terlebih dahulu.")
            st.stop()
        start_dt = datetime.combine(start_date_scrape, datetime.min.time()).replace(tzinfo=wib)
        end_dt = datetime.combine(end_date_scrape, datetime.max.time()).replace(tzinfo=wib)
        with st.spinner("Mengambil pesan dari Telegram..."):
            df_all = asyncio.run(scrape_messages_iter(group, start_dt, end_dt))
        if df_all is None or df_all.empty:
            st.error("Gagal mengambil data atau tidak ada data yang ditemukan dalam rentang tanggal.")
            st.stop()
        else:
            st.success(f"✅ Berhasil mengambil {len(df_all)} pesan mentah.")

    st.header("📈 Analisis Topik dari Semua Pertanyaan")
    with st.spinner("Membersihkan data dan mencari pertanyaan..."):
        df_all['text'] = df_all['text'].astype(str)
        df_all['text'] = df_all['text'].str.replace(r'http\S+|www\.\S+', '', regex=True)
        df_all['text'] = df_all['text'].str.strip()
        # Koreksi ejaan tidak baku ke baku (opsional jika file tersedia)
        if spelling_correction:
            df_all['text'] = df_all['text'].apply(lambda x: correct_spelling(x, spelling_correction))
        # Hapus baris dari sender tertentu
        df_all = df_all[~df_all['sender_name'].isin(['CS TokoLadang', 'Eko | TokLa', 'Vava'])]
        # Hapus kalimat tidak penting
        df_all = df_all[~df_all['text'].apply(is_unimportant_sentence)]
        # deduplicate (sender, text, date)
        dedup_cols = ['sender_id', 'text', 'date'] if 'sender_id' in df_all.columns else ['sender_name', 'text', 'date']
        df_all = df_all.drop_duplicates(subset=dedup_cols, keep='first').reset_index(drop=True)
        # detect questions
        df_all['is_question'] = df_all['text'].apply(is_question_like)
        df_questions = df_all[df_all['is_question']].copy()

    tab1, tab2 = st.tabs(["❓ Daftar Pertanyaan", "📊 Analisis Topik"])
    with tab1:
        st.subheader(f"❓ Ditemukan {len(df_questions)} Pesan Pertanyaan" if isinstance(df_questions, pd.DataFrame) else "❓ Ditemukan 0 Pesan Pertanyaan")
        if not df_questions.empty:
            display_cols = [c for c in ['date','sender_name','text'] if c in df_questions.columns]
            st.dataframe(df_questions[display_cols], use_container_width=True)
        else:
            st.info("Tidak ada pesan yang terdeteksi sebagai pertanyaan pada periode ini.")
    with tab2:
        analyze_all_topics(df_questions, sentence_model)

    st.markdown("---")
    st.success("Analisis Selesai!")

st.caption("Catatan: Pastikan API ID/API Hash Telegram benar dan akun yang digunakan memiliki akses ke grup. Model sentence-transformers akan diunduh saat pertama kali dijalankan. Hindari membagikan API credentials di publik.")
