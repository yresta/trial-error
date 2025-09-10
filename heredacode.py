import re
import os
from typing import List, Dict, Any, Tuple
from functools import lru_cache

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
import nltk
from typing import Callable, Optional, List
from collections import Counter

# Setup NLTK
nltk.download('punkt', quiet=True)

# Configuration / Constants
DEFAULT_MODEL_NAME = 'paraphrase-multilingual-mpnet-base-v2'
MIN_CLUSTER_SIZE = 8
MAX_RECURSIVE_DEPTH = 3
EMBEDDING_BATCH_SIZE = 128

IND_STOPWORDS = set("""
yang dan di ke dari untuk dengan pada oleh dalam atas sebagai adalah ada itu ini atau tidak sudah belum bisa akan harus sangat juga karena jadi kalau namun tapi serta agar supaya sehingga maka lalu kemudian setelah sebelum hingga sampai pun saya kak bapak ibu pak
""".split())

QUESTION_WORDS = set("""apa siapa kapan mengapa kenapa bagaimana gimana dimana apakah mana dimana saja kenapa ya tidak""".split())

FOCUS_KEYWORDS = {
    "dana","pencairan","cair","rekening","saldo","uang",
    "pembayaran","bayar","verifikasi","otp","autentikasi",
    "login","akses","akun","aplikasi",
    "produk","barang","pesanan","kurir","pengiriman","retur",
    "ppn","pajak","modal","talangan","data","toko","upload","unggah"
}

# Sentence model (cached globally)
@lru_cache(maxsize=1)
def get_sentence_model(model_name: str = DEFAULT_MODEL_NAME):
    return SentenceTransformer(model_name)

def get_sentence_embeddings(texts: List[str], model=None, model_name: str = DEFAULT_MODEL_NAME) -> np.ndarray:
    if not texts:
        return np.zeros((0, 384))
    if model is None:
        model = get_sentence_model(model_name)
    return np.asarray(model.encode(
        texts,
        show_progress_bar=False,
        batch_size=EMBEDDING_BATCH_SIZE,
        normalize_embeddings=True
    ))

# Spelling correction
def load_spelling_corrections(path: str) -> Dict[str, str]:
    if not os.path.exists(path):
        return {}
    try:
        df = pd.read_csv(path)
        if 'tidak_baku' in df.columns and 'kata_baku' in df.columns:
            return dict(zip(df['tidak_baku'].astype(str), df['kata_baku'].astype(str)))
    except Exception:
        pass
    return {}

def build_spelling_pattern(corrections: Dict[str, str]):
    if not corrections:
        return None
    pattern = re.compile(r'\b(' + '|'.join(re.escape(k) for k in corrections.keys()) + r')\b')
    return lambda text: pattern.sub(lambda m: corrections[m.group(0)], text)

spelling = load_spelling_corrections("kata_baku.csv")
apply_spelling = build_spelling_pattern(spelling)

# Cleaning & filtering
def is_unimportant_sentence(text: str) -> bool:
    if not isinstance(text, str) or not text.strip():
        return True
    txt = text.strip().lower()
    unimportant_phrases = {
        "siap","noted","oke","ok","baik","sip","thanks","makasih","terima kasih",
        "iya","ya","oh","ohh","mantap","mantul","keren","wah","hebat",
        "anggota baru","selamat berlibur"
    }
    if txt in unimportant_phrases:
        return True
    if len(txt.split()) <= 2 and not any(q in txt for q in QUESTION_WORDS):
        return True
    return False

def clean_text_for_clustering(text: Any, spelling_fn: Optional[Callable[[str], str]] = None) -> str:
    if not isinstance(text, str):
        return ''
    text = text.lower()
    text = re.sub(r'http\S+|www\.\S+', ' ', text)
    text = re.sub(r'@\w+', ' ', text)
    text = re.sub(r'#\w+', ' ', text)
    if spelling_fn:  
        text = spelling_fn(text)
    text = re.sub(r'[^0-9a-z\s\?%.,]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Clustering
def cluster_texts_embeddings(X: np.ndarray, num_clusters: int = 8) -> Tuple[np.ndarray, np.ndarray]:
    if X.shape[0] == 0:
        return np.array([]), np.array([])
    k = min(max(1, int(num_clusters)), X.shape[0])
    if k == 1:
        labels = np.zeros(X.shape[0], dtype=int)
        centers = X.mean(axis=0, keepdims=True)
        return labels, centers
    kmeans = MiniBatchKMeans(n_clusters=k, random_state=0, n_init=10)
    labels = kmeans.fit_predict(X)
    return labels, kmeans.cluster_centers_

def find_optimal_clusters(texts: List[str], min_k: int = 2, max_k: int = 10) -> int:
    if not texts or len(texts) < 2:
        return 1
    X = get_sentence_embeddings(texts)
    best_k, best_score = 1, -1
    max_k = min(max_k, len(texts))
    for k in range(min_k, max_k+1):
        try:
            labels = MiniBatchKMeans(n_clusters=k, random_state=0, n_init=5).fit_predict(X)
            if len(set(labels)) == 1:
                continue
            score = silhouette_score(X, labels)
        except Exception:
            score = -1
        if score > best_score:
            best_k, best_score = k, score
    return best_k

def recursive_clustering(texts: List[str], embeddings=None, 
                         min_cluster_size: int = MIN_CLUSTER_SIZE,
                         max_depth: int = MAX_RECURSIVE_DEPTH,
                         depth: int = 0) -> List[List[str]]:
    if embeddings is None:
        embeddings = get_sentence_embeddings(texts)
    if depth >= max_depth or len(texts) <= min_cluster_size:
        return [texts]
    num_clusters = find_optimal_clusters(texts, min_k=2, max_k=min(10, len(texts)//min_cluster_size+1))
    if num_clusters <= 1:
        return [texts]

    labels, _ = cluster_texts_embeddings(embeddings, num_clusters=num_clusters)
    clusters = []
    for cid in set(labels):
        members = [texts[i] for i, lbl in enumerate(labels) if lbl == cid]
        sub_embeds = embeddings[labels == cid]
        if len(members) <= min_cluster_size or depth+1 >= max_depth:
            clusters.append(members)
        else:
            clusters.extend(recursive_clustering(members, sub_embeds, min_cluster_size, max_depth, depth+1))
    return clusters

def extract_representative_keywords(texts: List[str], top_n: int = 2, max_features: int = 1000) -> List[str]:
    if not texts:
        return ["Topik"]
    try:
        vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1,2),
            stop_words=IND_STOPWORDS
        )
        X = vectorizer.fit_transform(texts)
        scores = np.asarray(X.sum(axis=0)).ravel()
        features = vectorizer.get_feature_names_out()
        sorted_idx = np.argsort(scores)[::-1]
        keywords: List[str] = []
        for i in sorted_idx:
            word = split_stuck_words(features[i])
            tokens = word.split()
            if not all(is_meaningful_word(tok) for tok in tokens):
                continue
            if any(tok in FOCUS_KEYWORDS for tok in tokens):
                keywords.append(word)
            elif not keywords:
                keywords.append(word)
            if len(keywords) >= top_n:
                break
        return keywords if keywords else ["Topik"]
    except Exception:
        tokens = ' '.join(texts).split()
        freqs = pd.Series(tokens).value_counts()
        return [w for w in freqs.index if is_meaningful_word(w)][:top_n] or ["Topik"]

# Keyword extraction
def split_stuck_words(word: str) -> str:
    patterns = [
        (r'(verifikasi)(pembayaran)', r'\1 \2'),
        (r'(pembayaran)(dana)', r'\1 \2'),
        (r'(dana)(masuk)', r'\1 \2'),
        (r'(modal)(talangan)', r'\1 \2')
    ]
    for pat, repl in patterns:
        word = re.sub(pat, repl, word)
    return word

CUSTOM_IGNORE = {"kak","min","ya","kah","buk","pak", "min","om","yaa","apaaaaa","omom","loh","lah","deh"}

def is_meaningful_word(word: str) -> bool:
    word = word.strip()
    if len(word) <= 3 or word.isdigit():
        return False
    if word in IND_STOPWORDS or word in QUESTION_WORDS or word in CUSTOM_IGNORE:
        return False
    return True

def assign_topic_names(cluster_texts_list: List[str], labels: np.ndarray, top_n: int = 2) -> Dict[int, str]:
    topic_names: Dict[int, str] = {}
    for cluster_id in set(labels.tolist()):
        members = [cluster_texts_list[i] for i, lbl in enumerate(labels) if lbl == cluster_id]
        kws = extract_representative_keywords(members, top_n=top_n) if members else ["Topik"]
        topic_names[cluster_id] = ' '.join(kws)
    return topic_names

spelling = load_spelling_corrections("kata_baku.csv")
apply_spelling = build_spelling_pattern(spelling)

# Normalisasi nama topik otomatis
CUSTOM_IGNORE = {
    "min","admin","kok","kah","map","kira","pernah","ya","izin",
    "sih","loh","lah","deh","dong","nih","aja","saja","selamat",
    "new"
}

TOPIC_IGNORE = {
    "seperti","lagi","adakah","kak","dong","nih","aja","saja",
    "min","admin","map","pernah","ya","izin","sih","loh","lah","deh"
}

GENERIC_IGNORE = {
    "apa","ini","itu","lagi","sudah","bisa","tidak","belum","akan",
    "ada","dapat","dengan","dari","untuk","pada","dilaporkan","kendala"
}

# ----- DEPRECATED: replaced by improved version below ----
# def normalize_topic_name(topic: str, cluster_texts: list[str] = None) -> Optional[str]:
#     clean = re.sub(r"[^0-9a-zA-Z\s]", " ", str(topic))
#     tokens = [t for t in clean.lower().split() 
#               if t not in CUSTOM_IGNORE 
#               and t not in TOPIC_IGNORE 
#               and t not in GENERIC_IGNORE 
#               and len(t) > 2]

#     QUESTION_WORDS = {
#         "apa","apakah","bagaimana","gimana","kenapa","mengapa",
#         "kok","ya","yang","maksudnya","gitu","jadi","kah","kan","harus"
#     }
#     GENERIC_BAD = {
#         "tolong","info","informasi","mohon","tanya","baru","harap","sudah","belum",
#         "iya","ok","oke","baik","nah","loh","yah","min","admin","kak","pak","bu",
#         "bro","gan","dong","lah","toko","akun","status","sekolah"
#     }
#     LOCATION_WORDS = {
#         "jakarta","bandung","surabaya","semarang","medan","palembang","makassar",
#         "jogja","yogyakarta","solo","depok","bogor","tangerang","bekasi","bali",
#         "ntb","ntt","papua","ambon","maluku","aceh","riau","lampung","banten",
#         "jabar","jateng","jatim","jambi","bengkulu","sumbar","sumut","sumsel",
#         "kalbar","kalteng","kalsel","kaltim","kalut","sulsel","sulbar",
#         "sulteng","sultra","gorontalo","kebumen","cilacap","magelang","tegal"
#     }

#     # kalau semua token cuma kata tanya / generik / lokasi → drop
#     if tokens and all(
#         t in QUESTION_WORDS or t in GENERIC_BAD or t in LOCATION_WORDS
#         for t in tokens
#     ):
#         return "lainnya"

#     # tambahan RULE: kalau hasil topik hanya 1–2 token & termasuk GENERIC_BAD → drop
#     if len(tokens) <= 2 and any(t in GENERIC_BAD for t in tokens):
#         return "lainnya"

#     focus_hits = [t for t in tokens if t in FOCUS_KEYWORDS]
#     if focus_hits:
#         return f"(new) {' '.join(focus_hits[:2]).title()}"

#     if tokens:
#         return f"(new) {' '.join(tokens[:2]).title()}"

#     # fallback cluster_texts
#     if cluster_texts:
#         words = []
#         for txt in cluster_texts:
#             words.extend(re.sub(r"[^0-9a-zA-Z\s]", " ", txt).lower().split())
#         counter = Counter([
#             w for w in words 
#             if len(w) > 2 
#             and w not in CUSTOM_IGNORE 
#             and w not in TOPIC_IGNORE 
#             and w not in GENERIC_IGNORE
#         ])
#         if counter:
#             top_two = [w.title() for w, _ in counter.most_common(2)]
#             # validasi ulang
#             if all(w.lower() in QUESTION_WORDS or w.lower() in GENERIC_BAD or w.lower() in LOCATION_WORDS for w in top_two):
#                 return "lainnya"
#             return f"(new) {' '.join(top_two)}"

#     return "lainnya"

# def auto_topic_name(texts: List[str], top_n: int = 2) -> str:
#     kws = extract_representative_keywords(texts, top_n=top_n)
#     clean = re.sub(r"[^0-9a-zA-Z\s]", " ", " ".join(kws))
#     tokens = [t for t in clean.lower().split() if t not in CUSTOM_IGNORE and len(t) > 2]

#     if not tokens or all(t in IND_STOPWORDS for t in tokens):
#         return "lainnya"

#     return " ".join(tokens[:3]).title()
#----------------------------------------------------------------------


def auto_topic_name(texts: List[str], top_n: int = 3) -> str:
    """Generate nama topik otomatis dengan label (new), hindari duplikasi 'Lainnya'."""
    kws = extract_representative_keywords(texts, top_n=top_n)
    clean = re.sub(r"[^0-9a-zA-Z\s]", " ", " ".join(kws))
    tokens = [t for t in clean.lower().split() if len(t) > 2]

    QUESTION_BAD = {
        "apa","apakah","bagaimana","gimana","kenapa","mengapa","kok","ya","iya",
        "berapakah","berapa","dimana","mana","kah","kan","kira","seperti"
    }
    GENERIC_BAD = {
        "tolong","info","informasi","mohon","baru","harap","sudah","belum","ok",
        "oke","baik","nah","yah","min","admin","kak","pak","bu","bro","gan","dong",
        "lah","nih","aja","saja","teman","kita","halaman","awal","siang","pembeli",
        "new"
    }

    # filter token
    tokens = [t for t in tokens if t not in IND_STOPWORDS
                                and t not in CUSTOM_IGNORE
                                and t not in TOPIC_IGNORE
                                and t not in GENERIC_IGNORE
                                and t not in QUESTION_BAD
                                and t not in GENERIC_BAD
                                and len(t) > 2]

    # === Fokus keywords ===
    focus_hits = [t for t in tokens if t in FOCUS_KEYWORDS]
    if focus_hits:
        name = " ".join(w.title() for w in focus_hits[:2])
        return name if name.lower() == "Lainnya" else "(new) " + name

    # === Kalau ada token valid ===
    if tokens:
        name = " ".join(tokens[:3]).title()
        return name if name.lower() == "Lainnya" else "(new) " + name

    # === Fallback: pakai kata paling sering di cluster ===
    words = []
    for txt in texts:
        words.extend(re.sub(r"[^0-9a-zA-Z\s]", " ", txt).lower().split())

    counter = Counter([w for w in words if len(w) > 3
                       and w not in QUESTION_BAD
                       and w not in GENERIC_BAD])

    if counter:
        top_two = [w.title() for w, _ in counter.most_common(2)]
        if top_two:
            name = " ".join(top_two)
            return name if name.lower() == "Lainnya" else "(new) " + name

    # === Kalau semua gagal ===
    return "Lainnya"

def normalize_topic_name(topic: str, cluster_texts: list[str] = None) -> str:
    """Normalisasi nama topik (supaya konsisten)."""
    if cluster_texts:
        return auto_topic_name(cluster_texts)
    return auto_topic_name([topic])

def is_bad_cluster(texts: List[str]) -> bool:
    if not texts or len(texts) <= 1:
        return True
    kws = extract_representative_keywords(texts, top_n=2)
    if kws == ["Topik"]:
        return True
    # Jika semua pesan sangat berbeda (misal, similarity antar embeddings rendah)
    embeds = get_sentence_embeddings(texts)
    if embeds.shape[0] > 1:
        sims = np.dot(embeds, embeds.T)
        avg_sim = (np.sum(sims) - np.trace(sims)) / (embeds.shape[0]**2 - embeds.shape[0])
        if avg_sim < 0.25:  # threshold bisa diubah
            return True
    return False

GENERIC_TOPICS = {
    "knapa","kenapa","toko","bank","status","bertanya","siang",
    "kira","pembeli","lainnya"
}

def merge_and_rename_clusters(df: pd.DataFrame, similarity_threshold: float = 0.6) -> pd.DataFrame:
    """
    Gabungkan cluster yang terlalu generik atau mirip dengan cluster lain.
    - df: DataFrame dengan kolom ['topic', 'question']
    - similarity_threshold: ambang batas similarity antar nama topik
    """
    # Ambil nama topik unik
    topics = df['topic'].unique().tolist()

    # TF-IDF vektor untuk nama topik
    vectorizer = TfidfVectorizer().fit(topics)
    topic_vecs = vectorizer.transform(topics)
    sim_matrix = cosine_similarity(topic_vecs)

    # mapping cluster lama -> cluster baru
    topic_mapping = {}

    for i, t in enumerate(topics):
        clean_t = t.lower().strip()

        # kalau nama terlalu generik → cari tetangga mirip
        if clean_t in GENERIC_TOPICS or len(clean_t) < 4:
            # cari topik lain dengan similarity tertinggi
            sims = list(enumerate(sim_matrix[i]))
            sims = sorted(sims, key=lambda x: x[1], reverse=True)
            for j, score in sims:
                if j != i and score >= similarity_threshold:
                    topic_mapping[t] = topics[j]  # merge ke tetangga mirip
                    break
            else:
                topic_mapping[t] = "Lainnya"
        else:
            topic_mapping[t] = t

    # Apply mapping ke DataFrame
    df['topic'] = df['topic'].map(topic_mapping)

    return df

# Integrasi keyword + clustering (FULL FIXED)
def integrate_clustering_with_keywords(df: pd.DataFrame,
                                       topik_keywords: Dict[str, Any],
                                       spelling_corrections: Dict[str, str] = None,
                                       min_cluster_size: int = MIN_CLUSTER_SIZE,
                                       max_recursive_depth: int = MAX_RECURSIVE_DEPTH,
                                       num_auto_clusters: int = 15
                                       ) -> pd.DataFrame:
    df = df.copy()
    df['original_index'] = df.index
    df['processed_text'] = df['text'].apply(lambda t: clean_text_for_clustering(t, apply_spelling))
    df['is_unimportant'] = df['processed_text'].apply(is_unimportant_sentence)
    df = df[~df['is_unimportant']].reset_index(drop=True)

    keyword_categorized, remaining = [], []
    for _, row in df.iterrows():
        idx, txt = int(row['original_index']), row['processed_text']
        matched = []
        for topik, patterns in topik_keywords.items():
            if not patterns:
                continue
            if isinstance(patterns, list) and patterns and isinstance(patterns[0], list):
                if any(all(pat in txt for pat in conj) for conj in patterns):
                    matched.append(topik)
            else:
                if any(pat in txt for pat in (patterns if isinstance(patterns, list) else [patterns])):
                    matched.append(topik)
        if matched:
            spesifik = [t for t in matched if t != 'bantuan_umum']
            # Topik hasil keyword → tidak pakai (new)
            keyword_categorized.append((idx, spesifik[0] if spesifik else matched[0]))
        else:
            remaining.append((idx, txt))

    auto_categorized = []
    if remaining:
        rem_texts = [t for _, t in remaining]
        rem_idxs = [i for i, _ in remaining]

        grouped_clusters = recursive_clustering(
            rem_texts,
            min_cluster_size=min_cluster_size,
            max_depth=max_recursive_depth
        )

        for group in grouped_clusters:
            if not group:
                continue
            indices_in_remaining = [i for i, txt in enumerate(rem_texts) if txt in group and txt is not None]
            # Mark sudah dipakai
            for pos in indices_in_remaining:
                rem_texts[pos] = None
            orig_indices = [rem_idxs[pos] for pos in indices_in_remaining]

            if is_bad_cluster(group):
                # cluster jelek → "lainnya"
                auto_categorized.extend((i, "Lainnya") for i in orig_indices)
            else:
                if len(group) <= min_cluster_size:
                    # cluster kecil → generate nama dari keywords
                    topic_name = auto_topic_name(group, top_n=2)
                    topic_name = normalize_topic_name(topic_name)  # tambahkan (new)
                    auto_categorized.extend((i, topic_name) for i in orig_indices)
                else:
                    # cluster cukup besar → pecah pakai embeddings
                    X = get_sentence_embeddings(group)
                    k = max(1, len(group) // min_cluster_size)
                    labels, _ = cluster_texts_embeddings(X, num_clusters=k)
                    topic_names = assign_topic_names(group, labels, top_n=2)
                    for j, lbl in enumerate(labels.tolist()):
                        if j < len(orig_indices):
                            normalized_name = normalize_topic_name(topic_names[lbl])  # tambahkan (new)
                            auto_categorized.append((orig_indices[j], normalized_name))

    # Gabungkan hasil keyword + auto
    mapping = {}
    for idx, topic in keyword_categorized + auto_categorized:
        if idx not in mapping:
            mapping[idx] = topic

    df['final_topic'] = df['original_index'].apply(lambda i: mapping.get(i, 'Lainnya'))
    return df[['original_index', 'text', 'processed_text', 'final_topic']]

# Example Run
if __name__ == '__main__':
    sample_data = {
        'text': [
            'Bagaimana cara verifikasi toko saya?',
            'Dana saya belum masuk rekening, tolong dicek.',
            'Kapan pencairan dana gelombang 2?',
            'Saya tidak bisa login ke aplikasi, ada masalah apa?',
            'Bagaimana cara upload produk massal?',
            'Ada kendala akses web, tidak bisa dibuka.',
            'Apakah ada info terbaru tentang pajak PPN?',
            'Saya ingin bertanya tentang etika penggunaan platform.',
            'Pembayaran saya pending, mohon dibantu.',
            'Barang yang dikirim rusak, bagaimana ini?',
            'Ini topik baru yang belum ada di list keyword sama sekali.',
            'Pesan ini juga tentang topik baru yang mirip dengan yang sebelumnya.',
            'Ini adalah pesan yang sangat berbeda dan harusnya jadi topik baru lagi.',
            'Saya butuh bantuan umum, tidak spesifik.',
            'Verifikasi pembayaran saya gagal, bagaimana solusinya?',
            'Tanda tangan elektronik saya tidak berfungsi.',
            'Bagaimana cara mengubah data toko?',
            'Pengajuan modal saya dibatalkan, kenapa ya?',
            'Ada masalah dengan autentikasi OTP.',
            'Saya tidak bisa mengunggah gambar produk.',
            'Kapan kurir akan menjemput barang?',
            'Bagaimana cara menggunakan fitur siplah?',
            'Status pesanan saya masih menggantung.',
            'Ada pertanyaan umum lainnya.'
        ]
    }
    df_sample = pd.DataFrame(sample_data)
    topik_keywords_example = {
        "verifikasi_toko": [["verifikasi", "toko"]],
        "dana_belum_masuk": [["dana", "belum", "masuk"]],
        "jadwal_cair_dana": [["kapan", "cair"], ["kapan", "pencairan"]],
        "kendala_akses": [["tidak", "bisa", "login"], ["kendala", "akses"]],
        "kendala_upload": [["upload", "produk"], ["unggah", "gambar"]],
        "pajak": [["pajak", "ppn"], ["ppn"]],
        "etika_penggunaan": [["etika", "penggunaan"]],
        "pembayaran_dana": ["pembayaran", "pending"],
        "pengiriman_barang": ["barang", "rusak"],
        "bantuan_umum": ["bantuan", "umum", "tanya"]
    }
    spelling = load_spelling_corrections('kata_baku.csv')
    print('Running optimized clustering + keyword matching...')
    result = integrate_clustering_with_keywords(df_sample.copy(), topik_keywords_example, spelling_corrections=spelling)
    print(result.to_string(index=False))