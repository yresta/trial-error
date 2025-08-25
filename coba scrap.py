import streamlit as st
from telethon import TelegramClient
from telethon.tl.functions.messages import GetHistoryRequest
from datetime import datetime
from zoneinfo import ZoneInfo  # Python ≥ 3.9
import pandas as pd
import asyncio
import nest_asyncio

# Agar async di Streamlit jalan
nest_asyncio.apply()

# Zona waktu WIB
wib = ZoneInfo("Asia/Jakarta")

st.title("Telegram Group Message Scraper 📩")

# === Input akun ===
api_id = st.text_input("Masukkan API ID", type="password")
api_hash = st.text_input("Masukkan API Hash", type="password")
session_name = st.text_input("Nama session", value="session_scraper")

# === Input target group ===
group = st.text_input("Username atau ID grup (contoh: @namagrup)")

# === Input tanggal ===
col1, col2 = st.columns(2)
with col1:
    start_str = st.date_input("Dari tanggal")
with col2:
    end_str = st.date_input("Sampai tanggal")

# Tombol untuk mulai scraping
if st.button("Scrape Pesan"):
    if not all([api_id, api_hash, group]):
        st.error("API ID, API Hash, dan group harus diisi!")
    else:
        # Konversi tanggal ke datetime WIB
        start_date = datetime.combine(start_str, datetime.min.time()).replace(tzinfo=wib)
        end_date = datetime.combine(end_str, datetime.max.time()).replace(tzinfo=wib)

        client = TelegramClient(session_name, int(api_id), api_hash)

        async def scrape():
            await client.start()
            entity = await client.get_entity(group)
            all_messages = []
            offset_id = 0
            limit_per_batch = 100

            while True:
                history = await client(GetHistoryRequest(
                    peer=entity,
                    limit=limit_per_batch,
                    offset_id=offset_id,
                    offset_date=None,
                    max_id=0,
                    min_id=0,
                    add_offset=0,
                    hash=0
                ))
                messages = history.messages
                if not messages:
                    break

                for msg in messages:
                    if not msg.message or not msg.date:
                        continue

                    msg_date_wib = msg.date.astimezone(wib)

                    if msg_date_wib < start_date:
                        break

                    if start_date <= msg_date_wib <= end_date:
                        sender_id = getattr(msg.from_id, 'user_id', None) if msg.from_id else None
                        all_messages.append({
                            'id': msg.id,
                            'sender_id': sender_id,
                            'text': msg.message,
                            'date': msg_date_wib.strftime("%Y-%m-%d %H:%M:%S")
                        })

                offset_id = messages[-1].id
                if messages[-1].date.astimezone(wib) < start_date:
                    break

            # Simpan hasil ke DataFrame
            df = pd.DataFrame(all_messages)
            return df

        # Jalankan scraping async
        with st.spinner("Mengambil pesan..."):
            df = asyncio.run(scrape())

        if not df.empty:
            st.success(f"✅ {len(df)} pesan berhasil diambil!")
            st.dataframe(df)

            # Tombol download CSV
            csv = df.to_csv(index=False, encoding='utf-8-sig')
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"messages_{start_str}_to_{end_str}.csv",
                mime="text/csv"
            )
        else:
            st.warning("Tidak ada pesan ditemukan pada rentang tanggal tersebut.")
