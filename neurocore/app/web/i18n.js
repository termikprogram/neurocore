/**
 * NeuroCore — arayüz dili katmanı (TR → EN)
 * =========================================
 * Arayüz Türkçe yazılmıştır. Bu dosya, **hiçbir kaynak dosyayı değiştirmeden**
 * görünen metinleri İngilizceye çevirir.
 *
 * Neden böyle?
 *   app.js ~2.000 satır ve metinler şablon içinde gömülü; tam i18n refaktörü
 *   riskli. Bu katman yalnız **metin düğümlerini** ve birkaç özniteliği
 *   (`title`, `placeholder`, `aria-label`) eşler. Çeviri yoksa Türkçe kalır —
 *   yani eksik sözlük **bozmaz**, sadece çevirmez.
 *
 * Kullanım:
 *   - Tarayıcı dili Türkçe değilse otomatik İngilizce
 *   - `?lang=en` veya `?lang=tr` ile zorla
 *   - `localStorage.neurocore_dil` ile kalıcı seçim
 */
(function () {
  "use strict";

  const SOZLUK = {
    // ── Başlık / durum ──
    "Sohbetler": "Chats",
    "SOHBETLER": "CHATS",
    "Özerklik": "Autonomy",
    "ÖZERKLİK": "AUTONOMY",
    "Akış": "Activity",
    "Dosyalar": "Files",
    "Denetim": "Audit",
    "Gönder": "Send",
    "model yok": "no model",
    "araç çağrısı": "tool calls",
    "araç": "tools",
    "çağrı": "calls",
    "token": "tokens",
    "Ayarlar": "Settings",
    "Kapat": "Close",
    "Vazgeç": "Cancel",
    "Tamam": "OK",
    "Sil": "Delete",
    "Kaydet": "Save",
    "Yenile": "Refresh",
    "İptal": "Cancel",
    "Dur": "Stop",

    // ── Özerklik seviyeleri ──
    "Danışman": "Advisor",
    "Denetimli": "Supervised",
    "Güvenilir": "Trusted",
    "Serbest": "Unrestricted",

    // ── Sohbet listesi ──
    "Yeni sohbet": "New chat",
    "Klasör seç": "Pick folder",
    "Kaldır": "Detach",
    "genel sohbet": "general chat",
    "Klasör yok": "no folder",
    "mesaj": "messages",

    // ── Karşılama / hızlı eylemler ──
    "Kod yazabilen, komut çalıştırabilen ve uygulamaları (Godot, Blender) kontrol edebilen yerel yazılım ajanı.":
      "A local software agent that can write code, run commands and drive apps (Godot, Blender).",
    "Proje haritası çıkar": "Map the project",
    "Testleri çalıştır ve düzelt": "Run and fix tests",
    "Araç zinciri sürümleri": "Toolchain versions",
    "Proje değerlendirmesi": "Review the project",

    // ── Orta panel ──
    "Henüz işlem yok.": "No activity yet.",
    "Bu görevde değişen dosya yok.": "No files changed in this task.",
    "Bir görev yazın... (Enter: gönder · Shift+Enter: yeni satır)":
      "Describe a task… (Enter: send · Shift+Enter: new line)",

    // ── Özerklik açıklaması ──
    "Yazma, çalıştırma ve yıkıcı işlemler onay ister. Okuma serbest. Önerilen varsayılan.":
      "Writes, commands and destructive operations require approval. Reads are free. Recommended default.",
  };

  // Uzun/açıklama metinleri için desen (tam eşleşme yetmediğinde)
  const DESENLER = [
    [/^(\d+) araç · (\d+) çağrı · ([\d.]+) token$/,
      (m) => `${m[1]} tools · ${m[2]} calls · ${m[3]} tokens`],
    [/^(\d+) mesaj$/, (m) => `${m[1]} messages`],
    [/^genel sohbet \(klasör yok\)$/, () => "general chat (no folder)"],
    [/^(\d+) mesaj · (.+)$/, (m) => `${m[1]} messages · ${m[2]}`],
    // ⚠ Alt bardaki havuz sayacı JS tarafından üretiliyor (dinamik metin)
    [/^(\d+) makine$/, (m) => `${m[1]} ${m[1] === "1" ? "machine" : "machines"}`],
    [/^havuz —$/, () => "pool —"],
  ];

  // ⚠ ÖLÇÜLDÜ: kaynak metinler sözlüktekiyle **birebir** aynı değil —
  // "Proje haritasını çıkar" (belirtme hâli), "Henüz bir işlem yok." (fazladan
  // "bir"), placeholder'da `…` (U+2026) ve satır içi çoklu boşluk var.
  // Bu yüzden karşılaştırma **normalize** edilerek yapılır.
  Object.assign(SOZLUK, {
    "Proje haritasını çıkar": "Map the project",
    "Henüz bir işlem yok.": "No activity yet.",
    "Henüz kayıt yok.": "No records yet.",
    "Henüz tur yok.": "No rounds yet.",
    "Testleri çalıştır ve düzelt": "Run and fix tests",
  });

  function normal(metin) {
    return String(metin)
      .replace(/\u00a0/g, " ")
      .replace(/\u2026/g, "...")
      .replace(/\s+/g, " ")
      .trim();
  }

  // normalize edilmiş anahtar → çeviri
  const NORMAL_SOZLUK = new Map();
  for (const k of Object.keys(SOZLUK)) NORMAL_SOZLUK.set(normal(k), SOZLUK[k]);

  // Uzun anahtarlar önce denenir: "genel sohbet (klasör yok)" gibi
  // birleşik metinlerde de doğru eşleşsin.
  const ANAHTARLAR = Array.from(NORMAL_SOZLUK.keys())
    .sort((a, b) => b.length - a.length);

  function cevirMetin(metin) {
    const kirp = metin.trim();
    if (!kirp) return metin;
    const n = normal(metin);
    if (NORMAL_SOZLUK.has(n)) {
      // baştaki/sondaki boşlukları koru
      const bas = metin.match(/^\s*/)[0];
      const son = metin.match(/\s*$/)[0];
      return bas + NORMAL_SOZLUK.get(n) + son;
    }
    for (const [kalip, uret] of DESENLER) {
      const m = n.match(kalip);
      if (m) return uret(m);
    }
    // ⚠ Alt-dize: arayüz metinleri sık sık cümle içinde geçiyor
    // (ör. "genel sohbet (klasör yok)", "0 mesaj", uyarı bandı).
    let sonuc = metin;
    for (const anahtar of ANAHTARLAR) {
      if (anahtar.length < 4) continue;
      // ⚠ Boşluk esnek: kaynakta çoklu boşluk/satır sonu olabilir
      const kalip = new RegExp(
        anahtar.replace(/[.*+?^${}()|[\]\\]/g, "\\$&").replace(/ /g, "\\s+"),
        "g");
      if (kalip.test(sonuc)) {
        sonuc = sonuc.replace(kalip, NORMAL_SOZLUK.get(anahtar));
      }
    }
    return sonuc;
  }

  function dugumleriCevir(kok) {
    const yuruyucu = document.createTreeWalker(kok, NodeFilter.SHOW_TEXT, {
      acceptNode(n) {
        const p = n.parentElement;
        if (!p) return NodeFilter.FILTER_REJECT;
        const etiket = p.tagName;
        if (etiket === "SCRIPT" || etiket === "STYLE" || etiket === "TEXTAREA") {
          return NodeFilter.FILTER_REJECT;
        }
        return NodeFilter.FILTER_ACCEPT;
      },
    });
    let sayi = 0;
    while (yuruyucu.nextNode()) {
      const d = yuruyucu.currentNode;
      const yeni = cevirMetin(d.nodeValue || "");
      if (yeni !== d.nodeValue) { d.nodeValue = yeni; sayi++; }
    }
    // öznitelikler
    kok.querySelectorAll?.("[title], [placeholder], [aria-label]").forEach((el) => {
      for (const ad of ["title", "placeholder", "aria-label"]) {
        const v = el.getAttribute(ad);
        if (!v) continue;
        const yeni = cevirMetin(v);
        if (yeni !== v) { el.setAttribute(ad, yeni); sayi++; }
      }
    });
    return sayi;
  }

  function dilSec() {
    const p = new URLSearchParams(location.search);
    const zorla = p.get("lang");
    if (zorla === "tr" || zorla === "en") return zorla;
    const kayit = localStorage.getItem("neurocore_dil");
    if (kayit === "tr" || kayit === "en") return kayit;
    const tarayici = (navigator.language || "en").toLowerCase();
    return tarayici.startsWith("tr") ? "tr" : "en";
  }

  function basla() {
    if (dilSec() !== "en") return;
    let toplam = dugumleriCevir(document.body);
    // Dinamik içerik: yeni düğümleri kısıtlı sıklıkta çevir
    const gozcu = new MutationObserver(() => {
      if (gozcu._zaman) return;
      gozcu._zaman = setTimeout(() => {
        gozcu._zaman = null;
        dugumleriCevir(document.body);
      }, 200);
    });
    gozcu.observe(document.body, { childList: true, subtree: true });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", () => setTimeout(basla, 60));
  } else {
    setTimeout(basla, 60);
  }
})();
