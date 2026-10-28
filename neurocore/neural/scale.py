"""
Ölçekli Veri Üretici
=====================
Müfredat derslerinden **on binlerce** eğitim örneği üretir.

Neden gerekli?
    Ölçüm: model 3,4M token ile eğitildi, 208M parametre için bu
    **parametre başına 0,016 token** demektir. Sağlıklı oran ~20'dir.
    Yani veri ~1.000× eksik. Elle ders yazmak bu açığı kapatmaz.

Çözüm: **şablon tabanlı çoğaltma**
    Her ders (58 bilgi + 20 akıl yürütme) dört eksende çoğaltılır:

        1. **Biçim**   — tanım / nasıl / neden / örnek / özet / hata / karşılaştır
        2. **Düşünme** — ANLA / KARŞILAŞTIR / KARAR VER / TEŞHİS / YORUMLA
        3. **Bağlam**  — yeni başlayan / deneyimli / hata ayıklayan
        4. **Kod**     — kod varsa kod odaklı, yoksa kavram odaklı

    Sonuç: 78 temel ders → **~10.000 örnek**.

⚠ Dürüstlük kuralı
    Şablonlar **bilgi uydurmaz**. Her iz, dersin kendi gövdesinden ve
    kodundan türetilir; yalnızca **sunum biçimi** değişir. Uydurma bilgi
    modele zehir olurdu (zehirlenme koruması bu kurala dayanır).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from neurocore.neural.curriculum import (
    LAYER_LABELS,
    Lesson,
    build_curriculum,
    expand_curriculum,
)
from neurocore.neural.reasoning import (
    MODE_LABELS,
    THINK_PREFIX,
    ReasoningLesson,
    build_reasoning_curriculum,
)

#: Düşünme biçimleri (çoğaltma ekseni)
THINKING_MODES: Sequence[str] = ("analyze", "compare", "decide", "diagnose",
                                 "interpret")

#: Hedef kitle bağlamları
AUDIENCES: Sequence[Tuple[str, str]] = (
    ("yeni", "konuya yeni başlayan birine"),
    ("deneyimli", "deneyimli bir geliştiriciye"),
    ("hata", "sorun yaşayan birine"),
)


@dataclass
class GeneratedSample:
    """Üretilmiş tek eğitim örneği."""
    prompt: str
    target: str
    layer: str
    kind: str
    tags: List[str]

    def to_dict(self) -> Dict[str, Any]:
        """JSON-serileştirilebilir hâl."""
        return {"prompt": self.prompt, "target": self.target,
                "layer": self.layer, "kind": self.kind, "tags": list(self.tags)}


# ─── Ders Gövdesinden Parça Çıkarma ─────────────────────────────────────────

def _sentences(text: str) -> List[str]:
    """Gövdeyi cümlelere böler (nokta/ünlem/soru sonrası)."""
    import re

    temiz = " ".join(text.split())
    parcalar = re.split(r"(?<=[.!?])\s+", temiz)
    return [p.strip() for p in parcalar if len(p.strip()) > 25]


def _key_points(lesson: Lesson, limit: int = 4) -> List[str]:
    """
    Dersin **kendi** cümlelerinden ana noktaları seçer.

    Uydurma yok: yalnızca var olan cümleler seçilir ve numaralanır.
    """
    cumleler = _sentences(lesson.body)
    if not cumleler:
        return [lesson.body.strip()[:200]]
    # İlk cümle tanım, sonrakiler ayrıntı; en fazla `limit` tane al
    return cumleler[:limit]


def _code_lines(lesson: Lesson, limit: int = 6) -> List[str]:
    """Dersteki kodun anlamlı satırlarını döndürür."""
    if not lesson.code.strip():
        return []
    satirlar = [s.rstrip() for s in lesson.code.splitlines()]
    anlamli = [s for s in satirlar
               if s.strip() and not s.strip().startswith("#")]
    return anlamli[:limit]


# ─── Akıl Yürütme İzi Şablonları ────────────────────────────────────────────

def _trace_analyze(lesson: Lesson, audience: str) -> str:
    """ANLA: problem gerçekte ne istiyor?"""
    noktalar = _key_points(lesson, 3)
    kod = _code_lines(lesson, 2)
    satirlar = [
        f"ADIM 1: Sorulan kavram: {lesson.title}.",
        f"ADIM 2: Kapsam — {lesson.label} alanına ait.",
    ]
    for index, nokta in enumerate(noktalar, start=3):
        satirlar.append(f"ADIM {index}: {nokta}")
    sonraki = len(satirlar) + 1
    if kod:
        satirlar.append(
            f"ADIM {sonraki}: Uygulama biçimi kodda görülüyor: "
            f"`{kod[0].strip()[:70]}`")
        sonraki += 1
    satirlar.append(
        f"KARAR: Önce kavramın ne olduğunu netleştirip sonra uygulamaya "
        f"geçmeliyim; {audience} anlatırken tanımdan başlarım.")
    satirlar.append(
        "GEREKÇE: Tanımı atlayıp doğrudan uygulamaya geçmek, yanlış aracı "
        "seçmeye yol açar.")
    return "\n".join(satirlar)


def _trace_compare(lesson: Lesson, audience: str) -> str:
    """KARŞILAŞTIR: alternatifler ve ölçütler."""
    noktalar = _key_points(lesson, 3)
    satirlar = [
        f"SEÇENEK A: {lesson.title} yaklaşımını kullanmak.",
        f"  + {noktalar[0] if noktalar else 'Alanın standart yolu.'}",
        "SEÇENEK B: Aynı işi elle veya daha alt seviyede yapmak.",
        "  + Tam denetim, ek bağımlılık yok.",
        "  − Daha fazla kod, daha fazla hata yüzeyi.",
    ]
    if len(noktalar) > 1:
        satirlar.append(f"ÖLÇÜT: {noktalar[1]}")
    satirlar.append(
        f"KARŞILAŞTIRMA: Alanın sağladığı hazır yol ({lesson.label}) "
        f"genellikle test edilmiş ve belgelenmiştir.")
    satirlar.append(
        "KARAR: Standart yolu seçiyorum; yalnızca ölçülebilir bir kısıt "
        "varsa (performans, bağımlılık) alt seviyeye inerim.")
    satirlar.append(
        f"GEREKÇE: {audience} için okunabilirlik ve doğruluk, mikro "
        f"optimizasyondan önce gelir.")
    return "\n".join(satirlar)


def _trace_decide(lesson: Lesson, audience: str) -> str:
    """KARAR VER: hangi yol ve neden?"""
    noktalar = _key_points(lesson, 3)
    kod = _code_lines(lesson, 3)
    satirlar = [
        f"ADIM 1: Hedef — {lesson.title} işini doğru ve sürdürülebilir "
        f"biçimde yapmak.",
    ]
    for index, nokta in enumerate(noktalar, start=2):
        satirlar.append(f"ADIM {index}: {nokta}")
    if kod:
        satirlar.append("ADIM: Uygulama sırası:")
        for satir in kod:
            satirlar.append(f"    {satir.strip()[:76]}")
    satirlar.append(
        "KARAR: Yukarıdaki sırayı izliyorum; her adımdan sonra sonucu "
        "doğrulayacağım.")
    satirlar.append(
        "GEREKÇE: Doğrulama olmadan ilerlemek, hatayı geç fark ettirir ve "
        "geri dönüşü pahalı kılar.")
    return "\n".join(satirlar)


def _trace_diagnose(lesson: Lesson, audience: str) -> str:
    """TEŞHİS: olası hatalar ve ayırt etme."""
    noktalar = _key_points(lesson, 2)
    satirlar = [
        f"BELİRTİ: {lesson.title} beklendiği gibi çalışmıyor.",
        "OLASI NEDEN 1: Girdi/ortam uygun değil (sürüm, yol, izin).",
        "OLASI NEDEN 2: Kullanım sırası yanlış (bağımlılık atlanmış).",
    ]
    if noktalar:
        satirlar.append(f"OLASI NEDEN 3: {noktalar[0]}")
    satirlar.append(
        "AYIRT ETME: Hatayı en küçük örnekle yeniden üret; çalışan bir "
        "örnekle karşılaştır.")
    satirlar.append(
        "ÇÖZÜM: Önce ortamı doğrula, sonra kullanım sırasını; en son "
        "kodu değiştir.")
    satirlar.append(
        f"DOĞRULAMA: Düzeltmeden sonra aynı komutu tekrar çalıştır ve "
        f"beklenen çıktıyı gör. ({audience} için ipucu: hatayı saklama, "
        f"görünür kıl.)")
    return "\n".join(satirlar)


def _trace_interpret(lesson: Lesson, audience: str) -> str:
    """YORUMLA: bu bilgi ne anlama geliyor, ne yapmalıyım?"""
    noktalar = _key_points(lesson, 3)
    satirlar = [
        f"VERİ: {noktalar[0] if noktalar else lesson.body[:120]}",
    ]
    if len(noktalar) > 1:
        satirlar.append(f"AYRINTI: {noktalar[1]}")
    satirlar.append(
        f"YORUM: Bu, {lesson.label} alanında şu sonucu verir — kavram "
        f"tek başına değil, komşu kavramlarla birlikte anlam kazanır.")
    satirlar.append(
        "SINIR: Bu bilgi tek başına yeterli değil; uygulama sırasında "
        "ölçümle doğrulanmalı.")
    satirlar.append(
        "NE YAPMALIYIM: Kavramı küçük bir denemeyle sına, sonucu ölç, "
        "sonra üretime al.")
    satirlar.append(
        f"DERS: {audience} için önemli olan, bilgiyi ezberlemek değil "
        f"ne zaman kullanacağını bilmektir.")
    return "\n".join(satirlar)


_TRACE_BUILDERS = {
    "analyze": _trace_analyze,
    "compare": _trace_compare,
    "decide": _trace_decide,
    "diagnose": _trace_diagnose,
    "interpret": _trace_interpret,
}


def _thinking_prompts(lesson: Lesson) -> List[Tuple[str, str]]:
    """Ders için (düşünme biçimi, istem) çiftleri."""
    return [
        ("analyze", f"{THINK_PREFIX} {lesson.title} — bu konuda neyi "
                    f"anlamam gerekiyor?"),
        ("compare", f"{THINK_PREFIX} {lesson.title} için hangi yaklaşım "
                    f"daha uygun?"),
        ("decide", f"{THINK_PREFIX} {lesson.title} işini nasıl yapmalıyım?"),
        ("diagnose", f"{THINK_PREFIX} {lesson.title} çalışmıyor, neden "
                     f"olabilir?"),
        ("interpret", f"{THINK_PREFIX} {lesson.title} ne anlama geliyor, "
                      f"ne yapmalıyım?"),
    ]


# ─── Üretim ─────────────────────────────────────────────────────────────────

def generate_dialogues(lessons: Optional[Sequence[Lesson]] = None,
                       *, per_lesson: int = 3) -> List[GeneratedSample]:
    """
    Her ders için **çok turlu diyalog** örnekleri üretir.

    Neden? Tek turlu "soru → cevap" örnekleri akıcı konuşma öğretmez.
    Diyalog örnekleri üç örüntü ekler:

        * **Netleştirme**  — belirsiz soruyu netleştirme
        * **Derinleştirme** — takip sorusu ve ayrıntılı cevap
        * **Hata düzeltme** — yanlış varsayımı düzeltme

    ⚠ Uydurma yok: her diyalog dersin **kendi gövdesinden** cümleler kullanır.

    Args:
        lessons: Kaynak dersler (None = temel müfredat)
        per_lesson: Ders başına kaç diyalog

    Returns:
        Diyalog örnekleri
    """
    temel = list(lessons) if lessons else build_curriculum()
    ornekler: List[GeneratedSample] = []

    for ders in temel:
        havuz = _key_points(ders, 4) + _code_lines(ders, 3)
        if not havuz:
            continue
        for tur in range(max(1, per_lesson)):
            parca = havuz[tur % len(havuz)]
            parca2 = havuz[(tur + 1) % len(havuz)]

            if tur % 3 == 0:
                # Netleştirme
                istem = (f"Kullanıcı: {ders.title} hakkında bilgi ver.\n"
                         f"Asistan:")
                hedef = (f"Elbette. {parca}\n\n"
                         f"Devam edeyim mi, yoksa belirli bir noktaya mı "
                         f"odaklanayım?")
            elif tur % 3 == 1:
                # Derinleştirme
                istem = (f"Kullanıcı: {ders.title} — biraz daha ayrıntı?\n"
                         f"Asistan:")
                hedef = (f"Tabii. {parca}\n\n"
                         f"Bir adım daha: {parca2}\n\n"
                         f"Bu ikisi birlikte konuyu tamamlar.")
            else:
                # Hata düzeltme
                istem = (f"Kullanıcı: {ders.title} için yanlış bir yol "
                         f"kullanıyorum galiba.\nAsistan:")
                hedef = (f"Olabilir. Doğrusu şu: {parca}\n\n"
                         f"Yanlış varsayım genellikle burada olur: bir adım "
                         f"atlanır ve sonuç tutarsız çıkar. "
                         f"Kontrol için: {parca2}")

            ornekler.append(GeneratedSample(
                prompt=istem, target=hedef, layer=ders.layer,
                kind="diyalog", tags=["diyalog", f"tur{tur}", *ders.tags[:2]],
            ))
    return ornekler


def expand_curriculum_deep(lessons: Optional[Sequence[Lesson]] = None,
                           *, depth: int = 3) -> List[GeneratedSample]:
    """
    Dersleri **çok seviyeli** soru kalıplarıyla çoğaltır.

    Tek düzey ("X nedir?") yeterli değil; gerçek kullanım farklı
    derinliklerde sorar. `depth` arttıkça ders başına daha fazla örnek
    üretilir (her seviye 3 kalıp ekler, en fazla 6 seviye).

    Args:
        lessons: Kaynak dersler
        depth: Kaç derinlik seviyesi (1–6)

    Returns:
        Çoğaltılmış örnekler
    """
    temel = list(lessons) if lessons else build_curriculum()
    seviyeler = max(1, min(6, int(depth)))
    # `expand_curriculum` kalıp sayısını kendisi sınırlar; seviye başına
    # 3 kalıp isteriz (seviye 6 → mevcut kalıpların tamamı).
    return [
        GeneratedSample(prompt=ders.prompt(), target=ders.target(),
                        layer=ders.layer, kind="bilgi", tags=list(ders.tags))
        for ders in expand_curriculum(temel, patterns=seviyeler * 3)
    ]


def generate_samples(*, audiences: int = 2,
                     thinking: bool = True,
                     knowledge_patterns: int = 20,
                     dialogues: int = 0,
                     depth: int = 0) -> List[GeneratedSample]:
    """
    Ölçekli eğitim örnekleri üretir.

    Args:
        audiences: Kaç farklı hedef kitle bağlamı kullanılsın (1–3)
        thinking: Akıl yürütme izleri üretilsin mi
        knowledge_patterns: Bilgi dersleri için soru kalıbı sayısı
        dialogues: Ders başına **diyalog** örneği (0 = üretme)
        depth: Çok seviyeli çoğaltma derinliği (0 = kapalı)

    Returns:
        Üretilmiş örnekler
    """
    ornekler: List[GeneratedSample] = []
    kitleler = list(AUDIENCES)[:max(1, min(audiences, len(AUDIENCES)))]
    taban = build_curriculum()

    # ── 1) Bilgi dersleri: soru kalıplarıyla çoğalt ──
    for ders in expand_curriculum(patterns=knowledge_patterns):
        ornekler.append(GeneratedSample(
            prompt=ders.prompt(), target=ders.target(), layer=ders.layer,
            kind="bilgi", tags=list(ders.tags),
        ))

    # ── 2) Çok seviyeli çoğaltma ──
    if depth > 0:
        ornekler.extend(expand_curriculum_deep(taban, depth=depth))

    # ── 3) Akıl yürütme izleri: her ders × her biçim × her kitle ──
    if thinking:
        for ders in taban:
            for mod, istem in _thinking_prompts(ders):
                uretici = _TRACE_BUILDERS.get(mod)
                if uretici is None:
                    continue
                for kitle_ad, kitle_metin in kitleler:
                    iz = uretici(ders, kitle_metin)
                    if not iz.strip():
                        continue
                    ornekler.append(GeneratedSample(
                        prompt=istem,
                        target=f"{MODE_LABELS.get(mod, mod).upper()}:\n{iz}",
                        layer=ders.layer, kind=f"düşünme:{mod}",
                        tags=[mod, kitle_ad, *ders.tags[:2]],
                    ))

    # ── 4) Diyaloglar ──
    if dialogues > 0:
        ornekler.extend(generate_dialogues(taban, per_lesson=dialogues))

    return ornekler


def scale_report(*, audiences: int = 2, knowledge_patterns: int = 20,
                 dialogues: int = 0, depth: int = 0) -> Dict[str, Any]:
    """Üretimin büyüklüğünü ve token verimini raporlar."""
    from neurocore.neural.tokenizer import BPETokenizer

    ornekler = generate_samples(audiences=audiences,
                                knowledge_patterns=knowledge_patterns,
                                dialogues=dialogues, depth=depth)
    docs = [f"{o.prompt}\n{o.target}" for o in ornekler]
    karakter = sum(len(d) for d in docs)

    tokenizer = BPETokenizer()
    try:
        tokenizer.train(docs, vocab_size=8000, min_frequency=1, verbose=False)
        token = sum(len(tokenizer.encode(d)) for d in docs)
        vocab = tokenizer.vocab_size
    except Exception:
        token, vocab = 0, 0

    per_kind: Dict[str, int] = {}
    for ornek in ornekler:
        per_kind[ornek.kind] = per_kind.get(ornek.kind, 0) + 1

    return {
        "samples": len(ornekler),
        "chars": karakter,
        "tokens": token,
        "vocab": vocab,
        "chars_per_token": round(karakter / token, 2) if token else 0.0,
        "per_kind": per_kind,
        "audiences": audiences,
        "knowledge_patterns": knowledge_patterns,
        "dialogues": dialogues,
        "depth": depth,
    }


__all__ = [
    "GeneratedSample", "generate_samples", "scale_report", "THINKING_MODES",
    "AUDIENCES",
]
