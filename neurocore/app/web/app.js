/* ─── NeuroCore Masaüstü Arayüzü ────────────────────────────────────────
   Çerçevesiz (vanilla) JavaScript. Derleme adımı yoktur; dosya doğrudan
   yüklenir. Bu, Node.js gerektirmeyen bir kurulum sağlar.

   Bölümler:
     1. Yardımcılar ve basit markdown
     2. Sohbet (gönder, akış, mesaj baloncukları)
     3. WebSocket (canlı olaylar)
     4. Onay modalı
     5. Yan panel (akış, dosyalar, denetim)
     6. Araçlar çekmecesi
     7. Ayarlar (dil modeli, çalışma alanı, sağlık)
     8. Görsel doğrulama
   ─────────────────────────────────────────────────────────────────────── */

const $  = (id) => document.getElementById(id);
const api = async (path, options = {}) => {
  const response = await fetch(path, {
    headers: { 'Content-Type': 'application/json' },
    ...options,
  });
  const text = await response.text();
  let data;
  try { data = text ? JSON.parse(text) : {}; }
  catch { data = { ok: false, error: text.slice(0, 400) }; }
  if (!response.ok && !data.error) data.error = `HTTP ${response.status}`;
  return data;
};
const esc = (s) => String(s ?? '').replace(/[&<>"']/g,
  (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
const fmtMs = (ms) => ms >= 1000 ? `${(ms / 1000).toFixed(1)}s` : `${Math.round(ms)}ms`;
const fmtTime = (seconds) => new Date(seconds * 1000).toLocaleTimeString('tr-TR');

function toast(message, kind = '') {
  const node = document.createElement('div');
  node.className = `toast ${kind}`;
  node.textContent = message;
  $('toasts').appendChild(node);
  setTimeout(() => { node.style.opacity = '0'; }, 4600);
  setTimeout(() => node.remove(), 5100);
}

/* ─── 1. Basit markdown ──────────────────────────────────────────────── */

function markdown(text) {
  if (!text) return '';
  const blocks = [];
  let out = String(text);

  // Kod bloklarını koru
  out = out.replace(/```(\w*)\n?([\s\S]*?)```/g, (_m, lang, code) => {
    blocks.push(`<pre><code class="lang-${esc(lang || 'text')}">${esc(code.replace(/\n$/, ''))}</code></pre>`);
    return `\u0000BLOCK${blocks.length - 1}\u0000`;
  });
  // Satır içi kod
  const inline = [];
  out = out.replace(/`([^`\n]+)`/g, (_m, code) => {
    inline.push(`<code>${esc(code)}</code>`);
    return `\u0000INL${inline.length - 1}\u0000`;
  });

  out = esc(out);

  // Başlıklar
  out = out.replace(/^######\s+(.+)$/gm, '<h3>$1</h3>')
           .replace(/^#####\s+(.+)$/gm, '<h3>$1</h3>')
           .replace(/^####\s+(.+)$/gm, '<h3>$1</h3>')
           .replace(/^###\s+(.+)$/gm, '<h3>$1</h3>')
           .replace(/^##\s+(.+)$/gm, '<h2>$1</h2>')
           .replace(/^#\s+(.+)$/gm, '<h1>$1</h1>');
  // Yatay çizgi
  out = out.replace(/^---+$/gm, '<hr>');
  // Alıntı
  out = out.replace(/^&gt;\s?(.*)$/gm, '<blockquote>$1</blockquote>');
  out = out.replace(/<\/blockquote>\n<blockquote>/g, '<br>');
  // Kalın / eğik
  out = out.replace(/\*\*([^*\n]+)\*\*/g, '<strong>$1</strong>')
           .replace(/(^|\s)\*([^*\n]+)\*/g, '$1<em>$2</em>');
  // Tablolar
  out = out.replace(/^\|(.+)\|\s*\n\|[\s:|-]+\|\s*\n((?:\|.*\|\s*\n?)*)/gm, (_m, head, body) => {
    const cells = (row) => row.split('|').slice(1, -1).map((c) => c.trim());
    const headHtml = cells(`|${head}|`).map((c) => `<th>${c}</th>`).join('');
    const bodyHtml = body.trim().split('\n').filter(Boolean)
      .map((row) => `<tr>${cells(row).map((c) => `<td>${c}</td>`).join('')}</tr>`).join('');
    return `<table><thead><tr>${headHtml}</tr></thead><tbody>${bodyHtml}</tbody></table>`;
  });
  // Listeler
  out = out.replace(/^(?:[-*]\s+.*(?:\n|$))+/gm, (match) =>
    '<ul>' + match.trim().split('\n')
      .map((line) => `<li>${line.replace(/^[-*]\s+/, '')}</li>`).join('') + '</ul>');
  out = out.replace(/^(?:\d+\.\s+.*(?:\n|$))+/gm, (match) =>
    '<ol>' + match.trim().split('\n')
      .map((line) => `<li>${line.replace(/^\d+\.\s+/, '')}</li>`).join('') + '</ol>');
  // Paragraflar
  out = out.split(/\n{2,}/).map((chunk) => {
    const trimmed = chunk.trim();
    if (!trimmed) return '';
    if (/^<(h[1-3]|ul|ol|pre|blockquote|table|hr|div)/.test(trimmed)) return trimmed;
    return `<p>${trimmed.replace(/\n/g, '<br>')}</p>`;
  }).join('\n');

  out = out.replace(/\u0000BLOCK(\d+)\u0000/g, (_m, i) => blocks[i]);
  out = out.replace(/\u0000INL(\d+)\u0000/g, (_m, i) => inline[i]);
  return out;
}

/* ─── 9. Öğrenme paneli ──────────────────────────────────────────────── */

const Learning = {
  init() {
    $('learn-train').onclick = () => this.train();
    $('learn-refresh').onclick = () => this.refresh();
    this.refresh();
  },

  async refresh() {
    const data = await api('/api/learning');
    if (!data.ok) {
      $('learn-status').innerHTML =
        `<div class="learn-row"><span class="learn-label">Durum</span>`
        + `<span class="learn-value">${esc(data.error || 'kullanılamıyor')}</span></div>`;
      return;
    }
    const stats = data.stats || {};
    const model = data.model || {};
    const ready = data.readiness || {};

    const steps = model.training_steps ?? 0;
    const target = model.min_generation_steps ?? 800;
    const percent = Math.min(100, Math.round((steps / target) * 100));

    const rows = [
      ['Kayıtlı deneyim', `${stats.records ?? 0} (${stats.instructive ?? 0} öğretici)`],
      ['Eğitimde kullanılan', `${stats.used_in_training ?? 0}`],
      ['Bekleyen (yeni)', `${stats.pending ?? 0}`],
      ['Evrime aktarılan', `${stats.observations ?? 0}`],
      ['Eğitim turu', `${stats.training_runs ?? 0}`],
      ['Model adımı', `${steps} / ${target}`],
      ['Görülen token', `${(model.tokens_seen ?? 0).toLocaleString('tr-TR')}`],
      ['Uzman / nesil', `${model.experts ?? '?'} / ${model.generation ?? '?'}`],
      ['Üretim yapabilir', model.can_generate ? 'evet' : 'hayır'],
    ];

    $('learn-status').innerHTML =
      rows.map(([label, value]) =>
        `<div class="learn-row"><span class="learn-label">${esc(label)}</span>`
        + `<span class="learn-value">${esc(String(value))}</span></div>`).join('')
      + `<div class="learn-bar"><span style="width:${percent}%"></span></div>`
      + `<div class="learn-row"><span class="learn-label">Üretim eşiğine</span>`
      + `<span class="learn-value">%${percent}</span></div>`;

    // Öğrenme **nasıl** ilerliyor: kullanıcı neden beklediğini anlamalı.
    const pending = stats.pending ?? 0;
    const threshold = ready.min_new_experiences ?? 5;
    let help;
    if (stats.training) {
      help = '⏳ Eğitim şu anda çalışıyor (veri kümesi kuruluyor, adımlar atılıyor)…';
    } else if (ready.ready) {
      help = 'Eğitim için hazır — arka plan işçisi 3 dakika içinde başlatacak.';
    } else if (pending < threshold) {
      help = `Her görev deneyim olarak kaydedilir. Öğretici kayıt: ${pending}/${threshold}. `
        + 'Dosya oluşturan/düzenleyen bir görev çalıştırın; eşik dolunca eğitim '
        + 'kendiliğinden başlar (veya "Şimdi eğit" ile hemen başlatın).';
    } else {
      help = `Eğitim beklemede: ${ready.reason}. Kaynaklar boşaldığında veya `
        + 'süre dolduğunda kendiliğinden başlar.';
    }
    $('learn-help').textContent = help;

    $('learn-train').disabled = Boolean(stats.training);

    const recent = data.recent || [];
    $('learn-recent').innerHTML = recent.length
      ? '<div class="settings-help" style="margin-top:12px">Son çalışmalar:</div>'
        + recent.slice().reverse().map((item) => {
            const cls = item.reward >= 0.6 ? 'pos' : (item.reward <= 0 ? 'neg' : 'mid');
            return `<div class="learn-record">
              <span class="lr-task" title="${esc(item.task)}">${esc(item.task)}</span>
              <span class="learn-badge ${item.instructive ? 'on' : ''}">`
              + `${item.instructive ? 'öğretici' : '—'}</span>
              <span class="lr-reward ${cls}">${item.reward >= 0 ? '+' : ''}${item.reward.toFixed(2)}</span>
            </div>`;
          }).join('')
      : '<div class="settings-help" style="margin-top:10px">Henüz kayıt yok. Bir görev çalıştırın.</div>';
  },

  async train() {
    $('learn-help').textContent = 'Eğitim başlatılıyor…';
    const result = await api('/api/learning/train', {
      method: 'POST',
      body: JSON.stringify({ max_minutes: 4 }),
    });
    if (result.ok) {
      toast('Eğitim başlatıldı', 'ok');
    } else {
      toast(result.error || 'eğitim başlatılamadı', 'err');
    }
    this.refresh();
  },
};

/* ─── 1. Model değiştirici (üst çubuk) ───────────────────────────────── */

/**
 * Üst çubuktaki model rozetine tıklanınca açılan sağlayıcı/model seçici.
 *
 * Neden üst çubukta? Sağlayıcı kotası dolduğunda (HTTP 429) kullanıcının
 * Ayarlar'a girip uğraşması gerekiyordu. Artık tek tıkla geçiş yapılabilir.
 */
const ModelSwitch = {
  open: false,

  init() {
    $('model-pill').onclick = (event) => {
      event.stopPropagation();
      this.toggle();
    };
    $('model-menu-refresh').onclick = (event) => {
      event.stopPropagation();
      this.load(true);
    };
    // Dışına tıklanınca kapan
    document.addEventListener('click', (event) => {
      if (!this.open) return;
      if (!$('model-switch').contains(event.target)) this.close();
    });
    document.addEventListener('keydown', (event) => {
      if (event.key === 'Escape' && this.open) this.close();
    });
  },

  toggle() {
    if (this.open) {
      this.close();
    } else {
      this.open = true;
      $('model-menu').hidden = false;
      $('model-pill').setAttribute('aria-expanded', 'true');
      this.load();
    }
  },

  close() {
    this.open = false;
    $('model-menu').hidden = true;
    $('model-pill').setAttribute('aria-expanded', 'false');
  },

  /** Sağlayıcı kataloğunu ve aktif durumu çeker, menüyü çizer. */
  async load(force = false) {
    const data = await api('/api/settings');
    if (!data.ok) {
      $('model-menu-body').innerHTML =
        `<div class="model-menu-foot">Yüklenemedi: ${esc(data.error || '?')}</div>`;
      return;
    }
    this.catalog = data.provider_catalog || [];
    const llm = data.llm || {};
    // Alan adları: `active` sağlayıcı **etiketi**, `active_model` model adı.
    // Dikkat: `active_key` sağlayıcı kimliği **değildir** (`openai_compatible`
    // gibi bir uç nokta türü döner). Aktif sağlayıcı, `providers[].active`
    // bayrağından okunur — tek güvenilir kaynak odur.
    this.active = llm.active || '';
    this.activeModel = llm.active_model || '';
    this.providers = llm.providers || [];
    const activeEntry = this.providers.find((item) => item.active);
    this.activeName = activeEntry ? activeEntry.name : '';
    this.render();
    if (force) toast('Model listesi yenilendi', 'ok');
  },

  render() {
    const byName = new Map(this.providers.map((item) => [item.name, item]));
    const rows = this.catalog.map((entry) => {
      const live = byName.get(entry.name) || {};
      const available = Boolean(live.available);
      const isActive = entry.name === this.activeName;
      const models = entry.suggested_models || [];
      const state = available
        ? '<span class="mp-state ready">hazır</span>'
        : (entry.needs_key
           ? `<span class="mp-state nokey">${live.active ? 'kota?' : 'anahtar yok'}</span>`
           : '<span class="mp-state nokey">kapalı</span>');

      const head = `
        <div class="model-provider ${isActive ? 'active' : ''} ${available ? '' : 'no-key'}"
             data-provider="${esc(entry.name)}">
          <div class="mp-row">
            <span class="mp-name">${esc(entry.label)}</span>
            ${state}
          </div>
          <div class="mp-model">${esc(live.model || entry.default_model || '—')}</div>
        </div>`;

      // Yalnızca **aktif** sağlayıcının modelleri listelenir (menü sade kalsın)
      if (!isActive || !models.length) return head;
      const items = models.map((name) => `
        <button class="model-item ${name === this.activeModel ? 'current' : ''}"
                data-model="${esc(name)}">
          <span>${esc(name)}</span>
          ${name === entry.default_model ? '<span class="mi-tag">varsayılan</span>' : ''}
        </button>`).join('');
      return head + items;
    }).join('');

    $('model-menu-body').innerHTML = rows;
    $('model-menu-foot').innerHTML = this.active
      ? `Aktif: <b>${esc(this.active)}</b> · ${esc(this.activeModel || '—')}<br>`
        + 'Kota dolarsa başka bir sağlayıcıya tıklayın; anahtarı varsa hemen geçer.'
      : 'Hiçbir sağlayıcı hazır değil. Ayarlar → Dil Modeli bölümünden anahtar girin.';

    // Sağlayıcıya tıkla → o sağlayıcıya geç
    $('model-menu-body').querySelectorAll('.model-provider').forEach((node) => {
      node.onclick = (event) => {
        event.stopPropagation();
        this.activate(node.dataset.provider, '');
      };
    });
    // Modele tıkla → o modelle o sağlayıcıya geç
    $('model-menu-body').querySelectorAll('.model-item').forEach((node) => {
      node.onclick = (event) => {
        event.stopPropagation();
        const provider = node.closest('.model-provider').dataset.provider;
        this.activate(provider, node.dataset.model);
      };
    });
  },

  /** Sağlayıcıya (ve isteğe bağlı modele) geçer. */
  async activate(provider, model) {
    if (provider === this.activeName && (!model || model === this.activeModel)) {
      this.close();
      return;
    }
    $('model-menu-foot').textContent = 'Geçiliyor…';
    const result = await api('/api/settings/llm', {
      method: 'POST',
      body: JSON.stringify({ provider, model: model || '' }),
    });
    if (result.ok) {
      toast(model ? `Geçildi: ${provider} / ${model}` : `Geçildi: ${provider}`, 'ok');
      // Rozeti **anında** güncelle: `/api/state` yoklaması birkaç saniye
      // sürebiliyor ve kullanıcı geçişin olduğunu hemen görmeli.
      const llm = result.llm || {};
      if (llm.available) {
        $('model-pill-text').textContent =
          `${llm.active} · ${llm.active_model || '?'}`;
        $('model-pill').classList.remove('quota-low');
      }
      Settings.renderLlmStatus(llm);
      await Settings.fill();
      await State.refresh();
      this.load();
    } else {
      toast(result.error || 'geçilemedi', 'err');
      this.load();
    }
  },
};

/* ─── 10. Çalışan komutlar ───────────────────────────────────────────── */

/**
 * Hangi komutun çalıştığını gösterir ve durdurma imkânı verir.
 *
 * Neden gerekli? Kullanıcı "Durdur" düğmesine bastığında ne olduğunu
 * göremiyordu: arka planda çalışan bir sunucu/derleme iptalden sonra da
 * devam ediyordu. Artık çalışan her süreç listelenir ve tek tek (veya
 * topluca) durdurulabilir.
 */
const Running = {
  timer: null,
  tick: 0,

  init() {
    $('running-kill-all').onclick = () => this.killAll();
    this.refresh();
    // Süre göstergesi saniye saniye ilerlesin
    this.timer = setInterval(() => this.refresh(), 2000);
  },

  async refresh() {
    const data = await api('/api/processes');
    if (!data.ok) return;
    const panel = $('running-panel');
    const items = data.running || [];

    if (!items.length) {
      panel.hidden = true;
      $('running-list').innerHTML = '';
      return;
    }
    panel.hidden = false;
    $('running-count').textContent =
      `${items.length} komut · en uzun ${Math.max(...items.map((i) => i.elapsed_seconds)).toFixed(0)} sn`;

    $('running-list').innerHTML = items.map((item) => `
      <div class="running-item">
        <span class="running-cmd" title="${esc(item.command)}">${esc(item.command)}</span>
        <span class="running-meta">${esc(item.kind)} · ${item.elapsed_seconds.toFixed(0)}sn</span>
        <button class="running-kill" data-pid="${item.pid}"
                title="Bu komutu durdur">✕</button>
      </div>`).join('');

    $('running-list').querySelectorAll('.running-kill').forEach((node) => {
      node.onclick = () => this.killOne(Number(node.dataset.pid));
    });
  },

  async killOne(pid) {
    const result = await api('/api/processes/kill', {
      method: 'POST', body: JSON.stringify({ pid }),
    });
    toast(result.killed ? `Komut durduruldu (PID ${pid})` : 'Süreç bulunamadı',
          result.killed ? 'ok' : 'warn');
    this.refresh();
  },

  async killAll() {
    const result = await api('/api/processes/kill', {
      method: 'POST', body: JSON.stringify({}),
    });
    toast(result.killed ? `${result.killed} komut durduruldu` : 'Çalışan komut yok',
          result.killed ? 'ok' : 'warn');
    this.refresh();
  },
};

/* ─── 11. Kendi kendine geliştirme paneli ───────────────────────────── */

/**
 * Öğretmen–öğrenci döngüsünün durumu.
 *
 * Kullanıcı hangi konunun öğrenildiğini, kaç kaynağın kullanıldığını ve
 * yerel modelin neyi yanlış yaptığını buradan görür.
 */
const Study = {
  init() {
    $('study-run').onclick = () => this.runNow();
    $('study-save').onclick = () => this.saveTopics();
    $('study-refresh').onclick = () => this.refresh();
    this.refresh();
  },

  async refresh() {
    const data = await api('/api/study');
    if (!data.ok) {
      $('study-status').innerHTML =
        `<div class="learn-row"><span class="learn-label">Durum</span>`
        + `<span class="learn-value">${esc(data.error || 'kapalı')}</span></div>`;
      return;
    }
    const stats = data.stats || {};
    const rounds = stats.rounds ?? 0;
    const localQ = stats.local_questions ?? 0;
    const rows = [
      ['Öğretmen model', data.teacher || '—'],
      ['Öğrenme turu', `${rounds} (${stats.ok ?? 0} başarılı)`],
      // "Soru sormayı öğrenme" ilerlemesi: yerel model kaç soruyu kendi üretti
      ['Yerel modelin soruları', `${localQ} / ${rounds}`],
      ['Sağlayıcı yedeklemesi', `${stats.teacher_switches ?? 0}`],
      ['İndirilen kaynak', `${stats.sources_fetched ?? 0}`],
      ['Kasaya not', `${stats.notes ?? 0}`],
      ['Model eleştirisi', `${stats.critiques ?? 0}`],
      ['Harcanan token', `${(stats.tokens ?? 0).toLocaleString('tr-TR')}`],
      ['Arka plan', data.worker_alive ? 'çalışıyor' : 'durdu'],
    ];
    $('study-status').innerHTML = rows.map(([label, value]) =>
      `<div class="learn-row"><span class="learn-label">${esc(label)}</span>`
      + `<span class="learn-value">${esc(String(value))}</span></div>`).join('');

    $('study-help').textContent = data.studying
      ? '⏳ Öğrenme turu çalışıyor (soru → kaynak indirme → cevap)…'
      : 'Akış: önce **yerel model** soru sorar → öğretmenin önerdiği kaynak '
        + 'indirilir → kaynağa dayalı cevap alınır → yerel modelin eksiği '
        + 'yazılır → kasaya not bırakılır. Yerel model henüz soru üretemiyorsa '
        + 'öğretmen sorar; tur hiç durmaz.';

    const box = $('study-topics');
    if (box && document.activeElement !== box && !box.dataset.dirty) {
      box.value = (data.topics || []).join('\n');
    }
    if (box) box.oninput = () => { box.dataset.dirty = '1'; };

    const recent = data.recent || [];
    $('study-recent').innerHTML = recent.length
      ? '<div class="settings-help" style="margin-top:12px">Son turlar:</div>'
        + recent.slice().reverse().map((item) => {
            const mark = item.ok ? '✓' : '✗';
            const src = (item.sources || []).length;
            return `<div class="learn-record">
              <span class="lr-task" title="${esc(item.question || item.error || '')}">`
              + `${mark} ${esc(item.topic)}</span>
              <span class="learn-badge ${src ? 'on' : ''}">${src} kaynak</span>
            </div>`;
          }).join('')
      : '<div class="settings-help" style="margin-top:10px">Henüz tur yok.</div>';

    $('study-run').disabled = Boolean(data.studying);
  },

  async runNow() {
    $('study-help').textContent = 'Öğrenme turu başlatılıyor…';
    const first = $('study-topics').value
      .split('\n').map((line) => line.trim()).filter(Boolean)[0] || '';
    const result = await api('/api/study/run', {
      method: 'POST', body: JSON.stringify({ topic: first }),
    });
    if (result.ok) {
      toast(first ? `Öğreniliyor: ${first.slice(0, 40)}` : 'Öğrenme turu başlatıldı', 'ok');
    } else {
      toast(result.error || 'başlatılamadı', 'err');
    }
    this.refresh();
  },

  async saveTopics() {
    const topics = $('study-topics').value
      .split('\n').map((line) => line.trim()).filter(Boolean);
    const result = await api('/api/study/topics', {
      method: 'POST', body: JSON.stringify({ topics }),
    });
    if (result.ok) {
      $('study-topics').dataset.dirty = '';
      toast(`${topics.length} konu kaydedildi`, 'ok');
    } else {
      toast(result.error || 'kaydedilemedi', 'err');
    }
  },
};

/* ─── 12. Sohbet listesi (her sohbetin kendi çalışma alanı) ─────────── */

/**
 * Soldaki sohbet sütunu.
 *
 * Kural: **sohbetin klasörü varsa çalışma alanı odur.** Klasör yoksa
 * "genel sohbet" modu çalışır (proje bağlamı olmadan). Ayarlar'daki genel
 * çalışma alanı yalnızca klasörsüz sohbetler için geçerlidir.
 */
const ChatList = {
  chats: [],
  activeId: '',
  active: null,

  init() {
    $('chats-new').onclick = () => this.create();
    $('chats-ws-pick').onclick = () => this.pickFolder();
    $('chats-ws-clear').onclick = () => this.clearFolder();
    this.activeId = localStorage.getItem('neurocore.activeChat') || '';
    this.refresh();
  },

  async refresh() {
    const data = await api('/api/chats');
    if (!data.ok) {
      $('chats-list').innerHTML =
        `<div class="chats-empty">${esc(data.error || 'sohbetler alınamadı')}</div>`;
      return;
    }
    this.chats = data.chats || [];
    if (data.active) this.activeId = data.active;
    if (!this.activeId && this.chats.length) this.activeId = this.chats[0].id;
    this.active = this.chats.find((item) => item.id === this.activeId) || null;
    this.renderWorkspace();
    this.render();
  },

  renderWorkspace() {
    const icon = $('chats-ws-icon');
    const name = $('chats-ws-name');
    const clearBtn = $('chats-ws-clear');
    const pickBtn = $('chats-ws-pick');
    if (!this.active) {
      icon.textContent = '·';
      name.textContent = 'sohbet seçin';
      name.className = 'chats-ws-name general';
      name.title = '';
      pickBtn.disabled = true;
      clearBtn.disabled = true;
      return;
    }
    pickBtn.disabled = false;
    if (this.active.workspace) {
      icon.textContent = '📁';
      name.textContent = this.active.workspace_name || this.active.workspace;
      name.className = 'chats-ws-name';
      name.title = this.active.workspace;
      clearBtn.disabled = false;
    } else {
      icon.textContent = '💬';
      name.textContent = 'genel sohbet (klasör yok)';
      name.className = 'chats-ws-name general';
      name.title = 'Bu sohbet bir klasöre bağlı değil — proje bağlamı kullanılmaz.';
      clearBtn.disabled = true;
    }
    const wsName = $('workspace-name');
    if (wsName) wsName.textContent = this.active.workspace_name || 'genel';
  },

  render() {
    const list = $('chats-list');
    if (!this.chats.length) {
      list.innerHTML = '<div class="chats-empty">Henüz sohbet yok. '
        + '「+」 ile başlayın; isterseniz klasör bağlayın.</div>';
      return;
    }
    list.innerHTML = this.chats.map((chat) => {
      const ws = chat.workspace
        ? `<span class="chat-item-ws" title="${esc(chat.workspace)}">📁 ${esc(chat.workspace_name)}</span>`
        : '<span class="chat-item-ws general">💬 genel</span>';
      const active = chat.id === this.activeId ? ' active' : '';
      return `<div class="chat-item${active}" data-id="${chat.id}">
        <div class="chat-item-title">${esc(chat.title)}</div>
        <div class="chat-item-meta">${ws}<span>${chat.message_count} mesaj</span></div>
        <div class="chat-item-actions">
          <button data-act="rename" title="Yeniden adlandır">✎</button>
          <button data-act="clear" title="Mesajları temizle">⟲</button>
          <button data-act="delete" class="danger" title="Sohbeti sil">✕</button>
        </div>
      </div>`;
    }).join('');

    list.querySelectorAll('.chat-item').forEach((node) => {
      const id = node.dataset.id;
      node.onclick = (event) => {
        const action = event.target.dataset.act;
        if (action === 'rename') { event.stopPropagation(); this.rename(id); return; }
        if (action === 'clear') { event.stopPropagation(); this.clear(id); return; }
        if (action === 'delete') { event.stopPropagation(); this.remove(id); return; }
        this.activate(id);
      };
    });
  },

  async create(workspace = '') {
    const result = await api('/api/chats', {
      method: 'POST', body: JSON.stringify({ workspace }),
    });
    if (!result.ok) { toast(result.error || 'oluşturulamadı', 'err'); return; }
    this.activeId = result.chat.id;
    localStorage.setItem('neurocore.activeChat', this.activeId);
    Chat.clearTranscript();
    await this.refresh();
    await State.refresh();
    toast(workspace
      ? `Yeni sohbet · ${result.chat.workspace_name}`
      : 'Yeni sohbet (genel — klasör bağlamak için "Klasör seç")', 'ok');
  },

  async activate(id) {
    if (id === this.activeId && this.active) return;
    const data = await api(`/api/chats/${id}/activate`, { method: 'POST' });
    if (!data.ok) { toast(data.error || 'açılamadı', 'err'); return; }
    this.activeId = id;
    localStorage.setItem('neurocore.activeChat', id);
    this.active = data.chat;
    Chat.clearTranscript();
    Chat.loadHistory(data.chat.messages || []);
    this.renderWorkspace();
    this.render();
    await State.refresh();
    toast(data.general
      ? 'Genel sohbet modu (klasör yok)'
      : `Çalışma alanı: ${data.chat.workspace_name}`, 'ok');
  },

  async pickFolder() {
    if (!this.active) { toast('Önce bir sohbet seçin', 'warn'); return; }
    Folder.open({
      title: 'Bu sohbet için çalışma klasörü',
      start: this.active.workspace || '',
      onChoose: async (path) => {
        const result = await api(`/api/chats/${this.activeId}/workspace`, {
          method: 'POST', body: JSON.stringify({ path }),
        });
        if (!result.ok) { toast(result.error || 'bağlanamadı', 'err'); return; }
        toast(`Klasör bağlandı: ${result.chat.workspace_name}`, 'ok');
        await this.refresh();
        await State.refresh();
      },
    });
  },

  async clearFolder() {
    if (!this.active) return;
    const result = await api(`/api/chats/${this.activeId}/workspace`, {
      method: 'POST', body: JSON.stringify({ path: '' }),
    });
    if (!result.ok) { toast(result.error || 'kaldırılamadı', 'err'); return; }
    toast('Genel sohbet moduna geçildi', 'ok');
    await this.refresh();
    await State.refresh();
  },

  async rename(id) {
    const chat = this.chats.find((item) => item.id === id);
    Prompt.open({
      title: 'Sohbeti yeniden adlandır',
      value: chat ? chat.title : '',
      onOk: async (value) => {
        const result = await api(`/api/chats/${id}/rename`, {
          method: 'POST', body: JSON.stringify({ title: value }),
        });
        if (result.ok) { toast('Yeniden adlandırıldı', 'ok'); await this.refresh(); }
        else toast(result.error || 'olmadı', 'err');
      },
    });
  },

  async clear(id) {
    const result = await api(`/api/chats/${id}/clear`, { method: 'POST' });
    if (!result.ok) { toast(result.error || 'temizlenemedi', 'err'); return; }
    toast('Mesajlar temizlendi (klasör bağlı kaldı)', 'ok');
    if (id === this.activeId) Chat.clearTranscript();
    await this.refresh();
  },

  async remove(id) {
    const chat = this.chats.find((item) => item.id === id);
    const label = chat ? chat.title : 'sohbet';
    if (!window.confirm(`"${label}" silinsin mi?\n\n`
                        + 'Sohbet geçmişi silinir; diskteki dosyalarınız etkilenmez.')) {
      return;
    }
    const result = await api(`/api/chats/${id}`, { method: 'DELETE' });
    if (!result.ok) { toast(result.error || 'silinemedi', 'err'); return; }
    if (id === this.activeId) { this.activeId = ''; this.active = null; }
    localStorage.removeItem('neurocore.activeChat');
    Chat.clearTranscript();
    toast('Sohbet silindi', 'ok');
    await this.refresh();
  },
};

/* ─── Klasör seçici ──────────────────────────────────────────────────── */

/** Basit klasör gezgini (sohbet çalışma alanı seçimi için). */
const Folder = {
  path: '',
  //: Kullanıcının **açıkça** seçtiği klasör. `path` gezilen dizindir;
  //: ikisini karıştırmak, gezinirken yanlış klasörün seçilmesine yol açar.
  selected: '',
  onChoose: null,

  init() {
    $('folder-close').onclick = () => this.close();
    $('folder-cancel').onclick = () => this.close();
    $('folder-up').onclick = () => {
      // Yukarı çıkmak **seçimi değiştirmez**: kullanıcı gezinirken açıkça
      // seçtiği klasör korunur.
      this.browse(this.parentOf(this.path));
    };
    $('folder-choose').onclick = async () => {
      // Seçim yoksa gezilen dizin kabul edilir (kullanıcı "bu klasör" der)
      const chosen = this.selected || this.path;
      if (!chosen) { toast('Bir klasör seçin', 'warn'); return; }
      const handler = this.onChoose;
      this.close();
      if (handler) await handler(chosen);
    };
    $('folder-path').addEventListener('keydown', (event) => {
      if (event.key === 'Enter') {
        this.selected = event.target.value.trim();
        this.browse(this.selected);
      }
    });
    // ── Yeni klasör (DSH'deki "New folder" karşılığı) ──
    // Klasör oluşturma **seçicide** yapılır; dosya ağacı yalnızca listeler.
    $('folder-new').onclick = () => this.showNewField();
    $('folder-new-ok').onclick = () => this.createFolder();
    $('folder-new-name').addEventListener('keydown', (event) => {
      if (event.key === 'Enter') this.createFolder();
      if (event.key === 'Escape') this.hideNewField();
    });
  },

  showNewField() {
    $('folder-new-name').hidden = false;
    $('folder-new-ok').hidden = false;
    $('folder-new').hidden = true;
    $('folder-new-name').value = '';
    $('folder-new-name').focus();
  },

  hideNewField() {
    $('folder-new-name').hidden = true;
    $('folder-new-ok').hidden = true;
    $('folder-new').hidden = false;
    $('folder-new-name').value = '';
  },

  /**
   * Seçili klasörün **altında** yeni klasör oluşturur.
   *
   * Oluşturulan klasör hemen seçilir (DSH davranışı) ki kullanıcı tek
   * adımda "bu klasörü kullan" diyebilsin.
   */
  async createFolder() {
    const name = $('folder-new-name').value.trim();
    if (!name) { toast('Klasör adı girin', 'warn'); return; }
    const parent = this.selected || this.path;
    const result = await api('/api/settings/mkdir', {
      method: 'POST', body: JSON.stringify({ parent, name }),
    });
    if (!result.ok) { toast(result.error || 'oluşturulamadı', 'err'); return; }
    toast(`Klasör oluşturuldu: ${result.name}`, 'ok');
    this.hideNewField();
    await this.enter(result.path);      // içine gir ve seç
  },

  open({ title = 'Klasör seç', start = '', onChoose = null } = {}) {
    this.onChoose = onChoose;
    // Başlangıç klasörü hem gezilir hem seçili sayılır
    this.selected = start || '';
    this.path = '';
    $('folder-title').textContent = title;
    $('folder-backdrop').hidden = false;
    this.browse(start || '');
  },

  close() {
    $('folder-backdrop').hidden = true;
  },

  parentOf(path) {
    if (!path) return '';
    const parts = path.replace(/[\\/]+$/, '').split(/[\\/]/);
    parts.pop();
    const parent = parts.join('\\');
    return parent.length === 2 && parent.endsWith(':') ? parent + '\\' : parent;
  },

  async browse(path) {
    const query = path ? `?path=${encodeURIComponent(path)}` : '';
    const data = await api('/api/settings/browse' + query);
    if (!data.ok) { toast(data.error || 'klasör okunamadı', 'err'); return; }
    this.path = data.path || '';
    $('folder-path').value = this.path;

    // Dikkat: sunucu **yalnızca dizinleri** döndürür (`name`, `path`,
    // `has_project`). Var olmayan bir `is_dir` alanına göre filtrelemek
    // listenin her zaman boş görünmesine yol açıyordu.
    const rows = data.entries || [];
    $('folder-list').innerHTML = rows.length
      ? rows.map((item) => `<div class="folder-item" data-path="${esc(item.path)}">
           <span class="fi-icon">📁</span>
           <span>${esc(item.name)}</span>
           ${item.has_project ? '<span class="fi-tag">proje</span>' : ''}
         </div>`).join('')
      : '<div class="folder-item">(alt klasör yok)</div>';

    $('folder-list').querySelectorAll('.folder-item[data-path]').forEach((node) => {
      node.onclick = () => this.enter(node.dataset.path);
    });
    // Seçimi **çizimden sonra** uygula: `innerHTML` sınıfları sıfırlar,
    // bu yüzden seçili öğe burada işaretlenir.
    this.markSelection();
  },

  /** Klasöre gir ve onu seçili yap (çift amaçlı tıklama). */
  async enter(path) {
    this.selected = path;
    await this.browse(path);
  },

  /** Seçili klasörü listede işaretler. */
  markSelection() {
    const nodes = $('folder-list').querySelectorAll('.folder-item[data-path]');
    nodes.forEach((node) => {
      node.classList.toggle('selected', node.dataset.path === this.selected);
    });
  },
};

/* ─── Metin girdisi ──────────────────────────────────────────────────── */

/** Tek satırlık girdi isteyen basit iletişim kutusu. */
const Prompt = {
  onOk: null,

  init() {
    $('prompt-close').onclick = () => this.close();
    $('prompt-cancel').onclick = () => this.close();
    $('prompt-ok').onclick = async () => {
      const value = $('prompt-input').value.trim();
      const handler = this.onOk;
      this.close();
      if (handler && value) await handler(value);
    };
    $('prompt-input').addEventListener('keydown', (event) => {
      if (event.key === 'Enter') $('prompt-ok').click();
      if (event.key === 'Escape') this.close();
    });
  },

  open({ title = '', value = '', onOk = null } = {}) {
    this.onOk = onOk;
    $('prompt-title').textContent = title;
    $('prompt-input').value = value;
    $('prompt-backdrop').hidden = false;
    setTimeout(() => { $('prompt-input').focus(); $('prompt-input').select(); }, 40);
  },

  close() {
    $('prompt-backdrop').hidden = true;
  },
};

/* ─── Başlat ─────────────────────────────────────────────────────────── */

window.addEventListener('DOMContentLoaded', () => {
  Chat.init();
  Approval.init();
  ModelSwitch.init();
  State.init();
  Tools.init();
  Settings.init();
  Vision.init();
  Learning.init();
  Running.init();
  Study.init();
  Folder.init();
  Prompt.init();
  ChatList.init();
  Tree.init();
  Socket.connect();

  // Takılı kalan onay penceresine karşı güvenlik ağı: arayüzü sunucudaki
  // **gerçek** bekleyen onaylarla düzenli olarak eşitle. WebSocket olayı
  // kaçırılmışsa (sayfa yenilendi, bağlantı koptu, istek zaman aşımına
  // uğradı) modal açık kalabilir; bu denetim onu kapatır.
  setTimeout(() => Approval.sync(), 1500);
  setInterval(() => Approval.sync(), 12000);

  // Sağlık uyarısını ilk açılışta göster
  api('/api/health').then((result) => {
    (result.issues || []).forEach((issue) => toast(issue, 'warn'));
  }).catch(() => {});
});/* ─── 13. Çalışma alanı dosya ağacı ─────────────────────────────────── */

/**
 * Sağ paneldeki **dosya ağacı** (DSH `workspaceFiles.list` modeli).
 *
 * Tasarım kararları (DSH'den):
 *   * Ağaç **oturumun çalışma alanına** köklenir; kökün üstüne çıkılamaz.
 *   * Her seviye **istenildiğinde** okunur (tüm ağaç bir seferde değil).
 *   * Sıralama: önce dizinler, sonra doğal ve büyük/küçük harf duyarsız ad.
 *   * Ağaç **salt okunur** — mutasyon (oluştur/sil/adlandır) yapmaz;
 *     klasör oluşturma dizin seçicide, dosya işlemleri ajan araçlarındadır.
 */
const Tree = {
  levels: {},     // yol → { loading, entries, error }
  expanded: {},   // yol → true
  selected: '',
  root: '',

  init() {
    $('tree-reload').onclick = () => this.reload();
    $('file-preview-close').onclick = () => this.closePreview();
  },

  /** Ağacı kökten yeniden yükler (oturum değişince çağrılır). */
  async reload() {
    this.levels = {};
    this.expanded = {};
    this.selected = '';
    this.closePreview();
    await this.load('');
  },

  async load(path) {
    const key = path || '';
    this.levels[key] = { loading: true, entries: [], error: '' };
    this.render();

    const session = ChatList.activeId || '';
    const query = `?path=${encodeURIComponent(path)}&session_id=${encodeURIComponent(session)}`;
    const data = await api('/api/workspace/tree' + query);

    if (!data.ok) {
      this.levels[key] = { loading: false, entries: [], error: data.error || 'okunamadı' };
    } else {
      this.root = data.root || this.root;
      this.levels[key] = { loading: false, entries: data.entries || [],
                           error: '', truncated: data.truncated };
    }
    this.render();
  },

  async toggle(path) {
    if (this.expanded[path]) {
      delete this.expanded[path];     // kapat: seviye önbellekte kalır
      this.render();
      return;
    }
    this.expanded[path] = true;
    if (!this.levels[path]) await this.load(path);
    else this.render();
  },

  render() {
    const body = $('tree-body');
    if (!body) return;
    $('tree-root').textContent = this.root || '(çalışma alanı)';
    $('tree-root').title = this.root || '';

    if (!this.levels['']) {
      this.load('');
      return;
    }
    body.innerHTML = this.renderLevel('');
  },

  renderLevel(path) {
    const level = this.levels[path || ''];
    if (!level) return '';
    if (level.loading) return '<div class="tree-note">yükleniyor…</div>';
    if (level.error) return `<div class="tree-error">${esc(level.error)}</div>`;
    const entries = level.entries || [];
    if (!entries.length) return '<div class="tree-note">(boş klasör)</div>';

    return entries.map((entry) => {
      const isDir = entry.type === 'directory';
      const open = Boolean(this.expanded[entry.path]);
      const caret = isDir
        ? `<span class="tree-caret ${open ? 'open' : ''}">▶</span>`
        : '<span class="tree-caret"></span>';
      const icon = isDir ? (open ? '📂' : '📁') : this.fileIcon(entry.language);
      const size = isDir ? '' : `<span class="tree-size">${this.humanSize(entry.size)}</span>`;
      const row = `
        <div class="tree-row ${entry.path === this.selected ? 'selected' : ''}"
             data-path="${esc(entry.path)}" data-type="${entry.type}"
             title="${esc(entry.path)}">
          ${caret}<span class="tree-icon">${icon}</span>
          <span class="tree-name">${esc(entry.name)}</span>${size}
        </div>`;
      const children = (isDir && open)
        ? `<div class="tree-children">${this.renderLevel(entry.path)}</div>`
        : '';
      return `<div class="tree-node">${row}${children}</div>`;
    }).join('');
  },

  fileIcon(language) {
    return {
      python: '🐍', javascript: '𝗝𝗦', typescript: '𝗧𝗦', html: '🌐',
      css: '🎨', json: '{}', markdown: '📝', text: '📄',
      yaml: '⚙', toml: '⚙', shell: '⌨', powershell: '⌨', batch: '⌨',
      gdscript: '🎮', godot: '🎮', blender: '🧊', svg: '🖼',
    }[language] || '📄';
  },

  humanSize(bytes) {
    const value = Number(bytes || 0);
    if (value < 1024) return `${value} b`;
    if (value < 1024 * 1024) return `${(value / 1024).toFixed(1)} KB`;
    return `${(value / 1024 / 1024).toFixed(1)} MB`;
  },

  /** Ağaç satırına tıklandı (olay delegasyonu). */
  async onClick(row) {
    const path = row.dataset.path;
    this.selected = path;
    if (row.dataset.type === 'directory') {
      await this.toggle(path);
    } else {
      this.render();
      await this.openFile(path);
    }
  },

  async openFile(path) {
    const data = await api('/api/workspace/file?path=' + encodeURIComponent(path));
    const panel = $('file-preview');
    panel.hidden = false;
    if (!data.ok) {
      $('file-preview-name').textContent = 'hata';
      $('file-preview-body').textContent = data.error || 'okunamadı';
      return;
    }
    $('file-preview-name').textContent =
      `${data.name} · ${this.humanSize(data.size)}${data.language ? ' · ' + data.language : ''}`;
    $('file-preview-body').textContent = data.binary
      ? '(ikili dosya — metin olarak gösterilemez)'
      : (data.text || '(boş dosya)');
  },

  closePreview() {
    $('file-preview').hidden = true;
    $('file-preview-body').textContent = '';
  },
};

/* ─── 2. Sohbet ──────────────────────────────────────────────────────── */

const Chat = {
  busy: false,
  currentAgentNode: null,
  steps: [],
  files: new Set(),

  init() {
    // Ayarlar → dil seçici (varsayılan İngilizce)
    const dilSecici = $('ui-lang');
    if (dilSecici) {
      dilSecici.value = (window.NC_DIL && window.NC_DIL.sec()) || 'en';
      dilSecici.onchange = () => {
        // ⚠ Sayfa yenilenmez: i18n katmanı özgün metinleri geri yükleyip
        // yeni dili hemen uygular.
        if (window.NC_DIL) window.NC_DIL.ayarla(dilSecici.value);
      };
    }

    // MCP düğmeleri
    const mcpEkleBtn = $('mcp-add');
    if (mcpEkleBtn) {
      mcpEkleBtn.onclick = async () => {
        const ad = ($('mcp-ad').value || '').trim();
        const komut = ($('mcp-komut').value || '').trim();
        const args = ($('mcp-args').value || '').trim();
        const sonuc = await api('/api/mcp', {
          method: 'POST',
          body: JSON.stringify({ ad, komut, args }),
        });
        $('mcp-help').textContent = sonuc && sonuc.ok
          ? `${ad} eklendi` : `hata: ${(sonuc && sonuc.error) || '?'}`;
        this.mcpRefresh();
      };
    }
    ['blender', 'godot'].forEach((preset) => {
      const btn = $(`mcp-preset-${preset}`);
      if (!btn) return;
      btn.onclick = async () => {
        const sonuc = await api('/api/mcp', {
          method: 'POST', body: JSON.stringify({ preset }),
        });
        $('mcp-help').textContent = sonuc && sonuc.ok
          ? `${preset} şablonu eklendi` : `hata: ${(sonuc && sonuc.error) || '?'}`;
        this.mcpRefresh();
      };
    });
    if ($('mcp-refresh')) $('mcp-refresh').onclick = () => this.mcpRefresh();

    // ⚠ Paneller hazır olsun: sekme tıklaması beklemeden bir kez yükle
    //    (ölçüldü: yalnız tıklamaya bağlıyken panel boş kalabiliyordu).
    // ⚠ init async değil: promise'leri beklemeyiz, hataları yutarız
    this.projectYenile().catch(() => {});
    this.puzzleYenile().catch(() => {});

    $('btn-send').onclick = () => this.send();
    $('btn-stop').onclick = () => this.cancel();
    const input = $('input');
    // ⚠ Geçmiş gezinme: ↑/↓ ile gönderilmiş önceki mesajlar arasında dolaş.
    //    Kural: imleç metnin başında/sonunda değilse ok tuşu normal çalışır
    //    (satır içi düzenlemeyi bozmamak için).
    this.gecmisIndeks = -1;
    this.gecmisTaslak = '';
    input.addEventListener('keydown', (event) => {
      if (event.key === 'Enter' && !event.shiftKey) { event.preventDefault(); this.send(); return; }
      if (event.key !== 'ArrowUp' && event.key !== 'ArrowDown') return;
      const gecmis = this.gecmis();
      if (!gecmis.length) return;
      const imlecSonda = input.selectionStart === input.value.length;
      const imlecBasta = input.selectionStart === 0;
      if (this.gecmisIndeks === -1) {
        if (event.key === 'ArrowDown') return;
        if (!imlecSonda && input.value) return;      // düzenleme sürüyor
        this.gecmisTaslak = input.value;
        this.gecmisIndeks = gecmis.length - 1;
      } else if (event.key === 'ArrowUp') {
        if (this.gecmisIndeks <= 0) return;
        this.gecmisIndeks -= 1;
      } else {
        if (!imlecBasta && input.value) return;
        if (this.gecmisIndeks < gecmis.length - 1) {
          this.gecmisIndeks += 1;
        } else {
          // listenin sonuna geldik → taslağa dön
          this.gecmisIndeks = -1;
          input.value = this.gecmisTaslak || '';
          input.dispatchEvent(new Event('input'));
          event.preventDefault();
          return;
        }
      }
      event.preventDefault();
      input.value = gecmis[this.gecmisIndeks] || '';
      input.dispatchEvent(new Event('input'));
      input.setSelectionRange(input.value.length, input.value.length);
    });
    input.addEventListener('input', () => {
      input.style.height = 'auto';
      input.style.height = Math.min(input.scrollHeight, 190) + 'px';
    });
    document.querySelectorAll('.example').forEach((button) => {
      button.onclick = () => { input.value = button.dataset.prompt; this.send(); };
    });
  },

  addMessage(role, content) {
    const welcome = $('welcome');
    if (welcome) welcome.remove();
    const node = document.createElement('div');
    node.className = `msg ${role}`;
    node.innerHTML = `
      <div class="msg-avatar">${role === 'user' ? 'S' : '◆'}</div>
      <div class="msg-body">
        <div class="msg-role">${role === 'user' ? 'Siz' : 'NeuroCore'}</div>
        <div class="msg-content">${role === 'user' ? esc(content) : markdown(content)}</div>
      </div>`;
    $('transcript').appendChild(node);
    this.scroll();
    return node.querySelector('.msg-content');
  },

  /**
   * Asistan mesajının altına 👍 / 👎 düğmeleri ekler.
   *
   * Bu puanlar **en değerli öğrenme sinyalidir**: model yanlış yaptığında
   * doğrudan düzeltilir. Kayıt zamanı (`at`) sunucuya gönderilir ki doğru
   * deneyime bağlansın.
   */
  addFeedbackBar(at) {
    const bar = document.createElement('div');
    bar.className = 'msg-feedback';
    bar.dataset.at = at;
    bar.innerHTML = `
      <button data-fb="1" title="Bu yanıt iyiydi">👍 İyi</button>
      <button data-fb="-1" title="Bu yanıt yanlıştı">👎 Kötü</button>
      <span class="fb-note" hidden></span>`;
    bar.querySelectorAll('button').forEach((button) => {
      button.onclick = async () => {
        const reward = Number(button.dataset.fb);
        const result = await api('/api/learning/feedback', {
          method: 'POST',
          body: JSON.stringify({ at, reward }),
        });
        const note = bar.querySelector('.fb-note');
        note.hidden = false;
        if (result.ok) {
          button.classList.add(reward > 0 ? 'given-up' : 'given-down');
          note.textContent = reward > 0
            ? 'kaydedildi — model bu örneği güçlendirecek'
            : 'kaydedildi — model bu örnekten kaçınacak';
          toast(reward > 0 ? 'Olumlu geri bildirim kaydedildi' : 'Olumsuz geri bildirim kaydedildi',
                reward > 0 ? 'ok' : 'warn');
        } else {
          note.textContent = result.error || 'kaydedilemedi';
        }
      };
    });
    $('transcript').appendChild(bar);
    this.scroll();
    return bar;
  },

  thinking() {
    const node = document.createElement('div');
    node.className = 'msg agent';
    node.id = 'thinking-node';
    node.innerHTML = `
      <div class="msg-avatar">◆</div>
      <div class="msg-body">
        <div class="msg-role">NeuroCore</div>
        <div class="msg-content">
          <div class="thinking">
            <span class="thinking-dots"><i></i><i></i><i></i></span>
            <span id="thinking-label">düşünüyor…</span>
          </div>
        </div>
      </div>`;
    $('transcript').appendChild(node);
    this.scroll();
  },

  setThinkingLabel(text) {
    const label = $('thinking-label');
    if (label) label.textContent = text;
  },

  clearThinking() {
    const node = $('thinking-node');
    if (node) node.remove();
    const pill = $('steps-pill');
    if (pill) pill.hidden = true;
  },

  /** Sohbet alanını boşaltır (sohbet değiştirildiğinde). */
  clearTranscript() {
    $('transcript').innerHTML = '';
    this.steps = [];
    const pill = $('steps-pill');
    if (pill) pill.hidden = true;
    this.clearThinking();
  },

  /**
   * Kayıtlı sohbet geçmişini ekrana çizer.
   *
   * Yalnızca **sohbet** mesajları gösterilir; araç adımları geçmişe
   * yazılmaz (onlar akış panelinde canlı görünür).
   */
  loadHistory(messages) {
    const items = messages || [];
    if (!items.length) {
      this.showWelcome();
      return;
    }
    items.forEach((item) => {
      const role = item.role === 'user' ? 'user' : 'agent';
      const content = String(item.content || '').trim();
      if (!content) return;
      this.addMessage(role, content);
    });
    this.scroll();
  },

  /** Karşılama ekranını geri getirir. */
  showWelcome() {
    const transcript = $('transcript');
    transcript.innerHTML = `
      <div class="welcome" id="welcome">
        <h1>NeuroCore</h1>
        <p class="welcome-sub">Donanımına göre ölçeklenen, öğrenen ve
        bilgisayarını senin yerine kullanan yerel yapay zekâ.</p>
        <div class="examples" id="examples">
          <button class="example" data-prompt="proje haritasını çıkar">Proje haritasını çıkar</button>
          <button class="example" data-prompt="testleri çalıştır ve başarısız olanları düzelt">Testleri çalıştır ve düzelt</button>
          <button class="example" data-prompt="kurulu araçların sürümleri neler">Araç zinciri sürümleri</button>
          <button class="example" data-prompt="bu projede neler eksik, ne geliştirilebilir">Proje değerlendirmesi</button>
        </div>
        <div class="welcome-note" id="welcome-note"></div>
      </div>`;
    // Örnek düğmelerini yeniden bağla (innerHTML ile dinleyiciler kayboldu)
    document.querySelectorAll('#examples .example').forEach((node) => {
      node.onclick = () => {
        $('input').value = node.dataset.prompt || '';
        $('input').focus();
      };
    });
  },

  /**
   * Adım sayacını günceller.
   *
   * "Adım sınırı" kullanıcıya soyut geliyordu; artık her adımda
   * "adım 7 / 24" biçiminde somut olarak görünür ve sınıra yaklaşınca
   * rengi değişir.
   */
  updateStepCounter() {
    const pill = $('steps-pill');
    if (!pill) return;
    const used = this.steps.filter((item) => item.kind !== 'answer').length;
    // ⚠ Adım sınırı artık varsayılan olarak YOK (0 = sınırsız):
    // sayaç "adım N" gösterir; sunucudan sınır gelirse "N / M" gösterilir.
    const limit = Number(this.stepLimit || 0);
    pill.hidden = false;
    pill.textContent = limit > 0 ? `adım ${used} / ${limit}` : `adım ${used}`;
    pill.classList.toggle('quota-low', limit > 0 && used >= limit - 3);
    pill.title = used >= limit - 3
      ? 'Adım sınırına yaklaşıldı. Sınır dolduğunda ajan durur ve o ana '
        + 'kadar yaptıklarını özetler. Görevi daha küçük parçalara bölmek '
        + 'çözümdür.'
      : `Her araç çağrısı bir adımdır. Sınır: ${limit}`;
  },

  scroll() {
    const transcript = $('transcript');
    transcript.scrollTop = transcript.scrollHeight;
  },

  /**
   * Bu sohbette gönderilmiş kullanıcı mesajları (en eski → en yeni).
   * Kaynak: localStorage + sunucudan gelen sohbet geçmişi.
   */
  /** HTML kaçışı (kullanıcı/sunucu metni güvenli basılsın). */
  esc(metin) {
    return String(metin == null ? '' : metin)
      .replace(/&/g, '&amp;').replace(/</g, '&lt;')
      .replace(/>/g, '&gt;').replace(/"/g, '&quot;');
  },

  gecmis() {
    const anahtar = `neurocore_gecmis_${this.chatId || 'genel'}`;
    let kayit = [];
    try {
      kayit = JSON.parse(localStorage.getItem(anahtar) || '[]');
    } catch (err) {
      kayit = [];
    }
    return Array.isArray(kayit) ? kayit.slice(-50) : [];
  },

  gecmiseEkle(metin) {
    const temiz = String(metin || '').trim();
    if (!temiz) return;
    const anahtar = `neurocore_gecmis_${this.chatId || 'genel'}`;
    const kayit = this.gecmis();
    if (kayit[kayit.length - 1] === temiz) return;      // tekrarı yazma
    kayit.push(temiz);
    try {
      localStorage.setItem(anahtar, JSON.stringify(kayit.slice(-50)));
    } catch (err) {
      /* kota dolu — geçmiş kritik değil */
    }
    this.gecmisIndeks = -1;
  },

  // ─── MCP bağlantıları ─────────────────────────────────────────────
  mcpCiz(veri) {
    const kap = $('mcp-list');
    if (!kap) return;
    const sunucular = (veri && veri.sunucular) || [];
    if (!sunucular.length) {
      kap.innerHTML = '<div class="empty-hint">Henüz MCP bağlantısı yok.</div>';
      return;
    }
    kap.innerHTML = sunucular.map((s) => {
      const durum = s.bagli
        ? `bağlı · ${s.arac_sayisi} araç`
        : (s.hata ? `hata: ${this.esc(s.hata)}` : 'bağlı değil');
      const araclar = (s.araclar || []).slice(0, 6).join(', ');
      return `<div class="mcp-row">
        <div><strong>${this.esc(s.ad)}</strong> — <span class="muted">${durum}</span>
        ${araclar ? `<div class="settings-help">${this.esc(araclar)}</div>` : ''}</div>
        <div class="settings-actions">
          <button class="btn btn-ghost" data-mcp-test="${this.esc(s.ad)}">Sına</button>
          <button class="btn btn-ghost" data-mcp-del="${this.esc(s.ad)}">Kaldır</button>
        </div>
      </div>`;
    }).join('');
    kap.querySelectorAll('[data-mcp-test]').forEach((b) => {
      b.onclick = () => this.mcpDene(b.dataset.mcpTest);
    });
    kap.querySelectorAll('[data-mcp-del]').forEach((b) => {
      b.onclick = () => this.mcpSil(b.dataset.mcpDel);
    });
  },

  async mcpRefresh() {
    const sonuc = await api('/api/mcp');
    if (sonuc && sonuc.ok) this.mcpCiz(sonuc);
  },

  async mcpDene(ad) {
    const yardim = $('mcp-help');
    if (yardim) yardim.textContent = `${ad} sınanıyor…`;
    const sonuc = await api('/api/mcp/test', {
      method: 'POST', body: JSON.stringify({ ad }),
    });
    if (yardim) {
      yardim.textContent = sonuc && sonuc.ok
        ? `${ad}: bağlı · ${sonuc.arac_sayisi} araç (${(sonuc.araclar || []).join(', ')})`
        : `${ad}: ${(sonuc && (sonuc.hata || sonuc.error)) || 'bağlanamadı'}`;
    }
    this.mcpRefresh();
  },

  async mcpSil(ad) {
    await api('/api/mcp/delete', { method: 'POST', body: JSON.stringify({ ad }) });
    this.mcpRefresh();
  },

  // ─── Proje kartı (mimari + kaldığın yer) ──────────────────────────
  async projectYenile() {
    const kap = $('project-card');
    if (!kap) return;
    const veri = await api('/api/project');
    const bos = $('project-empty');
    if (!veri || !veri.ok || !veri.kart) {
      kap.innerHTML = '';
      if (bos) bos.hidden = false;
      return;
    }
    if (bos) bos.hidden = true;
    const k = veri.kart;
    const m = k.mimari || {};
    const kontrol = veri.kontrol || {};
    const komutlar = (k.komutlar || []).map((c) =>
      `<code class="proj-komut" data-komut="${this.esc(c)}">${this.esc(c)}</code>`).join(' ');
    const klasorler = (m.klasorler || []).map((x) =>
      `<span class="proj-klasor">${this.esc(x.ad)}<b>${x.dosya}</b></span>`).join('');
    const adimlar = (kontrol.adimlar || []).slice(-6).reverse().map((a) =>
      `<div class="proj-adim">${a.ok ? '✓' : '✗'} <code>${this.esc(a.tool)}</code>
       <span class="muted">${this.esc(a.ozet || '')}</span></div>`).join('');
    kap.innerHTML = `
      <div class="proj-blok">
        <div class="proj-ad">📁 ${this.esc(k.proje || '?')}</div>
        <div class="settings-help">${this.esc(k.kok || '')}</div>
        <div class="proj-durum">${this.esc(k.durum || '')}
          · ${this.esc(String(k.adim ?? (kontrol.adimlar || []).length))} adım</div>
      </div>
      <div class="proj-blok">
        <h4>Mimari</h4>
        <div class="settings-help">yığın: ${this.esc((m.yigin || []).join(', ') || '—')}
          · ${this.esc(String(m.toplam_dosya ?? 0))} dosya</div>
        <div class="settings-help">giriş: ${this.esc((m.giris_noktalari || []).join(', ') || '—')}</div>
        <div class="proj-klasorler">${klasorler || '<span class="muted">—</span>'}</div>
      </div>
      <div class="proj-blok">
        <h4>Çalıştırma</h4>
        <div class="proj-komutlar">${komutlar || '<span class="muted">komut bulunamadı</span>'}</div>
      </div>
      <div class="proj-blok">
        <h4>Kaldığın yer</h4>
        <div class="settings-help">${this.esc(k.son_adim || 'henüz adım yok')}</div>
        ${adimlar}
        <button class="btn btn-ghost" id="proj-devam">Kaldığı yerden devam et</button>
      </div>`;
    kap.querySelectorAll('[data-komut]').forEach((el) => {
      el.onclick = () => {
        const input = $('input');
        if (input) { input.value = el.dataset.komut; input.focus(); }
      };
    });
    if ($('proj-devam')) {
      $('proj-devam').onclick = () => {
        const input = $('input');
        if (input) { input.value = 'devam et'; input.focus(); }
      };
    }
  },

  // ─── Puzzle (iç içe fazlar) ───────────────────────────────────────
  /** Ağacı özyinelemeli çizer; her parçaya revizyon düğmesi koyar. */
  puzzleDugum(parca) {
    if (!parca) return '';
    const ikon = { tamam: '✓', aktif: '▶', revize: '✎' }[parca.durum] || '·';
    const sinif = `puzzle-node ${parca.durum || 'bekliyor'}`;
    const sorular = (parca.sorular || []).map((s) =>
      `<div class="puzzle-q">? ${this.esc(s)}</div>`).join('');
    const cevap = parca.cevap
      ? `<div class="puzzle-a">↳ ${this.esc(parca.cevap)}</div>` : '';
    const not = parca.not
      ? `<div class="puzzle-not">✎ ${this.esc(parca.not)}</div>` : '';
    const cocuklar = (parca.cocuklar || []).map((c) => this.puzzleDugum(c)).join('');
    return `<div class="${sinif}">
      <div class="puzzle-row">
        <span class="puzzle-ikon">${ikon}</span>
        <span class="puzzle-id">[${this.esc(parca.id)}]</span>
        <span class="puzzle-baslik">${this.esc(parca.baslik)}</span>
        <button class="btn btn-ghost puzzle-btn" data-puzzle-revize="${this.esc(parca.id)}"
                title="Bu parçayı değiştir">✎</button>
      </div>
      ${parca.olcut ? `<div class="puzzle-olcut">ölçüt: ${this.esc(parca.olcut)}</div>` : ''}
      ${sorular}${cevap}${not}
      ${cocuklar ? `<div class="puzzle-cocuklar">${cocuklar}</div>` : ''}
    </div>`;
  },

  async puzzleYenile() {
    const kap = $('puzzle-tree');
    if (!kap) return;
    const veri = await api('/api/puzzle');
    const bos = $('puzzle-empty');
    if (!veri || !veri.ok || !veri.kok) {
      kap.innerHTML = '';
      if (bos) bos.hidden = false;
      return;
    }
    if (bos) bos.hidden = true;
    kap.innerHTML = this.puzzleDugum(veri.kok);
    kap.querySelectorAll('[data-puzzle-revize]').forEach((b) => {
      b.onclick = () => this.puzzleRevize(b.dataset.puzzleRevize);
    });
  },

  async puzzleRevize(kimlik) {
    const not = prompt(`[${kimlik}] Bu parçada ne değişsin?`);
    if (!not) return;
    const sonuc = await api('/api/puzzle/revize', {
      method: 'POST', body: JSON.stringify({ id: kimlik, not }),
    });
    if (sonuc && sonuc.ok) this.puzzleYenile();
    else alert((sonuc && sonuc.error) || 'revizyon uygulanamadı');
  },

  async send() {
    const input = $('input');
    const message = input.value.trim();
    if (!message || this.busy) return;

    // ⚠ ↑/↓ geçmişi: gönderilen mesaj kaydedilir, gezinme sıfırlanır
    this.gecmiseEkle(message);
    this.gecmisIndeks = -1;
    this.gecmisTaslak = '';
    this.addMessage('user', message);
    input.value = '';
    input.style.height = 'auto';
    this.busy = true;
    this.steps = [];
    this.files = new Set();
    $('btn-send').disabled = true;
    $('btn-stop').hidden = false;
    $('view-steps').innerHTML = '';
    $('view-files').innerHTML = '<div class="empty-hint">Bu görevde değişen dosya yok.</div>';
    this.thinking();

    const result = await api('/api/chat', {
      method: 'POST',
      body: JSON.stringify({
        message,
        // Etkin sohbete gönder: sunucu mesajı o sohbete yazar ve sohbetin
        // kendi çalışma alanını kullanır.
        session_id: ChatList.activeId || '',
        plan_first: $('plan-first').checked,
        // ⚠ Adım sınırı YOK: 0 = sunucu tarafında sınırsız (güvenlik tavanı
        // MUTLAK_ADIM=400). Ölçüldü: burada 24 gönderildiği için ajan sürekli
        // "adım sınırına ulaştım" diyordu.
        max_steps: 0,
        max_seconds: 1800,
      }),
    });

    if (!result.ok) {
      this.clearThinking();
      this.addMessage('agent', `⚠ ${result.error || 'görev başlatılamadı'}`);
      this.finish();
      return;
    }
    // Sunucu sohbeti kendiliğinden oluşturduysa listede göster
    if (result.session_id && result.session_id !== ChatList.activeId) {
      ChatList.activeId = result.session_id;
      localStorage.setItem('neurocore.activeChat', result.session_id);
      ChatList.refresh();
    }
  },

  async cancel() {
    await api('/api/cancel', { method: 'POST' });
    toast('İptal isteği gönderildi', 'warn');
  },

  finish() {
    this.busy = false;
    this.clearThinking();
    $('btn-send').disabled = false;
    $('btn-stop').hidden = true;
    this.refreshAudit();
  },

  /* Ajan akışından gelen adım */
  onStep(step) {
    this.steps.push(step);
    this.renderStep(step);
    // Adım sayacını canlı göster: kullanıcı "adım sınırı" kavramını somut
    // olarak görsün (kaçıncı adımda olduğunu ve sınırı bilsin).
    this.updateStepCounter();
    // Komut çalıştıran adımlarda çalışan süreç panelini hemen tazele
    if (step.kind === 'tool' && /^(code\.run_command|shell\.)/.test(step.tool || '')) {
      Running.refresh();
    }

    if (step.kind === 'tool') {
      const labels = {
        'code.workspace_map': 'projeyi haritalıyor…',
        'code.read_file': 'dosya okuyor…',
        'code.search_text': 'kodda arıyor…',
        'code.find_symbol': 'sembol arıyor…',
        'code.edit_file': 'dosyayı düzenliyor…',
        'code.create_file': 'dosya oluşturuyor…',
        'code.run_command': 'komut çalıştırıyor…',
        'code.run_tests': 'testleri çalıştırıyor…',
        'code.diagnose_error': 'hatayı inceliyor…',
        'shell.run': 'komut çalıştırıyor…',
        'shell.session_run': 'oturumda komut çalıştırıyor…',
        'shell.versions': 'araç sürümlerini okuyor…',
        'vision.screenshot': 'ekrana bakıyor…',
        'vision.verify_app': 'uygulamayı doğruluyor…',
      };
      this.setThinkingLabel(labels[step.tool] || `${step.tool} çalıştırıyor…`);

      if (step.args && step.args.path) this.addFile(step.args.path);
      if (['code.edit_file', 'code.create_file'].includes(step.tool) && step.args?.path) {
        this.addFile(step.args.path);
      }
    } else if (step.kind === 'approval') {
      this.setThinkingLabel('onayınızı bekliyor…');
    }
  },

  addFile(path) {
    this.files.add(path);
    const view = $('view-files');
    view.innerHTML = [...this.files].map((item) =>
      `<div class="file-item">◆ ${esc(item)}</div>`).join('');
  },

  renderStep(step) {
    const view = $('view-steps');
    const node = document.createElement('div');
    const okClass = step.kind === 'tool' ? (step.ok ? 'ok' : 'fail') : '';
    node.className = `step kind-${step.kind} ${okClass}`;

    const icons = {
      tool: step.ok ? '✓' : '✗',
      plan: '≡', error: '!', approval: '⚿', answer: '◆', thinking: '…',
    };

    let body = '';
    if (step.kind === 'tool') {
      const args = step.args ? Object.entries(step.args)
        .slice(0, 3)
        .map(([key, value]) => `${key}=${String(value).slice(0, 44)}`)
        .join(' · ') : '';
      const result = String(step.result || '');
      const snippet = result.length > 620 ? result.slice(0, 620) + '…' : result;
      body = `${args ? `<div>${esc(args)}</div>` : ''}`
           + (snippet ? `<pre>${esc(snippet)}</pre>` : '');
    } else {
      body = `<div>${esc(String(step.text || '').slice(0, 400))}</div>`;
    }

    node.innerHTML = `
      <div class="step-head">
        <span class="step-icon">${icons[step.kind] || '·'}</span>
        <span class="step-tool">${esc(step.tool || step.kind)}</span>
        <span class="step-time">${step.duration_ms ? fmtMs(step.duration_ms) : ''}</span>
      </div>
      ${body ? `<div class="step-body">${body}</div>` : ''}`;

    view.appendChild(node);
    view.scrollTop = view.scrollHeight;
  },

  onRunFinished(payload) {
    this.clearThinking();
    this.finish();

    const answer = payload.answer || (payload.ok ? '(yanıt üretilmedi)' : '');
    if (answer) {
      this.addMessage('agent', answer);
    } else if (payload.error) {
      this.addMessage('agent', `⚠ ${payload.error}`);
    }
    // Puanlama düğmeleri: yalnızca gösterilecek yanıt varsa anlamlıdır.
    this.lastRunHadAnswer = Boolean(answer);
    this.feedbackShown = false;
    if (answer && this.lastLearningAt) {
      this.addFeedbackBar(this.lastLearningAt);
      this.feedbackShown = true;
    }

    // Alt bilgi: süre, adım, araçlar
    const meta = document.createElement('div');
    const parts = [];
    if (payload.steps_count) parts.push(`${payload.steps_count} adım`);
    if (payload.duration_ms) parts.push(fmtMs(payload.duration_ms));
    if (payload.mode) parts.push(payload.mode === 'rules' ? 'kural tabanlı' : payload.mode);
    if (payload.reason && payload.reason !== 'tamamlandı') parts.push(payload.reason);
    if (payload.files_changed?.length) parts.push(`${payload.files_changed.length} dosya değişti`);
    meta.className = 'msg-role';
    meta.style.marginTop = '-12px';
    meta.textContent = parts.join(' · ');
    $('transcript').appendChild(meta);
    this.scroll();

    if (payload.files_changed?.length) {
      payload.files_changed.forEach((file) => this.addFile(file));
    }
    if (payload.answer) Side.setAnswer(payload.answer);

    // Puanlama çubuğu `onLearningRecorded` içinde eklenir (kaydın zaman
    // damgasını taşır ve yanıt gösterildikten sonra gelir).
  },

  /**
   * Öğrenme kaydı oluştuğunda puanlama düğmelerini ekler.
   *
   * Olay sırası garanti değildir (`learning.recorded` bazen
   * `agent.run_finished`'ten hemen önce gelir); bu yüzden zaman damgası
   * saklanır ve çubuk, yanıt gösterildiği anda eklenir.
   */
  onLearningRecorded(data) {
    this.lastLearningAt = data.at;
    if (this.lastRunHadAnswer && !this.feedbackShown) {
      this.addFeedbackBar(data.at);
      this.feedbackShown = true;
    }
  },

  async refreshAudit() {
    const data = await api('/api/autonomy');
    if (!data.ok) return;
    const view = $('view-audit');
    const rows = (data.audit || []).slice().reverse();
    if (!rows.length) {
      view.innerHTML = '<div class="empty-hint">Araç çağrısı kaydı boş.</div>';
      return;
    }
    view.innerHTML = rows.map((row) => `
      <div class="audit-row">
        <span class="a-time">${esc(row.time)}</span>
        <span class="a-tool" title="${esc(row.args)}">${esc(row.tool)}</span>
        <span class="${row.ok ? 'a-ok' : 'a-bad'}">${row.ok ? '✓' : (row.blocked ? '⚿' : '✗')}</span>
      </div>`).join('');
  },
};

/* ─── 3. WebSocket ───────────────────────────────────────────────────── */

const Socket = {
  socket: null,
  retry: 0,

  connect() {
    const protocol = location.protocol === 'https:' ? 'wss' : 'ws';
    this.socket = new WebSocket(`${protocol}://${location.host}/ws`);

    this.socket.onopen = () => {
      this.retry = 0;
      this.ping();
    };

    this.socket.onmessage = (event) => {
      let message;
      try { message = JSON.parse(event.data); } catch { return; }
      this.handle(message.type, message.data);
    };

    this.socket.onclose = () => {
      this.retry = Math.min(this.retry + 1, 6);
      setTimeout(() => this.connect(), 500 * this.retry);
    };

    this.socket.onerror = () => { /* onclose devralır */ };
  },

  ping() {
    setInterval(() => {
      if (this.socket?.readyState === WebSocket.OPEN) {
        this.socket.send(JSON.stringify({ type: 'ping' }));
      }
    }, 25000);
  },

  handle(type, data) {
    switch (type) {
      case 'connected':
        State.applyConnected(data);
        break;
      case 'agent.run_started':
        Chat.busy = true;
        Chat.setThinkingLabel(data.mode === 'rules' ? 'kural tabanlı çalışıyor…' : 'düşünüyor…');
        break;
      case 'agent.run_started':
        Chat.steps = [];
        // ⚠ 0 = sınırsız: '||' 0'ı yutuyordu, ?? kullanılır
    Chat.stepLimit = data.max_steps ?? 0;
        Chat.updateStepCounter();
        break;
      case 'agent.step':
        Chat.onStep(data);
        break;
      case 'agent.run_finished':
        Chat.onRunFinished(data);
        State.refresh();
        ChatList.refresh();
        break;
      case 'learning.recorded':
        Chat.onLearningRecorded(data);
        Learning.refresh();
        break;
      case 'learning.training_started':
        toast('Öğrenme başladı — yerel model eğitiliyor', 'ok');
        Learning.refresh();
        break;
      case 'learning.training_finished':
        if (data.ok) {
          const before = typeof data.initial_loss === 'number' ? data.initial_loss.toFixed(2) : '?';
          const after = typeof data.final_loss === 'number' ? data.final_loss.toFixed(2) : '?';
          toast(`Eğitim tamamlandı: ${data.steps_completed || 0} adım, `
                + `kayıp ${before} → ${after}`, 'ok');
        } else {
          toast(`Eğitim başarısız: ${data.error || 'bilinmeyen hata'}`, 'err');
        }
        Learning.refresh();
        break;
      case 'learning.note_written':
        toast(`Bilgi kasasına eklendi: ${data.title}`, 'ok');
        break;
      case 'study.round_started':
        toast(`Öğreniliyor (${data.technique_label || data.technique}): `
              + `${(data.topic || '').slice(0, 44)}`, 'ok');
        Study.refresh();
        break;
      case 'study.round_finished':
        if (data.ok) {
          toast(`Öğrenildi: ${(data.note_title || data.topic || '').slice(0, 48)}`
                + ` · ${data.sources || 0} kaynak`, 'ok');
        } else {
          toast(`Öğrenme turu başarısız: ${data.error || '?'}`, 'warn');
        }
        Study.refresh();
        Learning.refresh();
        break;
      case 'shell.process_started':
        Running.refresh();
        break;
      case 'shell.process_finished':
      case 'shell.processes_killed':
        Running.refresh();
        break;
      // Sunucu tarafında sohbet/klasör değiştiğinde listeyi tazele:
      // arayüz ile sunucu arasındaki bir yarışta etiket eskimesin.
      case 'chats.created':
      case 'chats.updated':
      case 'chats.activated':
      case 'chats.renamed':
      case 'chats.cleared':
      case 'chats.deleted':
      case 'chats.workspace_changed':
      case 'settings.workspace_changed':
        ChatList.refresh();
        break;
      case 'agent.approval_requested':
        Approval.enqueue(data);
        break;
      case 'agent.approval_granted':
        toast(`Onay verildi: ${data.tool}`, 'ok');
        break;
      case 'agent.approval_denied':
        toast(`Reddedildi: ${data.tool}`, 'warn');
        break;
      case 'agent.approval_resolved':
        Approval.dismiss(data.id, data.approved, data.how);
        break;
      case 'tool.executed':
        if (data.ok) {
          toast(`Araç tamamlandı: ${data.tool}`, 'ok');
          Chat.clearThinking();
          Chat.finish();
          if (data.render) {
            Chat.addMessage('agent', `**${data.tool}**\n\n${data.render}`);
          }
        } else if (data.denied) {
          toast(`Onay verilmedi: ${data.tool}`, 'warn');
        } else {
          toast(`Araç hatası: ${data.tool}`, 'err');
        }
        break;
      case 'agent.autonomy_changed':
        State.applyAutonomy(data);
        toast(`Özerklik: ${data.label}`, 'warn');
        break;
      case 'agent.cancelled':
        Chat.clearThinking();
        Chat.finish();
        break;
      case 'settings.llm_updated':
        State.refresh();
        break;
      case 'settings.workspace_changed':
        $('workspace-name').textContent = data.workspace.split(/[\\/]/).pop();
        $('workspace-name').title = data.workspace;
        toast(`Çalışma alanı: ${data.workspace}`, 'ok');
        break;
      case 'pong':
        break;
      default:
        break;
    }
  },
};

/* ─── 4. Onay modalı ─────────────────────────────────────────────────── */

const Approval = {
  queue: [],
  current: null,

  init() {
    $('approval-allow-once').onclick = () => this.decide(true, false);
    $('approval-allow-remember').onclick = () => this.decide(true, true);
    $('approval-deny').onclick = () => this.decide(false, false);
    document.addEventListener('keydown', (event) => {
      if (!this.current) return;
      if (event.key === 'Escape') this.decide(false, false);
      if (event.key === 'Enter' && (event.ctrlKey || event.metaKey)) this.decide(true, false);
    });
  },

  enqueue(approval) {
    if (!this.current) { this.show(approval); return; }
    this.queue.push(approval);
    this.renderQueue();
  },

  show(approval) {
    this.current = approval;
    $('approval-summary').textContent = approval.summary || approval.tool;
    $('approval-tool').textContent = approval.tool;
    const badge = $('approval-safety');
    badge.textContent = approval.safety_label || approval.safety;
    badge.className = `badge safety-${approval.safety}`;
    $('approval-reason').textContent = approval.reason || '';
    $('approval-args').textContent = JSON.stringify(approval.args || {}, null, 2);
    $('approval-backdrop').hidden = false;
    this.renderQueue();
    $('approval-allow-once').focus();
  },

  renderQueue() {
    const node = $('approval-queue');
    node.textContent = this.queue.length
      ? `Sırada ${this.queue.length} onay isteği daha var.`
      : '';
  },

  async decide(approved, remember) {
    const approval = this.current;
    if (!approval) return;
    this.current = null;
    $('approval-backdrop').hidden = true;

    if (this.queue.length) {
      this.show(this.queue.shift());
    }

    const result = await api('/api/approve', {
      method: 'POST',
      body: JSON.stringify({ id: approval.id, approved, remember }),
    });

    if (!result.ok) {
      // İstek zaman aşımına uğramış veya sunucu yeniden başlamış olabilir.
      // Kullanıcıyı modalda bırakmak yerine durumu açıkça bildir.
      toast(`Onay iletilemedi: ${result.error || 'bilinmeyen hata'}`, 'err');
      return;
    }
    if (approved) toast(remember ? `${approval.tool} her zaman izinli` : 'Bir kez izin verildi', 'ok');
    else toast(`${approval.tool} reddedildi`, 'warn');
  },

  /**
   * Bir onay isteğini kapatır (onaylandı, reddedildi veya zaman aşımı).
   *
   * Sunucu çözümlemeyi bildirir; aksi hâlde zaman aşımına uğramış bir istek
   * ekranda takılı kalır ve kullanıcı butona bassa da modal kapanmaz.
   */
  dismiss(id, approved, how) {
    if (this.current && this.current.id === id) {
      this.current = null;
      $('approval-backdrop').hidden = true;
      if (this.queue.length) this.show(this.queue.shift());
      if (how === 'zaman aşımı') {
        toast('Onay zaman aşımına uğradı — işlem reddedildi', 'warn');
      } else if (how === 'iptal' || how === 'kapatma') {
        toast('Onay isteği iptal edildi', 'warn');
      }
      return;
    }
    this.queue = this.queue.filter((item) => item.id !== id);
    this.renderQueue();
  },

  /**
   * Sunucudaki bekleyen onaylarla arayüzü eşitler.
   *
   * WebSocket olayı kaçırılmış olabilir (sayfa yenilendi, bağlantı koptu).
   * Bu denetim, olmayan bir onay için modalın açık kalmasını engeller ve
   * sunucuda bekleyen bir onay varsa modalı açar.
   */
  async sync() {
    const data = await api('/api/approvals');
    if (!data.ok) return;
    const pending = data.pending || [];

    if (this.current && !pending.some((item) => item.id === this.current.id)) {
      // Gösterdiğimiz onay artık sunucuda yok: kapat
      this.current = null;
      $('approval-backdrop').hidden = true;
    }
    this.queue = this.queue.filter((item) => pending.some((p) => p.id === item.id));

    if (!this.current && pending.length) {
      this.show(pending[0]);
      this.queue = pending.slice(1);
      this.renderQueue();
    }
  },
};

/* ─── 5. Yan panel ve üst durum ──────────────────────────────────────── */

const State = {
  data: null,

  init() {
    // ⚠ ?panel=project|puzzle|steps → sekme otomatik açılır (panel testi)
    try {
      const istenen = new URLSearchParams(location.search).get('panel');
      if (istenen) {
        const hedef = document.querySelector(`.side-tab[data-tab="${istenen}"]`);
        if (hedef) setTimeout(() => hedef.click(), 400);
      }
    } catch (e) { /* yok */ }

    document.querySelectorAll('.side-tab').forEach((tab) => {
      // ⚠ Puzzle sekmesine geçilince ağacı tazele
      tab.addEventListener('click', () => {
        if (tab.dataset.tab === 'puzzle') this.puzzleYenile();
        if (tab.dataset.tab === 'project') this.projectYenile();
      });
      tab.onclick = () => {
        document.querySelectorAll('.side-tab').forEach((t) => t.classList.remove('active'));
        document.querySelectorAll('.side-view').forEach((v) => v.classList.remove('active'));
        tab.classList.add('active');
        // ⚠ ÖNCE tüm panelleri kapat: aksi hâlde önceki panel görünür
        // kalıyordu (ölçüldü: Proje sekmesinde "No files changed" satırı).
        document.querySelectorAll('.side-view').forEach((v) => {
          v.classList.remove('active');
        });
        const hedefPanel = $(`view-${tab.dataset.tab}`);
        if (hedefPanel) hedefPanel.classList.add('active');
        // "Dosyalar" sekmesi çalışma alanı ağacını gösterir; ilk açılışta
        // kökü yükle.
        if (tab.dataset.tab === 'files') Tree.reload();
      };
    });
    // Ağaç satırları olay delegasyonuyla yönetilir (satırlar sık yeniden
    // çizilir; her satıra dinleyici bağlamak yerine kapsayıcı dinlenir).
    $('tree-body').addEventListener('click', (event) => {
      const row = event.target.closest('.tree-row');
      if (row) Tree.onClick(row);
    });
    $('autonomy-select').onchange = (event) => this.setAutonomy(event.target.value);
    this.refresh();
    setInterval(() => this.refresh(), 15000);
  },

  applyConnected(data) {
    if (data.workspace) {
      $('workspace-name').textContent = data.workspace.split(/[\\/]/).pop();
      $('workspace-name').title = data.workspace;
    }
    if (data.autonomy) this.applyAutonomy(data.autonomy);
    if (data.mode) this.applyMode(data.mode);
  },

  applyMode(mode) {
    const node = $('agent-mode');
    const labels = { tools: 'araç çağrısı', prompt: 'JSON istemi', rules: 'kural tabanlı' };
    node.textContent = labels[mode] || mode;
    node.className = `pill mode-${mode}`;
  },

  applyAutonomy(autonomy) {
    $('autonomy-select').value = autonomy.level;
    const hint = $('autonomy-hint');
    hint.textContent = autonomy.description || '';
    hint.className = `autonomy-hint level-${autonomy.level}`;
  },

  async setAutonomy(level) {
    const result = await api('/api/autonomy', {
      method: 'POST',
      body: JSON.stringify({ level }),
    });
    if (result.ok) {
      this.applyAutonomy({ ...result.autonomy, description: this.levelDescription(level) });
      toast(`Özerklik: ${result.autonomy.label}`, 'warn');
    } else {
      toast(result.error || 'seviye değiştirilemedi', 'err');
    }
    this.refresh();
  },

  levelDescription(level) {
    const map = {
      readonly: 'Hiçbir değişiklik yapılmaz. Yalnızca okuma, arama, analiz ve ekran görüntüsü.',
      supervised: 'Yazma, çalıştırma ve yıkıcı işlemler onay ister. Okuma serbest.',
      trusted: 'Yazma, test ve derleme serbest; silme ve sistem ayarı onay ister.',
      yolo: 'Her şey serbest; yalnızca kum havuzu sınırları geçerli.',
    };
    return map[level] || '';
  },

  async refresh() {
    const data = await api('/api/state');
    if (!data.ok) return;
    this.data = data;

    $('workspace-name').textContent = data.workspace_name || '';
    $('workspace-name').title = data.workspace || '';
    this.applyMode(data.agent_mode);

    // Model rozeti: artık **tıklanabilir** bir değiştirici (metin ayrı
    // elemanda tutulur ki ok işareti kaybolmasın).
    const pill = $('model-pill');
    const pillText = $('model-pill-text');
    const lastError = String(data.llm.last_error || '');
    const quotaHit = /429|quota|exceeded/i.test(lastError);
    if (data.llm.available) {
      pillText.textContent = `${data.llm.active} · ${data.llm.model || '?'}`;
      pill.title = quotaHit
        ? 'Kota doldu — değiştirmek için tıklayın'
        : `Araç çağrısı: ${data.llm.supports_tools ? 'var' : 'yok'} · değiştirmek için tıklayın`;
      pill.classList.toggle('quota-low', quotaHit);
      pill.classList.add('model-on');
    } else {
      pillText.textContent = 'model yok';
      pill.title = data.llm.note || 'Model seçmek için tıklayın';
      pill.classList.remove('model-on', 'quota-low');
    }

    if (data.autonomy?.level) this.applyAutonomy(data.autonomy);

    $('usage-tools').textContent = `${data.tools.total} araç`;
    $('usage-calls').textContent = `${data.tools.calls} çağrı`;
    const tokens = (data.llm.stats?.prompt_tokens || 0) + (data.llm.stats?.completion_tokens || 0);
    $('usage-tokens').textContent = `${tokens.toLocaleString('tr-TR')} token`;

    // ⚠ Ortak havuz: kaç makine bağlı? (SSH'sız; kalp atışından önbellek)
    const elHavuz = $('usage-pool');
    if (elHavuz) {
      const havuz = data.pool || {};
      const sayi = havuz.dugum_sayisi || 0;
      if (!sayi) {
        elHavuz.textContent = 'havuz —';
        elHavuz.title = 'Havuz kaydı henüz yok';
      } else {
        elHavuz.textContent = `${sayi} makine`;
        elHavuz.title = havuz.canli
          ? `Havuzda ${sayi} makine · bu düğüm canlı (${havuz.ip || '?'})`
          : `Son kayıt eski — bağlantı koptu olabilir`;
      }
    }

    if (!data.llm.available) {
      $('welcome-note').innerHTML =
        '⚠ Dil modeli yapılandırılmadı — kural tabanlı modda çalışıyorum. '
        + 'Serbest akıl yürütme için <strong>Ayarlar → Dil Modeli</strong> bölümünden '
        + 'bir API anahtarı girin.';
    } else {
      $('welcome-note').textContent = '';
    }
  },

  async setAnswer(text) { Side.answer = text; },
};

const Side = { answer: '' };

/* ─── 6. Araçlar çekmecesi ───────────────────────────────────────────── */

const Tools = {
  all: [],
  category: '',

  init() {
    $('btn-tools').onclick = () => this.open();
    document.querySelectorAll('[data-close]').forEach((button) => {
      button.onclick = () => { $(button.dataset.close).hidden = true; };
    });
    $('tools-search').oninput = () => this.render();
  },

  async open() {
    $('tools-backdrop').hidden = false;
    const data = await api('/api/tools');
    if (!data.ok) { toast(data.error || 'araçlar alınamadı', 'err'); return; }
    this.all = data.tools || [];

    const categories = { '': this.all.length, ...(data.categories || {}) };
    $('tools-categories').innerHTML = Object.entries(categories).map(([key, count]) =>
      `<span class="chip ${this.category === key ? 'active' : ''}" data-cat="${esc(key)}">`
      + `${key || 'tümü'} (${count})</span>`).join('');

    $('tools-categories').querySelectorAll('.chip').forEach((chip) => {
      chip.onclick = () => {
        this.category = chip.dataset.cat;
        this.open();
      };
    });
    this.render();
  },

  render() {
    const query = $('tools-search').value.toLowerCase();
    const tools = this.all.filter((tool) => {
      if (this.category && tool.category !== this.category) return false;
      if (!query) return true;
      return tool.name.toLowerCase().includes(query)
          || tool.description.toLowerCase().includes(query);
    });

    $('tools-list').innerHTML = tools.length ? tools.map((tool) => `
      <div class="tool-card">
        <div class="tool-card-head">
          <span class="tool-name">${esc(tool.name)}</span>
          <span class="badge safety-${esc(tool.safety)}">${esc(tool.safety_label)}</span>
        </div>
        <p class="tool-desc">${esc(tool.description)}</p>
        ${tool.parameters?.length
          ? `<p class="tool-params">parametreler: ${tool.parameters.map(esc).join(', ')}</p>`
          : ''}
      </div>`).join('') : '<div class="empty-hint">Eşleşen araç yok.</div>';
  },
};

/* ─── 7. Ayarlar ─────────────────────────────────────────────────────── */

const Settings = {
  catalog: [],
  browsePath: '',

  init() {
    $('btn-settings').onclick = () => this.open();
    $('llm-save').onclick = () => this.saveKey();
    $('llm-test').onclick = () => this.testKey();
    $('llm-clear').onclick = () => this.clearKey();
    $('llm-provider').onchange = () => this.providerChanged();
    $('ws-apply').onclick = () => this.applyWorkspace();
    $('ws-up').onclick = () => this.browse(this.parentOf(this.browsePath || $('ws-path').value));
    $('ws-path').addEventListener('keydown', (event) => {
      if (event.key === 'Enter') this.browse(event.target.value);
    });
    // Ayarlar **açılmadan** da doldur: üst çubuktaki model değiştirici
    // sağlayıcıyı değiştirdiğinde panelin tutarlı olması gerekir.
    this.fill();
  },

  async open() {
    $('settings-backdrop').hidden = false;
    await this.fill();
    const data = await api('/api/settings');
    $('ws-path').value = data.workspace || '';
    this.browse(data.workspace);
  },

  /**
   * Sağlayıcı ve model kontrollerini **şimdi** doldurur.
   *
   * `open()` yalnızca panel görünürken çalıştığı için, kullanıcı Ayarlar'ı
   * hiç açmadıysa oradaki liste boş kalıyordu — üst çubuktan sağlayıcı
   * değiştirildiğinde panel tutarsız görünüyordu.
   */
  async fill() {
    const data = await api('/api/settings');
    if (!data.ok) {
      toast(data.error || 'ayarlar alınamadı', 'err');
      return;
    }
    this.catalog = data.provider_catalog || [];

    const select = $('llm-provider');
    select.innerHTML = this.catalog.map((item) =>
      `<option value="${esc(item.name)}">${esc(item.label)}${item.local ? ' (yerel)' : ''}</option>`
    ).join('');
    // Aktif sağlayıcı `providers[].active` bayrağından bulunur: `active_key`
    // sağlayıcı kimliği **değildir** (`openai_compatible` gibi tür döner).
    const live = (data.llm || {}).providers || [];
    const activeEntry = live.find((item) => item.active);
    if (activeEntry && this.catalog.some((item) => item.name === activeEntry.name)) {
      select.value = activeEntry.name;
    }
    this.renderLlmStatus(data.llm);
    this.providerChanged();
  },

  renderLlmStatus(llm) {
    $('llm-note').textContent = llm?.available
      ? `Aktif: ${llm.active} (${llm.active_model || 'varsayılan'}) · `
        + `araç çağrısı: ${llm.supports_tools ? 'destekleniyor' : 'yok (JSON istemi kullanılır)'}`
      : (llm?.note || 'Dil modeli yapılandırılmadı.');

    $('llm-status').innerHTML = (llm?.providers || []).map((provider) => `
      <div class="provider-row ${provider.available ? 'available' : ''} ${provider.active ? 'active' : ''}">
        <span class="provider-dot"></span>
        <span>${esc(provider.label)}</span>
        ${provider.active ? '<span class="badge">aktif</span>' : ''}
        <span class="provider-model">${esc(provider.model || '')}</span>
      </div>`).join('');
  },

  providerChanged() {
    const provider = this.catalog.find((item) => item.name === $('llm-provider').value);
    if (!provider) return;
    $('llm-help').textContent = provider.needs_key
      ? `${provider.help} Ortam değişkeni: ${provider.env_var}`
      : provider.help;
    $('llm-key').placeholder = provider.needs_key ? 'sk-…' : 'yerel sunucu — anahtar gerekmez';
    $('llm-key').disabled = !provider.needs_key;

    // Model açılır listesi: adlar sık değişir (Google `gemini-2.0-flash` için
    // "artık kullanılamıyor" diyor), bu yüzden **çalışan** adlar listelenir.
    const select = $('llm-model');
    const suggested = provider.suggested_models || [];
    const known = suggested.length ? suggested : [provider.default_model].filter(Boolean);
    select.innerHTML = known.map((name, index) =>
      `<option value="${esc(name)}">${esc(name)}${index === 0 ? '  (önerilen)' : ''}</option>`
    ).join('') + '<option value="">— varsayılan —</option>';
    // Sunucudaki kayıtlı seçim varsa onu işaretle
    if (provider.model && known.includes(provider.model)) select.value = provider.model;
    $('llm-model-custom').value = '';
    $('llm-model-custom').placeholder = provider.model && !known.includes(provider.model)
      ? `şu an kayıtlı: ${provider.model}`
      : 'listede yoksa adı buraya yazın (isteğe bağlı)';
  },

  async saveKey() {
    const provider = $('llm-provider').value;
    // Özel alan doldurulmuşsa o kazanır; aksi hâlde açılır listedeki seçim.
    const custom = $('llm-model-custom').value.trim();
    const model = custom || $('llm-model').value.trim();
    const result = await api('/api/settings/llm', {
      method: 'POST',
      body: JSON.stringify({
        provider,
        api_key: $('llm-key').value.trim(),
        model,
      }),
    });
    if (result.ok) {
      $('llm-key').value = '';
      $('llm-model-custom').value = '';
      toast(model ? `Kaydedildi — ${provider} / ${model}` : 'Kaydedildi', 'ok');
      this.renderLlmStatus(result.llm);
      await State.refresh();
      ModelSwitch.load();
      this.loadHealth();
    } else {
      toast(result.error || 'kaydedilemedi', 'err');
    }
  },

  async testKey() {
    const node = $('llm-test-result');
    node.hidden = false;
    node.textContent = 'Bağlantı sınanıyor…';
    const result = await api('/api/settings/llm/test', {
      method: 'POST',
      body: JSON.stringify({ provider: $('llm-provider').value }),
    });
    if (result.ok) {
      node.textContent = `✓ Bağlantı başarılı\n`
        + `Sağlayıcı : ${result.provider}\n`
        + `Model     : ${result.model}\n`
        + `Yanıt     : ${result.reply}\n`
        + `Süre      : ${fmtMs(result.duration_ms)}\n`
        + `Token     : ${result.tokens}`;
      toast('Bağlantı başarılı', 'ok');
    } else {
      node.textContent = `✗ Başarısız\n${result.error || 'bilinmeyen hata'}`;
      toast('Bağlantı başarısız', 'err');
    }
  },

  async clearKey() {
    const provider = $('llm-provider').value;
    const result = await api('/api/settings/llm/clear', {
      method: 'POST',
      body: JSON.stringify({ provider }),
    });
    if (result.ok) {
      toast(`${provider} anahtarı silindi`, 'warn');
      this.renderLlmStatus(result.status);
      await State.refresh();
    }
  },

  parentOf(path) {
    if (!path) return '';
    const normalized = path.replace(/[\\/]+$/, '');
    const index = Math.max(normalized.lastIndexOf('\\'), normalized.lastIndexOf('/'));
    return index > 2 ? normalized.slice(0, index) : normalized;
  },

  async browse(path) {
    const result = await api(`/api/settings/browse?path=${encodeURIComponent(path || '')}`);
    if (!result.ok) { toast(result.error || 'dizin okunamadı', 'err'); return; }
    this.browsePath = result.path;
    $('ws-path').value = result.path;

    $('ws-browser').innerHTML = (result.entries || []).map((entry) => `
      <div class="browser-item" data-path="${esc(entry.path)}">
        ◆ ${esc(entry.name)}
        ${entry.has_project ? '<span class="proj">proje</span>' : ''}
      </div>`).join('') || '<div class="empty-hint">Alt dizin yok.</div>';

    $('ws-browser').querySelectorAll('.browser-item').forEach((item) => {
      item.onclick = () => this.browse(item.dataset.path);
    });
  },

  async applyWorkspace() {
    const result = await api('/api/settings/workspace', {
      method: 'POST',
      body: JSON.stringify({ path: $('ws-path').value.trim() }),
    });
    if (result.ok) {
      toast(`Çalışma alanı: ${result.workspace}`, 'ok');
      await State.refresh();
      await this.browse(result.workspace);
    } else {
      toast(result.error || 'değiştirilemedi', 'err');
    }
  },

  async loadHealth() {
    const result = await api('/api/health');
    if (!result.ok && !result.capabilities) return;
    const labels = {
      llm: 'Dil modeli', tools: 'Araç kayıt defteri', coding: 'Kodlama ajanı',
      shell: 'PowerShell / CMD', vision: 'Görsel doğrulama', godot: 'Godot',
      blender: 'Blender', ocr: 'OCR (tesseract)', ui_text: 'Arayüz metni okuma',
    };
    const rows = Object.entries(result.capabilities || {}).map(([key, value]) =>
      `<div class="health-row"><span class="health-dot ${value ? 'on' : 'off'}"></span>`
      + `${esc(labels[key] || key)}</div>`).join('');
    const issues = (result.issues || []).map((issue) =>
      `<div class="health-issue">⚠ ${esc(issue)}</div>`).join('');
    $('health-list').innerHTML = rows + issues;
  },
};

/* ─── 8. Görsel doğrulama ────────────────────────────────────────────── */

const Vision = {
  init() {
    $('btn-vision').onclick = () => { $('vision-backdrop').hidden = false; };
    $('vs-capture').onclick = () => this.capture();
    $('vs-windows').onclick = () => this.windows();
  },

  async capture() {
    $('vs-result').textContent = 'Ekran görüntüsü alınıyor…';
    const result = await api('/api/vision/screenshot', {
      method: 'POST',
      body: JSON.stringify({ tag: 'ui' }),
    });
    if (!result.ok) {
      $('vs-result').textContent = `✗ ${result.error || 'alınamadı'}`;
      return;
    }
    const analysis = result.analysis || {};
    $('vs-result').innerHTML = `
      <div class="health-row"><strong>${esc(result.verdict || '')}</strong></div>
      <div class="health-row">Boyut: ${result.width}×${result.height}</div>
      <div class="health-row">Parlaklık: ${analysis.mean_brightness} · Kontrast: ${analysis.contrast}</div>
      <div class="health-row">Renk sayısı: ${analysis.unique_colors} · Kenar yoğunluğu: ${analysis.edge_density}</div>
      <div class="health-row">Baskın renkler: ${(analysis.dominant_colors || []).join(' ')}</div>`;
    if (result.path) {
      const image = $('vs-image');
      image.src = `/api/vision/shot?path=${encodeURIComponent(result.path)}&t=${Date.now()}`;
      image.hidden = false;
    }
  },

  async windows() {
    $('vs-result').textContent = 'Pencereler listeleniyor…';
    const result = await api('/api/vision/windows?min_size=80');
    if (!result.ok) {
      $('vs-result').textContent = `✗ ${result.error || 'listelenemedi'}`;
      return;
    }
    $('vs-result').innerHTML = (result.windows || []).map((window) => `
      <div class="health-row" style="align-items:flex-start">
        <span class="badge">${window.width}×${window.height}</span>
        <span style="flex:1;overflow-wrap:anywhere">
          ${esc(window.title || '(başlıksız)')}
          <br><span style="color:var(--text-mute);font-size:11.5px">${esc(window.process)}</span>
        </span>
      </div>`).join('') || '<div class="empty-hint">Pencere bulunamadı.</div>';
  },
};


