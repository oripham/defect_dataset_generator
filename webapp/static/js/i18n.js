// i18n.js — English is the default UI language. This maps EN → JA only.
const EN_TO_JA = {
  // Navigation / Sidebar
  "🚀 Tuning Station": "🚀 チューニングステーション",
  "🛠 Data Setup": "🛠 データセット設定",
  "🖌 Mask Editor": "🖌 マスク描画",
  "Language": "言語",
  "● Connected": "● 接続済",
  "● Disconnected": "● 未接続",
  "AI Defect Dataset Generator": "欠陥データセットジェネレーター",
  "Dataset<br>Generator": "欠陥<br>ジェネレーター",

  // Setup Page
  "🛠 Data Setup (Server & Data Configuration)": "🛠 データセット設定",
  "Connect the generation server and configure image folders and defect classes.": "生成サーバーを接続し、正常/不良画像および欠陥クラスを設定します。",
  "🌐 1. Server Connection (GPU)": "🌐 1. サーバー接続 (GPU)",
  "Server URL": "サーバーURL",
  "API Key": "APIキー",
  "(RunPod API Key)": "(RunPod APIキー)",
  "🔁 Test Connection": "🔁 接続テスト",
  "📁 2. Data Folders": "📁 2. データフォルダ",
  "✅ Good Images Folder": "✅ 正常画像フォルダ",
  "Path will be auto-generated after upload...": "アップロード後にパスが生成されます...",
  "📁 Upload Folder": "📁 フォルダ参照",
  "Output Folder": "出力フォルダ",
  "Output subfolder name": "出力先サブフォルダ名",
  "Width (px)": "幅 (px)",
  "Height (px)": "高さ (px)",
  "💾 Save Folder Config": "💾 設定を保存",
  "✓ Saved!": "✓ 保存しました",
  "🏷 3. Defect Classes": "🏷 3. 欠陥クラス",
  "+ Add This Class": "+ このクラスを追加",
  "Class Name": "クラス名",
  "Class ID": "Class ID",
  "❌ Defect Reference Folder (NG Images)": "❌ 不良参考画像フォルダ",
  "(Reference images with actual defects)": "（実際の不良を含む参照画像）",
  "Upload images to auto-generate path...": "アップロードしてパス自動生成...",
  "💡 Masks will be saved automatically inside masks/ of this folder.": "💡 マスクはこのフォルダのmasks/に自動保存されます。",

  // Crop tool
  "✂ Extract Real Defect Sample (For realism)": "✂ 実際の欠陥サンプルを抽出 (リアルさ向上)",
  "This sample will be used as a base to generate more realistic defects": "このサンプルは、よりリアルな欠陥を生成するベースとして使用されます",
  "📤 Upload Pre-cropped Image": "📤 クロップ済み画像をアップロード",
  "✂ Manual Crop (Draw Region)": "✂ 手動クロップ (領域を描画)",
  "Click & Drag to select defect region → 💾 Save": "ドラッグで欠陥領域を選択し保存",
  "Images in folder:": "フォルダ内の画像:",
  "💾 Save Crop": "💾 領域保存",
  "🔧 Show Advanced AI Params (Prompts, Scales)": "🔧 詳細AIパラメータを表示 (プロンプト, スケール)",
  "Prompt (comma separated)": "プロンプト（カンマ区切り）",
  "Negative Prompt": "ネガティブプロンプト",
  "Save Class": "クラスを保存",
  "Delete Class": "クラスを削除",
  "Class Info: ": "クラス情報: ",

  // Defect type names
  "Scratch": "傷・スクラッチ",
  "Crack": "割れ・クラック",
  "Dent": "へこみ・打痕",
  "Bulge": "膨れ",
  "Chip": "欠け",
  "Rust": "サビ",
  "Burn": "焼け",
  "Micro Crack": "微小クラック",
  "Foreign Material": "異物付着",
  "Deform": "整形不良",
  "Flash": "バリ",
  "Sink mark": "ヒケ",
  "Bubble/Void": "気泡・ボイド",
  "Burn/Discoloration": "変色・焼け",
  "Crack/Chip": "欠け・クラック",
  "Foreign material": "異物",
  "All (39 prompts)": "すべて (39)",
  "➕ Add Custom Defect...": "➕ カスタム欠陥を追加...",
  "Enter English defect name (e.g. paint_peel, dirt):": "英語で欠陥名を入力 (例: paint_peel, dirt):",

  // Masking Page
  "🖌 Mask Editor": "🖌 マスクエディタ",
  "1. Draw Mask Region": "1. マスク描画",
  "On the good product image, use the brush to paint": "正常画像にて、ツールを使い",
  "where you want AI to generate defects": "欠陥生成位置を黒く塗ってください",
  "⚠ Warning: Class Has No Mask": "⚠ 警告: マスクがありません",
  "Following classes have no mask drawn:": "以下のクラスにはマスクが未設定です:",
  "Please select an image and continue drawing.": "画像を選択し、描画してください。",
  "Red = Defect Zone / Black = Keep": "赤 = 欠陥領域 / 黒 = 維持",
  "① Select Class": "① クラス選択",
  "-- Select Defect Class --": "-- クラスを選択 --",
  "② Select Good Image": "② 正常画像を選択",
  "Please select a class first": "先にクラスを選択してください",
  "View Ref Image": "参照画像を見る",
  "Hide Background": "背景を隠す",
  "③ Mask Status": "③ マスク状況",
  "④ Drawing Tools": "④ 描画ツール",
  "Brush": "ブラシ",
  "Rect": "矩形",
  "Ellipse": "楕円",
  "Eraser": "消しゴム",
  "Size:": "サイズ:",
  "↩ Undo": "↩ 元に戻す",
  "Clear": "クリア",
  "Invert": "反転",
  "⑤ Save Mask": "⑤ マスクを保存",
  "💾 Save Mask (Overwrite)": "💾 マスクを保存",
  "📂 Upload Mask from PC": "📂 PCからアップロード",

  // Tuning Station
  "🎛️ Tuning Station": "🎛️ チューニングステーション",
  "Model: Hybrid System (GenAI & CV)": "モデル: ハイブリッドシステム (GenAI & CV)",
  "1. Current Data (from Data Setup)": "1. 現在のデータ",
  "Active Good Images Folder:": "アクティブなGood Imagesフォルダ:",
  "Output Folder:": "出力フォルダ:",
  "Selected Defect Config:": "選択中の欠陥構成:",
  "Material: Plastic": "材質: プラスチック (Plastic)",
  "Material: Metal": "材質: 金属 (Metal)",
  "Material: Pharma": "材質: 医薬品 (Pharma)",
  "-- No classes defined --": "-- クラス未定義 --",
  "2. Tuning Parameters": "2. パラメータ調整",
  "Render Engine": "レンダリングエンジン",
  "Basic Mode (CV)": "基本モード (OpenCV)",
  "AI Realistic Defect Mode": "AIリアル欠陥生成モード",
  "Defect Severity (Intensity)": "欠陥の深刻度 (Intensity)",
  "Defect Edge Blend": "欠陥境界のブレンド",
  "Very Smooth": "非常に滑らか",
  "Fair": "普通",
  "Rough": "粗い",
  "Mask Rotation Jitter (± Deg)": "マスク回転ジッター (± 度)",
  "3. Auto QA Guard": "3. 自動品質保証 (QA Guard)",
  "3. Advanced AI Parameters (GenAI Only)": "3. 詳細AIパラメータ (GenAIのみ)",
  "SSIM Filter (Auto Discard)": "SSIMフィルタ (自動破棄)",
  "💡 Auto delete & regen if SSIM < 0.85": "💡 SSIM < 0.85 の場合自動削除＆再生成",
  "📊 FID score will be evaluated after completion.": "📊 完了後にFIDスコアを評価",
  "👁️ CLICK HERE TO GENERATE 1 PREVIEW": "👁️ ここをクリックしてプレビュー生成 (1枚)",
  "Ideate and Render.": "レンダリングを実行",
  "🚀 MASS BATCH GENERATION": "🚀 バッチ生成開始 (BATCH GENERATION)",
  "Batch Size": "生成枚数 (Batch)",
  "Imgs": "枚",
  "▶️ CONFIRM QUALITY & CREATE DATASET": "▶️ 品質確認＆データセット作成",
  "📦 DOWNLOAD FULL DATASET (.ZIP)": "📦 データセットを保存 (.ZIP)",
  "🖼️ Live Results Gallery": "🖼️ ライブ結果ギャラリー",
  "Generated images will load here after running...": "生成された画像はここに表示されます...",
  "Please select a defect config first": "欠陥構成を選択してください",
  "Strength (Denoise)": "強度(Strength / Denoise)",
  "Guidance Scale": "ガイダンス スケール",
  "Steps": "ステップ数",
  "IP Adapter Scale": "IPアダプタ強度",
  "ControlNet Scale": "ControlNet強度",
  "Inject Alpha": "Inject Alpha",
  "Defect Angularity (Epsilon)": "欠陥の角張度 (Epsilon)",
  "Higher = more jagged/angular edges (Good for Chips/Dents)": "値が高いほど、エッジが鋭く（ギザギザに）なります (欠けや打痕に有効)",

  // Progress / Alerts
  "🔄 Sending request to engine...": "🔄 エンジンにリクエスト送信中...",
  "✅ Batch generated successfully! Download now!": "✅ バッチ生成完了！ダウンロード可能です。",
  "❌ Render Error: ": "❌ レンダリングエラー: ",
  "❌ Network Error: ": "❌ ネットワークエラー: ",
  "❌ Server Error: ": "❌ サーバーエラー: ",
  "❌ Flask Error: ": "❌ Flaskエラー: ",
  "❌ JS Error: ": "❌ JSエラー: ",
  "🔄 Running GenAI... QA check... (": "🔄 生成処理中... QAチェック... (",
  "✅ Batch saved to Output! ": "✅ バッチ生成完了！出力先: ",
  "🎉 BATCH COMPLETE!\nDataset written to Output Dir: ": "🎉 バッチ完了！\nデータセット出力先: ",
  "Loading images from Output...": "出力先から画像を読み込み中...",
  "No images in Output yet...": "出力先に画像がありません...",
  "Gallery display error: ": "ギャラリーエラー: ",
  "📸 QA auto discarded ": "📸 QAシステムが自動破棄：",
  " bad noisy images!": " 枚のノイズ画像（低SSIM）",

  // Session
  "🗑 Clear Session": "🗑 セッションをクリア",
  "Are you sure you want to clear all data and reset the session?": "全データを削除してセッションをリセットしてもよろしいですか？",

  // Class list / status
  "No classes yet. Add a defect type to begin.": "欠陥クラスがありません",
  "✅ All classes have masks!": "✅ 全クラスのマスクが揃っています！",
  "⚠ Missing masks for: ": "⚠ マスク不足のクラス: ",
  "Go to Mask Editor →": "マスクエディタへ →",
  "Mask OK": "マスクOK",
  "No Mask": "マスク未設定",
  "✂ Crop OK": "切り抜きOK",
  "No ref images.": "参照画像がありません。",
  "No valid images or files > 10MB!": "有効な画像がないか、50MBを超えています",
  "Uploading ": "アップロード中: ",
  " files... Please wait!": " ファイル... お待ちください",
  "✅ Upload mapped ": "✅ アップロード成功 ",
  " images!": " 枚の画像",
  "Please select a larger region.": "より広い領域を選択してください",
  "✅ Saved!": "✅ 保存完了！",
  "Connecting...": "接続中...",
  "✅ Connected! ": "✅ 接続しました！ ",
  "❌ Failed: ": "❌ 失敗: ",
  "Not loaded...": "未読み込み...",
  "Not configured...": "未設定...",
  "Please load Good Images in Setup Tab": "Setupタブで正常画像を読み込んでください",
  "✅ Saved for image ": "✅ 画像の保存完了: ",
  "✅ Mask OK": "✅ マスクOK",
  "⚠ No mask drawn": "⚠ マスク未描画",
  "✅ Mask ready": "✅ マスク準備完了",
  "No NG images in folder": "フォルダにNG画像がありません",
  "(Empty)": "(空)",

  // Backend errors
  "Defect class does not exist": "欠陥クラスが存在しません",
  "Good images not loaded": "正常画像が読み込まれていません",
  "Good image not found": "正常画像が見つかりません",
  "Mask not found": "マスクが見つかりません",
  "Mask file not found": "マスクファイルが見つかりません",
  "No image received from engine": "サーバーから画像を受信できません",
  "Cannot connect to API: ": "APIに接続できません: ",
  "Not connected to Server API": "サーバーAPIに接続されていません",
  "Following classes lack masks: ": "マスクが不足しているクラス: ",
  "No valid good images.": "有効な正常画像がありません。",
  "No classes meet generation criteria.": "実行条件を満たすクラスがありません。",
  "Job ID not found": "JobIDが見つかりません",
  "Internal loop error: ": "内部ループエラー: ",
  "Job send failed: HTTP ": "ジョブ送信失敗: HTTP ",
  "Poll status error: HTTP ": "ステータスポールエラー: HTTP ",
  "Class name is empty": "クラス名が空です",
  "Class name already exists": "同じ名前のクラスがすでに存在します",
  "Class not found": "クラスが見つかりません",
  "Not found": "見つかりません",
  "Missing upload type": "アップロードタイプがありません",
  "No files loaded": "ファイルが読み込まれていません",
  "Invalid upload parameters": "不正なアップロードパラメータ",
  "Class or defect image folder not configured": "クラスまたは不良画像フォルダが未設定です",
  "Image decode error: ": "画像デコードエラー: ",
  "Invalid class or folder name": "クラスまたはフォルダ名が不正です",
  "Please select a subfolder": "サブフォルダを選択してください",
  "💡 Auto delete & regen if SSIM < ": "💡 SSIMが指定値を下回れば自動再生成: < ",
  "Defect: ": "欠陥: ",

  // QA Review Page
  "🔍 QA Review": "🔍 精度検証 (QA Review)",
  "No images loaded": "画像が読み込まれていません",
  "Review each generated image.": "生成された各画像をレビューします。",
  "Accept": "採用",
  "to keep or": "して保存、または",
  "Reject": "不採用",
  "to delete.": "して削除します。",
  "SSIM scores are shown when available.": "利用可能な場合、SSIMスコアが表示されます。",
  "Source Batch": "対象バッチ",
  "-- Select Batch --": "-- バッチを選択 --",
  "🔄 Refresh": "🔄 更新",
  "✅ Accept All": "✅ すべて採用",
  "🗑 Delete Rejected": "🗑 不採用分を削除",
  "accepted": "採用済み",
  "rejected": "不採用",
  "total": "合計",
  "Select a batch from the dropdown above, or generate images in the Tuning Station first.": "上のドロップダウンからバッチを選択するか、先にチューニングステーションで画像を生成してください。",
  "images, ": "枚の画像, ",
  "Select a batch above.": "バッチを選択してください。",
  "Loading...": "読み込み中...",
  "images — Batch ": "枚の画像 — バッチ ",
  "Drop": "除外",
  "Keep": "採用",
  "No images in this batch.": "このバッチに画像はありません。",
  "No images marked for rejection.": "削除対象としてマークされた画像はありません。",
  "Delete ": "削除: ",
  " rejected image(s)? This cannot be undone.": " 枚の不採用画像。この操作は取り消せません。",
  " — Click to close": " — クリックで閉じる"
};

function walkAndTranslate(node, lang) {
  if (node.nodeType === 3) {
    let text = node.nodeValue.trim();
    if (text) {
      if (EN_TO_JA[text]) {
        node.nodeValue = node.nodeValue.replace(text, EN_TO_JA[text]);
      } else {
        Object.keys(EN_TO_JA).forEach(key => {
          if (text.includes(key) && key.length > 4) {
            node.nodeValue = node.nodeValue.replace(key, EN_TO_JA[key]);
            text = node.nodeValue.trim();
          }
        });
      }
    }
  } else if (node.nodeType === 1 && node.nodeName !== 'SCRIPT' && node.nodeName !== 'STYLE') {
    ['placeholder', 'title'].forEach(attr => {
      let t = node.getAttribute(attr);
      if (t) {
        t = t.trim();
        if (EN_TO_JA[t]) node.setAttribute(attr, EN_TO_JA[t]);
      }
    });
    for (let i = 0; i < node.childNodes.length; i++) {
      walkAndTranslate(node.childNodes[i], lang);
    }
  }
}

function setLanguage(lang) {
  fetch('/api/set-language', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ language: lang })
  }).then(() => window.location.reload());
}

document.addEventListener('DOMContentLoaded', () => {
  const selectEl = document.getElementById('lang-select');
  if (selectEl) {
    const lang = selectEl.value;
    if (lang === 'ja') {
      walkAndTranslate(document.body, lang);
      if (EN_TO_JA[document.title]) {
        document.title = EN_TO_JA[document.title];
      }
    }
  }
});

// Helper for JS alerts/strings
window.t = function(text) {
  const lang = document.getElementById('lang-select')?.value || 'en';
  if (lang !== 'ja') return text;
  if (EN_TO_JA[text]) return EN_TO_JA[text];
  let translated = text;
  Object.keys(EN_TO_JA).forEach(key => {
    if (translated.includes(key) && key.length > 4) {
      translated = translated.replace(key, EN_TO_JA[key]);
    }
  });
  return translated;
};
