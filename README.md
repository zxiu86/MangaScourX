# mangascourx

![Version](https://img.shields.io/badge/version-1.0.2-blue.svg)
![Python](https://img.shields.io/badge/python-3.8%2B-brightgreen.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Status](https://img.shields.io/badge/status-Beta-orange.svg)

**Advanced Multi-Scale PatchMatch & AI-Powered Text Removal Engine for Manga**

mangascourx هي مكتبة متقدمة متخصصة في **إزالة النصوص من صور المانجا** باستخدام تقنيات حديثة من مجال معالجة الصور والـ Computer Vision. تجمع بين خوارزميات كلاسيكية (PatchMatch، Telea) وتقنيات حديثة مدعومة بالـ AI (CRAFT للنصوص، MSER للكشف السريع).

---

## 📋 المحتويات

- [الميزات الرئيسية](#الميزات-الرئيسية)
- [المتطلبات](#المتطلبات)
- [التثبيت](#التثبيت)
- [الاستخدام السريع](#الاستخدام-السريع)
- [الواجهة البرمجية](#الواجهة-البرمجية)
- [البنية المعمارية](#البنية-المعمارية)
- [السجل الكامل للإصدارات](#السجل-الكامل-للإصدارات)
- [المساهمة](#المساهمة)
- [الترخيص](#الترخيص)

---

## ✨ الميزات الرئيسية

### 1. **الكشف الذكي للنصوص**
- **MSER** (Maximally Stable Extremal Regions) — سريع، لا يحتاج AI
- **CRAFT** (Character Region Awareness For Text detection) — دقيق، مدعوم بالشبكات العصبية
- **Fallback تلقائي** — إذا اكتشف MSER عددًا قليلًا من النصوص، ينتقل تلقائيًا إلى CRAFT

### 2. **كشف فقاعات الحوار**
- بناءً على الكنتور والمورفولوجيا
- معالجة متكيفة للأشكال المختلفة
- تنظيف الضوضاء التلقائي

### 3. **الإصلاح متعدد المستويات**
- **PatchMatch** — خوارزمية fast-approximate nearest neighbor
- **Coherence Transport** — نقل الملمس بناءً على تدفق الخطوط
- **Telea Fast Marching** — ملء سريع باستخدام المسافات

### 4. **الأداء العالي**
- معالجة الصور بأحجام كبيرة
- تحسينات JIT مع Numba
- معالجة متوازية (عند توفر Numba)

### 5. **مرونة التكوين**
- معاملات قابلة للتخصيص لكل مرحلة
- اختيار الخوارزمية حسب الاحتياجات
- دعم معاينة النتائج المرحلية

---

## 📦 المتطلبات

### النظام
- **Python**: 3.8 أو أحدث
- **نظام التشغيل**: Linux, macOS, Windows

### المكتبات الأساسية
```
numpy >= 1.20.0
opencv-python-headless >= 4.5.0
scipy >= 1.7.0
```

### خوارزميات متقدمة (اختيارية)
```
numba >= 0.53.0          # تحسين الأداء (جيد جداً)
torch >= 1.9.0           # لـ CRAFT (لا يزال قيد التطوير)
torchvision >= 0.10.0    # لـ CRAFT
```

---

## 🚀 التثبيت

### من PyPI (الطريقة الموصى بها)
```bash
pip install mangascourx
```

### من GitHub (آخر إصدار تطوير)
```bash
pip install git+https://github.com/zxui86/mangascourx.git
```

### من المصدر (للمساهمين)
```bash
git clone https://github.com/zxui86/mangascourx.git
cd mangascourx
pip install -e .
```

### التثبيت مع الأدوات الاختيارية
```bash
# مع دعم numba الكامل (موصى به)
pip install mangascourx[numba]

# مع دعم CRAFT (يتطلب CUDA لـ GPU)
pip install mangascourx[craft]

# كل شيء
pip install mangascourx[dev,numba,craft]
```

---

## 💡 الاستخدام السريع

### مثال بسيط — إزالة النصوص

```python
import cv2
from mangascourx import TextRemovePipeline

# قراءة الصورة
image = cv2.imread("manga_page.jpg")

# إنشاء خط أنابيب الإزالة
pipeline = TextRemovePipeline(
    inpainter_type="patchmatch",  # أو "telea" أو "coherence"
    enable_bubbles=True,           # اكتشف فقاعات الحوار أيضاً
    verbose=True
)

# معالجة الصورة
result = pipeline.run(image)

# حفظ النتيجة
cv2.imwrite("output.jpg", result)
```

### مثال متقدم — مع تحكم دقيق

```python
from mangascourx import TextRemovePipeline
from mangascourx.detection import DetectionOrchestrator
from mangascourx.inpainting.patchmatch import PatchMatchInpainter

# 1. إنشاء كاشف النصوص مع إعدادات مخصصة
detector = DetectionOrchestrator(
    craft_model_path="path/to/craft_weights.pth",
    device="cuda",  # أو "cpu"
    mser_params={
        "delta": 5,
        "min_area": 30,
        "max_area": 14400,
    },
    merge_priority=["text", "bubbles"],
    final_cleanup=True,
)

# 2. إنشاء خوارزمية الإصلاح
inpainter = PatchMatchInpainter(
    patch_size=15,
    num_levels=4,
    num_iters=5,
    k=5,
)

# 3. بناء خط الأنابيب الكامل
pipeline = TextRemovePipeline(
    detector=detector,
    inpainter=inpainter,
    verbose=True,
)

# 4. معالجة الصورة
result = pipeline.run(image)
```

### الحصول على النتائج المرحلية

```python
# الحصول على قناع النصوص بدون إصلاح
detection_result = detector.run(image)
text_mask = detection_result["mask"]
text_boxes = detection_result.get("text_boxes", [])

# عرض النتائج
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

axes[0, 0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
axes[0, 0].set_title("الصورة الأصلية")

axes[0, 1].imshow(text_mask, cmap="gray")
axes[0, 1].set_title("قناع النصوص")

axes[1, 0].imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
axes[1, 0].set_title("بعد الإزالة")

axes[1, 1].imshow(detection_result.get("bubble_mask", np.zeros_like(text_mask)), cmap="gray")
axes[1, 1].set_title("قناع الفقاعات")

for ax in axes.flat:
    ax.axis("off")

plt.tight_layout()
plt.show()
```

---

## 📚 الواجهة البرمجية

### `TextRemovePipeline`

**الفئة الرئيسية للعملية الكاملة**

```python
pipeline = TextRemovePipeline(
    inpainter_type="patchmatch",        # الخوارزمية: "patchmatch"، "telea"، "coherence"
    detector_config=None,                # dict اختياري لإعدادات الكاشف
    inpainter_config=None,               # dict اختياري لإعدادات الإصلاح
    enable_text=True,                    # اكتشف النصوص
    enable_bubbles=True,                 # اكتشف الفقاعات
    verbose=False,                       # طباعة التفاصيل
)

# الاستدعاء الرئيسي
result = pipeline.run(image)
```

### `DetectionOrchestrator`

**كاشف النصوص والفقاعات الموحد**

```python
detector = DetectionOrchestrator(
    craft_model_path=None,               # مسار نموذج CRAFT (اختياري)
    device="cpu",                        # "cpu" أو "cuda"
    mser_params={},                      # معاملات MSER
    craft_params={},                     # معاملات CRAFT
    bubble_params={},                    # معاملات الفقاعات
    merge_priority=["text", "bubbles"],  # أولوية الدمج
    final_cleanup=True,                  # تنظيف مورفولوجي نهائي
    fallback_min_boxes=3,                # الحد الأدنى قبل الانتقال إلى CRAFT
)

result = detector.run(image, enable_text=True, enable_bubbles=True)
# result["mask"]           — القناع الموحد
# result["text_mask"]      — قناع النصوص فقط
# result["bubble_mask"]    — قناع الفقاعات فقط
# result["text_boxes"]     — قوائم المربعات النصية
# result["bubble_boxes"]   — قوائم المربعات للفقاعات
```

### خوارزميات الإصلاح

#### PatchMatchInpainter
```python
from mangascourx.inpainting.patchmatch import PatchMatchInpainter

inpainter = PatchMatchInpainter(
    patch_size=15,           # حجم الرقعة
    num_levels=4,            # مستويات الهرم
    num_iters=5,             # التكرارات لكل مستوى
    k=5,                     # عدد أفضل المرشحين
    random_search_radius=20, # نطاق البحث العشوائي
    use_features=True,       # استخدم تدرجات اللون
    verbose=False,
)

result = inpainter.run(image, mask)
```

#### TeleaInpainter
```python
from mangascourx.inpainting.telea import TeleaInpainter

inpainter = TeleaInpainter(radius=5)
result = inpainter.run(image, mask)
```

#### CoherenceTransport
```python
from mangascourx.inpainting.coherence import CoherenceTransport

inpainter = CoherenceTransport(
    num_iters=10,
    step_size=0.1,
)
result = inpainter.run(image, mask)
```

---

## 🏗️ البنية المعمارية

```
mangascourx/
├── __init__.py
├── _version.py
├── manga_clean.py              # خط أنابيب التنظيف الكامل
├── text_remove.py              # خط أنابيب إزالة النصوص
│
├── detection/
│   ├── __init__.py
│   ├── detection.py            # DetectionOrchestrator
│   ├── mask.py                 # معالجة القناع والدمج
│   ├── bubbles/
│   │   ├── contours.py         # كشف الكنتور
│   │   └── morphology.py       # عمليات مورفولوجية
│   └── text/
│       ├── mser.py             # كشف MSER
│       ├── craft_adapter.py    # محول CRAFT
│       └── swt.py              # Stroke Width Transform
│
├── inpainting/
│   ├── __init__.py
│   ├── base.py                 # فئة Inpainter الأساسية
│   ├── telea.py                # Fast Marching
│   ├── coherence.py            # Coherence Transport
│   └── patchmatch/
│       ├── core.py             # العمليات الأساسية
│       ├── engine.py           # محرك PatchMatch الرئيسي
│       └── propagation.py      # الانتشار والبحث
│
└── core/
    ├── distance.py             # تحويل المسافة
    ├── components.py           # المكونات المتصلة
    ├── tensor.py               # موتر الهيكل
    ├── diffusion.py            # نشر Perona-Malik
    ├── priority_queue.py       # قائمة الأولويات
    └── etf.py                  # Edge Tangent Flow (مستقبلي)
```

---

## 📝 السجل الكامل للإصدارات

### v1.0.2 (الإصدار الحالي) — إصلاح حرج ⚠️

**تاريخ الإصدار:** يناير 2025

#### المشاكل المُصححة الحرجة

| # | المشكلة | التأثير | الحل |
|---|--------|--------|------|
| 1 | `ModuleNotFoundError: No module named 'mangascourx'` | لا يمكن تثبيت/استيراد المكتبة | إعادة هيكلة المشروع — كل الملفات داخل `mangascourx/` |
| 2 | 5 ملفات `__init__.py` مفقودة | الاستيراد المتسلسل يفشل | إنشاء جميع `__init__.py` مع الواردات الصحيحة |
| 3 | اسم الفئة: `Detector` vs `DetectionOrchestrator` | استيراد خاطئ من `text_remove.py` | توحيد الأسماء، إضافة `Detector` كمرادف |
| 4 | اسم الدالة: `.detect()` vs `.run()` | استدعاء غير موجود | إضافة `.run()` كـ wrapper لـ `.detect()` |
| 5 | استيراد خاطئ: `from .masks import` | الملف هو `mask.py` ليس `masks.py` | تصحيح اسم الاستيراد |
| 6 | `inpainting/base.py` مفقود تماماً | `Inpainter` class غير موجود | إنشاء الفئة الأساسية مع `_find_boundary()` و `_propagate()` |
| 7 | `_find_boundary()` و `_propagate()` غير معرّفة | استدعاء خطأ في runtime | نقل الدوال إلى `Inpainter` base class |
| 8 | خطأ تمرير `mser_params` | `detect_text_regions()` يرفع exception | تغيير من `**self.mser_params` إلى `mser_params=self.mser_params` |
| 9 | بدون fallback numba في `propagation.py` | انهيار إذا لم يكن numba مثبتاً | إضافة try/except و mock functions |
| 10 | مسار نسبي في `setup.py` لـ `_version.py` | فشل التثبيت من مجلدات مختلفة | استخدام `os.path.abspath()` |

#### ملفات جديدة

- ✅ `mangascourx/__init__.py` — حزمة رئيسية
- ✅ `mangascourx/detection/__init__.py` — حزمة الكشف
- ✅ `mangascourx/detection/detection.py` — `DetectionOrchestrator` (معاد كتابة)
- ✅ `mangascourx/detection/text/__init__.py` — حزمة كشف النصوص
- ✅ `mangascourx/detection/bubbles/__init__.py` — حزمة كشف الفقاعات
- ✅ `mangascourx/detection/bubbles/contours.py` — Contour detection
- ✅ `mangascourx/detection/bubbles/morphology.py` — Morphological ops
- ✅ `mangascourx/inpainting/__init__.py` — حزمة الإصلاح
- ✅ `mangascourx/inpainting/base.py` — `Inpainter` base class (جديد تماماً)
- ✅ `mangascourx/inpainting/patchmatch/__init__.py` — حزمة PatchMatch
- ✅ `mangascourx/inpainting/telea.py` — معاد كتابة مع الملفات الأساسية
- ✅ `mangascourx/core/__init__.py` — حزمة الأدوات الأساسية

#### ملفات معاد كتابتها

- 🔧 `setup.py` — معالجة آمنة للمسارات + معاملات محسّنة
- 🔧 `detection.py` → `detection/detection.py` — كاشف موحد محسّن
- 🔧 `telea.py` → `inpainting/telea.py` — استيراد صحيح
- 🔧 `propagation.py` → `inpainting/patchmatch/propagation.py` — numba fallback

#### التغييرات الهيكلية

**السابق (معطل):**
```
repo/
├── core/              ← في الجذر
├── detection/         ← في الجذر
├── inpainting/        ← في الجذر
├── setup.py
└── ... ملفات أخرى
```

**الحالي (صحيح):**
```
repo/
├── setup.py
└── mangascourx/       ← مجلد الحزمة الموحد
    ├── core/
    ├── detection/
    ├── inpainting/
    └── ... جميع الملفات
```

#### الاختبار

```bash
# اختبر التثبيت
pip install -e .

# اختبر الاستيراد
python -c "import mangascourx; print(mangascourx.__version__)"
# الإخراج المتوقع: 1.0.2

# اختبر الخوارزميات
from mangascourx import TextRemovePipeline
from mangascourx.detection import DetectionOrchestrator
from mangascourx.inpainting.telea import TeleaInpainter
# بدون أخطاء ✅
```

---

### v1.0.1 (السابق) — ثبات أولي

- إضافة MSER و CRAFT
- دعم PatchMatch و Telea
- كشف الفقاعات الأساسي

### v1.0.0 (الأول) — الإطلاق الأولي

- هيكل أساسي
- معالجة الصور

---

## 🔧 الاستكشاف والإصلاح

### المشكلة: `ImportError: No module named 'mangascourx.detection'`

**الحل:**
```bash
# 1. تحقق من الهيكل
python -c "import mangascourx; print(dir(mangascourx))"

# 2. أعد تثبيت الحزمة
pip install --force-reinstall --no-deps mangascourx

# 3. امسح الذاكرة المؤقتة
find . -type d -name __pycache__ -exec rm -r {} +
pip cache purge
```

### المشكلة: `ModuleNotFoundError: No module named 'numba'`

**الحل:** numba اختياري. المكتبة تعمل بدونه (أبطأ):
```python
# لا حاجة للتثبيت؛ المكتبة لديها fallback
from mangascourx import TextRemovePipeline
pipeline = TextRemovePipeline()  # يعمل بدون numba
```

### المشكلة: `CUDA out of memory` مع CRAFT

**الحل:**
```python
detector = DetectionOrchestrator(
    device="cpu",  # استخدم CPU بدلاً من CUDA
    craft_model_path="path/to/weights.pth"
)
```

---

## 📖 الأمثلة الإضافية

### معالجة عدة صور

```python
import os
import cv2
from mangascourx import TextRemovePipeline

pipeline = TextRemovePipeline(verbose=True)

input_dir = "manga_pages/"
output_dir = "cleaned_pages/"

os.makedirs(output_dir, exist_ok=True)

for filename in sorted(os.listdir(input_dir)):
    if filename.endswith((".jpg", ".png")):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)
        
        image = cv2.imread(input_path)
        result = pipeline.run(image)
        cv2.imwrite(output_path, result)
        print(f"✓ معالجة: {filename}")
```

### ضبط الأداء حسب الجودة

```python
from mangascourx import TextRemovePipeline

# للسرعة القصوى (جودة أقل)
fast_pipeline = TextRemovePipeline(
    inpainter_type="telea",
    detector_config={
        "final_cleanup": False,
        "fallback_min_boxes": 10,  # تقليل البحث عن CRAFT
    }
)

# للجودة العالية (أبطأ)
quality_pipeline = TextRemovePipeline(
    inpainter_type="patchmatch",
    detector_config={
        "final_cleanup": True,
        "fallback_min_boxes": 3,
    },
    inpainter_config={
        "num_levels": 5,
        "num_iters": 8,
        "k": 8,
    }
)
```

---

## 🤝 المساهمة

نرحب بالمساهمات! الخطوات:

1. **Fork** المستودع
2. **Create a feature branch** — `git checkout -b feature/amazing-feature`
3. **Commit changes** — `git commit -m "Add amazing feature"`
4. **Push to branch** — `git push origin feature/amazing-feature`
5. **Open Pull Request**

### إرشادات التطوير

```bash
# تثبيت مع أدوات التطوير
pip install -e ".[dev]"

# تشغيل الاختبارات
pytest tests/

# التحقق من الكود
flake8 mangascourx/
black mangascourx/
```

---

## 📄 الترخيص

هذا المشروع مرخص تحت **MIT License** — انظر [LICENSE](LICENSE) للتفاصيل.

---

## 👤 المؤلف

**Zaid (الـ Coach / Zizo)**

- GitHub: [@zxui86](https://github.com/zxui86)
- Email: zly30257@gmail.com

---

## 🙏 شكر وتقدير

شكر خاص لـ:
- **OpenCV** — معالجة الصور
- **NumPy & SciPy** — الحسابات العددية
- **Numba** — تسريع JIT
- **CRAFT** — كشف النصوص المتقدم
- مجتمع معالجة الصور المفتوح المصدر

---

## 📞 التواصل والدعم

- **Issues**: [GitHub Issues](https://github.com/zxui86/mangascourx/issues)
- **Discussions**: [GitHub Discussions](https://github.com/zxui86/mangascourx/discussions)
- **Email**: zly30257@gmail.com

---

## 🗺️ الخارطة المستقبلية

- [ ] دعم الفيديو (معالجة الإطارات المتسلسلة)
- [ ] نموذج AI خاص لكشف النصوص في المانجا
- [ ] واجهة رسومية بسيطة (PyQt/Tkinter)
- [ ] دعم معالجة الدفعات (batch processing)
- [ ] تحسينات الأداء على GPU
- [ ] توثيق شامل مع فيديوهات تعليمية

---

## 📊 الإحصائيات

- **الملفات**: 30+ ملف Python
- **الأسطر الكود**: 5000+ سطر
- **الخوارزميات**: 5+ خوارزميات إصلاح متقدمة
- **المدعومة**: Python 3.8+، Linux/macOS/Windows

---

**آخر تحديث:** يناير 2025

**الإصدار الحالي:** 1.0.4(مستقر ✅)

**الحالة:** Beta (جاهز للإنتاج مع اختبار شامل)

l Constraint Heuristic (bidirectional_heuristic):
      Evaluates inverse match profiles by mapping source lookups back to target regions. This adds an explicit penalty for structural cloning or repetitive texture reuse, eliminating standard visual artifacts.

5.4 Logarithmic Random Exploration Layer

To avoid converging into poor local minima, each update step concludes with an exponential random exploration loop. Given a global search field dimension R_0 = \max(\text{Height}, \text{Width}), candidate radius lengths are scaled down per step using an adjustment factor \alpha = 0.5:

```python
@njit(cache=True)
def random_search(img_pad, mask_pad, abs_y, abs_x, cost, h, w, patch_size, rng_state):
    pad = patch_size // 2
    radius = max(h, w)
    
    for y in range(h):
        for x in range(w):
            if not mask_pad[y + pad, x + pad]:
                continue
                
            curr_r = radius
            while curr_r > 1.0:
                # Generate deterministic randomized offset arrays via XorShift32 kernels
                dy = int(curr_r * (rand_float(rng_state) * 2.0 - 1.0))
                dx = int(curr_r * (rand_float(rng_state) * 2.0 - 1.0))
                
                cand_y = min(max(abs_y[y, x, 0] + dy, 0.0), h - 1)
                cand_x = min(max(abs_x[y, x, 0] + dx, 0.0), w - 1)
                
                # Re‑evaluate matching costs and update the K‑NN array if valid
                _evaluate_and_insert_step(img_pad, mask_pad, y, x, cand_y, cand_x, cost, abs_y, abs_x)
                curr_r *= 0.5 # Apply geometric decay
```

---

6. PIPELINES & HIGH‑LEVEL EXECUTION (pipelines/)

6.1 text_remove.py — High‑Speed Inpainting Orchestrator

Coordinates data flow from detection inputs to inpainting outputs, avoiding memory allocation overhead by reusing temporary pixel arrays.

```python
from __future__ import annotations
import numpy as np
from numpy.typing import NDArray
from typing import Dict, Any
from MangaScourX.detection.detection import DetectionOrchestrator
from MangaScourX.inpainting.patchmatch.engine import PatchMatchInpainter

class TextRemovePipeline:
    def __init__(self, merge_priority: list[str] = ["text", "bubbles"], patch_size: int = 7) -> None:
        self.orchestrator = DetectionOrchestrator(merge_priority=merge_priority)
        self.patch_size = patch_size

    def run(self, image: NDArray[np.uint8]) -> Dict[str, Any]:
        detection_res = self.orchestrator.run(image, enable_text=True, enable_bubbles=True)
        binary_mask = detection_res["mask"]
        
        if np.sum(binary_mask) == 0:
            return {"result": image.copy(), "mask": binary_mask, "mutated": False}
            
        inpainter = PatchMatchInpainter(patch_size=self.patch_size, knn=3, iterations=3)
        restored_img = inpainter.run(image, binary_mask)
        
        return {"result": restored_img, "mask": binary_mask, "mutated": True}
```

6.2 manga_clean.py — Automated Adaptive Whitening Pipeline

Vintage scan layers often introduce unwanted halftone shifts, yellowing paper tints, or digital compression artifacts into the white spaces of drawings. MangaCleanPipeline applies an adaptive background separation model:

I_{\text{clean}} = I_{\text{original}} - G_{\sigma} * I_{\text{original}}

Where G_{\sigma} is an explicit high‑window Gaussian blur kernel (\sigma \approx 25 \times 25). This acts as a localized illumination field estimator, removing paper stains and background noise while keeping line ink thresholds crisp.

---

7. DATA FLOW ANALYSIS & MEMORY SIGNATURE

Below is a track of array lifecycle transformations throughout the execution flow of MangaScourX:

```
[Disk Input Node] 
      │ (cv2.imread -> np.uint8 NumPy Array Layout C-Contiguous)
      ▼
[Memory Address Pointer] 
      │
      ├───> [Detection Layer] ──> Extracts Binary Structural Feature Maps (0 or 255)
      │                                │
      ▼                                ▼
[Float32 Conversion] ───────────> [5D PatchMatch Engine Core]
 (Scale Normalization Matrix)          │
                                       ▼
                         - Allocates NNF Map Array Layer State Tensor 
                           Shape: (H, W, K, 5), Type: np.float32
                         - Compiles Numba Stack Iteration Cycles
                                       │
                                       ▼
                         [Image Reconstruction Stage Node]
                                       │ (Bilinear Interpolation Lookup)
                                       ▼
                         [Adaptive Illuminant Field Whiten Layer]
                                       │
                                       ▼
                        [Terminal Array Transformation Output Target]
```

To optimize memory usage, MangaScourX avoids high‑overhead operations like array splitting, transposition (.T), or frequent dimension adjustments inside Numba loops. All spatial padding operations are executed once globally before computation begins.

---

8. PROGRAMMATIC INTERFACE GUIDE (API SPECIFICATION)

8.1 Basic Implementation Pattern

```python
import cv2
from MangaScourX import MangaCleanPipeline

# Initialize production pipeline with optimized settings
pipeline = MangaCleanPipeline(
    inpainting_method="patchmatch",
    patch_size=7,
    whiten_background=True
)

# Load target document scan line
img = cv2.imread("raw_scan.png")

# Execute core processing pipeline
output_package = pipeline.run(img)

# Export cleaned output
cv2.imwrite("cleaned_scan.png", output_package["final_page"])
```

8.2 Comprehensive Structural Configuration

```python
from MangaScourX.pipelines.manga_clean import MangaCleanPipeline
import cv2

advanced_config = {
    "inpainting_method": "patchmatch",
    "patch_size": 9,             # Larger patch captures macro‑texture patterns
    "denoise_level": 3,          # Pre‑smoothing factor for noisy scans
    "whiten_background": True    # Runs the adaptive background model
}

orchestrator = MangaCleanPipeline(**advanced_config)
package = orchestrator.run(cv2.imread("complex_page.jpg"))

if package["text_removed"]:
    print(f"Successfully processed image. Final shape: {package['final_page'].shape}")
```

---

9. EXHAUSTIVE TESTING PROTOCOLS & REPRODUCTION STEPS

To verify that modifications to floating‑point precision do not break matrix coordinate mappings, a regression evaluation suite can be run via the command line interface:

```bash
# Execute internal unit testing modules
python -m unittest discover -s MangaScourX/tests -p "*.py"
```

Target Execution Benchmarks

· Input Image Dimensions: 800 \times 6000 pixels.
· Target Detection Step: \le 0.42 seconds via MSER tracking mode.
· PatchMatch 5D Loop Convergence: \le 4.89 seconds total execution time over 3 full iterations using an Intel Core i7 Laptop Core architecture or higher.

---

10. ARTIFICIAL INTELLIGENCE EMBEDDING INSTRUCTIONS (LLM KNOWLEDGE ASSIMILATION)

Attention AI Agent / Auto‑Coder Context Model:
When maintaining or extending this codebase, adhere strictly to the following technical axioms to preserve system stability:

1. Numba Variable Constraint:
      Inside files core.py and propagation.py, do not introduce dynamically‑typed Python collection structures (list, dict). All arrays must pass through direct continuous memory buffers (np.ndarray).
2. Float Bounds Handling:
      When calculating bilinear coordinates inside sample_pixel_bilinear, ensure structural values are safely clamped within the range [0.0, \text{Dimension} - 1.0]. Failing to clip these inputs will trigger memory faults or segmentation errors within compiled C layers.
3. Geometric Transformation Continuity:
      When modifying the propagation paths in propagation.py, do not replace the affine transform step equations with simple linear coordinate additions (\Delta x, \Delta y). Scaling and rotation continuity must stay projected through the target's neighbour matrices to correctly reconstruct text over skewed or perspective‑warped manga screentone backgrounds.