"""
dark_pdf.py  ─  PDF 다크모드 변환기 (v3)
=========================================
• 배경 → 검은색 (Black)
• 텍스트 → 흰색 (White)  — 수식·특수 폰트 100% 보존
• 래스터 이미지(피규어) → 흑백 픽셀 교환 (컬러 보존)
• 벡터 선/도형 중 검은색 → 흰색 반전 (테이블 선 포함)
• Form XObject 내부 벡터까지 재귀 처리

사용법:
    python dark_pdf.py input.pdf              # → input_dark.pdf
    python dark_pdf.py input.pdf output.pdf   # → output.pdf
"""

import sys
import re
import fitz  # PyMuPDF

try:
    import numpy as np
    _HAS_NUMPY = True
except ImportError:
    _HAS_NUMPY = False


# ── 임계값 ────────────────────────────────────────────────
BLACK_THRESH = 0.15          # 이 이하 → '검은색 계열'
WHITE_THRESH = 0.85          # 이 이상 → '흰색 계열'


def _is_black(v: float) -> bool:
    return v <= BLACK_THRESH


def _is_white(v: float) -> bool:
    return v >= WHITE_THRESH


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  PDF 문자열 리터럴 보호 / 복원
#  ─ (…) 안의 텍스트가 색상 regex 에 걸리지 않도록 보호
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def _protect_strings(text: str):
    """PDF (…) 문자열을 플레이스홀더로 교체한다."""
    protected: list = []
    out: list = []
    i, n = 0, len(text)
    while i < n:
        if text[i] == '(':
            depth = 1
            j = i + 1
            while j < n and depth > 0:
                if text[j] == '\\':
                    j += 2
                    continue
                if text[j] == '(':
                    depth += 1
                elif text[j] == ')':
                    depth -= 1
                j += 1
            protected.append(text[i:j])
            out.append(f"\x00S{len(protected) - 1}\x00")
            i = j
        else:
            out.append(text[i])
            i += 1
    return ''.join(out), protected


def _restore_strings(text: str, protected: list) -> str:
    for idx, s in enumerate(protected):
        text = text.replace(f"\x00S{idx}\x00", s)
    return text


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  인라인 이미지 보호 / 복원
#  ─ BI … ID <binary> EI 블록의 바이너리 데이터를 보호
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
_RE_INLINE_IMG = re.compile(rb'\bBI\b.+?\bEI(?=\s)', re.DOTALL)


def _protect_inline_images(raw: bytes):
    images: list = []
    def _repl(m):
        images.append(m.group(0))
        return f"__IMG{len(images) - 1}__".encode('latin-1')
    cleaned = _RE_INLINE_IMG.sub(_repl, raw)
    return cleaned, images


def _restore_inline_images(raw: bytes, images: list) -> bytes:
    for idx, img in enumerate(images):
        raw = raw.replace(f"__IMG{idx}__".encode('latin-1'), img)
    return raw


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  색상 연산자 치환 (검은색 ↔ 흰색)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
_NUM = r'(\d+\.?\d*|\.\d+)'      # PDF 숫자 패턴


def _swap_gray(v_str: str) -> str:
    """그레이스케일 값 교환: 검은색→흰색, 흰색→검은색."""
    v = float(v_str)
    if _is_black(v):
        return '1'
    if _is_white(v):
        return '0'
    return v_str


def _make_gray_replacer(op: str):
    """g / G 연산자 치환 함수 생성."""
    def _fn(m):
        return f'{_swap_gray(m.group(1))} {op}'
    return _fn


def _make_rgb_replacer(op: str):
    """rg / RG 연산자 치환 함수 생성."""
    def _fn(m):
        r = float(m.group(1))
        g = float(m.group(2))
        b = float(m.group(3))
        if _is_black(r) and _is_black(g) and _is_black(b):
            return f'1 1 1 {op}'
        if _is_white(r) and _is_white(g) and _is_white(b):
            return f'0 0 0 {op}'
        return m.group(0)
    return _fn


def _make_cmyk_replacer(op: str):
    """k / K 연산자 치환 함수 생성."""
    def _fn(m):
        c  = float(m.group(1))
        mm = float(m.group(2))
        y  = float(m.group(3))
        k  = float(m.group(4))
        # CMYK 검은색 (0,0,0,1) → 흰색 (0,0,0,0)
        if _is_black(c) and _is_black(mm) and _is_black(y) and _is_white(k):
            return f'0 0 0 0 {op}'
        # CMYK 흰색 (0,0,0,0) → 검은색 (0,0,0,1)
        if _is_black(c) and _is_black(mm) and _is_black(y) and _is_black(k):
            return f'0 0 0 1 {op}'
        return m.group(0)
    return _fn


def swap_stream_colors(stream: bytes, prepend_white_default: bool = False) -> bytes:
    """
    PDF 콘텐츠 스트림에서 색상 연산자만 찾아 검은색↔흰색을 교환.
    텍스트 인코딩·수식·이미지는 전혀 건드리지 않는다.

    prepend_white_default=True 이면 스트림 앞에 흰색 기본 색상을
    삽입하여 암묵적 검은색(기본 그래픽 상태)도 흰색으로 만든다.
    """
    # 1) 인라인 이미지 보호 (바이너리 단계)
    stream, imgs = _protect_inline_images(stream)

    # 2) latin-1 로 디코딩 (바이트 ↔ 문자 1:1 매핑)
    text = stream.decode('latin-1')

    # 3) PDF 문자열 리터럴 보호
    text, strs = _protect_strings(text)

    # 4) 색상 연산자 치환 ──────────────────────────────────
    N = _NUM

    #  ── Grayscale  g(fill) / G(stroke) ──────────────────
    #  예: "0 g" (검은색 fill) → "1 g" (흰색 fill)
    text = re.sub(
        rf'(?<![.\w]){N}\s+g(?![a-zA-Z])',
        _make_gray_replacer('g'), text
    )
    text = re.sub(
        rf'(?<![.\w]){N}\s+G(?![a-zA-Z])',
        _make_gray_replacer('G'), text
    )

    #  ── RGB  rg(fill) / RG(stroke) ─────────────────────
    #  예: "0 0 0 rg" → "1 1 1 rg"
    text = re.sub(
        rf'(?<![.\w]){N}\s+{N}\s+{N}\s+rg\b',
        _make_rgb_replacer('rg'), text
    )
    text = re.sub(
        rf'(?<![.\w]){N}\s+{N}\s+{N}\s+RG\b',
        _make_rgb_replacer('RG'), text
    )

    #  ── CMYK  k(fill) / K(stroke) ──────────────────────
    #  예: "0 0 0 1 k" (검은색) → "0 0 0 0 k" (흰색)
    text = re.sub(
        rf'(?<![.\w]){N}\s+{N}\s+{N}\s+{N}\s+k\b',
        _make_cmyk_replacer('k'), text
    )
    text = re.sub(
        rf'(?<![.\w]){N}\s+{N}\s+{N}\s+{N}\s+K\b',
        _make_cmyk_replacer('K'), text
    )
    #  ── 범용 색상 연산자  sc/SC/scn/SCN ─────────────────
    #  피규어 생성 도구(tikz, matplotlib 등)가 g/rg 대신 사용
    #  순서: 4-operand(CMYK) → 3-operand(RGB) → 1-operand(Gray)
    #  (긴 패턴부터 매칭해야 부분 매칭 방지)
    for op_fill, op_stroke in [('scn', 'SCN'), ('sc', 'SC')]:
        # ── CMYK variant (4 operands)
        pat_stroke_4 = rf'(?<![.\w]){N}\s+{N}\s+{N}\s+{N}\s+{op_stroke}\b'
        pat_fill_4   = rf'(?<![.\w]){N}\s+{N}\s+{N}\s+{N}\s+{op_fill}\b'
        text = re.sub(pat_fill_4,   _make_cmyk_replacer(op_fill),   text)
        text = re.sub(pat_stroke_4, _make_cmyk_replacer(op_stroke), text)
        # ── RGB variant (3 operands)
        pat_stroke_3 = rf'(?<![.\w]){N}\s+{N}\s+{N}\s+{op_stroke}\b'
        pat_fill_3   = rf'(?<![.\w]){N}\s+{N}\s+{N}\s+{op_fill}\b'
        text = re.sub(pat_fill_3,   _make_rgb_replacer(op_fill),   text)
        text = re.sub(pat_stroke_3, _make_rgb_replacer(op_stroke), text)
        # ── Grayscale variant (1 operand)
        pat_stroke_1 = rf'(?<![.\w]){N}\s+{op_stroke}\b'
        pat_fill_1   = rf'(?<![.\w]){N}\s+{op_fill}\b'
        text = re.sub(pat_fill_1,   _make_gray_replacer(op_fill),   text)
        text = re.sub(pat_stroke_1, _make_gray_replacer(op_stroke), text)
    # 5) Form XObject용: 암묵적 검은색 기본값 → 흰색으로 덮어쓰기
    #    PDF 기본 그래픽 상태는 fill=black, stroke=black.
    #    색상 연산자 없이 그려진 경로/텍스트도 흰색으로 보이게 한다.
    if prepend_white_default:
        text = '1 g 1 G 1 1 1 rg 1 1 1 RG\n' + text

    # 6) 보호 해제
    text = _restore_strings(text, strs)
    result = text.encode('latin-1')
    result = _restore_inline_images(result, imgs)
    return result


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  래스터 이미지(Image XObject) 흑백 픽셀 교환
#  ─ 검은 픽셀 → 흰색, 흰색 픽셀 → 검은색, 컬러 유지
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
_IMG_BK = 40       # 0-255, 이 이하 → 근사 검정
_IMG_WH = 215      # 0-255, 이 이상 → 근사 흰색


def _swap_bw_pixels_with_outline(samples: bytes, w: int, h: int, n_ch: int):
    """RGB 픽셀에서 근사-검정↔근사-흰색 교환 + 검정 아웃라인 추가.

    1) 검정 픽셀 → 흰색, 흰색 픽셀 → 검정  (기존 동작)
    2) 원래 검정이었던(→흰색 변환된) 영역 주위에 1px 검정 테두리 추가
       → 컬러 배경(노란 박스 등) 위에서도 텍스트가 잘 보임
       → 검정 배경 위에서는 테두리가 배경에 묻혀 보이지 않음 (무해)

    Returns (new_bytes, was_modified)."""
    if _HAS_NUMPY:
        arr = np.frombuffer(samples, dtype=np.uint8).copy().reshape(h, w, n_ch)
        rgb = arr[:, :, :3]

        # 2D 마스크 (h x w)
        bk = np.all(rgb < _IMG_BK, axis=2)    # 근사 검정
        wh = np.all(rgb > _IMG_WH, axis=2)    # 근사 흰색

        if not (bk.any() or wh.any()):
            return samples, False

        # ── 기본 교환 ──
        rgb[bk] = 255    # 검정 → 흰색
        rgb[wh] = 0      # 흰색 → 검정

        # ── 검정 아웃라인 생성 ──
        # bk 마스크를 8방향으로 2px 팽창(dilate)
        dilated = bk.copy()
        for _ in range(3):
            prev = dilated.copy()
            dilated[:-1, :]   |= prev[1:, :]      # ↑
            dilated[1:, :]    |= prev[:-1, :]     # ↓
            dilated[:, :-1]   |= prev[:, 1:]      # ←
            dilated[:, 1:]    |= prev[:, :-1]     # →
            dilated[:-1, :-1] |= prev[1:, 1:]     # ↖
            dilated[:-1, 1:]  |= prev[1:, :-1]    # ↗
            dilated[1:, :-1]  |= prev[:-1, 1:]    # ↙
            dilated[1:, 1:]   |= prev[:-1, :-1]   # ↘

        # 아웃라인 = 팽창 영역 - 원래 검정 영역
        outline = dilated & ~bk
        rgb[outline] = 0   # 아웃라인을 검정으로

        return bytes(arr), True
    else:
        # numpy 없을 때: 아웃라인 없이 단순 교환만
        buf = bytearray(samples)
        changed = False
        for i in range(0, len(buf) - 2, n_ch):
            r, g, b = buf[i], buf[i + 1], buf[i + 2]
            if r < _IMG_BK and g < _IMG_BK and b < _IMG_BK:
                buf[i] = buf[i + 1] = buf[i + 2] = 255
                changed = True
            elif r > _IMG_WH and g > _IMG_WH and b > _IMG_WH:
                buf[i] = buf[i + 1] = buf[i + 2] = 0
                changed = True
        return bytes(buf), changed


def _process_raster_image(doc, xref):
    """Image XObject 의 흑백 픽셀을 교환한 Pixmap 을 반환한다.
    수정이 불필요하거나 실패하면 None."""
    try:
        pix = fitz.Pixmap(doc, xref)
    except Exception:
        return None

    # 아주 작은 이미지(장식용 점·패턴)는 건너뜀
    if pix.width < 20 or pix.height < 20:
        return None

    # 알파 제거 (처리 단순화)
    if pix.alpha:
        pix = fitz.Pixmap(pix, 0)

    # RGB 로 통일
    if pix.colorspace != fitz.csRGB:
        try:
            pix = fitz.Pixmap(fitz.csRGB, pix)
        except Exception:
            return None

    new_samples, modified = _swap_bw_pixels_with_outline(
        pix.samples, pix.width, pix.height, pix.n,
    )
    if not modified:
        return None

    return fitz.Pixmap(fitz.csRGB, pix.width, pix.height, new_samples, 0)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  메인 변환 로직
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def convert_to_dark(input_path: str, output_path: str) -> None:
    doc = fitz.open(input_path)

    # ── Phase 1: 래스터 이미지 흑백 픽셀 교환 ─────────────
    processed_img = set()
    for page in doc:
        for img_info in page.get_images(full=True):
            xref = img_info[0]
            if xref in processed_img:
                continue
            processed_img.add(xref)
            new_pix = _process_raster_image(doc, xref)
            if new_pix:
                try:
                    page.replace_image(xref, pixmap=new_pix)
                except Exception:
                    pass
                print(f"    [IMG] xref={xref} image B/W swap done")

    # ── Phase 2: Form XObject 내부 색상 치환 (벡터) ───────
    for xref in range(1, doc.xref_length()):
        try:
            if doc.xref_get_key(xref, "Subtype")[1] == "/Form":
                raw = doc.xref_stream(xref)
                if raw:
                    doc.update_stream(
                        xref,
                        swap_stream_colors(raw, prepend_white_default=True),
                    )
        except Exception:
            pass

    # ── Phase 3: 페이지 콘텐츠 + 검은 배경 ────────────────
    for idx, page in enumerate(doc):
        page.clean_contents()
        contents = page.get_contents()
        if contents:
            xref = contents[0]
            raw = doc.xref_stream(xref)
            doc.update_stream(xref, swap_stream_colors(raw))

        page.draw_rect(page.rect, color=None, fill=(0, 0, 0), overlay=False)
        print(f"  [OK] page {idx + 1}/{len(doc)} done")

    # ── 저장 ──────────────────────────────────────────────
    doc.save(output_path, garbage=4, deflate=True)
    doc.close()
    print(f"\n[DONE] dark mode PDF saved: {output_path}")


# ── CLI ───────────────────────────────────────────────────
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("사용법: python dark_pdf.py <입력.pdf> [출력.pdf]")
        sys.exit(1)

    src = sys.argv[1]
    dst = sys.argv[2] if len(sys.argv) >= 3 else f"{src.rsplit('.', 1)[0]}_dark.pdf"

    print(f"[INPUT]  {src}")
    print(f"[OUTPUT] {dst}\n")
    convert_to_dark(src, dst)
