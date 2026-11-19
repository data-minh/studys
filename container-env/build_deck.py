"""Build the Kubernetes vs OpenShift deck on top of the BIDV template.

Usage: python build_deck.py <template.pptx> <assets_dir> <output.pptx>
assets_dir must contain: k8s_white.png, k8s_color.png, ocp_red.png, ocp_white.png
"""
import copy
import sys
from pathlib import Path

from lxml import etree
from pptx import Presentation
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.oxml.ns import qn
from pptx.util import Inches, Pt

TPL, ASSETS, OUT = sys.argv[1], Path(sys.argv[2]), sys.argv[3]

prs = Presentation(TPL)
ORIG = list(prs.slides)
LAYOUT = {l.name: l for l in prs.slide_layouts}
T_COVER, T_CONTENT, T_SECTION, T_CLOSING = ORIG[0], ORIG[1], ORIG[2], ORIG[9]

HEX = set("0123456789ABCDEFabcdef")
MONO = "Courier New"


# ---------------------------------------------------------------- XML helpers
def clr(color, alpha=None):
    tag = "a:srgbClr" if len(color) == 6 and set(color) <= HEX else "a:schemeClr"
    e = etree.Element(qn(tag), val=color)
    if alpha is not None:
        etree.SubElement(e, qn("a:alpha"), val=str(int(alpha * 1000)))
    return e


def solid(color, alpha=None):
    f = etree.Element(qn("a:solidFill"))
    f.append(clr(color, alpha))
    return f


def strip_style(shape):
    st = shape._element.find(qn("p:style"))
    if st is not None:
        shape._element.remove(st)


def inch(v):
    return Inches(v)


# ---------------------------------------------------------------- text
def add_par(txBody, runs, size=13, color="bg1", bold=False, align="l",
            bullet=False, after=0, before=0, font=None, italic=False, line=None):
    p = etree.SubElement(txBody, qn("a:p"))
    pPr = etree.SubElement(p, qn("a:pPr"), algn=align)
    if bullet:
        pPr.set("marL", "177800"); pPr.set("indent", "-177800")
    else:
        pPr.set("marL", "0"); pPr.set("indent", "0")
    if line:
        ls = etree.SubElement(pPr, qn("a:lnSpc"))
        etree.SubElement(ls, qn("a:spcPct"), val=str(int(line * 100000)))
    if before:
        sb = etree.SubElement(pPr, qn("a:spcBef"))
        etree.SubElement(sb, qn("a:spcPts"), val=str(int(before * 100)))
    if after:
        sa = etree.SubElement(pPr, qn("a:spcAft"))
        etree.SubElement(sa, qn("a:spcPts"), val=str(int(after * 100)))
    if bullet:
        etree.SubElement(pPr, qn("a:buSzPct"), val="100000")
        etree.SubElement(pPr, qn("a:buChar"), char="•")
    else:
        etree.SubElement(pPr, qn("a:buNone"))
    if isinstance(runs, str):
        runs = [(runs, {})]
    for text, o in runs:
        r = etree.SubElement(p, qn("a:r"))
        rPr = etree.SubElement(r, qn("a:rPr"), lang="en-US",
                               sz=str(int(o.get("size", size) * 100)), dirty="0")
        if o.get("bold", bold):
            rPr.set("b", "1")
        if o.get("italic", italic):
            rPr.set("i", "1")
        if o.get("spc"):
            rPr.set("spc", str(o["spc"]))
        rPr.append(solid(o.get("color", color)))
        f = o.get("font", font) or "Arial"
        if f:
            etree.SubElement(rPr, qn("a:latin"), typeface=f)
            etree.SubElement(rPr, qn("a:cs"), typeface=f)
        t = etree.SubElement(r, qn("a:t"))
        t.text = text
    return p


def fill_txbody(txBody, paras, **kw):
    for p in txBody.findall(qn("a:p")):
        txBody.remove(p)
    for para in paras:
        if isinstance(para, dict):
            args = dict(kw); args.update({k: v for k, v in para.items() if k != "runs"})
            add_par(txBody, para["runs"], **args)
        else:
            add_par(txBody, para, **kw)


def tb(s, x, y, w, h, paras, anchor="t", name=None, **kw):
    t = s.shapes.add_textbox(inch(x), inch(y), inch(w), inch(h))
    if name:
        t.name = name
    bp = t.text_frame._txBody.find(qn("a:bodyPr"))
    for k in ("lIns", "tIns", "rIns", "bIns"):
        bp.set(k, "0")
    bp.set("wrap", "square"); bp.set("anchor", anchor)
    for c in list(bp):
        bp.remove(c)
    if isinstance(paras, (str, tuple)):
        paras = [paras] if isinstance(paras, str) else [[paras]]
    fill_txbody(t.text_frame._txBody, paras, **kw)
    return t


# ---------------------------------------------------------------- shapes
def box(s, x, y, w, h, fill="accent2", alpha=None, line="accent1", lw=0.75,
        geom="rr", adj=0.05, name=None, line_alpha=None):
    g = {"rr": MSO_SHAPE.ROUNDED_RECTANGLE, "rect": MSO_SHAPE.RECTANGLE,
         "oval": MSO_SHAPE.OVAL, "chev": MSO_SHAPE.CHEVRON,
         "rarrow": MSO_SHAPE.RIGHT_ARROW}[geom]
    shp = s.shapes.add_shape(g, inch(x), inch(y), inch(w), inch(h))
    if name:
        shp.name = name
    if geom == "rr":
        shp.adjustments[0] = adj
    strip_style(shp)
    spPr = shp._element.spPr
    spPr.append(solid(fill, alpha) if fill else etree.Element(qn("a:noFill")))
    ln = etree.SubElement(spPr, qn("a:ln"), w=str(int(Pt(lw))) if line else "0")
    ln.append(solid(line, line_alpha) if line else etree.Element(qn("a:noFill")))
    return shp


def card(s, x, y, w, h, **kw):
    kw.setdefault("fill", "accent2"); kw.setdefault("line", "accent1")
    return box(s, x, y, w, h, **kw)


def arrow(s, x1, y1, x2, y2, color="bg1", w=1.5, head=True):
    c = s.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, inch(x1), inch(y1), inch(x2), inch(y2))
    strip_style(c)
    spPr = c._element.spPr
    ln = etree.SubElement(spPr, qn("a:ln"), w=str(int(Pt(w))))
    ln.append(solid(color))
    if head:
        etree.SubElement(ln, qn("a:tailEnd"), type="triangle", w="med", len="med")
    return c


def pic(s, name, x, y, w=None, h=None):
    return s.shapes.add_picture(str(ASSETS / name), inch(x), inch(y),
                                inch(w) if w else None, inch(h) if h else None)


def label_box(s, x, y, w, h, title, sub=None, fill="tx1", alpha=None, line="accent1",
              tcolor="bg1", scolor=None, tsize=12, ssize=11, align="l", lw=0.75, anchor="ctr",
              pad=0.12):
    card(s, x, y, w, h, fill=fill, alpha=alpha, line=line, lw=lw, adj=0.08)
    paras = [{"runs": [(title, {"bold": True})], "size": tsize, "color": tcolor, "align": align}]
    if sub:
        paras.append({"runs": sub, "size": ssize, "color": scolor or tcolor, "align": align})
    tb(s, x + pad, y + 0.05, w - 2 * pad, h - 0.1, paras, anchor=anchor)


# ---------------------------------------------------------------- slides
def clone(src):
    s = prs.slides.add_slide(src.slide_layout)
    for ph in list(s.placeholders):
        ph._element.getparent().remove(ph._element)
    for el in src.shapes._spTree.iterchildren():
        if el.tag in (qn("p:sp"), qn("p:cxnSp"), qn("p:graphicFrame")):
            s.shapes._spTree.append(copy.deepcopy(el))
    return s


def ph(s, idx):
    for el in s.shapes._spTree.iter(qn("p:sp")):
        p = el.find(".//" + qn("p:ph"))
        if p is not None and p.get("idx") == str(idx):
            return el
    raise KeyError(idx)


def set_ph(s, idx, paras, **kw):
    fill_txbody(ph(s, idx).find(qn("p:txBody")), paras, **kw)


def content(title_runs, notes=""):
    """title_runs: list of (text, highlight:bool)."""
    s = prs.slides.add_slide(T_CONTENT.slide_layout)
    for p in list(s.placeholders):
        p._element.getparent().remove(p._element)
    for el in T_CONTENT.shapes._spTree.iterchildren(qn("p:sp")):
        p = el.find(".//" + qn("p:ph"))
        if p is not None:  # title + slide number
            s.shapes._spTree.append(copy.deepcopy(el))
    runs = [(t, {"color": "accent1" if hl else "bg1"}) for t, hl in title_runs]
    set_ph(s, 101, [runs], size=28, bold=True)
    if notes:
        s.notes_slide.notes_text_frame.text = notes
    return s


def section(num, title, sub, notes=""):
    s = clone(T_SECTION)
    set_ph(s, 101, [[(num + "  ", {"color": "bg1"}), (title, {})]], size=40, bold=True, color="accent1")
    set_ph(s, 102, [sub], size=16)
    if notes:
        s.notes_slide.notes_text_frame.text = notes
    return s


def flow_row(s, y, items, h=0.95, hl_last=True, light=False, x0=0.6, total=8.8, gap=0.32,
             arrows=True):
    n = len(items); w = (total - gap * (n - 1)) / n
    for i, (t, sub) in enumerate(items):
        x = x0 + i * (w + gap)
        last = hl_last and i == n - 1
        if last:
            label_box(s, x, y, w, h, t, [(sub, {})], fill="accent1", line=None, tcolor="tx1")
        elif light:
            label_box(s, x, y, w, h, t, [(sub, {})], fill="bg1", alpha=22, line="bg1",
                      tcolor="bg1")
        else:
            label_box(s, x, y, w, h, t, [(sub, {})], fill="tx1", line="accent1")
        if arrows and i < n - 1:
            arrow(s, x + w + 0.02, y + h / 2, x + w + gap - 0.02, y + h / 2)
    return w


def table(s, x, y, colw, rows, rowh=0.4, size=11, header_size=12, first_bold=True):
    nr, nc = len(rows), len(colw)
    gf = s.shapes.add_table(nr, nc, inch(x), inch(y), inch(sum(colw)), inch(rowh * nr))
    tbl = gf.table
    tblPr = tbl._tbl.find(qn("a:tblPr"))
    for k in list(tblPr.attrib):
        del tblPr.attrib[k]
    for c in list(tblPr):
        tblPr.remove(c)
    for j, w in enumerate(colw):
        tbl.columns[j].width = inch(w)
    for i in range(nr):
        tbl.rows[i].height = inch(rowh)
        for j in range(nc):
            cell = tbl.cell(i, j)
            tc = cell._tc
            txBody = tc.find(qn("a:txBody"))
            val = rows[i][j]
            if i == 0:
                paras = [[(val, {"bold": True, "color": "accent1"})]]
                fill_txbody(txBody, paras, size=header_size)
            else:
                runs = val if isinstance(val, list) else [(val, {"bold": first_bold and j == 0})]
                fill_txbody(txBody, [runs], size=size)
            tcPr = tc.get_or_add_tcPr()
            for c in list(tcPr):
                tcPr.remove(c)
            tcPr.set("marL", "91440"); tcPr.set("marR", "91440")
            tcPr.set("marT", "36576"); tcPr.set("marB", "36576"); tcPr.set("anchor", "ctr")
            for side in ("lnL", "lnR", "lnT", "lnB"):
                ln = etree.SubElement(tcPr, qn("a:" + side), w="6350")
                ln.append(solid("accent3"))
            fillc = "tx1" if i == 0 else ("accent6" if i % 2 else "accent2")
            tcPr.append(solid(fillc))
    return gf


def caption(s, text, y=4.62, color="accent5", size=11):
    tb(s, 0.6, y, 8.8, 0.3, [text], size=size, color=color, italic=False)


# ============================================================== CONTENT
# 1. Cover
s = clone(T_COVER)
set_ph(s, 103, [[("KUBERNETES & OPENSHIFT", {})]], size=22, bold=True, color="accent1", align="ctr")
set_ph(s, 104, ["Nền tảng container: từ kiến trúc lõi đến lựa chọn doanh nghiệp"], size=13, align="ctr")
set_ph(s, 105, ["Trung tâm Dữ liệu và Phân tích"], size=24, bold=True)
set_ph(s, 106, ["Hà Nội, tháng 11/2026"], size=14, italic=True)
set_ph(s, 107, [[("Họ và tên", {"bold": True, "color": "accent1"}), (" · Nhóm / Phòng", {})]],
       size=12, italic=True)
s.notes_slide.notes_text_frame.text = (
    "Mở đầu: người nghe đã biết cơ bản Kubernetes. Mục tiêu buổi này là nhìn lại bản chất, "
    "rồi trả lời: OpenShift thêm gì vào Kubernetes, khác ở đâu khi dùng/vận hành, và khi nào nên chọn.")

# 2. Agenda
s = content([("Nội dung và ", False), ("mục tiêu", True)],
            "Ba phần: ôn nền tảng (nhanh), giải phẫu OpenShift, rồi so sánh theo cách dùng, nền tảng, vận hành.")
card(s, 0.6, 1.3, 3.9, 1.05, fill="accent2", alpha=85, line=None)
tb(s, 0.8, 1.4, 3.5, 0.22, [[("BỐI CẢNH", {"spc": 200})]], size=10, bold=True, color="accent5")
tb(s, 0.8, 1.66, 3.5, 0.6, ["Cả nhóm đã nắm cơ bản Kubernetes: Pod, Deployment, Service"],
   size=13, bold=True)
box(s, 4.6, 1.68, 0.4, 0.3, fill="accent1", line=None, geom="rarrow")
card(s, 5.1, 1.3, 4.3, 1.05, fill="accent1", line=None)
tb(s, 5.3, 1.4, 3.9, 0.22, [[("MỤC TIÊU", {"spc": 200})]], size=10, bold=True, color="tx1")
tb(s, 5.3, 1.66, 3.9, 0.6, ["Hiểu OpenShift thêm gì vào Kubernetes, đánh đổi gì và khi nào nên chọn"],
   size=13, bold=True, color="tx1")
cards = [
    ("01", "Nền tảng", "Container là gì, chuỗi runtime, kiến trúc và vòng lặp reconcile của Kubernetes.",
     "Container · CRI · Control plane", "k8s_white.png"),
    ("02", "OpenShift", "Kubernetes được đóng gói thành nền tảng: RHCOS, Operator, dịch vụ tích hợp.",
     "RHCOS · CVO · OperatorHub", "ocp_white.png"),
    ("03", "So sánh", "Khác biệt về bảo mật, cách dùng, mạng, vòng đời, chi phí và tiêu chí chọn.",
     "SCC · oc · Route · TCO", None),
]
for i, (n, t, d, lab, logo) in enumerate(cards):
    x = 0.6 + i * 3.0
    card(s, x, 2.6, 2.8, 2.0, adj=0.04)
    tb(s, x + 0.2, 2.72, 1.2, 0.5, [n], size=28, bold=True, color="accent1")
    if logo:
        pic(s, logo, x + 2.1, 2.78, 0.45, 0.45)
    tb(s, x + 0.2, 3.28, 2.4, 0.3, [t], size=16, bold=True)
    tb(s, x + 0.2, 3.62, 2.45, 0.62, [d], size=12)
    tb(s, x + 0.2, 4.28, 2.45, 0.22, [lab], size=11, bold=True, color="accent1")
s3_ocp_badge = (2 * 3.0 + 0.6)
pic(s, "k8s_white.png", s3_ocp_badge + 1.75, 2.8, 0.4, 0.4)
pic(s, "ocp_white.png", s3_ocp_badge + 2.2, 2.8, 0.4, 0.4)

# ---------------------------------------------------------------- Section 1
section("01", "Nền tảng container", "Ôn nhanh bản chất: container là gì, Kubernetes vận hành ra sao")

# 3. VM vs container
s = content([("Container: cô lập ", False), ("tiến trình", True), (", không phải VM", False)],
            "Nhấn mạnh: container chỉ là tiến trình Linux được kernel cô lập. Không có kernel riêng, "
            "nên khởi động nhanh và mật độ cao, nhưng ranh giới bảo mật mỏng hơn VM. Đây là lý do "
            "OpenShift siết chặt bảo mật mặc định (SCC, SELinux) mà ta sẽ thấy ở phần 3.")
# stacks
def stack_col(x, w, title, layers, tops):
    tb(s, x, 1.35, w, 0.28, [title], size=13, bold=True, color="accent1", align="ctr")
    yb = 4.5
    for name, fill, h in layers:
        yb -= h
        label_box(s, x, yb, w, h - 0.05, name, fill=fill, line=None, tsize=11, align="ctr", pad=0.04)
        yb -= 0.0
    return yb

y_top = stack_col(0.6, 2.05, "Máy ảo (VM)",
                  [("Phần cứng", "tx1", 0.5), ("Hypervisor", "accent6", 0.5)], None)
for k in range(2):
    xx = 0.6 + k * 1.05
    label_box(s, xx, y_top - 0.55, 1.0, 0.5, "Guest OS", fill="accent2", line="accent5",
              tsize=10, align="ctr", pad=0.03)
    label_box(s, xx, y_top - 1.1, 1.0, 0.5, "Libs", fill="accent2", line="accent5",
              tsize=10, align="ctr", pad=0.03)
    label_box(s, xx, y_top - 1.65, 1.0, 0.5, "App", fill="accent1", line=None, tcolor="tx1",
              tsize=11, align="ctr", pad=0.03)
y_top = stack_col(2.85, 2.05, "Container",
                  [("Phần cứng", "tx1", 0.5), ("Host OS · kernel Linux", "accent6", 0.5),
                   ("containerd / CRI-O", "accent2", 0.5)], None)
for k in range(3):
    xx = 2.85 + k * 0.69
    label_box(s, xx, y_top - 0.55, 0.64, 0.5, "Libs", fill="accent2", line="accent5",
              tsize=10, align="ctr", pad=0.02)
    label_box(s, xx, y_top - 1.1, 0.64, 0.5, "App", fill="accent1", line=None, tcolor="tx1",
              tsize=11, align="ctr", pad=0.02)
card(s, 5.15, 1.3, 4.25, 3.2, adj=0.04)
tb(s, 5.35, 1.42, 3.9, 0.3, ["Bản chất kỹ thuật"], size=16, bold=True, color="accent1")
tb(s, 5.35, 1.82, 3.9, 2.6, [
    [("Namespaces: ", {"bold": True}), ("cô lập PID, network, mount, UTS, IPC, user", {})],
    [("cgroups v2: ", {"bold": True}), ("giới hạn CPU, memory, I/O cho từng container", {})],
    [("Image OCI: ", {"bold": True}), ("các layer chỉ đọc xếp chồng bằng OverlayFS, thêm một layer ghi", {})],
    [("Dùng chung kernel: ", {"bold": True}), ("khởi động tính bằng mili giây, mật độ cao", {})],
], size=13, bullet=True, after=6)
caption(s, "Đánh đổi: nhẹ và nhanh hơn VM, nhưng ranh giới bảo mật mỏng hơn vì mọi container dùng chung một kernel")

# 4. Runtime chain
s = content([("Chuỗi runtime: từ ", False), ("kubelet", True), (" tới tiến trình", False)],
            "Kubernetes không biết Docker. kubelet gọi CRI (gRPC) tới runtime cấp cao (containerd "
            "hoặc CRI-O), runtime này gọi runc/crun theo OCI runtime spec. dockershim bị gỡ từ "
            "Kubernetes 1.24. Image build bằng Docker vẫn chạy vì cùng chuẩn OCI image. "
            "OpenShift dùng CRI-O, một runtime chỉ phục vụ Kubernetes, phát hành cùng nhịp với Kubernetes.")
flow_row(s, 1.35, [("1. kubelet", "Agent trên node"), ("2. CRI", "Giao diện gRPC"),
                   ("3. containerd / CRI-O", "Runtime cấp cao"), ("4. runc / crun", "OCI runtime"),
                   ("5. Tiến trình", "namespaces + cgroups")], h=1.0)
tb(s, 0.6, 2.62, 8.8, 0.25, [[("CHUẨN VÀ BỐI CẢNH", {"spc": 200})]], size=10, bold=True, color="accent5")
flow_row(s, 2.95, [("OCI Image Spec", "Định dạng image, layer"),
                   ("OCI Runtime Spec", "Bundle + config.json"),
                   ("OCI Distribution", "API pull/push registry"),
                   ("Docker Engine", "dockershim gỡ từ 1.24"),
                   ("OpenShift", "Mặc định dùng CRI-O")], h=1.0, light=True, arrows=False)
caption(s, "Kubernetes chỉ giao tiếp qua CRI. Image build bằng Docker vẫn chạy bình thường vì cùng chuẩn OCI.",
        y=4.2)

# 5. K8s architecture
s = content([("Kiến trúc K8s: ", False), ("control plane", True), (" và worker node", False)],
            "API server là cổng duy nhất, mọi thành phần khác giao tiếp qua watch trên API server. "
            "Chỉ API server đọc/ghi etcd. Scheduler chỉ gán nodeName; kubelet là bên thực thi. "
            "OpenShift giữ nguyên các thành phần này và bổ sung thêm thành phần riêng (slide 10).")
card(s, 0.6, 1.3, 4.0, 3.2, fill="accent2", adj=0.04)
tb(s, 0.8, 1.38, 3.0, 0.3, ["Control plane"], size=14, bold=True, color="accent1")
pic(s, "k8s_white.png", 4.05, 1.38, 0.38, 0.38)
label_box(s, 0.8, 1.85, 3.6, 0.62, "kube-apiserver", [("Cổng duy nhất: REST, xác thực, admission, watch", {})],
          fill="accent1", line=None, tcolor="tx1", tsize=12, ssize=10)
label_box(s, 0.8, 2.6, 1.75, 0.82, "etcd", [("Lưu toàn bộ trạng thái cluster", {})], tsize=12, ssize=10)
label_box(s, 2.65, 2.6, 1.75, 0.82, "kube-scheduler", [("Chọn node cho Pod mới", {})], tsize=12, ssize=10)
label_box(s, 0.8, 3.55, 1.75, 0.82, "controller-manager", [("Các vòng lặp reconcile", {})], tsize=11, ssize=10)
label_box(s, 2.65, 3.55, 1.75, 0.82, "cloud-controller", [("LB, volume, node của cloud", {})], tsize=11, ssize=10)
for k, yy in enumerate((1.3, 2.95)):
    card(s, 5.2, yy, 4.2, 1.55, fill="accent2", adj=0.05)
    tb(s, 5.38, yy + 0.08, 3.0, 0.28, [f"Worker node {k + 1}"], size=13, bold=True, color="accent1")
    for j, comp in enumerate(("kubelet", "kube-proxy", "runtime")):
        label_box(s, 5.38 + j * 1.3, yy + 0.42, 1.2, 0.42, comp, fill="tx1", line="accent5",
                  tsize=11, align="ctr", pad=0.03)
    for j in range(4):
        label_box(s, 5.38 + j * 0.98, yy + 0.97, 0.88, 0.45, "Pod", fill="accent1", line=None,
                  tcolor="tx1", tsize=11, align="ctr", pad=0.03)
arrow(s, 4.6, 2.16, 5.2, 2.05)
arrow(s, 4.6, 2.16, 5.2, 3.7)
caption(s, "Mọi thành phần chỉ giao tiếp qua API server. Chỉ API server được đọc và ghi etcd.")

# 6. Reconcile loop
s = content([("Cốt lõi của Kubernetes: ", False), ("vòng lặp reconcile", True)],
            "Mô hình declarative: ta khai báo trạng thái mong muốn, controller liên tục so sánh và "
            "kéo trạng thái thực tế về đó. Các điểm mở rộng (CRD, Operator, CNI, CSI, webhook) là "
            "nền móng mà OpenShift dùng để xây toàn bộ nền tảng.")
flow_row(s, 1.35, [("1. Khai báo", "YAML trạng thái mong muốn"), ("2. API server", "Xác thực, admission"),
                   ("3. etcd", "Lưu desired state"), ("4. Controller", "Watch và so sánh"),
                   ("5. Hành động", "Tạo, xóa, sửa đối tượng")], h=1.0)
# loop-back
arrow(s, 8.4, 2.4, 8.4, 2.62, color="accent1", head=False)
arrow(s, 8.4, 2.62, 6.2, 2.62, color="accent1", head=False)
arrow(s, 6.2, 2.62, 6.2, 2.4, color="accent1")
tb(s, 6.3, 2.66, 2.0, 0.25, ["lặp liên tục"], size=10, italic=True, color="accent1", align="ctr")
tb(s, 0.6, 2.75, 5.0, 0.25, [[("ĐIỂM MỞ RỘNG", {"spc": 200})]], size=10, bold=True, color="accent5")
flow_row(s, 3.05, [("CRD", "Kiểu tài nguyên mới"), ("Operator", "Controller theo domain"),
                   ("CNI", "Mạng cho Pod"), ("CSI", "Lưu trữ"),
                   ("Webhook", "Policy khi admission")], h=0.95, light=True, arrows=False)
caption(s, "OpenShift xây trên chính các điểm mở rộng này: mọi thành phần nền tảng đều là Operator + CRD.",
        y=4.2)

# 7. K8s is a framework
s = content([("Kubernetes là bộ khung, ", False), ("phải tự lắp ráp", True)],
            "Đây là cầu nối sang OpenShift: một cluster Kubernetes thuần chưa phải nền tảng cho "
            "developer. Mỗi tổ chức phải tự chọn, tích hợp, nâng cấp và chịu trách nhiệm cho từng "
            "mảnh ghép. Gateway API là API thế hệ mới thay cho Ingress.")
table(s, 0.6, 1.3, [2.1, 3.0, 3.7], [
    ["Nhu cầu", "Kubernetes cung cấp", "Phải tự chọn và tích hợp"],
    ["Đưa traffic vào", "API Ingress / Gateway API", "Controller: HAProxy, Traefik, Envoy Gateway"],
    ["Image registry", "Không có", "Harbor, Nexus, Quay"],
    ["Build và CI/CD", "Không có", "GitLab CI, Jenkins, Tekton, Argo CD"],
    ["Giám sát và log", "metrics-server cơ bản", "Prometheus, Grafana, Loki, EFK"],
    ["Xác thực người dùng", "Cert, token, OIDC cấu hình tay", "Keycloak, Dex kết hợp RBAC"],
    ["Policy và bảo mật", "RBAC, Pod Security Admission", "Kyverno, OPA Gatekeeper, Falco"],
], rowh=0.43, size=12)
caption(s, "Mỗi tổ chức tự ghép một “distribution” riêng. OpenShift là distribution đã ghép sẵn, kiểm thử và có hỗ trợ.",
        y=4.45)

# ---------------------------------------------------------------- Section 2
section("02", "Red Hat OpenShift", "Kubernetes được đóng gói thành nền tảng ứng dụng cho doanh nghiệp")

# 8. OpenShift layers
s = content([("OpenShift = Kubernetes + ", False), ("các lớp tích hợp sẵn", True)],
            "Lõi vẫn là Kubernetes chuẩn (đạt chứng nhận CNCF conformance). OpenShift thêm OS bất "
            "biến RHCOS bên dưới và các dịch vụ nền tảng, developer bên trên. OKD là bản cộng đồng "
            "upstream. OCP 4.22 (GA 07/2026) dựa trên Kubernetes 1.35 và CRI-O 1.35.")
layers = [
    ("Ứng dụng và workload", None, "bg1", 22, "bg1"),
    ("Dịch vụ developer", "Console · S2I · Pipelines · GitOps · Serverless", "accent2", None, "bg1"),
    ("Dịch vụ nền tảng", "OVN-K · Router · Registry · Monitoring · OLM", "accent2", None, "bg1"),
    ("Kubernetes + CRI-O", "Lõi chuẩn, đạt chứng nhận CNCF", "accent1", None, "tx1"),
    ("RHCOS", "OS bất biến, cập nhật theo cluster", "accent6", None, "bg1"),
    ("Hạ tầng", "Bare metal · vSphere · OpenStack · cloud", "tx1", None, "bg1"),
]
yy = 1.3
for name, sub, fill, alpha, tc in layers:
    card(s, 0.6, yy, 5.3, 0.48, fill=fill, alpha=alpha, line=None if fill != "accent2" else "accent1", adj=0.1)
    tb(s, 0.78, yy, 1.75 if sub else 4.9, 0.48, [name], size=12, bold=True, color=tc, anchor="ctr")
    if sub:
        tb(s, 2.45, yy, 3.0, 0.48, [sub], size=10, color=tc, anchor="ctr")
    yy += 0.54
pic(s, "k8s_color.png", 5.42, 1.3 + 3 * 0.54 + 0.06, 0.36, 0.36)
card(s, 6.15, 1.3, 3.25, 3.2, adj=0.04)
pic(s, "ocp_red.png", 6.35, 1.42, 0.42, 0.42)
tb(s, 6.85, 1.42, 2.4, 0.42, ["Các phiên bản"], size=16, bold=True, color="accent1", anchor="ctr")
tb(s, 6.35, 1.98, 2.95, 2.45, [
    [("OKD: ", {"bold": True}), ("bản cộng đồng, miễn phí", {})],
    [("OKE: ", {"bold": True}), ("lõi Kubernetes + RHCOS", {})],
    [("OCP: ", {"bold": True}), ("nền tảng đầy đủ", {})],
    [("Platform Plus: ", {"bold": True}), ("thêm ACM, ACS, Quay", {})],
    [("Managed: ", {"bold": True}), ("ROSA, ARO, OpenShift Dedicated", {})],
], size=12, bullet=True, after=5)
caption(s, "OpenShift Container Platform 4.22 (GA 07/2026) dựa trên Kubernetes 1.35 và CRI-O 1.35")

# 9. OpenShift architecture
s = content([("Kiến trúc OpenShift: ", False), ("Operator", True), (" quản lý tất cả", False)],
            "Control plane có thêm openshift-apiserver (API riêng như Route, Project, BuildConfig), "
            "oauth-server. Infra node chạy router, registry, monitoring để tách khỏi workload (và "
            "không tính subscription). Cluster Version Operator điều phối toàn bộ Cluster Operator; "
            "Machine Config Operator quản lý OS của mọi node. Không SSH vào node để sửa cấu hình.")
cols = [
    (0.6, 3.0, "Control plane (3 node)", [
        ("kube-apiserver · etcd", "tx1"), ("scheduler · controller-mgr", "tx1"),
        ("openshift-apiserver", "accent6"), ("oauth-server", "accent6"),
        ("Cluster Version Operator", "accent1")]),
    (3.8, 2.7, "Infra node", [
        ("Router (HAProxy)", "accent6"), ("Image registry", "accent6"),
        ("Monitoring (Prometheus)", "accent6"), ("Logging (Loki)", "accent6")]),
    (6.7, 2.7, "Worker node", [
        ("kubelet + CRI-O", "tx1"), ("Pod ứng dụng", "accent1"), ("Pod ứng dụng", "accent1"),
        ("Pod ứng dụng", "accent1")]),
]
for x, w, title, comps in cols:
    card(s, x, 1.3, w, 2.7, fill="accent2", adj=0.04)
    tb(s, x + 0.18, 1.38, w - 0.36, 0.3, [title], size=13, bold=True, color="accent1")
    for k, (c, f) in enumerate(comps):
        yy = 1.75 + k * 0.44
        label_box(s, x + 0.18, yy, w - 0.36, 0.38, c, fill=f, line=None if f == "accent1" else "accent5",
                  tcolor="tx1" if f == "accent1" else "bg1", tsize=11, pad=0.1)
card(s, 0.6, 4.12, 8.8, 0.45, fill="accent1", line=None, adj=0.12)
tb(s, 0.8, 4.12, 8.4, 0.45, [[("Machine Config Operator", {"bold": True}),
                              (": quản lý cấu hình RHCOS (kernel args, CRI-O, kubelet) của mọi node theo khai báo", {})]],
   size=12, color="tx1", anchor="ctr")

# 10. Install & upgrade with operators
s = content([("Cài đặt và nâng cấp ", False), ("tự động bằng Operator", True)],
            "IPI: installer tự tạo hạ tầng (cloud, vSphere). UPI: tự chuẩn bị hạ tầng. Agent-based: "
            "phù hợp bare metal/air-gapped. Nâng cấp: chọn channel, CVO kéo release payload "
            "(một image chứa phiên bản đã kiểm thử của mọi thành phần), lần lượt nâng các Cluster "
            "Operator, rồi MCO drain và reboot từng node. Bản số chẵn là EUS, cho phép nâng EUS lên EUS.")
tb(s, 0.6, 1.25, 4.0, 0.25, [[("CÀI ĐẶT", {"spc": 200})]], size=10, bold=True, color="accent5")
flow_row(s, 1.52, [("install-config", "Khai báo cluster"), ("Installer", "IPI · UPI · Agent"),
                   ("Bootstrap", "Dựng control plane"), ("Operators", "Tự cấu hình cluster"),
                   ("Sẵn sàng", "Console, oc login")], h=0.92)
tb(s, 0.6, 2.67, 4.0, 0.25, [[("NÂNG CẤP OTA", {"spc": 200})]], size=10, bold=True, color="accent5")
flow_row(s, 2.94, [("Channel", "stable · fast · eus"), ("CVO", "Tải release payload"),
                   ("Cluster Operator", "Nâng lần lượt"), ("MCO", "Drain, reboot từng node"),
                   ("Đồng bộ", "OS, K8s, add-on")], h=0.92)
caption(s, "So với kubeadm: OS, runtime, Kubernetes và add-on được nâng cấp như một khối đã kiểm thử.",
        y=4.1)

# ---------------------------------------------------------------- Section 3
section("03", "So sánh Kubernetes và OpenShift", "Khác biệt về nền tảng, cách sử dụng và vận hành")

# 11. Overview table
s = content([("Tổng quan ", False), ("khác biệt", True)],
            "Bảng tóm tắt; các slide sau đi sâu vào bảo mật, cách dùng, mạng/lưu trữ và vòng đời. "
            "Lưu ý: Kubernetes ở đây là upstream tự dựng; các bản managed (EKS, AKS, GKE) thu hẹp "
            "khoảng cách ở phần cài đặt và nâng cấp control plane.")
table(s, 0.6, 1.25, [1.9, 3.3, 3.6], [
    ["Khía cạnh", "Kubernetes (upstream)", "OpenShift"],
    ["Bản chất", "Dự án mã nguồn mở CNCF, bộ khung", "Distribution thương mại của Red Hat"],
    ["OS của node", "Tự chọn: Ubuntu, RHEL, Flatcar…", "RHCOS bất biến, do cluster quản lý"],
    ["Runtime", "containerd, CRI-O…", "CRI-O"],
    ["Mạng mặc định", "Tự chọn CNI: Calico, Cilium…", "OVN-Kubernetes"],
    ["Ingress", "Tự cài controller", "Router HAProxy + Route có sẵn"],
    ["Bảo mật mặc định", "PSA, mặc định khá mở", "SCC restricted-v2: cấm root"],
    ["Nâng cấp", "Thủ công theo từng thành phần", "OTA qua Cluster Version Operator"],
    ["Hỗ trợ, chi phí", "Miễn phí, tự hỗ trợ", "Subscription theo core, Red Hat hỗ trợ"],
], rowh=0.37, size=11)
pic(s, "k8s_white.png", 5.45, 1.3, 0.27, 0.27)
pic(s, "ocp_white.png", 9.05, 1.3, 0.27, 0.27)

# 12. Security
s = content([("Bảo mật mặc định: ", False), ("PSA", True), (" so với ", False), ("SCC", True)],
            "Đây là khác biệt gây vấp nhiều nhất khi chuyển ứng dụng. OpenShift gán UID ngẫu nhiên "
            "(trong dải của project) và GID 0, nên image phải chạy được với UID bất kỳ. OpenShift "
            "cũng đồng bộ nhãn PSA cho namespace, nhưng SCC vẫn là cơ chế chính.")
for i, (title, lab, labc, items) in enumerate([
    ("Kubernetes · Pod Security Admission", "Đặc điểm", "accent5", [
        "3 mức privileged, baseline, restricted gắn qua label namespace",
        "Không cấu hình thì không ép buộc",
        "Image yêu cầu root vẫn chạy được"]),
    ("OpenShift · SCC", "Đặc điểm", "accent5", [
        "restricted-v2 áp mặc định cho mọi Pod",
        "UID ngẫu nhiên theo project, cấm root",
        "SELinux MCS tách biệt giữa các project"]),
]):
    x = 0.6 + i * 4.5
    card(s, x, 1.3, 4.3, 1.75, adj=0.04)
    tb(s, x + 0.2, 1.4, 3.9, 0.3, [title], size=15, bold=True, color="accent1")
    tb(s, x + 0.2, 1.74, 3.9, 0.22, [lab], size=11, bold=True, color=labc)
    tb(s, x + 0.2, 2.0, 3.95, 1.2, [[(t, {})] for t in items], size=12, bullet=True, after=3)
tb(s, 0.6, 3.2, 6.0, 0.3, ["Hệ quả khi chuyển ứng dụng sang OpenShift"], size=15, bold=True, color="accent1")
for i, (t, d) in enumerate([("1 · Image chạy root", "Lỗi khi khởi động, cần chạy được với UID bất kỳ"),
                            ("2 · Quyền ghi", "chgrp 0 và chmod g=u cho thư mục cần ghi"),
                            ("3 · Port < 1024", "Lắng nghe port cao như 8080"),
                            ("4 · Cấp quyền", "Gán SCC riêng qua RBAC, tránh privileged")]):
    x = 0.6 + i * 2.25
    card(s, x, 3.6, 2.05, 1.0, fill="accent1", line=None, adj=0.08)
    tb(s, x + 0.14, 3.68, 1.8, 0.85, [{"runs": [(t, {"bold": True})], "size": 12},
                                    {"runs": [(d, {})], "size": 11}], color="tx1")

# 13. kubectl vs oc
s = content([("Cách sử dụng: ", False), ("kubectl", True), (" và ", False), ("oc", True)],
            "oc là superset của kubectl, mọi lệnh kubectl đều chạy với oc. Điểm khác nằm ở các lệnh "
            "cấp cao cho developer: new-project (self-service), new-app (deploy từ image hoặc source "
            "qua S2I), expose (tạo Route), login (OAuth của cluster).")
pic(s, "k8s_white.png", 2.65, 1.28, 0.3, 0.3)
tb(s, 3.0, 1.28, 2.5, 0.3, ["kubectl"], size=14, bold=True, color="accent1", anchor="ctr", font=MONO)
pic(s, "ocp_white.png", 6.1, 1.28, 0.3, 0.3)
tb(s, 6.45, 1.28, 2.5, 0.3, ["oc"], size=14, bold=True, color="accent1", anchor="ctr", font=MONO)
rows = [
    ("Không gian làm việc", "kubectl create ns demo", "oc new-project demo"),
    ("Deploy từ image", "kubectl create deploy web \\\n  --image=reg/web:1.0", "oc new-app reg/web:1.0"),
    ("Build từ source", "Cần CI ngoài + registry", "oc new-app https://git/app.git"),
    ("Mở ra bên ngoài", "Ingress YAML + controller", "oc expose service web"),
    ("Đăng nhập", "kubeconfig: cert hoặc OIDC", "oc login --server=https://api…"),
]
for i, (lab, k8, oc) in enumerate(rows):
    yy = 1.68 + i * 0.58
    tb(s, 0.6, yy, 1.9, 0.5, [lab], size=12, bold=True, anchor="ctr")
    for x, txt in ((2.6, k8), (6.05, oc)):
        mono = not txt.startswith(("Cần", "Ingress", "kubeconfig"))
        card(s, x, yy, 3.35, 0.5, fill="tx1", line="accent3", adj=0.1)
        tb(s, x + 0.12, yy, 3.15, 0.5, txt.split("\n"), size=10 if mono else 11,
           font=MONO if mono else None, color="bg1" if mono else "accent5", anchor="ctr")
caption(s, "oc là superset của kubectl: mọi lệnh kubectl đều chạy được với oc.", y=4.63)

# 14. OpenShift-specific concepts
s = content([("Khái niệm riêng ", False), ("của OpenShift", True)],
            "DeploymentConfig bị deprecated từ OCP 4.14, nên dùng Deployment chuẩn. Route có trước "
            "Ingress; OpenShift vẫn hỗ trợ Ingress (tự chuyển thành Route) và Gateway API. "
            "Builds for OpenShift (dựa trên Shipwright) là hướng mới thay cho BuildConfig.")
grid = [
    ("Namespace → Project", "Namespace kèm metadata và quyền; developer tự tạo qua template có quota."),
    ("Ingress → Route", "TLS edge, passthrough, re-encrypt; Ingress và Gateway API vẫn dùng được."),
    ("DeploymentConfig", "Đã deprecated từ 4.14, nên dùng Deployment chuẩn của Kubernetes."),
    ("ImageStream", "Theo dõi tag image, tự kích hoạt redeploy khi image thay đổi."),
    ("BuildConfig · S2I", "Build image ngay trong cluster từ source code, không cần Dockerfile."),
    ("OperatorHub · OLM", "Cài và nâng cấp Operator có kiểm soát, như kho ứng dụng của cluster."),
]
for i, (t, d) in enumerate(grid):
    x = 0.6 + (i % 3) * 3.0; yy = 1.3 + (i // 3) * 1.7
    card(s, x, yy, 2.8, 1.5, adj=0.05)
    tb(s, x + 0.18, yy + 0.14, 2.45, 0.3, [t], size=14, bold=True, color="accent1")
    tb(s, x + 0.18, yy + 0.55, 2.45, 0.9, [d], size=12)

# 15. Network, storage, CI/CD
s = content([("Mạng, lưu trữ và CI/CD: ", False), ("tự chọn", True), (" hay ", False), ("tích hợp", True)],
            "OVN-Kubernetes là CNI mặc định; OpenShift SDN đã bị gỡ từ 4.17. Multus cho phép gắn "
            "nhiều interface vào Pod (hay dùng cho telco/VM). ODF dựa trên Ceph (Rook). Pipelines "
            "dựa trên Tekton, GitOps dựa trên Argo CD, cài qua OperatorHub và có hỗ trợ.")
for i, (n, t, k8, ocp) in enumerate([
    ("01", "Mạng", "Tự chọn CNI: Calico, Cilium, Flannel", "OVN-Kubernetes mặc định, Multus, egress IP"),
    ("02", "Lưu trữ", "CSI driver theo nhà cung cấp", "CSI + Data Foundation (Ceph): block, file, object"),
    ("03", "CI/CD", "Tự tích hợp GitLab CI, Argo CD", "Pipelines (Tekton), GitOps (Argo CD) có hỗ trợ"),
]):
    x = 0.6 + i * 3.0
    card(s, x, 1.3, 2.8, 3.3, adj=0.04)
    tb(s, x + 0.2, 1.4, 1.0, 0.5, [n], size=28, bold=True, color="accent1")
    tb(s, x + 0.2, 1.95, 2.4, 0.3, [t], size=16, bold=True)
    pic(s, "k8s_white.png", x + 0.2, 2.42, 0.26, 0.26)
    tb(s, x + 0.55, 2.42, 2.0, 0.26, ["Kubernetes"], size=11, bold=True, color="accent5", anchor="ctr")
    tb(s, x + 0.2, 2.75, 2.45, 0.6, [k8], size=12)
    pic(s, "ocp_white.png", x + 0.2, 3.45, 0.26, 0.26)
    tb(s, x + 0.55, 3.45, 2.0, 0.26, ["OpenShift"], size=11, bold=True, color="accent1", anchor="ctr")
    tb(s, x + 0.2, 3.78, 2.45, 0.7, [ocp], size=12)

# 16. Lifecycle stats
s = content([("Vòng đời phiên bản và ", False), ("vận hành", True)],
            "Upstream ra khoảng 3 minor mỗi năm, mỗi minor được vá khoảng 14 tháng, nên phải nâng "
            "cấp thường xuyên. OpenShift đi sau upstream 1–2 minor (4.22 dựa trên 1.35, trong khi "
            "upstream đã có bản mới hơn), đổi lại release payload đã kiểm thử và bản EUS hỗ trợ dài hơn.")
for i, (big, d) in enumerate([("~4 tháng", "Chu kỳ ra minor mới của Kubernetes"),
                              ("~14 tháng", "Thời gian vá lỗi cho mỗi minor upstream"),
                              ("1.35", "Nền Kubernetes của OCP 4.22 (07/2026)"),
                              ("EUS", "Bản OCP số chẵn hỗ trợ dài, nâng EUS lên EUS")]):
    x = 0.6 + i * 2.27
    card(s, x, 1.3, 2.0, 1.25, adj=0.05)
    tb(s, x + 0.16, 1.38, 1.75, 0.55, [big], size=24, bold=True, color="accent1")
    tb(s, x + 0.16, 1.95, 1.75, 0.55, [d], size=11)
for i, (t, items) in enumerate([
    ("Kubernetes tự vận hành", ["Nâng tuần tự control plane, node, add-on",
                                "Tự kiểm thử tương thích CNI, CSI, ingress",
                                "Được dùng ngay tính năng upstream mới"]),
    ("OpenShift", ["Một payload đã kiểm thử: OS + K8s + operator",
                   "Nâng cấp một lệnh hoặc một click qua CVO",
                   "Đi sau upstream 1–2 minor, đổi lấy ổn định"]),
]):
    x = 0.6 + i * 4.5
    card(s, x, 2.75, 4.3, 1.85, adj=0.04)
    tb(s, x + 0.2, 2.85, 3.9, 0.3, [t], size=15, bold=True, color="accent1")
    tb(s, x + 0.2, 3.25, 3.95, 1.3, [[(it, {})] for it in items], size=12, bullet=True, after=4)

# 17. When to choose
s = content([("Khi nào chọn ", False), ("Kubernetes", True), (", khi nào ", False), ("OpenShift", True)],
            "Không có đáp án tuyệt đối. Với ngân hàng: on-premise, yêu cầu tuân thủ và kiểm toán, "
            "cần hỗ trợ có SLA thường nghiêng về OpenShift. Nếu đội platform mạnh, chạy chủ yếu "
            "trên cloud thì Kubernetes managed (EKS, AKS, GKE) hoặc tự dựng vẫn hợp lý. "
            "Hãy so TCO gồm cả nhân sự vận hành, không chỉ phí license.")
for i, (t, logo, items) in enumerate([
    ("Kubernetes · vanilla hoặc managed", "k8s_white.png",
     ["Đội platform mạnh, muốn tự chọn từng thành phần", "Chạy chủ yếu trên public cloud",
      "Ưu tiên giảm license, chấp nhận chi phí nhân sự"]),
    ("OpenShift", "ocp_white.png",
     ["On-premise, hybrid, yêu cầu tuân thủ cao", "Cần hỗ trợ thương mại, SLA, chứng nhận",
      "Một nền tảng đồng nhất cho nhiều đội dev"]),
]):
    x = 0.6 + i * 4.5
    card(s, x, 1.3, 4.3, 1.75, adj=0.04)
    pic(s, logo, x + 0.2, 1.42, 0.3, 0.3)
    tb(s, x + 0.6, 1.4, 3.6, 0.34, [t], size=15, bold=True, color="accent1", anchor="ctr")
    tb(s, x + 0.2, 1.8, 3.9, 0.22, ["Phù hợp khi"], size=11, bold=True, color="accent5")
    tb(s, x + 0.2, 2.06, 3.95, 1.15, [[(it, {})] for it in items], size=12, bullet=True, after=3)
tb(s, 0.6, 3.2, 6.0, 0.3, ["Tiêu chí quyết định"], size=15, bold=True, color="accent1")
for i, (t, d) in enumerate([("1 · Đội ngũ", "Ai vận hành, vá lỗi và trực 24/7?"),
                            ("2 · Tuân thủ", "Yêu cầu kiểm toán, bảo mật mặc định"),
                            ("3 · Hạ tầng", "On-premise, cloud hay hybrid"),
                            ("4 · TCO", "License so với nhân sự và tích hợp")]):
    x = 0.6 + i * 2.25
    card(s, x, 3.6, 2.05, 1.0, fill="accent1", line=None, adj=0.08)
    tb(s, x + 0.14, 3.68, 1.8, 0.85, [{"runs": [(t, {"bold": True})], "size": 12},
                                    {"runs": [(d, {})], "size": 11}], color="tx1")

# 18. Key takeaways
s = content([("Thông điệp ", False), ("chính", True)],
            "Chốt lại 5 ý, sau đó mở phần hỏi đáp.")
for i, t in enumerate([
    "Container: tiến trình Linux cô lập bằng namespaces, cgroups, đóng gói theo OCI",
    "Kubernetes: bộ khung điều phối declarative, mở rộng qua CRD, Operator, CNI, CSI",
    "OpenShift: Kubernetes + RHCOS + dịch vụ nền tảng, tất cả quản lý bằng Operator",
    "Khác biệt lớn nhất khi dùng: SCC chặt mặc định và trải nghiệm developer tích hợp",
    "Chọn nền tảng theo năng lực đội ngũ, yêu cầu tuân thủ và tổng chi phí sở hữu",
]):
    yy = 1.3 + i * 0.66
    card(s, 0.6, yy, 8.8, 0.56, adj=0.12)
    tb(s, 0.8, yy, 0.6, 0.56, [f"{i + 1:02d}"], size=18, bold=True, color="accent1", anchor="ctr")
    tb(s, 1.5, yy, 7.8, 0.56, [t], size=13, anchor="ctr")

# 19. Closing
s = clone(T_CLOSING)
s.notes_slide.notes_text_frame.text = "Hỏi đáp."

# ---------------------------------------------------------------- drop template slides
sldIdLst = prs.slides._sldIdLst
for sld in ORIG:
    for sid in list(sldIdLst):
        if prs.part.related_part(sid.rId) is sld.part:
            prs.part.drop_rel(sid.rId)
            sldIdLst.remove(sid)
prs.save(OUT)
print("saved", OUT, "slides:", len(prs.slides))
