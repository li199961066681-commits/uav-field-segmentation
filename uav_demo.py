"""
uav_demo.py —— 无人机正射影像「农田图斑」分割（纯 OpenCV，无深度学习）

用法：把图片放在脚本同目录，运行  python uav_demo.py
输出：output/overlay_<图片名>.png  （原图上画出每个田块的红线）

思路一句话：按颜色圈出像农田的地方 → 单独找出「田埂/道路/沟渠」当缝 →
用缝把农田切成一块块 → 去掉太小/太细/太毛糙的 → 描出轮廓。
"""
import glob
import os

import cv2
import numpy as np

# ============================ 参数（要调就调这里） ============================
P = dict(
    h_range=(12, 70),       # 农田色相范围：12~70 覆盖青绿稻 → 枯黄稻/麦茬
    s_min=45,               # 饱和度过低的是灰白道路，排除
    v_range=(42, 240),      # 太暗（阴影/水）和过曝的都排除
    ms_sp=8, ms_sr=30,      # 均值漂移：抹平作物纹理、保留田块交界；调大更平滑但边界更少
    ms_pct=88,              # 梯度阈值取 88 分位 → 自适应，换图不用重调
    canny_pct=92,           # 边缘阈值同样取分位
    road=(60, 145),         # 机耕道：饱和度 < 60 且亮度 > 145
    dark=52,                # 亮度 < 52 视为沟渠/阴影线
    dilate=3,               # 把缝加粗几像素，缝太细会漏
    open_k=5,               # 开运算去小碎点；不要调大，会把细田埂一起抹掉
    repair_k=11,            # 单块内部补缺口；不要调大，会吃掉田角
    min_area=0.0018,        # 图斑面积下限（占整幅图的比例）
    min_extent=0.50,        # 面积/外接矩形面积，太小说明形状太碎
    min_solidity=0.60,      # 面积/凸包面积，太小说明太凹凸
    max_std=26.0,           # 内部纹理标准差，树带/屋顶毛糙，会被剔掉
    eps=0.010,              # 轮廓抽稀系数：压掉锯齿又保留拐角
    thick=3,                # 红线粗细
)
# ==========================================================================

BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, "output")


def field_mask(bgr):
    """第一步：按颜色 + 纹理，圈出「像农田」的像素"""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (P["h_range"][0], P["s_min"], P["v_range"][0]),
                       (P["h_range"][1], 255, P["v_range"][1]))
    # 局部标准差：小块内像素的跳跃程度。树带/建筑很跳，田块很平
    g = cv2.GaussianBlur(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), (5, 5), 0).astype(np.float32)
    m1, m2 = cv2.blur(g, (15, 15)), cv2.blur(g * g, (15, 15))
    mask[np.sqrt(np.maximum(m2 - m1 * m1, 0)) > P["max_std"]] = 0
    return mask


def barrier(bgr):
    """第二步：找「缝」—— 田埂、道路、沟渠、作物行之间的真实断层"""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    s, v = hsv[:, :, 1], hsv[:, :, 2]
    b = ((s < P["road"][0]) & (v > P["road"][1])).astype(np.uint8) * 255   # 亮而灰=机耕道
    b |= (v < P["dark"]).astype(np.uint8) * 255                            # 很暗=沟渠/阴影

    # (a) 主力线索：先均值漂移抹掉纹理，再取 Lab 三通道最大梯度
    #     这样留下的是「两块田之间的色调断层」，而不是作物行纹理
    small = cv2.resize(bgr, (bgr.shape[1] // 2, bgr.shape[0] // 2),
                       interpolation=cv2.INTER_AREA)
    ms = cv2.pyrMeanShiftFiltering(small, P["ms_sp"], P["ms_sr"], maxLevel=1)
    lab = cv2.cvtColor(ms, cv2.COLOR_BGR2LAB).astype(np.float32)
    gmax = np.zeros(lab.shape[:2], np.float32)
    for i in range(3):
        gx = cv2.Sobel(lab[:, :, i], cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(lab[:, :, i], cv2.CV_32F, 0, 1, ksize=3)
        gmax = np.maximum(gmax, cv2.magnitude(gx, gy))
    thr = max(30.0, float(np.percentile(cv2.GaussianBlur(gmax, (3, 3), 0), P["ms_pct"])))
    e = cv2.resize((gmax > thr).astype(np.uint8) * 255,
                   (bgr.shape[1], bgr.shape[0]), interpolation=cv2.INTER_NEAREST)
    b |= e

    # (b) 降采样 Canny：作物行纹理在低分辨率下消失，真正的田界还在
    gray = cv2.pyrDown(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY))
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    hi = max(8.0, float(np.percentile(cv2.magnitude(gx, gy), P["canny_pct"])))
    b |= cv2.resize(cv2.Canny(gray, 0.4 * hi, hi),
                    (bgr.shape[1], bgr.shape[0]), interpolation=cv2.INTER_NEAREST)

    return cv2.dilate(b, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (P["dilate"],) * 2))


def fields(bgr):
    """主流程：候选减缝 → 清理 → 逐块过滤 → 描轮廓"""
    H, W = bgr.shape[:2]
    bgr = cv2.bilateralFilter(bgr, 9, 45, 45)          # 保边平滑：压噪声不糊田埂
    mask = cv2.bitwise_and(field_mask(bgr), cv2.bitwise_not(barrier(bgr)))

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (P["open_k"],) * 2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)

    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    m1, m2 = cv2.blur(g, (15, 15)), cv2.blur(g * g, (15, 15))
    std = np.sqrt(np.maximum(m2 - m1 * m1, 0))

    n, lab, st, _ = cv2.connectedComponentsWithStats(mask, 8)
    keep, fills = [], []
    min_area = P["min_area"] * H * W
    for i in sorted(range(1, n), key=lambda j: -st[j, 4]):      # 从大到小处理
        if st[i, 4] < min_area:                                 # 太小
            continue
        x, y, w, h = st[i, 0], st[i, 1], st[i, 2], st[i, 3]
        sub = (lab[y:y + h, x:x + w] == i).astype(np.uint8) * 255
        # 只在自身框内补缺口，不会把相邻的田连起来
        sub = cv2.morphologyEx(sub, cv2.MORPH_CLOSE,
                               cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (P["repair_k"],) * 2))
        cs, _ = cv2.findContours(sub, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cs:
            continue
        c = max(cs, key=cv2.contourArea) + np.array([[[x, y]]], np.int32)
        a = cv2.contourArea(c)
        bx, by, bw, bh = cv2.boundingRect(c)
        if a < min_area or a / (bw * bh) < P["min_extent"]:      # 太细长
            continue
        if a / (cv2.contourArea(cv2.convexHull(c)) + 1e-6) < P["min_solidity"]:  # 太凹凸
            continue
        own = np.zeros((H, W), np.uint8)
        cv2.drawContours(own, [c], -1, 255, -1)
        if cv2.countNonZero(own) == 0:
            continue
        if float(std[own > 0].mean()) > P["max_std"]:             # 内部太毛糙
            continue
        if any(cv2.countNonZero(cv2.bitwise_and(own, f)) > 0.8 * cv2.countNonZero(own)
               for f in fills):                                   # 被大块包住的碎片
            continue
        keep.append(cv2.approxPolyDP(c, P["eps"] * cv2.arcLength(c, True), True))
        fills.append(own)
    return keep


def main():
    os.makedirs(OUT, exist_ok=True)
    paths = [p for e in ("png", "jpg", "jpeg", "PNG", "JPG")
             for p in glob.glob(os.path.join(BASE, f"*.{e}"))]
    if not paths:
        print("没找到图片，把 jpg/png 放到脚本同目录再运行")
        return
    for path in sorted(paths):
        img = cv2.imread(path)
        cnts = fields(img)
        out = img.copy()
        cv2.drawContours(out, cnts, -1, (0, 0, 255), P["thick"])
        name = os.path.splitext(os.path.basename(path))[0]
        cv2.imwrite(os.path.join(OUT, f"overlay_{name}.png"), out)
        print(f"{name}: {len(cnts)} 个图斑 → output/overlay_{name}.png")


if __name__ == "__main__":
    main()
