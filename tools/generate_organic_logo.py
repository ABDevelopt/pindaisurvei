import cv2
import numpy as np
import os

def generate_svg():
    img_path = r'static\img\reference_emblem.png'
    img = cv2.imread(img_path)
    if img is None:
        print("Image not found:", img_path)
        return
    
    h, w, _ = img.shape
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)

    scale_up = 4
    thresh_large = cv2.resize(thresh, (w * scale_up, h * scale_up), interpolation=cv2.INTER_CUBIC)
    smoothed = cv2.GaussianBlur(thresh_large, (5, 5), 1.5)
    _, smoothed = cv2.threshold(smoothed, 127, 255, cv2.THRESH_BINARY)

    contours, hierarchy = cv2.findContours(smoothed, cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_L1)

    valid_cnts = [c for c in contours if cv2.contourArea(c) > 200]
    all_pts = np.vstack([c.reshape(-1, 2) for c in valid_cnts])
    min_x, min_y = np.min(all_pts, axis=0)
    max_x, max_y = np.max(all_pts, axis=0)
    bw, bh = max_x - min_x, max_y - min_y

    # Fit into 512x512
    target_h = 440.0
    target_scale = target_h / bh
    target_w = bw * target_scale

    offset_x = (512.0 - target_w) / 2.0 - min_x * target_scale
    offset_y = (512.0 - target_h) / 2.0 - min_y * target_scale

    path_parts = []
    for cnt in valid_cnts:
        epsilon = 0.8
        approx = cv2.approxPolyDP(cnt, epsilon, True).reshape(-1, 2)
        
        first = approx[0]
        fx = first[0] * target_scale + offset_x
        fy = first[1] * target_scale + offset_y
        seg = [f"M {fx:.2f} {fy:.2f}"]
        for pt in approx[1:]:
            px = pt[0] * target_scale + offset_x
            py = pt[1] * target_scale + offset_y
            seg.append(f"L {px:.2f} {py:.2f}")
        seg.append("Z")
        path_parts.append(" ".join(seg))

    combined_d = " ".join(path_parts)
    print(f"Total path segments: {len(path_parts)}")

    # 1. Standalone Mark (Obsidian & Transparent versions)
    # Standalone with delicate glow and gradient
    svg_mark = f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 512 512" width="512" height="512" fill="none">
  <defs>
    <!-- 4-Stage Organic Spectrum Gradient (Deep Navy -> Cyan -> Spring Green -> Warm Amber) -->
    <linearGradient id="organicGrad" x1="0" y1="476" x2="0" y2="36" gradientUnits="userSpaceOnUse">
      <stop offset="0%" stop-color="#0B3B95"/>
      <stop offset="12%" stop-color="#0055B8"/>
      <stop offset="35%" stop-color="#00AEEF"/>
      <stop offset="65%" stop-color="#76BC21"/>
      <stop offset="85%" stop-color="#F7941D"/>
      <stop offset="100%" stop-color="#F58220"/>
    </linearGradient>

    <!-- Ambient Shadow Filter -->
    <filter id="softGlow" x="-20%" y="-20%" width="140%" height="140%">
      <feDropShadow dx="0" dy="8" stdDeviation="12" flood-color="#00AEEF" flood-opacity="0.25"/>
      <feDropShadow dx="0" dy="2" stdDeviation="4" flood-color="#0B3B95" flood-opacity="0.3"/>
    </filter>
  </defs>

  <!-- The Organic Data Tree Emblem -->
  <g filter="url(#softGlow)">
    <path d="{combined_d}" fill="url(#organicGrad)" fill-rule="evenodd"/>
  </g>
</svg>'''

    with open(r'static\img\pindaisurvei_logo.svg', 'w', encoding='utf-8') as f:
        f.write(svg_mark)
    print("Saved static/img/pindaisurvei_logo.svg")

    # 2. Horizontal Master Brandmark (Emblem + Typography)
    # Dimensions: 720 x 200 (Compact horizontal lockup)
    # Emblem scaled to height=160 (from y=20 to y=180), x from 20 to 140
    # Text from x=155
    scale_h = 156.0 / bh
    scale_w = scale_h
    h_offset_x = 24.0 - min_x * scale_w
    h_offset_y = 22.0 - min_y * scale_h

    h_path_parts = []
    for cnt in valid_cnts:
        epsilon = 0.8
        approx = cv2.approxPolyDP(cnt, epsilon, True).reshape(-1, 2)
        first = approx[0]
        fx = first[0] * scale_w + h_offset_x
        fy = first[1] * scale_h + h_offset_y
        seg = [f"M {fx:.2f} {fy:.2f}"]
        for pt in approx[1:]:
            px = pt[0] * scale_w + h_offset_x
            py = pt[1] * scale_h + h_offset_y
            seg.append(f"L {px:.2f} {py:.2f}")
        seg.append("Z")
        h_path_parts.append(" ".join(seg))

    h_combined_d = " ".join(h_path_parts)

    svg_horizontal = f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 680 200" width="680" height="200" fill="none">
  <defs>
    <!-- Organic Spectrum Gradient for Horizontal Lockup -->
    <linearGradient id="hTreeGrad" x1="0" y1="180" x2="0" y2="20" gradientUnits="userSpaceOnUse">
      <stop offset="0%" stop-color="#0B3B95"/>
      <stop offset="12%" stop-color="#0055B8"/>
      <stop offset="35%" stop-color="#00AEEF"/>
      <stop offset="65%" stop-color="#76BC21"/>
      <stop offset="85%" stop-color="#F7941D"/>
      <stop offset="100%" stop-color="#F58220"/>
    </linearGradient>

    <!-- Typography Gradient -->
    <linearGradient id="textGrad" x1="0%" y1="0%" x2="100%" y2="0%">
      <stop offset="0%" stop-color="#0B3B95"/>
      <stop offset="100%" stop-color="#0077C8"/>
    </linearGradient>

    <!-- Subtitle Cyan Gradient -->
    <linearGradient id="subGrad" x1="0%" y1="0%" x2="100%" y2="0%">
      <stop offset="0%" stop-color="#0284C7"/>
      <stop offset="100%" stop-color="#0D9488"/>
    </linearGradient>
  </defs>

  <!-- Left: The Organic Data Tree Emblem -->
  <g>
    <path d="{h_combined_d}" fill="url(#hTreeGrad)" fill-rule="evenodd"/>
  </g>

  <!-- Vertical Divider Line (Subtle) -->
  <line x1="162" y1="36" x2="162" y2="164" stroke="#CBD5E1" stroke-width="1.5" stroke-linecap="round" opacity="0.7"/>

  <!-- Right: Typography Lockup -->
  <!-- Primary Brand Name: PindaiSurvei -->
  <g id="brand-text">
    <text x="182" y="96" font-family="system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif" font-size="44" font-weight="800" letter-spacing="-0.03em" fill="#0F172A">
      Pindai<tspan fill="url(#textGrad)">Survei</tspan>
    </text>

    <!-- Descriptor / Category -->
    <text x="184" y="128" font-family="system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif" font-size="16" font-weight="600" letter-spacing="0.04em" fill="#0284C7">
      SISTEM DIGITALISASI DOKUMEN SURVEI
    </text>

    <!-- Institutional Endorsement: BPS Kab. Penajam Paser Utara -->
    <text x="184" y="152" font-family="system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif" font-size="13" font-weight="500" letter-spacing="0.02em" fill="#64748B">
      Badan Pusat Statistik Kab. Penajam Paser Utara
    </text>
  </g>
</svg>'''

    with open(r'static\img\pindaisurvei_logo_horizontal.svg', 'w', encoding='utf-8') as f:
        f.write(svg_horizontal)
    print("Saved static/img/pindaisurvei_logo_horizontal.svg")

    # 3. Favicon (Clean 64x64 or 32x32 viewbox)
    svg_favicon = f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 512 512" width="64" height="64" fill="none">
  <defs>
    <linearGradient id="favGrad" x1="0" y1="480" x2="0" y2="32" gradientUnits="userSpaceOnUse">
      <stop offset="0%" stop-color="#0B3B95"/>
      <stop offset="25%" stop-color="#00AEEF"/>
      <stop offset="65%" stop-color="#76BC21"/>
      <stop offset="100%" stop-color="#F58220"/>
    </linearGradient>
  </defs>
  <path d="{combined_d}" fill="url(#favGrad)" fill-rule="evenodd"/>
</svg>'''

    with open(r'static\img\favicon.svg', 'w', encoding='utf-8') as f:
        f.write(svg_favicon)
    print("Saved static/img/favicon.svg")

if __name__ == '__main__':
    generate_svg()
