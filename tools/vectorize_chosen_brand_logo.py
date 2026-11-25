import cv2
import numpy as np

def vectorize_emblem():
    # Load original minimal brand image
    img = cv2.imread(r'static\img\pindaisurvei_minimal_brand.jpg')
    h, w, _ = img.shape

    # Emblem region: x in [250, 510], y in [245, 525]
    crop = img[245:525, 250:510]
    ch, cw, _ = crop.shape

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

    # 1. Blue Mask: H in [95, 135], S > 50, V > 50
    lower_blue = np.array([95, 50, 40])
    upper_blue = np.array([135, 255, 255])
    blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # 2. Green Mask: H in [35, 95], S > 50, V > 50
    lower_green = np.array([35, 50, 40])
    upper_green = np.array([95, 255, 255])
    green_mask = cv2.inRange(hsv, lower_green, upper_green)

    # Upscale 4x with cubic interpolation for silky-smooth curves
    scale_up = 4
    w4, h4 = cw * scale_up, ch * scale_up

    # Process Blue
    blue_large = cv2.resize(blue_mask, (w4, h4), interpolation=cv2.INTER_CUBIC)
    blue_smooth = cv2.GaussianBlur(blue_large, (7, 7), 2.0)
    _, blue_thresh = cv2.threshold(blue_smooth, 127, 255, cv2.THRESH_BINARY)

    # Process Green
    green_large = cv2.resize(green_mask, (w4, h4), interpolation=cv2.INTER_CUBIC)
    green_smooth = cv2.GaussianBlur(green_large, (7, 7), 2.0)
    _, green_thresh = cv2.threshold(green_smooth, 127, 255, cv2.THRESH_BINARY)

    # Find total emblem bounding box on the 4x canvas
    total_mask = cv2.bitwise_or(blue_thresh, green_thresh)
    tot_cnts, _ = cv2.findContours(total_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    all_pts = np.vstack([c.reshape(-1, 2) for c in tot_cnts if cv2.contourArea(c) > 500])
    min_x, min_y = np.min(all_pts, axis=0)
    max_x, max_y = np.max(all_pts, axis=0)
    bw = max_x - min_x
    bh = max_y - min_y
    print(f"Emblem bbox on 4x: ({min_x}, {min_y}) to ({max_x}, {max_y}), size={bw}x{bh}")

    # Map to 512x512 with margin (target height = 400)
    target_h = 400.0
    s = target_h / bh
    target_w = bw * s

    offset_x = (512.0 - target_w) / 2.0 - min_x * s
    offset_y = (512.0 - target_h) / 2.0 - min_y * s

    def contours_to_paths(mask, epsilon=1.0):
        cnts, hier = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_KCOS)
        paths = []
        for c in cnts:
            if cv2.contourArea(c) < 200:
                continue
            approx = cv2.approxPolyDP(c, epsilon, True).reshape(-1, 2)
            if len(approx) < 3:
                continue
            first = approx[0]
            fx = first[0] * s + offset_x
            fy = first[1] * s + offset_y
            seg = [f"M {fx:.2f} {fy:.2f}"]
            for pt in approx[1:]:
                px = pt[0] * s + offset_x
                py = pt[1] * s + offset_y
                seg.append(f"L {px:.2f} {py:.2f}")
            seg.append("Z")
            paths.append(" ".join(seg))
        return " ".join(paths)

    blue_d = contours_to_paths(blue_thresh, epsilon=0.9)
    green_d = contours_to_paths(green_thresh, epsilon=0.9)

    print(f"Blue path length: {len(blue_d)}, Green path length: {len(green_d)}")

    # -------------------------------------------------------------------------
    # 1. Master Square Emblem: static/img/pindaisurvei_logo.svg (512x512)
    # -------------------------------------------------------------------------
    svg_square = f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 512 512" width="512" height="512" fill="none">
  <defs>
    <!-- BPS Royal Blue Gradient -->
    <linearGradient id="bpsRoyal" x1="0%" y1="0%" x2="0%" y2="100%">
      <stop offset="0%" stop-color="#1B3C97"/>
      <stop offset="100%" stop-color="#102A72"/>
    </linearGradient>

    <!-- Vibrant Natural Leaf Green Gradient -->
    <linearGradient id="leafGreen" x1="0%" y1="100%" x2="100%" y2="0%">
      <stop offset="0%" stop-color="#1DBF73"/>
      <stop offset="50%" stop-color="#25B878"/>
      <stop offset="100%" stop-color="#34D399"/>
    </linearGradient>

    <!-- Soft Ambient Drop Shadow -->
    <filter id="logoShadow" x="-10%" y="-10%" width="120%" height="120%">
      <feDropShadow dx="0" dy="10" stdDeviation="14" flood-color="#1B3C97" flood-opacity="0.16"/>
    </filter>
  </defs>

  <g filter="url(#logoShadow)">
    <!-- Blue Stem & Teardrop Loop of 'P' -->
    <path d="{blue_d}" fill="url(#bpsRoyal)" fill-rule="evenodd"/>
    <!-- Flowing Green Ribbon & Leaf Loop of 'P' -->
    <path d="{green_d}" fill="url(#leafGreen)" fill-rule="evenodd"/>
  </g>
</svg>'''

    with open(r'static\img\pindaisurvei_logo.svg', 'w', encoding='utf-8') as f:
        f.write(svg_square)
    print("Saved static/img/pindaisurvei_logo.svg")

    # -------------------------------------------------------------------------
    # 2. Master Horizontal Lockup: static/img/pindaisurvei_logo_horizontal.svg
    # Dimensions: 820 x 220
    # Emblem on the left, "PindaiSurvei" and "SURVEY INTELLIGENCE" on right
    # -------------------------------------------------------------------------
    h_target_h = 150.0
    hs = h_target_h / bh
    h_offset_x = 36.0 - min_x * hs
    h_offset_y = 35.0 - min_y * hs

    def h_contours_to_paths(mask, epsilon=1.0):
        cnts, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_KCOS)
        paths = []
        for c in cnts:
            if cv2.contourArea(c) < 200:
                continue
            approx = cv2.approxPolyDP(c, epsilon, True).reshape(-1, 2)
            if len(approx) < 3:
                continue
            first = approx[0]
            fx = first[0] * hs + h_offset_x
            fy = first[1] * hs + h_offset_y
            seg = [f"M {fx:.2f} {fy:.2f}"]
            for pt in approx[1:]:
                px = pt[0] * hs + h_offset_x
                py = pt[1] * hs + h_offset_y
                seg.append(f"L {px:.2f} {py:.2f}")
            seg.append("Z")
            paths.append(" ".join(seg))
        return " ".join(paths)

    h_blue_d = h_contours_to_paths(blue_thresh, epsilon=0.9)
    h_green_d = h_contours_to_paths(green_thresh, epsilon=0.9)

    svg_horizontal = f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 820 220" width="820" height="220" fill="none">
  <defs>
    <linearGradient id="hRoyal" x1="0%" y1="0%" x2="0%" y2="100%">
      <stop offset="0%" stop-color="#1B3C97"/>
      <stop offset="100%" stop-color="#102A72"/>
    </linearGradient>

    <linearGradient id="hGreen" x1="0%" y1="100%" x2="100%" y2="0%">
      <stop offset="0%" stop-color="#1DBF73"/>
      <stop offset="50%" stop-color="#25B878"/>
      <stop offset="100%" stop-color="#34D399"/>
    </linearGradient>
  </defs>

  <!-- Left: The Chosen Minimalist Emblem -->
  <g id="emblem">
    <path d="{h_blue_d}" fill="url(#hRoyal)" fill-rule="evenodd"/>
    <path d="{h_green_d}" fill="url(#hGreen)" fill-rule="evenodd"/>
  </g>

  <!-- Right: Exact Typography Matching Minimal Brand Identity -->
  <g id="typography">
    <!-- PindaiSurvei Wordmark -->
    <text x="210" y="112" 
          font-family="system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Montserrat', 'Plus Jakarta Sans', sans-serif" 
          font-size="64" 
          font-weight="600" 
          letter-spacing="-0.025em" 
          fill="#172946">
      PindaiSurvei
    </text>

    <!-- Subtitle: SURVEY INTELLIGENCE -->
    <text x="214" y="152" 
          font-family="system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Montserrat', 'Plus Jakarta Sans', sans-serif" 
          font-size="19" 
          font-weight="600" 
          letter-spacing="0.14em" 
          fill="#334155">
      SURVEY INTELLIGENCE
    </text>

    <!-- Institutional Endorsement -->
    <text x="214" y="178" 
          font-family="system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif" 
          font-size="13" 
          font-weight="500" 
          letter-spacing="0.04em" 
          fill="#64748B">
      Badan Pusat Statistik Kab. Penajam Paser Utara
    </text>
  </g>
</svg>'''

    with open(r'static\img\pindaisurvei_logo_horizontal.svg', 'w', encoding='utf-8') as f:
        f.write(svg_horizontal)
    print("Saved static/img/pindaisurvei_logo_horizontal.svg")

    # -------------------------------------------------------------------------
    # 3. Master Favicon: static/img/favicon.svg (64x64)
    # -------------------------------------------------------------------------
    fav_target_h = 420.0
    fav_s = fav_target_h / bh
    fav_offset_x = (512.0 - bw * fav_s) / 2.0 - min_x * fav_s
    fav_offset_y = (512.0 - bh * fav_s) / 2.0 - min_y * fav_s

    def fav_contours_to_paths(mask, epsilon=1.2):
        cnts, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_KCOS)
        paths = []
        for c in cnts:
            if cv2.contourArea(c) < 300:
                continue
            approx = cv2.approxPolyDP(c, epsilon, True).reshape(-1, 2)
            if len(approx) < 3:
                continue
            first = approx[0]
            fx = first[0] * fav_s + fav_offset_x
            fy = first[1] * fav_s + fav_offset_y
            seg = [f"M {fx:.2f} {fy:.2f}"]
            for pt in approx[1:]:
                px = pt[0] * fav_s + fav_offset_x
                py = pt[1] * fav_s + fav_offset_y
                seg.append(f"L {px:.2f} {py:.2f}")
            seg.append("Z")
            paths.append(" ".join(seg))
        return " ".join(paths)

    fav_blue_d = fav_contours_to_paths(blue_thresh, epsilon=1.0)
    fav_green_d = fav_contours_to_paths(green_thresh, epsilon=1.0)

    svg_fav = f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 512 512" width="64" height="64" fill="none">
  <path d="{fav_blue_d}" fill="#1B3C97" fill-rule="evenodd"/>
  <path d="{fav_green_d}" fill="#25B878" fill-rule="evenodd"/>
</svg>'''

    with open(r'static\img\favicon.svg', 'w', encoding='utf-8') as f:
        f.write(svg_fav)
    print("Saved static/img/favicon.svg")

if __name__ == '__main__':
    vectorize_emblem()
