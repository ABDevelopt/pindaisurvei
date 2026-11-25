def create_master_minimal_svgs():
    # 1. Master Square Emblem (512x512) - Option B (The Flowing Leaf Checkmark 'P')
    svg_square = '''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 512 512" width="512" height="512" fill="none">
  <defs>
    <!-- Left Organic Wave (BPS Navy to Sky Cyan) -->
    <linearGradient id="minStemGrad" x1="0%" y1="100%" x2="50%" y2="0%">
      <stop offset="0%" stop-color="#0B3B95"/>
      <stop offset="60%" stop-color="#0284C7"/>
      <stop offset="100%" stop-color="#00AEEF"/>
    </linearGradient>

    <!-- Right Organic Wave (Emerald to Lime Checkmark) -->
    <linearGradient id="minCheckGrad" x1="0%" y1="100%" x2="100%" y2="0%">
      <stop offset="0%" stop-color="#059669"/>
      <stop offset="60%" stop-color="#10B981"/>
      <stop offset="100%" stop-color="#84CC16"/>
    </linearGradient>

    <!-- Soft Ambient Breathing Shadow -->
    <filter id="cleanShadow" x="-20%" y="-20%" width="140%" height="140%">
      <feDropShadow dx="0" dy="12" stdDeviation="16" flood-color="#0284C7" flood-opacity="0.18"/>
    </filter>
  </defs>

  <g filter="url(#cleanShadow)">
    <!-- Stroke 1 (Left): Smooth Natural Spine / Leaf Arcing Upwards -->
    <path d="M 160 416
             C 156 312, 172 212, 216 138
             C 248 84, 296 58, 340 64
             C 316 90, 288 132, 264 190
             C 236 256, 218 336, 208 416
             C 208 426, 160 426, 160 416 Z" 
          fill="url(#minStemGrad)"/>

    <!-- Stroke 2 (Right): Graceful Organic Checkmark (✓) & Loop of 'P' -->
    <path d="M 200 282
             C 226 288, 258 298, 290 268
             C 340 222, 380 154, 412 94
             C 404 150, 372 242, 310 318
             C 276 360, 238 366, 208 340
             C 192 326, 190 300, 200 282 Z" 
          fill="url(#minCheckGrad)"/>

    <!-- Single Digital Sensor Node (Data Precision) -->
    <circle cx="412" cy="94" r="10" fill="#84CC16"/>
    <circle cx="412" cy="94" r="4" fill="#FFFFFF"/>
  </g>
</svg>'''

    # 2. Master Horizontal Lockup (Emblem + Wordmark) - 640x160
    svg_horizontal = '''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 640 160" width="640" height="160" fill="none">
  <defs>
    <linearGradient id="hStemGrad" x1="0%" y1="100%" x2="50%" y2="0%">
      <stop offset="0%" stop-color="#0B3B95"/>
      <stop offset="60%" stop-color="#0284C7"/>
      <stop offset="100%" stop-color="#00AEEF"/>
    </linearGradient>

    <linearGradient id="hCheckGrad" x1="0%" y1="100%" x2="100%" y2="0%">
      <stop offset="0%" stop-color="#059669"/>
      <stop offset="60%" stop-color="#10B981"/>
      <stop offset="100%" stop-color="#84CC16"/>
    </linearGradient>

    <linearGradient id="hTextGrad" x1="0%" y1="0%" x2="100%" y2="0%">
      <stop offset="0%" stop-color="#0B3B95"/>
      <stop offset="100%" stop-color="#0284C7"/>
    </linearGradient>
  </defs>

  <!-- Left: The Minimalist Natural Curve Emblem (Scaled to 120x120 at X=16, Y=20) -->
  <g transform="translate(16, 16) scale(0.24)">
    <path d="M 160 416
             C 156 312, 172 212, 216 138
             C 248 84, 296 58, 340 64
             C 316 90, 288 132, 264 190
             C 236 256, 218 336, 208 416
             C 208 426, 160 426, 160 416 Z" 
          fill="url(#hStemGrad)"/>

    <path d="M 200 282
             C 226 288, 258 298, 290 268
             C 340 222, 380 154, 412 94
             C 404 150, 372 242, 310 318
             C 276 360, 238 366, 208 340
             C 192 326, 190 300, 200 282 Z" 
          fill="url(#hCheckGrad)"/>

    <circle cx="412" cy="94" r="10" fill="#84CC16"/>
    <circle cx="412" cy="94" r="4" fill="#FFFFFF"/>
  </g>

  <!-- Subtle Elegant Divider -->
  <line x1="148" y1="36" x2="148" y2="124" stroke="#E2E8F0" stroke-width="1.5" stroke-linecap="round"/>

  <!-- Typography: Pure, Spacious, Modern Minimalist -->
  <g transform="translate(168, 0)">
    <!-- Primary Brand Wordmark: PindaiSurvei -->
    <text x="0" y="72" font-family="system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif" font-size="38" font-weight="700" letter-spacing="-0.03em" fill="#0F172A">
      Pindai<tspan fill="url(#hTextGrad)">Survei</tspan>
    </text>

    <!-- Subtitle Category Tagline -->
    <text x="2" y="102" font-family="system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif" font-size="13" font-weight="600" letter-spacing="0.08em" fill="#0284C7">
      SISTEM DIGITALISASI DOKUMEN SURVEI
    </text>

    <!-- Institutional Endorsement -->
    <text x="2" y="122" font-family="system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif" font-size="11.5" font-weight="500" letter-spacing="0.02em" fill="#64748B">
      Badan Pusat Statistik Kab. Penajam Paser Utara
    </text>
  </g>
</svg>'''

    # 3. Clean Favicon (64x64)
    svg_favicon = '''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 512 512" width="64" height="64" fill="none">
  <defs>
    <linearGradient id="favStem" x1="0%" y1="100%" x2="50%" y2="0%">
      <stop offset="0%" stop-color="#0B3B95"/>
      <stop offset="100%" stop-color="#00AEEF"/>
    </linearGradient>
    <linearGradient id="favCheck" x1="0%" y1="100%" x2="100%" y2="0%">
      <stop offset="0%" stop-color="#059669"/>
      <stop offset="100%" stop-color="#84CC16"/>
    </linearGradient>
  </defs>
  <path d="M 160 416 C 156 312, 172 212, 216 138 C 248 84, 296 58, 340 64 C 316 90, 288 132, 264 190 C 236 256, 218 336, 208 416 C 208 426, 160 426, 160 416 Z" fill="url(#favStem)"/>
  <path d="M 200 282 C 226 288, 258 298, 290 268 C 340 222, 380 154, 412 94 C 404 150, 372 242, 310 318 C 276 360, 238 366, 208 340 C 192 326, 190 300, 200 282 Z" fill="url(#favCheck)"/>
  <circle cx="412" cy="94" r="14" fill="#84CC16"/>
  <circle cx="412" cy="94" r="5" fill="#FFFFFF"/>
</svg>'''

    with open('static/img/pindaisurvei_logo.svg', 'w', encoding='utf-8') as f:
        f.write(svg_square)
    with open('static/img/pindaisurvei_logo_horizontal.svg', 'w', encoding='utf-8') as f:
        f.write(svg_horizontal)
    with open('static/img/favicon.svg', 'w', encoding='utf-8') as f:
        f.write(svg_favicon)

    print("Updated master static/img SVGs with minimalist organic mark.")

if __name__ == '__main__':
    create_master_minimal_svgs()
