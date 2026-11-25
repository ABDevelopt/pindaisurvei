import math

def generate_minimalist_logos():
    # -------------------------------------------------------------
    # OPTION A: Monogram "P" Daun Organik (The Organic Leaf 'P')
    # A single continuous, sweeping organic curve that loops to form 'P'
    # -------------------------------------------------------------
    svg_opt_a = '''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 512 512" width="512" height="512" fill="none">
  <defs>
    <linearGradient id="gradLeafA" x1="0%" y1="100%" x2="100%" y2="0%">
      <stop offset="0%" stop-color="#0284C7"/>
      <stop offset="50%" stop-color="#00AEEF"/>
      <stop offset="100%" stop-color="#10B981"/>
    </linearGradient>
    <linearGradient id="scanGlowA" x1="0%" y1="0%" x2="100%" y2="0%">
      <stop offset="0%" stop-color="#00AEEF" stop-opacity="0.2"/>
      <stop offset="100%" stop-color="#10B981"/>
    </linearGradient>
    <filter id="softShadowA" x="-20%" y="-20%" width="140%" height="140%">
      <feDropShadow dx="0" dy="12" stdDeviation="16" flood-color="#0284C7" flood-opacity="0.2"/>
    </filter>
  </defs>

  <!-- Clean Minimalist Monogram Leaf 'P' -->
  <g filter="url(#softShadowA)">
    <!-- Main Sweeping Organic Ribbon forming Stem and Loop of 'P' -->
    <path d="M 180 420 
             C 176 330, 180 230, 184 170
             C 188 110, 240 82, 305 88
             C 365 94, 396 142, 388 200
             C 380 256, 332 292, 260 292
             C 216 292, 196 280, 184 270
             L 184 316
             C 200 326, 228 334, 264 334
             C 360 334, 432 276, 442 196
             C 452 112, 400 48, 300 42
             C 204 36, 136 86, 132 176
             C 128 244, 126 340, 128 420
             C 128 436, 180 436, 180 420 Z" 
          fill="url(#gradLeafA)"/>

    <!-- Inner Organic Scanner Eye / Sprout Leaf Core (Negative Space Accent) -->
    <path d="M 230 180 
             C 230 148, 258 132, 292 136
             C 324 140, 340 166, 336 196
             C 332 224, 308 244, 274 244
             C 246 244, 230 228, 230 180 Z" 
          fill="#00AEEF" opacity="0.18"/>

    <!-- Subtle Single Optical Scan Dot (Sensor Node) -->
    <circle cx="340" cy="196" r="9" fill="#10B981"/>
    <circle cx="340" cy="196" r="3.5" fill="#FFFFFF"/>
  </g>
</svg>'''

    # -------------------------------------------------------------
    # OPTION B: Dua Lengkungan Daun & Centang (Twin Leaf Checkmark)
    # 2 graceful flowing strokes: Left stem (Blue/Cyan) + Right checkmark leaf (Green/Gold)
    # Extremely clean, intuitive verification + scanning + nature
    # -------------------------------------------------------------
    svg_opt_b = '''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 512 512" width="512" height="512" fill="none">
  <defs>
    <!-- Left Curve: BPS Deep Blue to Sky Cyan -->
    <linearGradient id="gradLeftB" x1="0%" y1="100%" x2="50%" y2="0%">
      <stop offset="0%" stop-color="#0B3B95"/>
      <stop offset="60%" stop-color="#0284C7"/>
      <stop offset="100%" stop-color="#00AEEF"/>
    </linearGradient>

    <!-- Right Curve: Fresh Green to Warm Lime -->
    <linearGradient id="gradRightB" x1="0%" y1="100%" x2="100%" y2="0%">
      <stop offset="0%" stop-color="#059669"/>
      <stop offset="50%" stop-color="#10B981"/>
      <stop offset="100%" stop-color="#84CC16"/>
    </linearGradient>

    <!-- Drop Shadow -->
    <filter id="softB" x="-20%" y="-20%" width="140%" height="140%">
      <feDropShadow dx="0" dy="10" stdDeviation="14" flood-color="#0284C7" flood-opacity="0.18"/>
    </filter>
  </defs>

  <g filter="url(#softB)">
    <!-- Stroke 1 (Left): Elegant Upward Arcing Leaf / Open Document Spine (Forms Left of 'P') -->
    <path d="M 148 412
             C 144 310, 160 210, 204 136
             C 236 82, 284 56, 328 62
             C 304 88, 276 130, 252 188
             C 224 254, 206 334, 196 412
             C 196 422, 148 422, 148 412 Z" 
          fill="url(#gradLeftB)"/>

    <!-- Stroke 2 (Right): Dynamic Flowing Leaf that forms the Verification Checkmark (✓) and loop of 'P' -->
    <path d="M 188 280
             C 214 286, 246 296, 278 266
             C 328 220, 368 152, 400 92
             C 392 148, 360 240, 298 316
             C 264 358, 226 364, 196 338
             C 180 324, 178 298, 188 280 Z" 
          fill="url(#gradRightB)"/>

    <!-- Single Digital Data Accent: Precision Sensor Node at the Apex -->
    <circle cx="400" cy="92" r="10" fill="#84CC16"/>
    <circle cx="400" cy="92" r="4" fill="#FFFFFF"/>
  </g>
</svg>'''

    # -------------------------------------------------------------
    # OPTION C: Tiga Lengkungan Pohon Hayat Minimalis (Tri-Flow Sprout)
    # Exactly 3 clean organic curved petals: Left (Blue), Center (Emerald), Right (Amber)
    # Balanced, symmetrical, serene, ample breathing room
    # -------------------------------------------------------------
    svg_opt_c = '''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 512 512" width="512" height="512" fill="none">
  <defs>
    <!-- Left Petal Gradient (Blue to Cyan) -->
    <linearGradient id="gradLeftC" x1="0%" y1="100%" x2="0%" y2="0%">
      <stop offset="0%" stop-color="#0B3B95"/>
      <stop offset="100%" stop-color="#00AEEF"/>
    </linearGradient>

    <!-- Center Petal Gradient (Emerald to Lime) -->
    <linearGradient id="gradCenterC" x1="0%" y1="100%" x2="0%" y2="0%">
      <stop offset="0%" stop-color="#0284C7"/>
      <stop offset="50%" stop-color="#10B981"/>
      <stop offset="100%" stop-color="#F59E0B"/>
    </linearGradient>

    <!-- Right Petal Gradient (Cyan to Emerald) -->
    <linearGradient id="gradRightC" x1="0%" y1="100%" x2="0%" y2="0%">
      <stop offset="0%" stop-color="#0B3B95"/>
      <stop offset="100%" stop-color="#00AEEF"/>
    </linearGradient>

    <filter id="softC" x="-20%" y="-20%" width="140%" height="140%">
      <feDropShadow dx="0" dy="12" stdDeviation="16" flood-color="#0B3B95" flood-opacity="0.2"/>
    </filter>
  </defs>

  <g filter="url(#softC)">
    <!-- Base Crescent: Open Document Curve / Scanner Horizon -->
    <path d="M 160 410 C 220 435, 292 435, 352 410 C 310 422, 202 422, 160 410 Z" 
          fill="#0B3B95" opacity="0.6"/>

    <!-- Left Flowing Leaf / Branch (Graceful Organic Curve) -->
    <path d="M 240 390
             C 200 360, 140 280, 140 190
             C 140 140, 170 100, 190 85
             C 178 120, 175 180, 205 240
             C 225 280, 245 340, 240 390 Z" 
          fill="url(#gradLeftC)"/>
    <circle cx="190" cy="85" r="9" fill="#00AEEF"/>
    <circle cx="190" cy="85" r="3.5" fill="#FFFFFF"/>

    <!-- Right Flowing Leaf / Branch (Symmetrical Balance) -->
    <path d="M 272 390
             C 312 360, 372 280, 372 190
             C 372 140, 342 100, 322 85
             C 334 120, 337 180, 307 240
             C 287 280, 267 340, 272 390 Z" 
          fill="url(#gradRightC)"/>
    <circle cx="322" cy="85" r="9" fill="#00AEEF"/>
    <circle cx="322" cy="85" r="3.5" fill="#FFFFFF"/>

    <!-- Center Spine / Growing Leaf & Scanner Laser (Vertical Ascendance) -->
    <path d="M 256 60
             C 272 120, 274 240, 262 370
             C 256 372, 250 370, 250 370
             C 238 240, 240 120, 256 60 Z" 
          fill="url(#gradCenterC)"/>
    <circle cx="256" cy="60" r="10" fill="#F59E0B"/>
    <circle cx="256" cy="60" r="4" fill="#FFFFFF"/>
  </g>
</svg>'''

    # Write Option files
    with open('static/img/pindaisurvei_minimal_a.svg', 'w', encoding='utf-8') as f:
        f.write(svg_opt_a)
    with open('static/img/pindaisurvei_minimal_b.svg', 'w', encoding='utf-8') as f:
        f.write(svg_opt_b)
    with open('static/img/pindaisurvei_minimal_c.svg', 'w', encoding='utf-8') as f:
        f.write(svg_opt_c)

    print("Saved Option A, B, and C SVGs successfully.")

if __name__ == '__main__':
    generate_minimalist_logos()
