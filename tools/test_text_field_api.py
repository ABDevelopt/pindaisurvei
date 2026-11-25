import urllib.request
import json

def test_api():
    for p in [1, 2, 7]:
        url = f"http://127.0.0.1:8000/api/booklets/booklet_01/pages/{p}/boxes"
        req = urllib.request.urlopen(url)
        d = json.loads(req.read().decode("utf-8"))
        tf = d.get("layers", {}).get("text_fields", [])
        cnt = d.get("counts", {}).get("text_fields", 0)
        print(f"Page {p} boxes: count={cnt}, actual={len(tf)}")
        if tf:
            f0 = tf[0]
            print(f"  Field 0: id={f0.get('id')} label={f0.get('label')} cell_id={f0.get('cell_id')} crop_url={f0.get('crop_url')}")

    # Test overlay
    url_ov = "http://127.0.0.1:8000/api/booklets/booklet_01/overlay/1?type=text"
    req_ov = urllib.request.urlopen(url_ov)
    print(f"Text Overlay: status={req_ov.status}, Content-Type={req_ov.headers.get('Content-Type')}, length={len(req_ov.read())}")

    # Test crops Tier 8
    url_cr = "http://127.0.0.1:8000/api/booklets/booklet_01/crops/8?page=1"
    req_cr = urllib.request.urlopen(url_cr)
    d_cr = json.loads(req_cr.read().decode("utf-8"))
    crops = d_cr.get("crops", [])
    print(f"Tier 8 Crops (Page 1): count={len(crops)}")
    if crops:
        print(f"  Crop 0: id={crops[0].get('id')}, label={crops[0].get('label')}, url={crops[0].get('crop_url')}")

if __name__ == "__main__":
    test_api()
