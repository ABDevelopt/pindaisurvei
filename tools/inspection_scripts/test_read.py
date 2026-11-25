
import joblib
clf = joblib.load(r'd:_Kedinasan_BPS\Pengenalan Kuesioner (OCR)\model.joblib')
print('Model type:', type(clf))
if hasattr(clf, 'n_features_in_'):
    print('n_features_in_:', clf.n_features_in_)
if hasattr(clf, 'classes_'):
    print('classes_:', clf.classes_)
